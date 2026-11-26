"""QD-PPO 2.0 (agente I 2.0, docs/qd_ppo_spec.md §I 2.0): archivo, enfrentamientos pareados, entrenador."""
from __future__ import annotations

import json

import numpy as np
import pytest

from showdown.rl.qd_ppo.archive import Archive
from showdown.rl.qd_ppo.matchups import MatchResults, make_matchups, paired_diff, play_matchups
from showdown.rl.qd_ppo.mosaic import MOSAICS, make_sampler
from showdown.rl.qd_ppo.train2 import calibrate_bounds


def test_archive_bounds_accumulated_fitness_and_qd_score(tmp_path):
    ar = Archive(bounds=((0.6, 1.0), (0.0, 0.2)))
    assert ar.key_of(0.8, 0.1) == (5, 5)
    assert ar.key_of(0.1, 0.9) == (0, 9)          # fuera de límites: celda del borde
    assert ar.try_insert(bd1=0.8, bd2=0.1, fitness=0.0, state_dict={'w': np.ones(1)},
                         returns=[1.0, -1.0, 1.0, 1.0])
    e = ar.cells[(5, 5)]
    assert e.n_eval == 4 and e.fitness == pytest.approx(0.5)
    e.add_games([-1.0, -1.0, -1.0, -1.0])
    assert e.n_eval == 8 and e.fitness == pytest.approx(-0.25)
    assert ar.qd_score() == pytest.approx(0.75)    # Σ (f + 1)
    assert not ar.try_insert(bd1=0.8, bd2=0.1, fitness=-0.5, state_dict={'w': np.zeros(1)})
    assert ar.try_insert(bd1=0.8, bd2=0.1, fitness=-0.5, state_dict={'w': np.zeros(1)}, force=True)
    ar.save(tmp_path / 'a.pt')
    br = Archive()
    br.load(tmp_path / 'a.pt')
    assert br.bounds == ((0.6, 1.0), (0.0, 0.2)) and br.cells[(5, 5)].fitness == -0.5


def test_calibrate_bounds_percentiles_and_min_width():
    rng = np.random.default_rng(0)
    bds = [(0.8 + 0.1 * rng.random(), 0.05 * rng.random()) for _ in range(1000)]
    (lo1, hi1), (lo2, hi2) = calibrate_bounds(bds)
    assert 0.80 <= lo1 < 0.82 and 0.88 < hi1 <= 0.90
    assert hi2 - lo2 >= 0.045
    (a, b), _ = calibrate_bounds([(0.9, 0.0)] * 50)
    assert b - a == pytest.approx(0.05)


def test_paired_diff_ignores_crashes():
    a = MatchResults([1.0, None, -1.0, 1.0])
    b = MatchResults([-1.0, 1.0, -1.0, None])
    assert paired_diff(a, b) == pytest.approx(1.0)   # solo partidas 0 y 2
    assert a.fitness == pytest.approx(1 / 3)


def test_mosaic_samplers():
    rng = np.random.default_rng(1)
    s = make_sampler('ppo_v2')
    draws = [s(rng) for _ in range(4000)]
    foes, probs = MOSAICS['ppo_v2']
    for f, p in zip(foes, probs):
        assert abs(draws.count(f) / 4000 - p) < 0.03
    m = make_matchups(rng, 5, make_sampler('heur'), split='val')
    assert len(m) == 5 and all(x['split'] == 'val' and x['opponent'] != 'random' for x in m)


class _FakeVenv:
    """Cada partida dura 1 + seed % 4 pasos y devuelve +1 si seed es par; el worker 1 cae una vez."""

    def __init__(self, n):
        self.num_envs = n
        self.cur = {}
        self.left = {}
        self.pending = []
        self.started = []
        self.crashed = False

    def set_reward(self, _c):
        pass

    def _start(self, i, m):
        self.cur[i] = m
        self.left[i] = 1 + int(m['seed']) % 4
        self.started.append(int(m['seed']))

    def start_matches(self, idx, matches):
        for i, m in zip(idx, matches):
            self._start(int(i), m)

    def last_obs_mask(self, idx):
        return np.zeros((len(idx), 3)), np.ones((len(idx), 2))

    def mask2_subset(self, idx, a1s):
        return np.ones((len(idx), 2))

    def step_async(self, idx, actions, on_done):
        self.pending.extend(zip(idx, on_done))   # como el entorno real: se acumulan hasta el wait

    def drain(self):
        out = []
        for i, nxt in self.pending:
            self.left[i] -= 1
            base = {'obs': np.zeros(3), 'mask': np.ones(2)}
            if self.left[i] > 0:
                out.append((i, dict(base, terminated=False, truncated=False, info={})))
                continue
            m = self.cur[i]
            if i == 1 and not self.crashed:      # caída: la partida no cuenta
                self.crashed = True
                info = {'trunc_reason': 'crash', 'episode_return': 0.0}
                out.append((i, dict(base, terminated=False, truncated=True, info=info, worker_restarted=True)))
                continue
            info = {'episode_return': 1.0 if int(m['seed']) % 2 == 0 else -1.0,
                    'archive_eligible': True, 'bd_raw': (int(m['seed']) / 100.0, 0.0)}
            if nxt is not None:
                self._start(i, nxt)
            out.append((i, dict(base, terminated=True, truncated=False, info=info)))
        self.pending = []
        return out


class _FakeAgent:
    def act_batch(self, obs, m1, m2_fn, greedy=True):
        k = len(obs)
        return np.zeros(k, int), np.zeros(k, int), np.zeros(k), np.zeros(k), np.ones((k, 2))


def test_play_vec_plays_each_match_once_in_order(monkeypatch):
    import showdown.rl.vec_rollout as vr
    monkeypatch.setattr(vr, 'is_vec_env', lambda e: True)
    matches = [{'opponent': 'greedy_joint', 'seed': s, 'split': None} for s in range(20)]
    venv = _FakeVenv(3)
    res = play_matchups(_FakeAgent(), venv, matches)
    assert sorted(venv.started) == list(range(20))           # cada enfrentamiento, una vez
    crashed = [k for k, r in enumerate(res.returns) if r is None]
    assert len(crashed) == 1                                  # la caída no cuenta como empate
    for k, r in enumerate(res.returns):
        if r is not None:
            assert r == (1.0 if k % 2 == 0 else -1.0)         # cada resultado en su índice
            assert res.bds[k] == (k / 100.0, 0.0)


def test_train2_smoke_qd_and_control(tmp_path, monkeypatch):
    monkeypatch.setenv('RL_OBS_CALC', '0')
    from showdown.rl.qd_ppo.train2 import run_training
    s = run_training(seed=0, smoke=True, checkpoint_dir=tmp_path / 'qd')
    assert s['children'] >= 1 and s['cells'] >= 1 and s['best_reeval'] is not None
    assert (tmp_path / 'qd' / 'best_elite.pt').is_file()
    rows = json.loads((tmp_path / 'qd' / 'reeval.json').read_text())
    assert rows and all('archive_fitness' in r for r in rows)
    c = run_training(seed=0, smoke=True, control=True, checkpoint_dir=tmp_path / 'ctrl')
    assert c['global_steps'] > 0 and (tmp_path / 'ctrl' / 'final.pt').is_file()


def test_sync_vec_start_match_and_next_match_dict(monkeypatch):
    monkeypatch.setenv('RL_OBS_CALC', '0')
    from showdown.rl.vec_env import EnvSpec, SyncVecVGCEnv
    venv = SyncVecVGCEnv([EnvSpec(opponent='mock'), EnvSpec(opponent='mock')], seed=0)
    try:
        venv.start_matches([0, 1], [{'opponent': None, 'seed': 11, 'split': None},
                                    {'opponent': None, 'seed': 12, 'split': None}])
        obs, mask = venv.last_obs_mask([0, 1])
        assert obs.shape[0] == 2 and mask.shape[0] == 2
        nxt = {'opponent': None, 'seed': 99, 'split': None}
        done = False
        for _ in range(200):
            m = venv.last_obs_mask([0])[1][0]
            a = int(np.flatnonzero(m)[0])
            venv.step_async([0], [(a, 0)], [nxt])
            (_i, r), = venv.drain()
            if r['terminated'] or r['truncated']:
                done = True
                assert 'obs' in r and r['next_opponent'] == 'mock'
                break
            venv._last_om[0] = (r['obs'], r['mask'])
        assert done
    finally:
        venv.close()


def test_bd2_focus_fire_default_and_switch_mode(monkeypatch):
    """BD2 desde I 2.0: foco de ataque (mismo rival en turnos con dos ataques de un solo objetivo)."""
    from showdown.rl.behavioral import BehavioralTracker
    from showdown.rl.snapshot import SlotAction
    monkeypatch.delenv('RL_BD2', raising=False)
    tr = BehavioralTracker()
    mv = lambda t: SlotAction(kind='move', move_index=0, target=t)  # noqa: E731
    tr.on_turn_actions([mv(0), mv(0)], ['Flare Blitz', 'Wood Hammer'])   # foco
    tr.on_turn_actions([mv(0), mv(1)], ['Flare Blitz', 'Wood Hammer'])   # reparto
    tr.on_turn_actions([mv(0), mv(1)], ['Rock Slide', 'Wood Hammer'])    # área: no cuenta
    tr.on_turn_actions([mv(1), mv(1)], ['Moonblast', 'Flare Blitz'])     # foco
    tr.on_turn_actions([mv(0), SlotAction(kind='switch', switch_index=0)], ['Flare Blitz', ''])
    assert tr.paired_attacks == 3 and tr.focus_turns == 2
    assert tr.bd_vector()[1] == pytest.approx(2 / 3)
    monkeypatch.setenv('RL_BD2', 'switch')
    assert tr.bd_vector()[1] == pytest.approx(1 / 5)


def test_play_multi_assigns_results_per_policy(monkeypatch):
    import showdown.rl.vec_rollout as vr
    from showdown.rl.qd_ppo.matchups import play_matchups_multi
    monkeypatch.setattr(vr, 'is_vec_env', lambda e: True)
    venv = _FakeVenv(3)
    venv.step_wait_any = venv.drain
    venv.crashed = True  # sin caídas en este test
    g0 = [{'opponent': 'x', 'seed': s, 'split': None} for s in (0, 1, 2, 3)]
    g1 = [{'opponent': 'x', 'seed': s, 'split': None} for s in (10, 11, 12)]
    r0, r1 = play_matchups_multi([_FakeAgent(), _FakeAgent()], venv, [g0, g1])
    assert sorted(venv.started) == [0, 1, 2, 3, 10, 11, 12]
    assert r0.returns == [1.0, -1.0, 1.0, -1.0] and r1.returns == [1.0, -1.0, 1.0]
    assert r1.bds[2] == (0.12, 0.0)


def test_evaluate_insert_batch_rules(monkeypatch):
    import showdown.rl.qd_ppo.batch_insert as bi
    from showdown.rl.qd_ppo.matchups import MatchResults
    ar = Archive(bounds=((0.0, 1.0), (0.0, 1.0)))
    ar.try_insert(bd1=0.15, bd2=0.15, fitness=0.0, state_dict={'id': 'inc'}, returns=[0.0] * 4)
    plan = {'a': [1.0] * 4, 'b': [-1.0] * 4, 'c': [1.0, 1.0, -1.0, 1.0], 'inc': [0.0] * 4}
    bds = {'a': (0.15, 0.15), 'b': (0.15, 0.15), 'c': (0.85, 0.85), 'inc': (0.15, 0.15)}

    def fake_multi(agents, env, groups, **kw):
        return [MatchResults(list(plan[a]), [bds[a]] * 4) for a in agents]
    monkeypatch.setattr(bi, 'play_matchups_multi', fake_multi)
    stats = {}
    out = bi.evaluate_insert_batch(ar, None, lambda sd: sd['id'],
                                   [({'id': 'a'}, {}), ({'id': 'b'}, {}), ({'id': 'c'}, {})],
                                   np.random.default_rng(0), lambda r: 'x', paired_games=4, g_min=1,
                                   stats=stats)
    assert [o['inserted'] for o in out] == [True, False, True]   # a gana al ocupante; c celda nueva
    assert ar.cells[(1, 1)].state_dict['id'] == 'a' and ar.cells[(8, 8)].state_dict['id'] == 'c'
    assert stats['replaced'] == 1 and stats['new_cells'] == 1
