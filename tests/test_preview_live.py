"""Team Preview live: API por lotes del entorno, procedencia mock/live y
entrenadores de preview recorriendo el camino live (pool simulado o real)."""
import os
import sys

sys.path.insert(0, '.')

import numpy as np
import pytest

from showdown.rl.team_preview_action import action_to_team_string
from showdown.rl.team_preview_env import TeamPreviewEnv
from showdown.rl.team_preview_obs import OBS_DIM


class FakePool:
    """Sustituto de LivePreviewPool: registra los BO1 pedidos y devuelve Cand. 1."""

    pilot = 'greedy_joint'
    num_workers = 2
    max_turns = 40

    def __init__(self, *args, **kwargs):
        self.matches = []
        self.closed = False
        self._rng = np.random.default_rng(0)

    def run(self, matches):
        self.matches.extend(matches)
        return self._rng.choice([-1.0, 0.0, 1.0], size=len(matches)).astype(np.float32)

    def summary(self):
        n = len(self.matches)
        return {'pilot': self.pilot, 'workers': self.num_workers, 'battles': n, 'wins': 0,
                'losses': 0, 'ties': n, 'capped': 0, 'failures': 0, 'retries': 0,
                'mean_turns': 0.0, 'battles_per_s': 0.0, 'worker_restarts': 0}

    def close(self):
        self.closed = True


# --------------------------------------------------------------------------- #
# API por lotes (backend mock)
# --------------------------------------------------------------------------- #

def test_reset_batch_matches_sequential_resets():
    a = TeamPreviewEnv(seed=7)
    b = TeamPreviewEnv(seed=7)
    obs_batch, infos = a.reset_batch(5)
    seq = [b.reset()[0] for _ in range(5)]
    assert obs_batch.shape == (5, OBS_DIM)
    np.testing.assert_allclose(obs_batch, np.stack(seq))
    assert all(info['action_mask'].shape == (360,) for info in infos)


def test_step_and_pair_label_batches_are_well_formed():
    env = TeamPreviewEnv(seed=3)
    env.reset_batch(8)
    r = env.step_batch(list(range(8)))
    assert r.shape == (8,) and set(np.unique(r)).issubset({-1.0, 0.0, 1.0})
    env.reset_batch(6)
    lab = env.pair_label_batch([0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0])
    assert set(np.unique(lab)).issubset({0.0, 0.5, 1.0})
    with pytest.raises(ValueError):
        env.step_batch([0, 1])  # 2 acciones para 6 contextos


def test_mock_provenance_is_never_reported_as_live():
    env = TeamPreviewEnv(seed=0)
    assert env.provenance == 'mock'
    env.reset()
    _o, _r, _t, _tr, info = env.step(0)
    assert info['pi_c'] == 'mock_proxy'


def test_live_backend_requires_a_pool():
    with pytest.raises(ValueError):
        TeamPreviewEnv(seed=0, backend='live')


# --------------------------------------------------------------------------- #
# Camino live con pool simulado
# --------------------------------------------------------------------------- #

def test_live_env_uses_real_teams_and_forwards_orders():
    pool = FakePool()
    env = TeamPreviewEnv(seed=1, backend='live', live_pool=pool)
    assert env.provenance == 'live' and env.team_source == 'metavgc'
    obs, infos = env.reset_batch(4)
    assert obs.shape == (4, OBS_DIM)
    assert all(info['our_team'] != info['foe_team'] for info in infos)
    acts = [0, 59, 120, 359]
    r = env.step_batch(acts)
    assert r.shape == (4,)
    assert [m.our_order for m in pool.matches] == [action_to_team_string(a) for a in acts]
    assert all(m.foe_order is None for m in pool.matches)  # rival: preview propio de π_c
    assert all('Ability:' in m.our_export and 'Ability:' in m.foe_export for m in pool.matches)

    pool.matches.clear()
    env.reset_batch(3)
    lab = env.pair_label_batch([0, 1, 2], [10, 11, 12])
    assert set(np.unique(lab)).issubset({0.0, 0.5, 1.0})
    assert [m.foe_order for m in pool.matches] == [action_to_team_string(j) for j in (10, 11, 12)]


@pytest.mark.parametrize('trainer', ['ppo', 'nash', 'qd'])
def test_preview_trainers_live_path_records_provenance(trainer, tmp_path, monkeypatch):
    import json

    from showdown.rl import preview_live

    monkeypatch.setattr(preview_live, 'LivePreviewPool', FakePool)
    live = {'backend': 'live', 'num_workers': 2}
    if trainer == 'ppo':
        from showdown.rl.team_preview_ppo.train import train
        train(seed=0, smoke=True, updates=1, checkpoint_dir=tmp_path, live=live)
    elif trainer == 'nash':
        from showdown.rl.team_preview_nash.train import train
        train(seed=0, smoke=True, updates=1, checkpoint_dir=tmp_path, live=live)
    else:
        from showdown.rl.team_preview_qd_pga.train import train_and_save
        train_and_save(seed=0, smoke=True, checkpoint_dir=tmp_path, live=live)
    meta = json.loads((tmp_path / 'train_meta.json').read_text(encoding='utf-8'))
    assert meta['provenance'] == 'live'
    assert meta['live']['battles'] > 0
    assert 'provenance: live' in (tmp_path / 'reporte.txt').read_text(encoding='utf-8')


def test_preview_trainer_mock_reports_mock(tmp_path):
    import json

    from showdown.rl.team_preview_ppo.train import train

    train(seed=0, smoke=True, updates=1, checkpoint_dir=tmp_path)
    meta = json.loads((tmp_path / 'train_meta.json').read_text(encoding='utf-8'))
    assert meta['provenance'] == 'mock' and 'live' not in meta


def test_pool_rejects_unknown_pilot():
    from showdown.rl.preview_live import LivePreviewPool

    with pytest.raises(ValueError):
        LivePreviewPool(1, pilot='random')


# --------------------------------------------------------------------------- #
# BO1 reales (opcional: IAPSVGC_LIVE_TESTS=1, requiere Showdown compilado)
# --------------------------------------------------------------------------- #

_LIVE = os.environ.get('IAPSVGC_LIVE_TESTS', '').strip() in ('1', 'true', 'yes')


@pytest.mark.skipif(not _LIVE, reason='define IAPSVGC_LIVE_TESTS=1 para jugar BO1 reales')
def test_real_bo1_pool_plays_and_reports():
    from showdown.inference_sim.team_factory import active_teams, to_showdown_export
    from showdown.rl.preview_live import LivePreviewPool, PreviewMatch

    teams = active_teams()
    matches = [
        PreviewMatch(to_showdown_export(teams[0]), to_showdown_export(teams[1]), our_order='1234'),
        PreviewMatch(to_showdown_export(teams[2]), to_showdown_export(teams[3]),
                     our_order='5612', foe_order='3412'),
    ]
    port = int(os.environ.get('IAPSVGC_LIVE_TEST_PORT', '8990'))
    with LivePreviewPool(2, envs_per_server=2, base_port=port, start_ps=True) as pool:
        r = pool.run(matches)
        s = pool.summary()
    assert r.shape == (2,) and set(np.unique(r)).issubset({-1.0, 0.0, 1.0})
    assert s['battles'] == 2 and s['failures'] == 0
