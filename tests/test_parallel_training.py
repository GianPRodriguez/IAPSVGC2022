"""Entrenamiento paralelo: recursos, entornos vectorizados, PPO/DDQN en lote,
trials de tuning paralelos y combinación de shards del robust E2E."""
import sys

sys.path.insert(0, '.')

import numpy as np
import pytest

from showdown.rl import parallel
from showdown.rl.constants import JOINT_ACTION_N, OBS_DIM, PASS_ID, SLOT_ACTION_N
from showdown.rl.vec_env import EnvSpec, SubprocVecVGCEnv, SyncVecVGCEnv, make_vec_env


# --------------------------------------------------------------------------- #
# Recursos
# --------------------------------------------------------------------------- #

def test_plan_env_layout_groups_envs_per_server():
    lay = parallel.plan_env_layout(10, envs_per_server=4, base_port=9000)
    assert lay.ports == [9000, 9001, 9002]
    assert lay.env_ports == [9000] * 4 + [9001] * 4 + [9002] * 2
    assert lay.num_servers == 3


def test_default_num_envs_reserves_cores_and_caps():
    assert parallel.default_num_envs(32) == 28
    assert parallel.default_num_envs(12) == 8
    assert parallel.default_num_envs(2) == 1


def test_resolve_device_auto_and_explicit():
    assert parallel.resolve_device('cpu') == 'cpu'
    assert parallel.resolve_device('cuda:1') == 'cuda:1'
    assert parallel.resolve_device('auto') in ('cpu', 'cuda')


def test_raise_nofile_limit_never_lowers():
    try:
        import resource
    except ImportError:
        assert parallel.raise_nofile_limit() is None
        return
    soft_before, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    parallel.raise_nofile_limit()
    soft_after, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert soft_after >= soft_before


# --------------------------------------------------------------------------- #
# Entornos vectorizados (rival mock: sin Showdown)
# --------------------------------------------------------------------------- #

def _mock_specs(n, obs_kind='ppo', mask_kind='slot', turns=6):
    return [
        EnvSpec(opponent='mock', obs_kind=obs_kind, mask_kind=mask_kind,
                mock_max_turns=turns, account_prefix=f'T{i}')
        for i in range(n)
    ]


def _first_legal(mask):
    legal = np.flatnonzero(np.asarray(mask) > 0.5)
    return int(legal[0]) if legal.size else PASS_ID


@pytest.mark.parametrize('cls', [SyncVecVGCEnv, SubprocVecVGCEnv])
def test_vec_env_reset_step_and_autoreset(cls):
    venv = cls(_mock_specs(3, turns=4), seed=1)
    try:
        obs, m1 = venv.reset()
        assert obs.shape == (3, OBS_DIM)
        assert m1.shape == (3, SLOT_ACTION_N)
        dones = 0
        for _ in range(30):
            a1 = [_first_legal(m) for m in m1]
            m2 = venv.mask2(a1)
            a2 = [_first_legal(m) for m in m2]
            res = venv.step(list(zip(a1, a2)))
            assert len(res) == 3
            for r in res:
                assert r['obs'].shape == (OBS_DIM,)
                if r['terminated'] or r['truncated']:
                    dones += 1
                    assert r['final_obs'].shape == (OBS_DIM,)
            obs = np.stack([r['obs'] for r in res])
            m1 = np.stack([r['mask'] for r in res])
        assert dones >= 3  # cada entorno terminó y se reinició solo
    finally:
        venv.close()


@pytest.mark.parametrize('cls', [SyncVecVGCEnv, SubprocVecVGCEnv])
def test_vec_env_async_step_and_drain(cls):
    venv = cls(_mock_specs(4, obs_kind='ddqn', mask_kind='joint'), seed=2)
    try:
        obs, masks = venv.reset()
        assert masks.shape == (4, JOINT_ACTION_N)
        from showdown.rl.action_space import decode_joint_action

        acts = [decode_joint_action(_first_legal(m)) for m in masks]
        venv.step_async([0, 1, 2, 3], acts)
        got = venv.step_wait_any()
        got += venv.drain()
        assert sorted(i for i, _ in got) == [0, 1, 2, 3]
        assert venv.in_flight() == []
    finally:
        venv.close()


def test_vec_env_on_done_opponent_switch_is_applied():
    venv = SyncVecVGCEnv(_mock_specs(1, turns=2), seed=3)
    try:
        _obs, m1 = venv.reset()
        switched = False
        for _ in range(10):
            a1 = _first_legal(m1[0])
            a2 = _first_legal(venv.mask2([a1])[0])
            # Rival mock -> mock: el flujo de cambio por episodio no debe romper.
            res = venv.step([(a1, a2)], ['mock'])[0]
            m1 = res['mask'][None]
            if res['terminated'] or res['truncated']:
                switched = res.get('next_opponent') == 'mock'
                break
        assert switched
    finally:
        venv.close()


def test_subproc_worker_crash_is_restarted_and_reported():
    venv = SubprocVecVGCEnv(_mock_specs(2), seed=4, step_timeout_s=30)
    try:
        venv.reset()
        venv._procs[0].kill()  # simula worker muerto
        venv._procs[0].join(timeout=5)
        res = venv.step([(PASS_ID, PASS_ID), (PASS_ID, PASS_ID)])
        assert res[0]['info']['trunc_reason'] == 'crash'
        assert res[0]['info']['skip_buffer'] is True
        assert res[0].get('worker_restarted') is True
        assert venv.restarts[0] == 1
        # El worker relanzado sigue funcionando.
        res2 = venv.step([(PASS_ID, PASS_ID), (PASS_ID, PASS_ID)])
        assert res2[0]['obs'].shape == (OBS_DIM,)
    finally:
        venv.close()


def test_make_vec_env_assigns_unique_accounts():
    venv = make_vec_env(EnvSpec(opponent='mock', account_prefix='Ppo'), num_envs=3,
                        subprocess=False)
    try:
        assert [s.account_prefix for s in venv.specs] == ['Ppo0', 'Ppo1', 'Ppo2']
        assert all(s.port is None for s in venv.specs)  # mock: sin Showdown
    finally:
        venv.close()


# --------------------------------------------------------------------------- #
# PPO / DDQN en lote
# --------------------------------------------------------------------------- #

torch = pytest.importorskip('torch')


def _fill(buf, rewards, values, terms):
    for r, v, t in zip(rewards, values, terms):
        buf.add(obs=np.zeros(OBS_DIM), a1=0, a2=0, logp=0.0, reward=r, value=v,
                terminated=t, m1=np.ones(SLOT_ACTION_N), m2=np.ones(SLOT_ACTION_N))


def test_vec_rollout_buffer_gae_matches_independent_streams():
    from showdown.rl.ppo.buffer import RolloutBuffer, VecRolloutBuffer

    streams = [
        ([1.0, 0.0, 2.0], [0.5, 0.2, 0.1], [False, False, True]),
        ([0.0, 1.0], [0.3, 0.4], [False, False]),
    ]
    last_values = [0.0, 0.7]
    vec = VecRolloutBuffer(2, capacity=8)
    singles = []
    for k, (r, v, t) in enumerate(streams):
        for rr, vv, tt in zip(r, v, t):
            vec.add(env_index=k, obs=np.zeros(OBS_DIM), a1=0, a2=0, logp=0.0, reward=rr,
                    value=vv, terminated=tt, m1=np.ones(SLOT_ACTION_N),
                    m2=np.ones(SLOT_ACTION_N))
        b = RolloutBuffer(capacity=8)
        _fill(b, r, v, t)
        b.compute_gae(last_value=last_values[k], gamma=0.9, lam=0.8, normalize_adv=False)
        singles.append(b)
    vec.compute_gae(last_value=last_values, gamma=0.9, lam=0.8, normalize_adv=False)
    for k, b in enumerate(singles):
        n = len(b)
        np.testing.assert_allclose(vec.buffers[k]._adv[:n], b._adv[:n], rtol=1e-6)
        np.testing.assert_allclose(vec.buffers[k]._ret[:n], b._ret[:n], rtol=1e-6)
    assert len(vec) == 5
    # Con normalización global: media 0 y desviación 1 sobre todo el lote.
    vec.compute_gae(last_value=last_values, gamma=0.9, lam=0.8, normalize_adv=True)
    adv = vec.get()['adv']
    assert abs(float(adv.mean())) < 1e-5
    assert abs(float(adv.std()) - 1.0) < 1e-4


def test_ppo_act_batch_respects_masks():
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig

    agent = PPOAgent(PPOConfig(seed=0))
    n = 5
    obs = np.random.default_rng(0).normal(size=(n, OBS_DIM)).astype(np.float32)
    m1 = np.zeros((n, SLOT_ACTION_N), dtype=np.float32)
    m1[:, 3] = 1.0
    m1[2, :] = 0.0  # sin acciones legales -> PASS
    m2_rows = np.zeros((n, SLOT_ACTION_N), dtype=np.float32)
    m2_rows[:, 5] = 1.0
    a1, a2, logp, value, m2 = agent.act_batch(obs, m1, lambda a1s: m2_rows)
    assert a1.shape == a2.shape == logp.shape == value.shape == (n,)
    assert all(a1[i] == 3 for i in (0, 1, 3, 4))
    assert a1[2] == PASS_ID
    assert (a2 == 5).all()
    assert np.isfinite(logp).all()


def test_ppo_use_vec_buffer_rollout_full_by_total():
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig

    agent = PPOAgent(PPOConfig(seed=0, rollout_steps=6))
    agent.use_vec_buffer(3)
    for step in range(2):
        for k in range(3):
            agent.observe(obs=np.zeros(OBS_DIM), a1=0, a2=0, logp=0.0, reward=0.1,
                          value=0.0, terminated=False, m1=np.ones(SLOT_ACTION_N),
                          m2=np.ones(SLOT_ACTION_N), env_index=k)
    assert agent.is_rollout_full()
    metrics = agent.update(last_value=np.zeros(3))
    assert 'loss' in metrics
    assert len(agent.buffer) == 0


def test_ddqn_act_batch_only_legal_actions():
    from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig

    agent = DDQNAgent(DDQNConfig(seed=0, warmup_steps=0, eps_start=0.0, eps_end=0.0))
    n = 4
    obs = np.zeros((n, OBS_DIM), dtype=np.float32)
    masks = np.zeros((n, JOINT_ACTION_N), dtype=np.float32)
    masks[0, 10] = 1.0
    masks[1, [3, 7]] = 1.0
    masks[3, :] = 1.0
    out = agent.act_batch(obs, masks, epsilon=0.0)
    assert out.shape == (n,)
    assert out[0] == 10
    assert out[1] in (3, 7)
    for i in (0, 1, 3):
        assert masks[i, out[i]] > 0.5


@pytest.mark.parametrize('module', ['ppo', 'ddqn'])
def test_vectorized_training_smoke_mock(module, tmp_path):
    if module == 'ppo':
        from showdown.rl.ppo.train import main
        extra = ['--rollout-steps', '64']
    else:
        from showdown.rl.ddqn.train import main
        extra = []
    rc = main([
        '--seed', '0', '--opponent', 'mock', '--steps', '200', '--num-envs', '3',
        '--checkpoint-dir', str(tmp_path), '--suite-games', '0',
        '--log-every', '100000', '--log-every-sec', '3600', *extra,
    ])
    assert rc == 0
    assert (tmp_path / 'final.pt').is_file()


# --------------------------------------------------------------------------- #
# Tuning con trials paralelos
# --------------------------------------------------------------------------- #

def test_parallel_trials_assign_distinct_port_blocks(tmp_path, monkeypatch):
    from showdown.inference_sim import rl_hp_tune as tune

    monkeypatch.setenv('PS_PORT', '8000')
    env0 = tune._trial_slot_env(0, port_stride=50)
    env2 = tune._trial_slot_env(2, port_stride=50)
    assert env0['PS_PORT'] != env2['PS_PORT']

    class _T:
        def __init__(self, v):
            self.overrides = {'lr': v}

    trials = [_T(0.1), _T(0.2), _T(0.3)]

    def build_cmd(trial, trial_dir):
        code = (
            'import os,sys; open(sys.argv[1],"w").write(str(%r)+" "+os.environ["PS_PORT"])'
            % trial.overrides['lr']
        )
        return [sys.executable, '-c', code, str(trial_dir / 'score.txt')]

    def parse(trial_dir):
        p = trial_dir / 'score.txt'
        return float(p.read_text().split()[0]) if p.is_file() else None

    res = tune._run_trials_parallel(
        trials, build_cmd=build_cmd, parse_score_from_dir=parse,
        root=tmp_path, parallel=2, port_stride=50,
    )
    assert {i: r[0] for i, r in res.items()} == {1: 0.1, 2: 0.2, 3: 0.3}
    ports = {(tmp_path / f'trial_{i:02d}' / 'score.txt').read_text().split()[1]
             for i in (1, 2, 3)}
    assert len(ports) >= 2  # al menos dos bloques distintos en uso


# --------------------------------------------------------------------------- #
# Robust E2E repartido: combinación de diagnósticos por proceso
# --------------------------------------------------------------------------- #

def test_joint_diag_snapshot_merge_adds_counts():
    from collections import Counter

    from showdown.battle_bots.greedy_joint import diag

    diag.reset()
    diag.record_battle(True, Counter({'fakeout': 2}))
    snap = diag.snapshot()
    diag.reset()
    diag.record_battle(False, Counter({'redirect': 1}))
    diag.merge(snap)
    merged = diag.snapshot()
    assert merged['battles_win'] == 1 and merged['battles_loss'] == 1
    assert merged['turns_win'] == {'fakeout': 2}
    assert merged['turns_loss'] == {'redirect': 1}
    diag.reset()


def test_divergence_merge_sums_counters_and_caps_samples():
    from showdown.battle_bots.expectimax_bayesian import divergence as dv

    dv.reset_divergence_stats()
    other = dv.DivergenceStats(n_choose=5, n_diverged=2, sum_delta_u=1.5,
                               samples=[{'k': i} for i in range(50)])
    dv.merge_divergence_stats(other)
    dv.merge_divergence_stats(other)
    snap = dv.snapshot_divergence_stats()
    assert snap.n_choose == 10 and snap.n_diverged == 4
    assert abs(snap.sum_delta_u - 3.0) < 1e-9
    assert len(snap.samples) == snap.max_samples
    dv.reset_divergence_stats()


# --------------------------------------------------------------------------- #
# Cola secuencial de entrenamiento
# --------------------------------------------------------------------------- #

def _write_queue(tmp_path, steps):
    import json

    q = tmp_path / 'cola.json'
    q.write_text(json.dumps({'name': 'prueba', 'steps': steps}), encoding='utf-8')
    return q


def test_train_queue_runs_in_order_and_resumes(tmp_path, capsys):
    from showdown.inference_sim import run_train_queue as tq

    q = _write_queue(tmp_path, [
        {'name': 'uno', 'module': 'platform', 'args': []},
        {'name': 'dos', 'module': 'platform', 'args': ['--terse']},
    ])
    run_dir = tmp_path / 'run'
    assert tq.main([str(q), '--run-dir', str(run_dir)]) == 0
    assert (run_dir / 'uno.done').is_file() and (run_dir / 'dos.done').is_file()
    assert (run_dir / 'uno.log').read_text(encoding='utf-8').count('START uno') == 1
    capsys.readouterr()
    # Relanzar: los pasos completados se saltan.
    assert tq.main([str(q), '--run-dir', str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert 'uno: ya completado' in out and 'dos: ya completado' in out


def test_train_queue_stops_on_failure(tmp_path):
    from showdown.inference_sim import run_train_queue as tq

    q = _write_queue(tmp_path, [
        {'name': 'falla', 'module': 'modulo_que_no_existe_xyz', 'args': []},
        {'name': 'despues', 'module': 'platform', 'args': []},
    ])
    run_dir = tmp_path / 'run'
    assert tq.main([str(q), '--run-dir', str(run_dir)]) == 1
    assert not (run_dir / 'falla.done').exists()
    assert not (run_dir / 'despues.log').exists()  # nunca se lanzó


def test_train_queue_placeholders_and_dry_run(tmp_path, capsys):
    from showdown.inference_sim import run_train_queue as tq

    q = _write_queue(tmp_path, [
        {'name': 'p', 'module': 'platform',
         'args': ['--x', '{num_envs}', '{device}', '{run_dir}/sub']},
    ])
    run_dir = tmp_path / 'run'
    assert tq.main([str(q), '--run-dir', str(run_dir), '--dry-run']) == 0
    out = capsys.readouterr().out
    assert '{num_envs}' not in out and str(run_dir) in out
    assert not (run_dir / 'p.done').exists()
    bad = _write_queue(tmp_path, [{'name': 'b', 'module': 'platform', 'args': ['{nope}']}])
    with pytest.raises(ValueError):
        tq.main([str(bad), '--run-dir', str(run_dir), '--dry-run'])


# --------------------------------------------------------------------------- #
# Node.js fuera del PATH (nvm en sesiones no interactivas)
# --------------------------------------------------------------------------- #

def test_node_executable_prefers_env_then_path_then_latest_nvm(tmp_path, monkeypatch):
    import shutil

    from showdown.inference_sim import paths

    monkeypatch.setenv('NODE_BINARY', '/opt/custom/node')
    assert paths.node_executable() == '/opt/custom/node'

    monkeypatch.delenv('NODE_BINARY')
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    for ver in ('v20.1.0', 'v26.13.1', 'v9.9.9'):
        b = tmp_path / 'versions' / 'node' / ver / 'bin'
        b.mkdir(parents=True)
        (b / 'node').write_text('')
    monkeypatch.setenv('NVM_DIR', str(tmp_path))
    from pathlib import Path

    assert Path(paths.node_executable()).parts[-3:] == ('v26.13.1', 'bin', 'node')
    # La carpeta de nvm queda en el PATH: `pokemon-showdown` ejecuta `node build` solo.
    import os
    assert str(tmp_path / 'versions' / 'node' / 'v26.13.1' / 'bin') in os.environ['PATH'].split(os.pathsep)

    monkeypatch.setenv('NVM_DIR', str(tmp_path / 'vacio'))
    assert paths.node_executable() == 'node'


# --------------------------------------------------------------------------- #
# Servidores Showdown: arranque en paralelo y apagado del árbol de procesos
# --------------------------------------------------------------------------- #

def test_ensure_ps_ports_spawns_all_down_ports_at_once(monkeypatch):
    from showdown.inference_sim import ps_server

    up = set()
    spawned = []

    class _FakeProc:
        def __init__(self, port):
            self.port = port
            self.pid = 10_000 + port

        def poll(self):
            return None

    def fake_spawn(port):
        spawned.append(port)
        return _FakeProc(port)

    def fake_up(host='localhost', port=0, timeout=1.0):
        # Ningún puerto responde hasta que se lanzaron todos: si el arranque
        # fuera secuencial, el primero esperaría hasta el timeout.
        if len(spawned) == 3:
            up.update(spawned)
        return port in up

    monkeypatch.setattr(ps_server, '_spawn_ps', fake_spawn)
    monkeypatch.setattr(ps_server, 'is_ps_up', fake_up)
    monkeypatch.setattr(ps_server, '_ps_procs', {})
    monkeypatch.setattr(ps_server, 'ensure_ps_build', lambda *a, **k: True)
    monkeypatch.delenv('PS_START_PARALLEL', raising=False)
    assert ps_server.ensure_ps_ports([9001, 9002, 9003], start=True, wait_s=5)
    assert spawned == [9001, 9002, 9003]


def test_ensure_ps_ports_starts_in_waves(monkeypatch):
    from showdown.inference_sim import ps_server

    spawned, up = [], set()

    class _P:
        pid = 1

        def poll(self):
            return None

    def fake_spawn(port):
        spawned.append(port)
        up.add(port)  # responde en cuanto se lanza
        return _P()

    monkeypatch.setattr(ps_server, '_spawn_ps', fake_spawn)
    monkeypatch.setattr(ps_server, 'is_ps_up', lambda host='localhost', port=0, timeout=1.0: port in up)
    monkeypatch.setattr(ps_server, '_ps_procs', {})
    monkeypatch.setattr(ps_server, 'ensure_ps_build', lambda *a, **k: True)
    monkeypatch.setenv('PS_START_PARALLEL', '2')
    assert ps_server.ensure_ps_ports([9101, 9102, 9103, 9104, 9105], start=True, wait_s=5)
    assert spawned == [9101, 9102, 9103, 9104, 9105]


def test_ps_start_argv_skips_build_when_compiled(monkeypatch):
    from showdown.inference_sim import ps_server

    monkeypatch.setattr(ps_server, 'ps_is_built', lambda: True)
    assert '--skip-build' in ps_server.ps_start_argv(9000)
    monkeypatch.setattr(ps_server, 'ps_is_built', lambda: False)
    assert '--skip-build' not in ps_server.ps_start_argv(9000)


def test_stop_started_ps_kills_each_process_tree(monkeypatch):
    from showdown.inference_sim import ps_server

    killed = []
    monkeypatch.setattr(ps_server, '_kill_process_tree', lambda proc, timeout=8.0: killed.append(proc))
    monkeypatch.setattr(ps_server, '_ps_procs', {9001: 'a', 9002: 'b'})
    ps_server.stop_started_ps()
    assert killed == ['a', 'b'] and ps_server._ps_procs == {}


def test_spawned_ps_gets_its_own_process_group(monkeypatch, tmp_path):
    import os
    import subprocess as sp

    from showdown.inference_sim import ps_server

    seen = {}

    def fake_popen(argv, **kwargs):
        seen.update(kwargs)

        class _P:
            pid = 1

        return _P()

    (tmp_path / 'config').mkdir()
    monkeypatch.setattr(ps_server, 'POKEMON_SHOWDOWN_ROOT', tmp_path)
    monkeypatch.setattr(sp, 'Popen', fake_popen)
    monkeypatch.setattr(ps_server, '_cleanup_registered', True)
    ps_server._spawn_ps(9005)
    if os.name == 'nt':
        assert 'creationflags' in seen
    else:
        assert seen.get('start_new_session') is True
    assert seen['env']['PSPORT'] == '9005'


# --------------------------------------------------------------------------- #
# Familia QD vectorizada (QD-PPO / QD-PGA / QD-PPGA)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('module', ['qd_ppo', 'qd_pga', 'qd_ppga'])
def test_qd_trainers_vectorized_smoke_mock(module, tmp_path):
    import importlib

    train = importlib.import_module(f'showdown.rl.{module}.train')
    rc = train.main([
        '--seed', '0', '--smoke', '--num-envs', '2', '--suite-games', '0',
        '--checkpoint-dir', str(tmp_path),
    ])
    assert rc == 0
    assert (tmp_path / 'archive.pt').is_file()


def test_qd_evaluate_policy_vec_plays_exactly_n_games():
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    from showdown.rl.qd_ppo.evaluate import evaluate_policy

    venv = SyncVecVGCEnv(_mock_specs(3, turns=4), seed=5)
    try:
        agent = PPOAgent(PPOConfig(seed=0))
        res = evaluate_policy(agent, venv, rng=np.random.default_rng(0), n_games=7, g_min=0)
        assert res.n_games == 7
        assert np.isfinite(res.fitness)
        assert venv.in_flight() == []
    finally:
        venv.close()


def test_qd_mosaic_never_turns_mock_envs_live():
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    from showdown.rl.qd_ppo.improve import ppo_improve

    venv = SyncVecVGCEnv(_mock_specs(2, turns=4), seed=6)
    try:
        agent = PPOAgent(PPOConfig(seed=0, rollout_steps=16))
        venv2, taken, _m = ppo_improve(agent, venv, steps=40, rng=np.random.default_rng(0),
                                       global_steps=0, t_total=10_000, use_mosaic=True)
        assert venv2 is venv and taken == 40
        assert all(s.opponent == 'mock' for s in venv.specs)
    finally:
        venv.close()


def test_vec_env_set_reward_switches_candidate():
    venv = SyncVecVGCEnv(_mock_specs(2), seed=7)
    try:
        venv.set_reward('candidate_1')
        assert all(s.reward == 'candidate_1' for s in venv.specs)
        assert all(r.env.reward_candidate.value == 'candidate_1' for r in venv._runners)
        venv.set_reward('candidate_2')
        assert all(r.env.reward_candidate.value == 'candidate_2' for r in venv._runners)
    finally:
        venv.close()


# --------------------------------------------------------------------------- #
# Semilla fija de batallas de Showdown
# --------------------------------------------------------------------------- #

def test_set_battle_seed_exports_env_for_children(monkeypatch):
    import os

    from showdown.inference_sim import ps_server

    for var in ('PS_BATTLE_SEED', 'PYTHONHASHSEED', 'PS_SEED_LOG'):
        monkeypatch.delenv(var, raising=False)
    ps_server.set_battle_seed(42, log_path='/tmp/seeds.jsonl')
    assert os.environ['PS_BATTLE_SEED'] == '42'
    assert os.environ['PYTHONHASHSEED'] == '0'      # orden de set/dict estable en los hijos
    assert os.environ['PS_SEED_LOG'] == '/tmp/seeds.jsonl'
    ps_server.set_battle_seed(None)                  # None no cambia nada
    assert os.environ['PS_BATTLE_SEED'] == '42'


def test_ps_start_argv_uses_seeded_launcher_only_with_seed_and_build(monkeypatch):
    from showdown.inference_sim import ps_server

    monkeypatch.setattr(ps_server, 'ps_is_built', lambda: True)
    monkeypatch.delenv('PS_BATTLE_SEED', raising=False)
    assert not any('ps_seeded_start' in a for a in ps_server.ps_start_argv(9000))
    monkeypatch.setenv('PS_BATTLE_SEED', '7')
    argv = ps_server.ps_start_argv(9000)
    assert any(a.endswith('ps_seeded_start.js') for a in argv) and argv[-1] == '9000'
    # Sin build no se puede usar el lanzador (no ejecuta `node build`).
    monkeypatch.setattr(ps_server, 'ps_is_built', lambda: False)
    assert not any('ps_seeded_start' in a for a in ps_server.ps_start_argv(9000))


def test_unseeded_server_already_running_is_warned_once(monkeypatch, capsys):
    from showdown.inference_sim import ps_server

    monkeypatch.setenv('PS_BATTLE_SEED', '9')
    monkeypatch.setattr(ps_server, '_ps_procs', {})
    monkeypatch.setattr(ps_server, '_seed_warned', set())
    monkeypatch.setattr(ps_server, 'is_ps_up', lambda host='localhost', port=0, timeout=1.0: True)
    assert ps_server.ensure_ps(port=9555)
    assert ps_server.ensure_ps(port=9555)
    assert capsys.readouterr().out.count('9555') == 1


def test_seeded_launcher_script_ships_with_repo():
    from showdown.inference_sim.paths import REPO_ROOT

    js = (REPO_ROOT / 'showdown' / 'tools' / 'ps_seeded_start.js').read_text(encoding='utf-8')
    assert 'PS_BATTLE_SEED' in js and 'Rooms.createBattle' in js and 'sodium,' in js
