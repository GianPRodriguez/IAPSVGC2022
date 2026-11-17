"""Replay de DDQN en GPU (ddqn_spec 1.13): mismos lotes y misma aumentación que en CPU."""
import sys
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, '.')

from showdown.rl.constants import JOINT_ACTION_N, OBS_DIM
from showdown.rl.ddqn.replay import ReplayBuffer

torch = pytest.importorskip('torch')
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='sin CUDA')


def _fill(rb, n, seed=0):
    rng = np.random.default_rng(seed)
    for k in range(n):
        s = (rng.random(OBS_DIM) < 0.2).astype(np.float32) * rng.random(OBS_DIM).astype(np.float32)
        m = (rng.random(JOINT_ACTION_N) < 0.05)
        m[0] = True
        rb.add(s=s, a=int(rng.integers(0, JOINT_ACTION_N)), r=float(rng.normal()),
               s_next=np.roll(s, 3), terminated=bool(k % 7 == 0), m_next=m,
               opponent_code=int(rng.integers(0, 4)))


def test_opponent_counts_incremental_with_wraparound():
    rb = ReplayBuffer(capacity=50, obs_dim=OBS_DIM)
    _fill(rb, 130)
    exact = Counter(int(c) for c in rb._opp[: rb._n].tolist())
    assert {k: v for k, v in rb._opp_counts.items() if v} == dict(exact)
    fr = rb.origin_fractions()
    assert abs(sum(fr.values()) - 1.0) < 1e-9


@needs_cuda
def test_gpu_sample_identical_to_cpu():
    cpu, gpu = ReplayBuffer(capacity=600, device='cpu'), ReplayBuffer(capacity=600, device='cuda')
    _fill(cpu, 700)
    _fill(gpu, 700)
    a = cpu.sample(64, np.random.default_rng(5))
    b = gpu.sample(64, np.random.default_rng(5))
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))


@needs_cuda
def test_gpu_augmentation_identical_to_numpy():
    from showdown.rl import symmetry as SYM
    cpu, gpu = ReplayBuffer(capacity=600, device='cpu'), ReplayBuffer(capacity=600, device='cuda')
    _fill(cpu, 500)
    _fill(gpu, 500)
    r1, r2 = np.random.default_rng(9), np.random.default_rng(9)
    s, a, _r, s2, _t, m2, _o = cpu.sample(128, r1)
    g = SYM.sample_elements(s, r1)
    s_a, a_a = SYM.augment_obs(s, g), SYM.augment_joint(a, g)
    h = SYM.sample_elements(s2, r1)
    s2_a, m2_a = SYM.augment_obs(s2, h), SYM.augment_joint_mask(m2, h)

    st, at, _rt, s2t, _dt, m2t, _ot = gpu.sample_torch(128, r2)
    gt = SYM.sample_elements_t(st, r2)
    assert np.array_equal(g, gt)
    ht = SYM.sample_elements_t(s2t, r2)
    assert np.array_equal(h, ht)
    assert np.array_equal(s_a, SYM.augment_obs_t(st, gt).cpu().numpy())
    assert np.array_equal(a_a, SYM.augment_joint_t(at, gt).cpu().numpy())
    assert np.array_equal(s2_a, SYM.augment_obs_t(s2t, ht).cpu().numpy())
    assert np.array_equal(m2_a, SYM.augment_joint_mask_t(m2t, ht).cpu().numpy())


@needs_cuda
def test_learn_same_loss_cpu_and_gpu_replay(monkeypatch):
    """Mismos pesos y mismo generador → mismo lote → misma pérdida (salvo redondeo de GPU)."""
    from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
    monkeypatch.delenv('RL_REPLAY_DEVICE', raising=False)
    torch.manual_seed(0)
    # dropout=0: el dropout (por defecto desde ddqn_spec 1.20) usa el generador global de torch
    # y haría distintas las dos pérdidas; aquí solo se compara el muestreo del replay.
    ag_c = DDQNAgent(DDQNConfig(device='cuda', replay_device='cpu', replay_capacity=800, seed=3,
                                dropout=0.0))
    ag_g = DDQNAgent(DDQNConfig(device='cuda', replay_device='auto', replay_capacity=800, seed=3,
                                dropout=0.0))
    assert not ag_c.replay.on_gpu and ag_g.replay.on_gpu
    ag_g.online.load_state_dict(ag_c.online.state_dict())
    ag_g.target.load_state_dict(ag_c.target.state_dict())
    _fill(ag_c.replay, 600)
    _fill(ag_g.replay, 600)
    lc, lg = ag_c.learn(), ag_g.learn()
    assert abs(lc['loss'] - lg['loss']) < 1e-5
    assert lc.get('sym_frac_our') == lg.get('sym_frac_our')


def test_replay_device_env_override(monkeypatch):
    from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
    monkeypatch.setenv('RL_REPLAY_DEVICE', 'cpu')
    assert not DDQNAgent(DDQNConfig(device='cpu')).replay.on_gpu
