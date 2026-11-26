"""QD-PGA 2.0 (agente J 2.0 = PGA-ME fiel, acciones discretas; docs/qd_pga_map_elites_spec.md §J 2.0)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from showdown.rl.qd_pga.pgame import (  # noqa: E402
    N_JOINT, JointReplay, TwinCritic, expected_q, iso_line, joint_log_probs, pg_steps, to_torch,
)
from showdown.rl.ppo.network import ActorCritic  # noqa: E402

D = 32  # observación pequeña para los tests (codificador plano)


def _net():
    torch.manual_seed(0)
    return ActorCritic(D, encoder='flat')


def _jmask(rng, b):
    m = (rng.random((b, 40, 40)) < 0.3).astype(np.float32)
    m[:, 39, 39] = 1.0  # siempre hay una acción legal
    return torch.as_tensor(m.reshape(b, N_JOINT))


def test_joint_log_probs_is_the_autoregressive_distribution():
    rng = np.random.default_rng(0)
    net = _net()
    obs = torch.randn(4, D)
    jm = _jmask(rng, 4)
    lp = joint_log_probs(net, obs, jm)
    p = torch.exp(lp)
    assert torch.allclose(p.sum(-1), torch.ones(4), atol=1e-5)
    assert float(p[jm < 0.5].abs().max()) == 0.0
    # Comparación con π1(a1)·π2(a2|a1) calculado con las máscaras de hueco
    from showdown.rl.ppo.network import apply_mask_logits
    with torch.no_grad():
        h = net.features(obs)
        j3 = jm.view(4, 40, 40)
        m1 = (j3.sum(-1) > 0).float()
        p1 = torch.softmax(apply_mask_logits(net.policy1(h), m1), -1)
        b, a1 = 2, int(torch.argmax(p1[2]))
        z2 = net.slot2_logits(h[b:b + 1], torch.tensor([a1]))
        p2 = torch.softmax(apply_mask_logits(z2, j3[b, a1].unsqueeze(0)), -1)[0]
        ref = p1[b, a1] * p2
    assert torch.allclose(p.view(4, 40, 40)[b, a1], ref, atol=1e-5)


def test_expected_pg_increases_expected_q():
    rng = np.random.default_rng(1)
    net = _net()
    rep = JointReplay(512, D)
    for _ in range(300):
        rep.add({'obs': rng.normal(size=D), 'action': 1599, 'reward': 0.0, 'next_obs': rng.normal(size=D),
                 'terminated': True, 'truncated': False, 'mask': _jmask(rng, 1)[0].numpy(), 'next_mask': None})

    class _C:  # crítico fijo: Q alto en la acción 1599 (39, 39)
        device = torch.device('cpu')

        def q1(self, s):
            q = torch.zeros(s.shape[0], N_JOINT)
            q[:, 1599] = 1.0
            return q
    crit = _C()
    b = to_torch(rep.sample(256, rng), crit.device)
    before = float(expected_q(net, crit.q1(b['s']), b['s'], b['m']).mean())
    pg_steps(net, crit, rep, rng, n_steps=30, lr=1e-2, batch=64)
    after = float(expected_q(net, crit.q1(b['s']), b['s'], b['m']).mean())
    assert after > before + 0.1


def test_twin_critic_learns_terminal_rewards_and_soft_target():
    rng = np.random.default_rng(2)
    rep = JointReplay(256, D)
    for _ in range(200):
        a = int(rng.integers(0, 2))
        m = np.zeros(N_JOINT, np.float32)
        m[[0, 1]] = 1
        rep.add({'obs': np.ones(D), 'action': a, 'reward': 1.0 if a == 1 else -1.0, 'next_obs': np.ones(D),
                 'terminated': True, 'truncated': False, 'mask': m, 'next_mask': None})
    from showdown.rl.ddqn import network as qn
    orig = qn.QNetwork.__init__

    def _init(self, obs_dim, *a, **k):
        k.setdefault('encoder', 'flat')
        orig(self, obs_dim, *a, **k)
    qn.QNetwork.__init__ = _init
    try:
        crit = TwinCritic(D, torch.device('cpu'), lr=1e-3)
    finally:
        qn.QNetwork.__init__ = orig
    t0 = [p.clone() for p in crit.q_targ.parameters()]
    for _ in range(300):
        crit.update(to_torch(rep.sample(64, rng), crit.device), _net())
    q = crit.q1(torch.ones(1, D))[0]
    assert float(q[1]) > 0.7 and float(q[0]) < -0.7
    moved = sum(float((a - b).abs().sum()) for a, b in zip(crit.q_targ.parameters(), t0))
    assert moved > 0.0   # τ = 0,005: las redes objetivo se mueven despacio pero se mueven


def test_iso_line_formula():
    rng = np.random.default_rng(3)
    a = {'w': torch.zeros(1000), 'n': torch.tensor(3)}
    b = {'w': torch.ones(1000), 'n': torch.tensor(3)}
    c = iso_line(a, b, np.random.default_rng(3), iso_sigma=0.005, line_sigma=0.05)
    xi = np.random.default_rng(3).normal() * 0.05
    resid = (c['w'] - xi).numpy()          # θ1 = 0, θ2 − θ1 = 1 → θ' = ε + ξ
    assert abs(resid.std() - 0.005) < 0.001 and abs(resid.mean()) < 0.001
    assert int(c['n']) == 3                # tensores no flotantes intactos
    same = iso_line(a, a, rng, iso_sigma=0.0, line_sigma=0.05)
    assert torch.equal(same['w'], a['w'])  # mismo padre y σ_iso = 0: sin cambio


def test_replay_masks_roundtrip_and_truncation_flag():
    rng = np.random.default_rng(4)
    rep = JointReplay(10, D)
    m = np.zeros(N_JOINT, np.float32)
    m[[5, 77, 1599]] = 1
    rep.add({'obs': np.ones(D), 'action': 77, 'reward': 0.5, 'next_obs': np.zeros(D), 'terminated': False,
             'truncated': True, 'mask': m, 'next_mask': m})
    rep.add({'obs': np.ones(D), 'action': 5, 'reward': 0.0, 'next_obs': np.zeros(D), 'terminated': False,
             'truncated': False, 'mask': np.ones(40), 'next_mask': None})   # sin máscara conjunta: se ignora
    assert len(rep) == 1
    b = rep.sample(3, rng)
    assert np.array_equal(b['m'][0], m) and np.array_equal(b['m2'][0], m)
    assert b['trunc'][0] == 1.0 and b['a'][0] == 77


def test_train_j2_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv('RL_OBS_CALC', '0')
    from showdown.rl.qd_pga.train_j2 import run_training
    s = run_training(seed=0, smoke=True, checkpoint_dir=tmp_path / 'j2')
    assert s['children'] >= 1 and s['critic_steps'] >= 4 and s['replay'] > 0
    assert (tmp_path / 'j2' / 'best_elite.pt').is_file()
