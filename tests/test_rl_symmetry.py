"""Simetrías (aumentación) y codificador compartido (rl_env_spec v1.20)."""
from __future__ import annotations

import numpy as np
import pytest

from showdown.rl import symmetry as SYM
from showdown.rl.constants import (
    JOINT_ACTION_N, MEGA_BASE, OBS_DIM, PASS_ID, SLOT_ACTION_N, SWITCH_BASE,
)
from showdown.rl.observation import build_observation
from tests.test_rl_obs_v2 import _snap


def _asym_snap(**kw):
    """Snapshot con diferencias entre posiciones (si no, el test sería trivial)."""
    s = _snap(**kw)
    s.our.active[0].hp_pct = 41.0
    s.our.active[1].boosts = {'spe': 1}
    s.foe.active[0].hp_pct = 63.0
    s.foe.active[1].boosts = {'atk': -1}
    s.our.reserve[0].hp_pct = 77.0
    s.force_switch = (False, False)
    return s


def _mirror(s, g: int):
    if g & SYM.BIT_OUR:
        s.our.active = [s.our.active[1], s.our.active[0]]
        s.avail_moves = (s.avail_moves[1], s.avail_moves[0])
        s.force_switch = (s.force_switch[1], s.force_switch[0])
    if g & SYM.BIT_FOE:
        s.foe.active = [s.foe.active[1], s.foe.active[0]]
    if g & SYM.BIT_BENCH:
        s.our.reserve = [s.our.reserve[1], s.our.reserve[0]] + list(s.our.reserve[2:])
        s.our.bench = list(s.our.reserve[:2])
    return s


@pytest.mark.parametrize('g', range(SYM.N_ELEMENTS))
def test_obs_permutation_matches_mirrored_battle(g):
    base = build_observation(_asym_snap())
    mirrored = build_observation(_mirror(_asym_snap(), g))
    perm = SYM.augment_obs(base[None], np.array([g]))[0]
    bad = np.flatnonzero(~np.isclose(perm, mirrored, atol=1e-6))
    assert bad.size == 0, (g, bad[:10])
    if g:
        assert not np.array_equal(base, mirrored)  # la permutación no es trivial


def test_group_elements_are_commuting_involutions():
    P, J, S = SYM.obs_perms(), SYM.joint_perms(), SYM.slot_perms()
    for g in range(SYM.N_ELEMENTS):
        assert np.array_equal(P[g][P[g]], np.arange(OBS_DIM))
        assert np.array_equal(J[g][J[g]], np.arange(JOINT_ACTION_N))
        assert np.array_equal(S[g][S[g]], np.arange(SLOT_ACTION_N))
        for h in range(SYM.N_ELEMENTS):
            assert np.array_equal(P[g][P[h]], P[g ^ h])
    # σ_o intercambia slots; σ_f objetivos rival 0↔1 (también en mega); σ_b cambios 32↔33
    j = 40 * 5 + 33
    assert J[SYM.BIT_OUR][j] == 40 * 33 + 5
    assert S[SYM.BIT_FOE][4 * 2 + 0] == 4 * 2 + 1 and S[SYM.BIT_FOE][MEGA_BASE + 1] == MEGA_BASE + 0
    assert S[SYM.BIT_FOE][4 * 2 + 3] == 4 * 2 + 3  # sin objetivo / spread: igual
    assert S[SYM.BIT_BENCH][SWITCH_BASE] == SWITCH_BASE + 1 and S[SYM.BIT_BENCH][PASS_ID] == PASS_ID


def test_joint_action_stays_legal_under_augmentation():
    rng = np.random.default_rng(0)
    obs = np.stack([build_observation(_asym_snap())] * 64)
    masks = (rng.random((64, JOINT_ACTION_N)) < 0.2).astype(np.float32)
    acts = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
    g = SYM.sample_elements(obs, rng)
    a2, m2 = SYM.augment_joint(acts, g), SYM.augment_joint_mask(masks, g)
    assert np.all(m2[np.arange(64), a2] > 0.5)
    assert np.array_equal(m2.sum(1), masks.sum(1))


def test_bench_swap_only_when_both_reserves_alive():
    s = _asym_snap()
    s.our.reserve[1].fainted = True
    s.our.reserve[1].hp_pct = 0.0
    obs = build_observation(s)[None]
    assert not SYM.bench_swap_valid(obs)[0]
    g = SYM.sample_elements(np.repeat(obs, 200, 0), np.random.default_rng(1))
    assert not np.any(g & SYM.BIT_BENCH)
    assert SYM.bench_swap_valid(build_observation(_asym_snap())[None])[0]


def test_ppo_batch_keeps_actions_legal_and_skips_slot_swap_after_mega_or_switch():
    rng = np.random.default_rng(2)
    n = 300
    obs = np.stack([build_observation(_asym_snap())] * n)
    m1 = np.zeros((n, SLOT_ACTION_N), np.float32)
    m1[:, :16] = 1.0
    m1[:, MEGA_BASE:MEGA_BASE + 16] = 1.0
    m1[:, SWITCH_BASE:SWITCH_BASE + 2] = 1.0
    a1 = rng.choice(np.flatnonzero(m1[0]), size=n)
    m2 = np.zeros_like(m1)
    m2[:, :16] = 1.0
    m2[:, SWITCH_BASE:SWITCH_BASE + 2] = 1.0
    for i in range(n):  # condicionada: sin mega si a1 fue mega; sin el cambio tomado
        if not (MEGA_BASE <= a1[i] < MEGA_BASE + 16):
            m2[i, MEGA_BASE:MEGA_BASE + 16] = 1.0
        if SWITCH_BASE <= a1[i] < SWITCH_BASE + 4:
            m2[i, a1[i]] = 0.0
    a2 = np.array([rng.choice(np.flatnonzero(m)) for m in m2])
    out = SYM.ppo_symmetry_batch(obs, a1, a2, m1, m2, rng)
    rows = np.arange(n)
    assert np.all(out['m1'][rows, out['a1']] > 0.5)
    assert np.all(out['m2'][rows, out['a2']] > 0.5)
    special = (a1 >= MEGA_BASE) & (a1 < SWITCH_BASE + 4)
    assert not np.any(out['g'][special] & SYM.BIT_OUR)
    assert np.any(out['g'][~special] & SYM.BIT_OUR)
    sw = (out['g'] & SYM.BIT_OUR) > 0
    # nuevo slot 2 = slot 1 original condicionado: nunca dos megas ni el mismo cambio
    mega_a1 = (out['a1'] >= MEGA_BASE) & (out['a1'] < MEGA_BASE + 16)
    assert np.all(out['m2'][sw & mega_a1, MEGA_BASE:MEGA_BASE + 16] == 0.0)


def test_ally_target_is_relative_to_slot():
    from types import SimpleNamespace

    from showdown.rl.pe_adapter import spec_target_to_pe

    assert spec_target_to_pe(0, 2) == -2 and spec_target_to_pe(1, 2) == -1
    assert spec_target_to_pe(0, 0) == 1 and spec_target_to_pe(1, 1) == 2
    assert spec_target_to_pe(0, 3) == 0
    acu = SimpleNamespace(target=SimpleNamespace(name='ADJACENT_ALLY_OR_SELF'))
    assert spec_target_to_pe(0, 3, acu) == -1 and spec_target_to_pe(1, 3, acu) == -2


# --------------------------------------------------------------------------- #
# Codificador compartido
# --------------------------------------------------------------------------- #

torch = pytest.importorskip('torch')


def _n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_shared_encoder_shapes_and_fewer_params(monkeypatch):
    from showdown.rl.ddqn.network import QNetwork
    from showdown.rl.ppo.network import ActorCritic
    from showdown.rl.team_preview_nash.network import LowRankPayoffNet
    from showdown.rl.team_preview_obs import OBS_DIM as P_OBS
    from showdown.rl.team_preview_ppo.network import PreviewActorCritic
    from showdown.rl.team_preview_qd_pga.critic import SharedPreviewCritic

    x = torch.zeros(3, OBS_DIM)
    xp = torch.zeros(3, P_OBS)
    q = QNetwork()
    assert q(x).shape == (3, JOINT_ACTION_N)
    z1, z2, v = ActorCritic()(x, a1=torch.zeros(3, dtype=torch.long))
    assert z1.shape == (3, SLOT_ACTION_N) and z2.shape == (3, SLOT_ACTION_N) and v.shape == (3,)
    pac = PreviewActorCritic()
    lg, pv = pac(xp)
    assert lg.shape == (3, 360) and pv.shape == (3,)
    assert SharedPreviewCritic()(xp).shape == (3,)
    assert LowRankPayoffNet().payoff_matrix(xp[0]).shape == (360, 360)

    monkeypatch.setenv('RL_OBS_ENCODER', 'flat')
    q_flat, pac_flat = QNetwork(), PreviewActorCritic()
    assert q_flat(x).shape == (3, JOINT_ACTION_N)
    assert _n_params(q) < 0.5 * _n_params(q_flat)
    assert _n_params(pac) < 0.35 * _n_params(pac_flat)
    monkeypatch.setenv('RL_OBS_ENCODER', 'nope')
    with pytest.raises(ValueError):
        QNetwork()


def test_same_mon_in_any_block_gets_the_same_embedding():
    from showdown.rl.encoders import combat_encoder

    enc = combat_encoder()
    obs = torch.as_tensor(build_observation(_asym_snap()))[None]
    out = enc(obs)
    g = enc.groups[0]
    blk = torch.zeros_like(obs)
    blk[0, 2 * g.dim:3 * g.dim] = obs[0, 0:g.dim]  # bloque 0 copiado en el bloque 2
    e = enc(blk)
    assert torch.allclose(out[0, 0:g.out], e[0, 2 * g.out:3 * g.out])


def test_symmetry_off_by_default(monkeypatch):
    from showdown.rl.symmetry import sym_aug_enabled
    monkeypatch.delenv('RL_SYM_AUG', raising=False)
    assert not sym_aug_enabled(True)
    monkeypatch.setenv('RL_SYM_AUG', '1')
    assert sym_aug_enabled(True) and not sym_aug_enabled(False)


def test_ddqn_and_ppo_updates_run_with_symmetry(monkeypatch):
    monkeypatch.setenv('RL_SYM_AUG', '1')  # apagada por defecto desde el 16-11-2026
    from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig

    rng = np.random.default_rng(3)
    obs = build_observation(_asym_snap())
    agent = DDQNAgent(DDQNConfig(batch_size=8, warmup_steps=0, train_freq=1))
    mask = np.zeros(JOINT_ACTION_N, np.float32)
    mask[:200] = 1.0
    out = None
    for _ in range(10):
        out = agent.observe(s=obs, a=int(rng.integers(0, 200)), r=0.1, s_next=obs,
                            terminated=False, m_next=mask)
    assert out is not None and np.isfinite(out['loss']) and 'sym_frac_our' in out

    ppo = PPOAgent(PPOConfig(rollout_steps=16, minibatch_size=8, epochs=1))
    m = np.zeros(SLOT_ACTION_N, np.float32)
    m[:16] = 1.0
    m[PASS_ID] = 1.0
    for t in range(16):
        a1, a2, logp, v = ppo.act(obs, m, lambda _a: m)
        ppo.observe(obs=obs, a1=a1, a2=a2, logp=logp, reward=0.0, value=v,
                    terminated=(t == 15), m1=m, m2=m)
    met = ppo.update()
    assert np.isfinite(met['loss']) and 'sym_pi_loss' in met and 'sym_v_loss' in met
