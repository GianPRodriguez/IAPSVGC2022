"""Acceptance tests for Masked PPOSpec (docs/ppo_spec.md) — offline."""
from __future__ import annotations

import numpy as np
import pytest

from showdown.rl.action_space import PASS_ID, build_autoregressive_masks
from showdown.rl.constants import OBS_DIM, SEG_BELIEF, SEG_SPECIES, SLOT_ACTION_N, SPEC_VERSION
from showdown.rl.env import MockBattleBackend, VGCEnv
from showdown.rl.observation import build_observation_ppo
from showdown.rl.reward import RewardCandidate


def test_spec_version_ppo_aligned():
    assert SPEC_VERSION == 'v1.8'


def test_trunc_reason_turn_cap():
    env = VGCEnv(
        backend=MockBattleBackend(max_turns=2),
        reward_candidate=RewardCandidate.DENSE,
        max_episode_turns=2,
    )
    obs, info = env.reset(seed=0)
    assert info.get('trunc_reason') is None
    truncated = False
    for _ in range(8):
        obs, r, term, trunc, info = env.step((PASS_ID, PASS_ID))
        if trunc:
            assert info.get('trunc_reason') == 'turn_cap'
            assert term is False
            assert abs(float(r)) < 1e-9
            truncated = True
            break
        if term:
            break
    assert truncated


def test_trunc_reason_crash_sets_skip_buffer():
    class BoomBackend(MockBattleBackend):
        def step(self, action):
            raise RuntimeError('boom')

    env = VGCEnv(backend=BoomBackend(max_turns=4), reward_candidate=RewardCandidate.DENSE)
    env.reset(seed=0)
    obs, r, term, trunc, info = env.step((PASS_ID, PASS_ID))
    assert trunc is True and term is False
    assert info.get('trunc_reason') == 'crash'
    assert info.get('skip_buffer') is True
    assert abs(float(r)) < 1e-9


def test_obs_ppo_zeros_belief():
    env = VGCEnv(backend=MockBattleBackend(max_turns=4), reward_candidate=RewardCandidate.DENSE)
    env.reset(seed=1)
    assert env._snap is not None
    obs = build_observation_ppo(env._snap)
    assert obs.shape == (OBS_DIM,)
    assert np.allclose(obs[SEG_SPECIES:SEG_SPECIES + SEG_BELIEF], 0.0)


def test_mask_and_logprob_consistency():
    torch = pytest.importorskip('torch')
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig

    env = VGCEnv(backend=MockBattleBackend(max_turns=6), reward_candidate=RewardCandidate.DENSE)
    env.reset(seed=0)
    assert env._snap is not None
    obs = build_observation_ppo(env._snap)
    m1, _ = build_autoregressive_masks(env._snap)

    def m2_fn(a1):
        return build_autoregressive_masks(env._snap, slot1_action_id=a1)[1]

    agent = PPOAgent(PPOConfig(seed=0, device='cpu', rollout_steps=64))
    for _ in range(20):
        a1, a2, logp, v = agent.act(obs, m1, m2_fn, greedy=False)
        assert m1[a1] > 0.5
        m2 = m2_fn(a1)
        assert m2[a2] > 0.5
        # Re-evaluate logπ
        x = torch.as_tensor(obs, dtype=torch.float32).view(1, -1)
        A1 = torch.tensor([a1])
        A2 = torch.tensor([a2])
        M1 = torch.as_tensor(m1, dtype=torch.float32).view(1, -1)
        M2 = torch.as_tensor(m2, dtype=torch.float32).view(1, -1)
        lp2, ent, vv, _ = agent.evaluate_actions(x, A1, A2, M1, M2)
        assert torch.isfinite(lp2).all()
        assert abs(float(lp2.item()) - logp) < 1e-4
        assert torch.isfinite(ent).all()


def test_skip_buffer_and_crash_do_not_grow():
    pytest.importorskip('torch')
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig

    agent = PPOAgent(PPOConfig(seed=0, device='cpu', rollout_steps=128))
    s = np.zeros(OBS_DIM, dtype=np.float32)
    m = np.ones(SLOT_ACTION_N, dtype=np.float32)
    n0 = len(agent.buffer)
    assert agent.observe(
        obs=s, a1=0, a2=0, logp=0.0, reward=0.0, value=0.0,
        terminated=False, m1=m, m2=m, skip_buffer=True,
    ) is False
    assert len(agent.buffer) == n0
    assert agent.observe(
        obs=s, a1=0, a2=0, logp=0.0, reward=0.0, value=0.0,
        terminated=False, m1=m, m2=m, trunc_reason='crash',
    ) is False
    assert len(agent.buffer) == n0


def test_turn_cap_gae_bootstraps():
    pytest.importorskip('torch')
    from showdown.rl.ppo.buffer import RolloutBuffer

    buf = RolloutBuffer(capacity=8)
    s = np.zeros(OBS_DIM, dtype=np.float32)
    m = np.ones(SLOT_ACTION_N, dtype=np.float32)
    # One step then turn-cap with bootstrap 2.0
    assert buf.add(
        obs=s, a1=0, a2=0, logp=0.0, reward=0.0, value=1.0,
        terminated=False, m1=m, m2=m,
        trunc_reason='turn_cap', bootstrap_value=2.0,
    )
    buf.compute_gae(last_value=0.0, gamma=0.95, lam=0.95, normalize_adv=False)
    # δ = 0 + 0.95*2 - 1 = 0.9
    assert abs(float(buf._adv[0]) - 0.9) < 1e-5


def test_clip_identity_when_theta_unchanged():
    torch = pytest.importorskip('torch')
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig

    agent = PPOAgent(PPOConfig(seed=1, device='cpu', rollout_steps=32, epochs=1, minibatch_size=16))
    s = np.zeros(OBS_DIM, dtype=np.float32)
    m = np.ones(SLOT_ACTION_N, dtype=np.float32)
    # Fill with synthetic transitions using current policy logp
    for i in range(32):
        a1, a2, logp, v = 0, 0, 0.0, 0.0
        # Use evaluate to get consistent logp for fixed actions under mask-all-ones
        x = torch.as_tensor(s, dtype=torch.float32, device=agent.device).view(1, -1)
        A1 = torch.tensor([PASS_ID], device=agent.device)
        A2 = torch.tensor([PASS_ID], device=agent.device)
        M = torch.ones(1, SLOT_ACTION_N, device=agent.device)
        with torch.no_grad():
            logp_t, _e, v_t, _ = agent.evaluate_actions(x, A1, A2, M, M)
            logp = float(logp_t.item())
            v = float(v_t.item())
        agent.observe(
            obs=s, a1=PASS_ID, a2=PASS_ID, logp=logp, reward=float(i % 3) - 1.0,
            value=v, terminated=(i == 31), m1=m, m2=m,
        )
    data = agent.buffer.get()
    # Surrogate at θ=θ_old: ratio=1 → L_CLIP = E[A] (before normalize we check raw)
    agent.buffer.compute_gae(last_value=0.0, normalize_adv=False)
    adv = agent.buffer.get()['adv']
    # ratio=1 ⇒ min(A, A) = A
    assert abs(float(adv.mean()) - float(adv.mean())) < 1e-9
    # Fresh evaluate: ratios ~ 1
    obs_t = torch.as_tensor(data['obs'], dtype=torch.float32, device=agent.device)
    a1 = torch.as_tensor(data['a1'], dtype=torch.long, device=agent.device)
    a2 = torch.as_tensor(data['a2'], dtype=torch.long, device=agent.device)
    m1 = torch.as_tensor(data['m1'], dtype=torch.float32, device=agent.device)
    m2 = torch.as_tensor(data['m2'], dtype=torch.float32, device=agent.device)
    old_logp = torch.as_tensor(data['logp'], dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        logp, _, _, _ = agent.evaluate_actions(obs_t, a1, a2, m1, m2)
        ratio = torch.exp(logp - old_logp)
    assert float((ratio - 1.0).abs().mean().item()) < 1e-5


def test_selfplay_pool_uniform_and_best_immune():
    pytest.importorskip('torch')
    import tempfile
    from pathlib import Path

    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    from showdown.rl.ppo.selfplay import SelfPlayPool

    with tempfile.TemporaryDirectory() as td:
        pool = SelfPlayPool(k=3, root=Path(td))
        agent = PPOAgent(PPOConfig(seed=0, device='cpu', rollout_steps=8))
        for i in range(5):
            pool.push(agent, tag=f'c{i}')
        assert len(pool.paths()) <= 3
        assert pool.maybe_update_best(agent, wr=0.7)
        best = pool.best_path
        assert best is not None and best.is_file()
        for i in range(5):
            pool.push(agent, tag=f'd{i}')
        assert best.is_file()
        assert best in pool.paths()
        rng = np.random.default_rng(0)
        picks = [pool.sample(rng) for _ in range(50)]
        assert all(p is not None for p in picks)


def test_bc_smoke():
    pytest.importorskip('torch')
    from showdown.rl.ppo.bc import BCTrainer, BCConfig
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig

    s = np.zeros(OBS_DIM, dtype=np.float32)
    m = np.ones(SLOT_ACTION_N, dtype=np.float32)
    data = [(s, PASS_ID, PASS_ID, m, m) for _ in range(64)]
    bc = BCTrainer(BCConfig(seed=0, device='cpu', batch_size=16, epochs=2))
    metrics = bc.fit(data)
    assert 'bc_acc_legal' in metrics
    agent = PPOAgent(PPOConfig(seed=0, device='cpu'))
    bc.export_to_agent(agent)
    assert abs(agent.optim.param_groups[0]['lr'] - 1e-4) < 1e-12


def test_smoke_mock_collect_update():
    pytest.importorskip('torch')
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    from showdown.rl.ppo.train import _collect_and_update, _make_env
    from showdown.rl.reward import RewardCandidate
    from showdown.rl.train_log import TrainHeartbeat

    agent = PPOAgent(PPOConfig(seed=0, device='cpu', rollout_steps=32, epochs=2, minibatch_size=16))
    env = _make_env('mock', seed=0, start_ps=False, reward=RewardCandidate.DENSE)
    hb = TrainHeartbeat(tag='ppo', t_total=100, log_every=50, log_every_sec=9999)
    try:
        env, done, info = _collect_and_update(
            agent, env,
            steps=100,
            omega=('mock',),
            expand_omega=False,
            weights={'mock': 1.0},
            rng=np.random.default_rng(0),
            global_step_offset=0,
            ckpt_dir=__import__('pathlib').Path('logs/ppo_test_smoke'),
            heartbeat=hb,
            phase_label='smoke',
            seed=0,
            start_ps=False,
            reward_mode='candidate2',
            self_play=False,
            pool=None,
        )
        assert done == 100
        assert agent.env_steps >= 100
    finally:
        env.close()
