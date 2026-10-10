"""Acceptance tests for DDQNSpec (docs/ddqn_spec.md) — offline."""
from __future__ import annotations

import numpy as np
import pytest

from showdown.rl.action_space import (
    PASS_ID,
    decode_joint_action,
    encode_joint_action,
    legal_joint_mask,
)
from showdown.rl.constants import JOINT_ACTION_N, OBS_DIM, SEG_BELIEF, SEG_SPECIES, SPEC_VERSION
from showdown.rl.env import MockBattleBackend, VGCEnv
from showdown.rl.observation import build_observation_ddqn
from showdown.rl.reward import RewardCandidate


def test_spec_version_v14():
    assert SPEC_VERSION == 'v1.6'
    assert JOINT_ACTION_N == 576


def test_parse_phases_and_schedule():
    from showdown.rl.ddqn.train import (
        parse_phases,
        resolve_schedule,
        default_curriculum_phases,
    )

    phases = parse_phases('random:100000,greedy_solo:50k,greedy_joint:1.5k')
    assert [(p.opponent, p.steps) for p in phases] == [
        ('random', 100_000),
        ('greedy_solo', 50_000),
        ('greedy_joint', 1_500),
    ]
    single = resolve_schedule(opponent='greedy_joint', steps=1234, phases=None)
    assert len(single) == 1 and single[0].opponent == 'greedy_joint'
    assert single[0].steps == 1234
    multi = resolve_schedule(
        opponent='mock', steps=1,
        phases='random:10,greedy_solo:20',
    )
    assert len(multi) == 2
    cur = resolve_schedule(opponent='curriculum', steps=1, phases=None, curriculum_phase_steps=100)
    assert [p.opponent for p in cur] == [
        'random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian',
    ]
    assert all(p.steps == 100 for p in cur)
    assert len(default_curriculum_phases(7)) == 4


def test_expand_omega_monotonic_and_weights():
    from showdown.rl.ddqn.curriculum import (
        expand_phases_from_exclusive,
        omega_sampling_probs,
        sample_from_omega,
    )
    from showdown.rl.ddqn.train import TrainPhase

    exclusive = [
        TrainPhase('random', 10),
        TrainPhase('greedy_solo', 20),
        TrainPhase('greedy_joint', 30),
        TrainPhase('expectimax_bayesian', 40),
    ]
    expand = expand_phases_from_exclusive(exclusive)
    assert expand[0].omega == ('random',)
    assert expand[1].omega == ('random', 'greedy_solo')
    assert expand[2].omega == ('random', 'greedy_solo', 'greedy_joint')
    assert expand[3].omega == (
        'random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian',
    )
    probs = omega_sampling_probs(expand[3].omega)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs['expectimax_bayesian'] < probs['random']
    rng = np.random.default_rng(0)
    picks = [sample_from_omega(expand[1].omega, rng) for _ in range(200)]
    assert set(picks) <= {'random', 'greedy_solo'}


def test_replay_tracks_opponent_origin():
    from showdown.rl.ddqn.replay import ReplayBuffer
    from showdown.rl.ddqn.curriculum import opponent_code

    buf = ReplayBuffer(capacity=32)
    s = np.zeros(OBS_DIM, dtype=np.float32)
    m = np.ones(JOINT_ACTION_N, dtype=np.float32)
    for _ in range(6):
        buf.add(
            s=s, a=0, r=0.0, s_next=s, terminated=False, m_next=m,
            opponent_code=opponent_code('random'),
        )
    for _ in range(2):
        buf.add(
            s=s, a=1, r=0.0, s_next=s, terminated=False, m_next=m,
            opponent_code=opponent_code('greedy_joint'),
        )
    frac = buf.origin_fractions()
    assert abs(frac['random'] - 0.75) < 1e-6
    assert abs(frac['greedy_joint'] - 0.25) < 1e-6
    _s, _a, _r, _s2, _t, _m, opp = buf.sample(4, np.random.default_rng(1))
    batch_frac = ReplayBuffer.batch_origin_fractions(opp)
    assert abs(sum(batch_frac.values()) - 1.0) < 1e-6


def test_joint_codec_idempotent():
    for a1 in range(24):
        for a2 in (0, 7, 20, 23):
            j = encode_joint_action(a1, a2)
            b1, b2 = decode_joint_action(j)
            assert (b1, b2) == (a1, a2)
            assert 0 <= j < JOINT_ACTION_N


def test_legal_joint_mask_shape_and_pass():
    env = VGCEnv(backend=MockBattleBackend(max_turns=4), reward_candidate=RewardCandidate.DENSE)
    obs, info = env.reset(seed=0)
    assert env._snap is not None
    m = legal_joint_mask(env._snap)
    assert m.shape == (576,)
    assert m.dtype == np.float32
    assert np.any(m > 0.5)
    # pass/pass should be legal on fresh mock
    assert m[encode_joint_action(PASS_ID, PASS_ID)] > 0.5


def test_obs_ddqn_zeros_belief():
    env = VGCEnv(backend=MockBattleBackend(max_turns=4), reward_candidate=RewardCandidate.DENSE)
    env.reset(seed=1)
    assert env._snap is not None
    obs = build_observation_ddqn(env._snap)
    assert obs.shape == (OBS_DIM,)
    assert np.allclose(obs[SEG_SPECIES:SEG_SPECIES + SEG_BELIEF], 0.0)


def test_skip_buffer_does_not_grow_replay():
    torch = pytest.importorskip('torch')
    from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig

    agent = DDQNAgent(DDQNConfig(seed=0, device='cpu', warmup_steps=10_000))
    s = np.zeros(OBS_DIM, dtype=np.float32)
    m = np.ones(JOINT_ACTION_N, dtype=np.float32)
    n0 = len(agent.replay)
    ok = agent.replay.add(
        s=s, a=0, r=0.0, s_next=s, terminated=False, m_next=m,
        skip_buffer=True,
    )
    assert ok is False
    assert len(agent.replay) == n0
    ok2 = agent.replay.add(
        s=s, a=0, r=0.0, s_next=s, terminated=False, m_next=m,
        truncated=True,
    )
    assert ok2 is False
    assert len(agent.replay) == n0
    assert agent.replay.add(
        s=s, a=0, r=0.1, s_next=s, terminated=False, m_next=m,
    )
    assert len(agent.replay) == n0 + 1


def test_double_update_uses_online_argmax():
    torch = pytest.importorskip('torch')
    from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig

    agent = DDQNAgent(DDQNConfig(
        seed=0, device='cpu', warmup_steps=0, batch_size=8,
        train_freq=1, replay_capacity=64,
    ))
    # Fill replay
    s = np.random.randn(OBS_DIM).astype(np.float32)
    m = np.ones(JOINT_ACTION_N, dtype=np.float32)
    for i in range(16):
        agent.replay.add(
            s=s, a=i % JOINT_ACTION_N, r=0.1, s_next=s,
            terminated=False, m_next=m,
        )
    # Make online prefer action 3, target prefer 7 on next-state
    with torch.no_grad():
        agent.online.net[-1].bias.zero_()
        agent.target.net[-1].bias.zero_()
        agent.online.net[-1].bias[3] = 50.0
        agent.target.net[-1].bias[7] = 50.0
    # Spy: run one learn and check a_star would be 3 from online
    s_b, a_b, r_b, s2_b, term_b, m2_b, _opp = agent.replay.sample(8, agent._rng)
    s2_t = torch.as_tensor(s2_b, dtype=torch.float32)
    m2_t = torch.as_tensor(m2_b, dtype=torch.float32)
    with torch.no_grad():
        q_on = agent.online(s2_t) + (1.0 - m2_t) * (-1e9)
        a_star = int(q_on[0].argmax().item())
    assert a_star == 3
    metrics = agent.learn()
    assert 'loss' in metrics


def test_greedy_respects_mask():
    torch = pytest.importorskip('torch')
    from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig

    agent = DDQNAgent(DDQNConfig(seed=0, device='cpu'))
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    mask = np.zeros(JOINT_ACTION_N, dtype=np.float32)
    mask[42] = 1.0
    mask[100] = 1.0
    a = agent.act(obs, mask, epsilon=0.0)
    assert mask[a] > 0.5


def test_smoke_mock_env_steps():
    torch = pytest.importorskip('torch')
    from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
    from showdown.rl.action_space import decode_joint_action

    agent = DDQNAgent(DDQNConfig(seed=1, device='cpu', warmup_steps=10_000))
    env = VGCEnv(
        backend=MockBattleBackend(max_turns=40),
        reward_candidate=RewardCandidate.DENSE,
        max_episode_turns=40,
    )
    obs, info = env.reset(seed=1)
    assert env._snap is not None
    obs = build_observation_ddqn(env._snap)
    for _ in range(100):
        mask = legal_joint_mask(env._snap)
        joint = agent.act(obs, mask, epsilon=1.0)
        a1, a2 = decode_joint_action(joint)
        obs, r, term, trunc, info = env.step((a1, a2))
        assert env._snap is not None
        obs = build_observation_ddqn(env._snap)
        if term or trunc:
            obs, info = env.reset()
            assert env._snap is not None
            obs = build_observation_ddqn(env._snap)
    env.close()
