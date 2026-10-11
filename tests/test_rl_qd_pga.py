"""Acceptance tests for QD-PGA-MAP-Elites (docs/qd_pga_map_elites_spec.md)."""
from __future__ import annotations

import numpy as np

from showdown.rl.constants import SPEC_VERSION
from showdown.rl.qd_pga.archive import Archive, bin_bd, is_actor_key
from showdown.rl.qd_pga.mutate import mutate_actor_state
from showdown.rl.qd_pga.replay import ValueReplay
from showdown.rl.qd_pga.train import derived_n_outer


def test_spec_version_qd_pga_aligned():
    assert SPEC_VERSION == 'v1.7'


def test_derived_n_outer_pga():
    assert derived_n_outer() == 219
    assert derived_n_outer(t_total=1_000_000, b_boot=100_000, n_emit=4, t_emit=1024) == 219


def test_actor_only_keys():
    assert is_actor_key('fc1.weight')
    assert is_actor_key('policy1.bias')
    assert not is_actor_key('value.weight')
    assert not is_actor_key('value.bias')


def test_mutation_does_not_write_replay_contract():
    """Mut burn-rollout must not grow the shared replay."""
    from showdown.rl.env import MockBattleBackend, VGCEnv
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    from showdown.rl.qd_pga.critic import SharedCritic
    from showdown.rl.qd_pga.improve import mut_burn_rollout, pg_improve
    from showdown.rl.reward import RewardCandidate

    rng = np.random.default_rng(0)
    cfg = PPOConfig(seed=0, device='cpu', rollout_steps=64)
    agent = PPOAgent(cfg)
    critic = SharedCritic(device='cpu')
    replay = ValueReplay(capacity=10_000)
    env = VGCEnv(
        reward_candidate=RewardCandidate.DENSE,
        backend=MockBattleBackend(max_turns=40),
        max_episode_turns=40,
    )
    try:
        before = len(replay)
        env, taken, _ = mut_burn_rollout(
            agent, env, critic,
            steps=32, rng=rng, global_steps=0, t_total=10_000, seed=0,
        )
        assert taken > 0
        assert len(replay) == before
        env, taken2, _ = pg_improve(
            agent, env, critic, replay,
            steps=64, rng=rng, global_steps=taken, t_total=10_000, seed=1,
            k_pg=1, g_c=1,
        )
        assert taken2 > 0
        assert len(replay) > before
    finally:
        env.close()

    sd = {'fc1.weight': np.zeros((2, 2), dtype=np.float32)}
    mut = mutate_actor_state(sd, rng, sigma=0.02)
    assert not np.allclose(mut['fc1.weight'], sd['fc1.weight'])


def test_replay_rejects_skip_and_crash():
    replay = ValueReplay(capacity=16)
    s = np.zeros(1024, dtype=np.float32)
    assert not replay.add(s=s, r=0.0, s_next=s, terminated=False, skip_buffer=True)
    assert not replay.add(
        s=s, r=0.0, s_next=s, terminated=False, trunc_reason='crash',
    )
    assert replay.add(s=s, r=0.1, s_next=s, terminated=False)
    assert len(replay) == 1


def test_archive_actor_only_insert():
    arch = Archive()
    sd = {'fc1.weight': np.array([1.0]), 'value.weight': np.array([9.0])}
    # Simulate actor-only payload
    actor_sd = {k: v for k, v in sd.items() if is_actor_key(k)}
    assert 'value.weight' not in actor_sd
    assert arch.try_insert(bd1=0.1, bd2=0.2, fitness=0.3, state_dict=actor_sd)
    elite = arch.cells[bin_bd(0.1, 0.2)]
    assert 'value.weight' not in elite.state_dict


def test_smoke_train_mock():
    from showdown.rl.qd_pga.train import run_training

    arch = run_training(seed=0, smoke=True, checkpoint_dir='logs/qd_pga_test_smoke')
    assert isinstance(arch, Archive)
    assert arch.coverage() >= 0.0
