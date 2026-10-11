"""Tests for Team Preview Deep Payoff Nash (spec v1.0)."""
from __future__ import annotations

import numpy as np
import pytest

from showdown.rl.team_preview_action import N_ACTIONS
from showdown.rl.team_preview_nash.nash_solver import solve_mixed_strategy, support_size


def test_nash_solver_simplex():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 8))
    p, q, v, method = solve_mixed_strategy(A, timeout_s=2.0, fp_iters=100)
    assert p.shape == (8,) and q.shape == (8,)
    assert abs(p.sum() - 1.0) < 1e-5
    assert abs(q.sum() - 1.0) < 1e-5
    assert p.min() >= -1e-9
    assert method in ('lp', 'fp', 'softmax')
    assert np.isfinite(v)


def test_nash_dominated_row_mass():
    # Row 1 strictly dominated by row 0
    A = np.array([
        [2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.5, 0.5],
    ], dtype=np.float64)
    p, _q, _v, method = solve_mixed_strategy(A, timeout_s=2.0)
    assert method in ('lp', 'fp', 'softmax')
    assert p[1] < 1e-6


def test_payoff_matrix_shape_and_range():
    torch = pytest.importorskip('torch')
    from showdown.rl.team_preview_nash.network import LowRankPayoffNet
    from showdown.rl.team_preview_obs import OBS_DIM

    net = LowRankPayoffNet()
    obs = torch.zeros(OBS_DIM)
    M = net.payoff_matrix(obs).detach()
    assert M.shape == (N_ACTIONS, N_ACTIONS)
    assert float(M.min()) >= 0.0 - 1e-5
    assert float(M.max()) <= 1.0 + 1e-5


def test_agent_act_eval_and_sample():
    torch = pytest.importorskip('torch')
    del torch
    gymnasium = pytest.importorskip('gymnasium')
    del gymnasium
    from showdown.rl.team_preview_env import TeamPreviewEnv
    from showdown.rl.team_preview_nash.agent import PreviewNashAgent, PreviewNashConfig

    env = TeamPreviewEnv(seed=1, backend='mock')
    obs, info = env.reset()
    agent = PreviewNashAgent(PreviewNashConfig(seed=1, device='cpu'))
    a_g, inf_g = agent.act(obs, greedy=True, mask=info['action_mask'])
    assert 0 <= a_g < N_ACTIONS
    assert abs(inf_g['entropy_p']) >= 0.0
    assert support_size(np.ones(N_ACTIONS) / N_ACTIONS) == N_ACTIONS
    # stochastic path runs
    a_s, _ = agent.act(obs, greedy=False, mask=info['action_mask'])
    assert 0 <= a_s < N_ACTIONS
    assert len(inf_g['team']) == 4


def test_pair_label_wr_scale():
    gymnasium = pytest.importorskip('gymnasium')
    del gymnasium
    from showdown.rl.team_preview_env import TeamPreviewEnv

    env = TeamPreviewEnv(seed=0, backend='mock')
    env.reset()
    r = env.pair_label(0, 1)
    assert r in (0.0, 0.5, 1.0)


def test_nash_smoke_train():
    torch = pytest.importorskip('torch')
    del torch
    gymnasium = pytest.importorskip('gymnasium')
    del gymnasium
    from showdown.rl.team_preview_nash.train import train

    agent = train(seed=0, smoke=True, updates=1, device='cpu')
    assert agent.env_pairs == 64
    assert agent.update_steps == 1
