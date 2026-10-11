"""Train Team Preview Nash Value Net (docs/team_preview_nash_spec.md)."""
from __future__ import annotations

import argparse
from typing import Optional, Tuple

import numpy as np

from showdown.rl.team_preview_action import N_ACTIONS
from showdown.rl.team_preview_env import TeamPreviewEnv
from showdown.rl.team_preview_nash.agent import PreviewNashAgent, PreviewNashConfig
from showdown.rl.team_preview_obs import OBS_DIM


def collect_uniform_pairs(
    env: TeamPreviewEnv,
    agent: PreviewNashAgent,
    n_pairs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """v1.0: uniform independent (i, j) + env.pair_label → r ∈ {0, 0.5, 1}."""
    n = int(n_pairs)
    obs_buf = np.zeros((n, OBS_DIM), dtype=np.float32)
    i_buf = np.zeros(n, dtype=np.int64)
    j_buf = np.zeros(n, dtype=np.int64)
    r_buf = np.zeros(n, dtype=np.float32)
    rng = agent._rng
    for k in range(n):
        obs, _info = env.reset()
        i = int(rng.integers(0, N_ACTIONS))
        j = int(rng.integers(0, N_ACTIONS))
        r = float(env.pair_label(i, j))
        obs_buf[k] = obs
        i_buf[k] = i
        j_buf[k] = j
        r_buf[k] = r
        agent.env_pairs += 1
    return obs_buf, i_buf, j_buf, r_buf


def train(
    *,
    seed: int = 0,
    smoke: bool = False,
    updates: int = 1,
    device: str = 'cpu',
) -> PreviewNashAgent:
    cfg = PreviewNashConfig(seed=seed, device=device)
    n = cfg.smoke_pairs if smoke else cfg.rollout_pairs
    agent = PreviewNashAgent(cfg)
    env = TeamPreviewEnv(seed=seed, backend='mock')
    for u in range(int(updates)):
        obs, i, j, r = collect_uniform_pairs(env, agent, n)
        metrics = agent.update_from_batch(obs, i, j, r)
        # quick sanity: one Nash solve on last obs
        _a, info = agent.act(obs[-1], greedy=True)
        print(
            f'update={u} pairs={agent.env_pairs} '
            f'loss={metrics.get("loss", 0):.4f} '
            f'v*={info["v_star"]:.3f} '
            f'H={info["entropy_p"]:.3f} '
            f'supp={info["support"]} '
            f'solver={info["solver"]}',
            flush=True,
        )
    return agent


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description='Team Preview Deep Payoff Nash')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--smoke', action='store_true', help='N=64 pairs / update')
    p.add_argument('--updates', type=int, default=1)
    args = p.parse_args(argv)
    train(seed=args.seed, smoke=args.smoke, updates=args.updates, device=args.device)
    print('DONE team_preview_nash', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
