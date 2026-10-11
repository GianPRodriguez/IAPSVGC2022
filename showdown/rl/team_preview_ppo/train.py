"""Train Team Preview Masked PPO (docs/team_preview_masked_ppo_spec.md)."""
from __future__ import annotations

import argparse
from typing import Optional

from showdown.rl.team_preview_env import TeamPreviewEnv
from showdown.rl.team_preview_ppo.agent import PreviewPPOAgent, PreviewPPOConfig


def collect_rollout(env: TeamPreviewEnv, agent: PreviewPPOAgent, n_episodes: int) -> None:
    for _ in range(int(n_episodes)):
        obs, info = env.reset()
        mask = info.get('action_mask')
        if mask is None:
            mask = env.action_masks()
        a, logp, v = agent.act(obs, mask, greedy=False)
        _obs2, reward, terminated, truncated, _info = env.step(a)
        assert terminated or truncated
        agent.buffer.add(obs, a, reward, logp, v, mask)
        agent.env_episodes += 1


def train(
    *,
    seed: int = 0,
    smoke: bool = False,
    updates: int = 1,
    device: str = 'cpu',
) -> PreviewPPOAgent:
    cfg = PreviewPPOConfig(seed=seed, device=device)
    n = cfg.smoke_episodes if smoke else cfg.rollout_episodes
    # For smoke, shrink buffer capacity to n
    if smoke:
        cfg.rollout_episodes = n
    agent = PreviewPPOAgent(cfg)
    env = TeamPreviewEnv(seed=seed, backend='mock')
    for u in range(int(updates)):
        collect_rollout(env, agent, n)
        metrics = agent.update()
        print(
            f'update={u} episodes={agent.env_episodes} '
            f'pg={metrics.get("pg_loss", 0):.4f} '
            f'v={metrics.get("v_loss", 0):.4f} '
            f'H={metrics.get("entropy", 0):.4f}',
            flush=True,
        )
    return agent


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description='Team Preview Masked PPO')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--smoke', action='store_true', help='N=64 episodes / update')
    p.add_argument('--updates', type=int, default=1)
    args = p.parse_args(argv)
    train(seed=args.seed, smoke=args.smoke, updates=args.updates, device=args.device)
    print('DONE team_preview_ppo', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
