"""Train Team Preview Masked PPO (docs/team_preview_masked_ppo_spec.md)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    lr: Optional[float] = None,
    clip_eps: Optional[float] = None,
    checkpoint_dir: str | Path = 'logs/team_preview_ppo_checkpoints',
) -> PreviewPPOAgent:
    cfg = PreviewPPOConfig(seed=seed, device=device)
    if lr is not None:
        cfg.lr = float(lr)
    if clip_eps is not None:
        cfg.clip_eps = float(clip_eps)
    n = cfg.smoke_episodes if smoke else cfg.rollout_episodes
    if smoke:
        cfg.rollout_episodes = n
    agent = PreviewPPOAgent(cfg)
    env = TeamPreviewEnv(seed=seed, backend='mock')
    last_metrics = {}
    for u in range(int(updates)):
        collect_rollout(env, agent, n)
        metrics = agent.update()
        last_metrics = metrics
        print(
            f'update={u} episodes={agent.env_episodes} '
            f'pg={metrics.get("pg_loss", 0):.4f} '
            f'v={metrics.get("v_loss", 0):.4f} '
            f'H={metrics.get("entropy", 0):.4f}',
            flush=True,
        )
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Score for HP tune: higher is better → negative pg_loss (proxy when no suite).
    pg = float(last_metrics.get('pg_loss', 0.0) or 0.0)
    score = -pg
    meta = {
        'seed': seed,
        'device': device,
        'smoke': smoke,
        'updates': updates,
        'lr': float(cfg.lr),
        'clip_eps': float(cfg.clip_eps),
        'last_metrics': {k: float(v) for k, v in last_metrics.items()},
        'MEAN_SCORE': score,
        'note': 'MEAN_SCORE = -pg_loss (proxy); live BO1 suite is separate.',
    }
    (ckpt_dir / 'train_meta.json').write_text(
        json.dumps(meta, indent=2) + '\n', encoding='utf-8',
    )
    (ckpt_dir / 'reporte.txt').write_text(
        '=== Team Preview PPO train ===\n'
        f'seed: {seed}\n'
        f'lr: {cfg.lr}\n'
        f'clip_eps: {cfg.clip_eps}\n'
        f'updates: {updates}\n'
        f'pg_loss: {pg:.6f}\n'
        f'MEAN_SCORE: {score:.6f}\n'
        f'MEAN_WR: {score:.6f}\n',
        encoding='utf-8',
    )
    final_pt = agent.save(ckpt_dir / 'final.pt')
    meta['final_pt'] = str(final_pt)
    (ckpt_dir / 'train_meta.json').write_text(
        json.dumps(meta, indent=2) + '\n', encoding='utf-8',
    )
    print(f'MEAN_SCORE: {score:.6f}', flush=True)
    print(f'checkpoint: {final_pt}', flush=True)
    return agent


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description='Team Preview Masked PPO')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--smoke', action='store_true', help='N=64 episodes / update')
    p.add_argument('--updates', type=int, default=1)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--clip-eps', type=float, default=None)
    p.add_argument('--checkpoint-dir', default='logs/team_preview_ppo_checkpoints')
    args = p.parse_args(argv)
    train(
        seed=args.seed,
        smoke=args.smoke,
        updates=args.updates,
        device=args.device,
        lr=args.lr,
        clip_eps=args.clip_eps,
        checkpoint_dir=args.checkpoint_dir,
    )
    print('DONE team_preview_ppo', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
