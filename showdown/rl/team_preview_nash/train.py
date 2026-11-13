"""Train Team Preview Nash Value Net (docs/team_preview_nash_spec.md)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    if n <= 0:
        return obs_buf, i_buf, j_buf, r_buf
    # Por lotes: N contextos y N pares (i, j); en live, N BO1 en paralelo.
    obs, _infos = env.reset_batch(n)
    for k in range(n):
        i_buf[k] = int(rng.integers(0, N_ACTIONS))
        j_buf[k] = int(rng.integers(0, N_ACTIONS))
    r_buf[:] = env.pair_label_batch(i_buf, j_buf)
    obs_buf[:] = obs
    agent.env_pairs += n
    return obs_buf, i_buf, j_buf, r_buf


def train(
    *,
    seed: int = 0,
    smoke: bool = False,
    updates: int = 1,
    device: str = 'cpu',
    checkpoint_dir: str | Path = 'logs/team_preview_nash_checkpoints',
    lr: Optional[float] = None,
    live: Optional[dict] = None,
) -> PreviewNashAgent:
    from showdown.rl.parallel import resolve_device
    from showdown.rl.preview_live import make_preview_env, provenance_lines, provenance_meta

    device = resolve_device(device)
    cfg = PreviewNashConfig(seed=seed, device=device)
    if lr is not None:
        cfg.lr = float(lr)
    n = cfg.smoke_pairs if smoke else cfg.rollout_pairs
    agent = PreviewNashAgent(cfg)
    env, pool = make_preview_env(seed, live)
    last_metrics = {}
    last_info = {}
    try:
        for u in range(int(updates)):
            obs, i, j, r = collect_uniform_pairs(env, agent, n)
            metrics = agent.update_from_batch(obs, i, j, r)
            last_metrics = metrics
            # quick sanity: one Nash solve on last obs
            _a, info = agent.act(obs[-1], greedy=True)
            last_info = info
            print(
                f'update={u} pairs={agent.env_pairs} '
                f'loss={metrics.get("loss", 0):.4f} '
                f'v*={info["v_star"]:.3f} '
                f'H={info["entropy_p"]:.3f} '
                f'supp={info["support"]} '
                f'solver={info["solver"]} '
                f'backend={env.provenance}',
                flush=True,
            )
        prov = provenance_meta(env, pool)
    finally:
        if pool is not None:
            pool.close()
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_pt = agent.save(ckpt_dir / 'final.pt')
    loss = float(last_metrics.get('loss', 0.0) or 0.0)
    score = -loss  # higher is better for HP tune
    meta = {
        'seed': seed,
        'device': device,
        'smoke': smoke,
        'updates': updates,
        'lr': float(cfg.lr),
        'last_metrics': {k: float(v) for k, v in last_metrics.items()},
        'v_star': float(last_info.get('v_star', 0.0) or 0.0),
        'MEAN_SCORE': score,
        'note': 'MEAN_SCORE = -loss (proxy); independent of MCTS combat.',
        'final_pt': str(final_pt),
        **prov,
    }
    (ckpt_dir / 'train_meta.json').write_text(
        json.dumps(meta, indent=2) + '\n', encoding='utf-8',
    )
    (ckpt_dir / 'reporte.txt').write_text(
        '=== Team Preview Nash train ===\n'
        f'seed: {seed}\n'
        f'lr: {cfg.lr}\n'
        f'updates: {updates}\n'
        f'loss: {loss:.6f}\n'
        f'v_star: {meta["v_star"]:.6f}\n'
        f'MEAN_SCORE: {score:.6f}\n'
        f'MEAN_WR: {score:.6f}\n'
        + provenance_lines(prov),
        encoding='utf-8',
    )
    print(f'MEAN_SCORE: {score:.6f}', flush=True)
    print(f'checkpoint: {final_pt}', flush=True)
    return agent


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description='Team Preview Deep Payoff Nash')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cpu', help='cpu | cuda | auto')
    p.add_argument('--smoke', action='store_true', help='N=64 pairs / update')
    p.add_argument('--updates', type=int, default=1)
    p.add_argument('--lr', type=float, default=None, help='Override PreviewNashConfig.lr')
    p.add_argument('--checkpoint-dir', default='logs/team_preview_nash_checkpoints')
    from showdown.rl.preview_live import add_live_cli_args, live_options_from_args
    add_live_cli_args(p)
    args = p.parse_args(argv)
    train(
        seed=args.seed,
        smoke=args.smoke,
        updates=args.updates,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        lr=args.lr,
        live=live_options_from_args(args),
    )
    print('DONE team_preview_nash', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
