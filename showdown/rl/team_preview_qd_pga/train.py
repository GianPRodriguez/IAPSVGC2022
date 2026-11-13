"""Train Team Preview QD-PGA (docs/team_preview_qd_pga_spec.md)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from showdown.rl.team_preview_env import TeamPreviewEnv
from showdown.rl.team_preview_obs import OBS_DIM
from showdown.rl.team_preview_ppo.network import PreviewActorCritic, apply_mask_logits
from showdown.rl.team_preview_qd_pga.archive import (
    Archive,
    actor_state_dict,
    load_actor_into,
)
from showdown.rl.team_preview_qd_pga.critic import SharedPreviewCritic
from showdown.rl.team_preview_qd_pga.evaluate import evaluate_actor
from showdown.rl.team_preview_qd_pga.improve import burn_rollout, pg_improve
from showdown.rl.team_preview_qd_pga.mutate import mutate_actor_state

B_BOOT = 4096
T_EMIT = 128
N_EMIT = 4
N_OUTER = 256
P_MUT = 0.5
SIGMA = 0.02
K_PG = 2
G_FIT = 20
G_MIN = 10


def _make_act_fn(net, device, rng):
    import torch

    def act_fn(obs, mask, greedy=True):
        net.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            m = torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = net(o)
            z = apply_mask_logits(logits, m)
            if greedy:
                return int(torch.argmax(z, dim=-1).item())
            dist = torch.distributions.Categorical(logits=z)
            return int(dist.sample().item())

    return act_fn


def train(
    *,
    seed: int = 0,
    smoke: bool = False,
    device: str = 'cpu',
    lr: Optional[float] = None,
    clip_eps: Optional[float] = None,
    live: Optional[dict] = None,
) -> Archive:
    import torch

    from showdown.rl.parallel import resolve_device
    from showdown.rl.preview_live import make_preview_env, provenance_meta

    device = resolve_device(device)

    b_boot = 64 if smoke else B_BOOT
    t_emit = 16 if smoke else T_EMIT
    n_emit = 2 if smoke else N_EMIT
    n_outer = 2 if smoke else N_OUTER
    g_fit = 4 if smoke else G_FIT
    g_min = 2 if smoke else G_MIN
    actor_lr = 3e-4 if lr is None else float(lr)
    actor_clip = 0.2 if clip_eps is None else float(clip_eps)

    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    dev = torch.device(device)
    env, pool = make_preview_env(seed, live)
    try:
        archive = _train_loop(
            env, dev, rng, b_boot=b_boot, t_emit=t_emit, n_emit=n_emit,
            n_outer=n_outer, g_fit=g_fit, g_min=g_min, actor_lr=actor_lr,
            actor_clip=actor_clip,
        )
        archive.provenance = provenance_meta(env, pool)
    finally:
        if pool is not None:
            pool.close()
    return archive


def _train_loop(env, dev, rng, *, b_boot, t_emit, n_emit, n_outer, g_fit, g_min,
                actor_lr, actor_clip) -> Archive:
    import torch

    net = PreviewActorCritic(OBS_DIM).to(dev)
    critic = SharedPreviewCritic(OBS_DIM).to(dev)
    actor_optim = torch.optim.Adam(
        [p for n, p in net.named_parameters() if not n.startswith('value.')],
        lr=actor_lr,
    )
    critic_optim = torch.optim.Adam(critic.parameters(), lr=actor_lr)
    archive = Archive(10, 10)
    act_fn = _make_act_fn(net, dev, rng)

    # Bootstrap: PG improves + try insert seed policy
    print(f'bootstrap episodes={b_boot}', flush=True)
    pg_improve(
        env, net, critic, actor_optim, critic_optim, dev,
        n_episodes=b_boot, epochs=K_PG, clip_eps=actor_clip,
    )
    f, bd, g, meta = evaluate_actor(env, act_fn, G=g_fit, g_min=g_min, greedy=True)
    if meta['eligible']:
        archive.try_insert(
            bd1=bd[0], bd2=bd[1], fitness=f,
            state_dict=actor_state_dict(net),
            meta={'phase': 'boot'},
        )
    print(
        f'boot done cov={archive.coverage():.2f} f={f:.3f} bd={bd}',
        flush=True,
    )

    for outer in range(int(n_outer)):
        for _e in range(int(n_emit)):
            filled = archive.filled_keys()
            if filled:
                key = filled[int(rng.integers(0, len(filled)))]
                parent_sd = archive.cells[key].state_dict
                load_actor_into(net, parent_sd)
            else:
                parent_sd = actor_state_dict(net)

            do_mut = bool(rng.random() < P_MUT)
            if do_mut:
                mut_sd = mutate_actor_state(parent_sd, rng, sigma=SIGMA)
                load_actor_into(net, mut_sd)
                burn_rollout(env, net, dev, n_episodes=t_emit, rng=rng)
            else:
                pg_improve(
                    env, net, critic, actor_optim, critic_optim, dev,
                    n_episodes=t_emit, epochs=K_PG, clip_eps=actor_clip,
                )

            f, bd, g, meta = evaluate_actor(
                env, act_fn, G=g_fit, g_min=g_min, greedy=True,
            )
            if meta['eligible']:
                archive.try_insert(
                    bd1=bd[0], bd2=bd[1], fitness=f,
                    state_dict=actor_state_dict(net),
                    meta={'outer': outer, 'mut': do_mut},
                )

        best = archive.best()
        bf = best.fitness if best else float('nan')
        print(
            f'outer={outer} cov={archive.coverage():.2f} '
            f'qd={archive.qd_score():.3f} best_f={bf:.3f}',
            flush=True,
        )

    return archive


def train_and_save(
    *,
    seed: int = 0,
    smoke: bool = False,
    device: str = 'cpu',
    checkpoint_dir: str | Path = 'logs/team_preview_qd_pga_checkpoints',
    lr: Optional[float] = None,
    clip_eps: Optional[float] = None,
    live: Optional[dict] = None,
) -> Archive:
    from showdown.rl.preview_live import provenance_lines

    actor_lr = 3e-4 if lr is None else float(lr)
    actor_clip = 0.2 if clip_eps is None else float(clip_eps)
    arch = train(
        seed=seed, smoke=smoke, device=device,
        lr=actor_lr, clip_eps=actor_clip, live=live,
    )
    prov = dict(getattr(arch, 'provenance', None) or {'provenance': 'mock'})
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    archive_pt = arch.save(ckpt_dir / 'archive.pt')
    best = arch.best()
    best_path = None
    best_f = float('nan')
    if best is not None:
        import torch
        best_path = ckpt_dir / 'best_elite.pt'
        best_f = float(best.fitness)
        torch.save({'state_dict': best.state_dict, 'fitness': best.fitness}, best_path)
    # Score for HP tune: best elite fitness (Cand. 1); NaN → 0 so trial is rejectable.
    score = 0.0 if (best_f != best_f) else best_f
    meta = {
        'seed': seed,
        'device': device,
        'smoke': smoke,
        'lr': actor_lr,
        'clip_eps': actor_clip,
        'coverage': arch.coverage(),
        'qd_score': arch.qd_score(),
        'best_f': None if (best_f != best_f) else best_f,
        'MEAN_SCORE': score,
        'note': 'MEAN_SCORE = best_f (elite fitness); independent of QD combat.',
        'archive_pt': str(archive_pt),
        'best_elite_pt': str(best_path) if best_path else None,
        **prov,
    }
    (ckpt_dir / 'train_meta.json').write_text(
        json.dumps(meta, indent=2) + '\n', encoding='utf-8',
    )
    (ckpt_dir / 'reporte.txt').write_text(
        '=== Team Preview QD-PGA train ===\n'
        f'seed: {seed}\n'
        f'lr: {actor_lr}\n'
        f'clip_eps: {actor_clip}\n'
        f'coverage: {arch.coverage():.6f}\n'
        f'qd_score: {arch.qd_score():.6f}\n'
        f'best_f: {best_f:.6f}\n'
        f'MEAN_SCORE: {score:.6f}\n'
        f'MEAN_WR: {score:.6f}\n'
        + provenance_lines(prov),
        encoding='utf-8',
    )
    print(f'MEAN_SCORE: {score:.6f}', flush=True)
    print(f'checkpoint: {archive_pt}', flush=True)
    return arch


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description='Team Preview QD-PGA')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cpu', help='cpu | cuda | auto')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--lr', type=float, default=None, help='Override actor/critic Adam lr')
    p.add_argument('--clip-eps', type=float, default=None, help='Override PPO clip_eps')
    p.add_argument('--checkpoint-dir', default='logs/team_preview_qd_pga_checkpoints')
    from showdown.rl.preview_live import add_live_cli_args, live_options_from_args
    add_live_cli_args(p)
    args = p.parse_args(argv)
    arch = train_and_save(
        seed=args.seed,
        smoke=args.smoke,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        lr=args.lr,
        clip_eps=args.clip_eps,
        live=live_options_from_args(args),
    )
    print(
        f'DONE team_preview_qd_pga cov={arch.coverage():.3f} '
        f'cells={len(arch.cells)}',
        flush=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
