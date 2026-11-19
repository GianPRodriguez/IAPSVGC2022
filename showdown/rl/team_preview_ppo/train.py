"""Train Team Preview Masked PPO (docs/team_preview_masked_ppo_spec.md)."""
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Optional

from showdown.rl.preview_eval import (
    DEFAULT_EVAL_SEED,
    add_eval_cli_args,
    evaluate_preview_policy,
    resolve_eval_n,
)
from showdown.rl.preview_metrics import formation_stats
from showdown.rl.team_preview_env import TeamPreviewEnv
from showdown.rl.team_preview_obs import OBS_DIM, OBS_VERSION
from showdown.rl.train_metrics import MetricsLog
from showdown.rl.team_preview_ppo.agent import PreviewPPOAgent, PreviewPPOConfig


def collect_rollout(env: TeamPreviewEnv, agent: PreviewPPOAgent, n_episodes: int) -> dict:
    """Rollout por lotes: N contextos, N acciones y N BO1 (en paralelo si live).

    Devuelve las métricas de estrategia del lote (``preview_metrics.formation_stats``).
    """
    n = int(n_episodes)
    if n <= 0:
        return {}
    obs, infos = env.reset_batch(n)
    bundles = [b for (b, _ctx) in env._batch]
    masks = []
    acts, logps, values = [], [], []
    for k in range(n):
        mask = infos[k].get('action_mask')
        if mask is None:
            mask = env.action_masks()
        a, logp, v = agent.act(obs[k], mask, greedy=False)
        masks.append(mask)
        acts.append(a)
        logps.append(logp)
        values.append(v)
    rewards = env.step_batch(acts)
    for k in range(n):
        agent.buffer.add(obs[k], acts[k], float(rewards[k]), logps[k], values[k], masks[k])
        agent.env_episodes += 1
    return formation_stats(bundles, acts, rewards)


def train(
    *,
    seed: int = 0,
    smoke: bool = False,
    updates: int = 1,
    device: str = 'cpu',
    lr: Optional[float] = None,
    clip_eps: Optional[float] = None,
    checkpoint_dir: str | Path = 'logs/team_preview_ppo_checkpoints',
    canonical: bool = False,
    ent_coef: Optional[float] = None,
    live: Optional[dict] = None,
    eval_episodes: int = 512,
    eval_baseline: str = 'fixed',
    eval_seed: int = DEFAULT_EVAL_SEED,
    eval_every: int = 5,
    eval_periodic_episodes: int = 256,
    obs_abilities: Optional[bool] = None,
) -> PreviewPPOAgent:
    from showdown.rl.parallel import resolve_device
    from showdown.rl.preview_live import make_preview_env, provenance_lines, provenance_meta

    device = resolve_device(device)
    cfg = PreviewPPOConfig(seed=seed, device=device)
    cfg.canonical = bool(canonical)
    if ent_coef is not None:
        cfg.ent_coef = float(ent_coef)
    if lr is not None:
        cfg.lr = float(lr)
    if clip_eps is not None:
        cfg.clip_eps = float(clip_eps)
    n = cfg.smoke_episodes if smoke else cfg.rollout_episodes
    if smoke:
        cfg.rollout_episodes = n
    if obs_abilities is not None:
        cfg.obs_abilities = bool(obs_abilities)
    # Observación del preview (y de sus entornos, que heredan la variable): sigue a la config (P5).
    os.environ['RL_OBS_ABILITIES'] = '1' if cfg.obs_abilities else '0'
    print(f'[preview] observación: corrección de habilidades (P5) = {bool(cfg.obs_abilities)}', flush=True)
    agent = PreviewPPOAgent(cfg)
    env, pool = make_preview_env(seed, live)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    mlog = MetricsLog(ckpt_dir / 'train_metrics.jsonl', run_meta={
        'agent': 'team_preview_ppo', 'seed': seed, 'device': device, 'smoke': smoke,
        'updates': updates, 'episodes_per_update': n, 'lr': float(cfg.lr),
        'clip_eps': float(cfg.clip_eps), 'obs_version': OBS_VERSION, 'obs_dim': OBS_DIM,
        'obs_abilities': bool(cfg.obs_abilities),
        'backend': env.provenance, 'eval_baseline': eval_baseline, 'eval_seed': eval_seed,
    })
    n_periodic = resolve_eval_n(eval_periodic_episodes, smoke=smoke, smoke_n=32)
    best_wr = -1.0
    last_metrics = {}

    def _greedy(o, m):
        return agent.act(o, m, greedy=True)[0]

    try:
        for u in range(int(updates)):
            strat = collect_rollout(env, agent, n)
            metrics = agent.update()
            last_metrics = metrics
            mlog.write('update', update=u, episodes=agent.env_episodes,
                       **{**metrics, **strat})
            print(
                f'update={u} episodes={agent.env_episodes} '
                f'wr_rollout={strat.get("rollout_wr", float("nan")):.3f} '
                f'pg={metrics.get("pg_loss", 0):.4f} '
                f'v={metrics.get("v_loss", 0):.4f} '
                f'H={metrics.get("entropy", 0):.3f} '
                f'eff_A={metrics.get("effective_actions", 0):.1f} '
                f'kl={metrics.get("approx_kl", 0):.4f} '
                f'clip={metrics.get("clip_frac", 0):.3f} '
                f'ev={metrics.get("explained_var", 0):.3f} '
                f'backend={env.provenance}',
                flush=True,
            )
            last_u = (u + 1) == int(updates)
            if n_periodic > 0 and eval_every > 0 and ((u + 1) % eval_every == 0) and not last_u:
                pev = evaluate_preview_policy(env, _greedy, n=n_periodic,
                                              baseline=eval_baseline, eval_seed=eval_seed)
                mlog.write('eval', update=u, episodes=agent.env_episodes, final=False, **pev.as_dict())
                print(f'[eval] update={u} ' + pev.lines().replace('\n', ' | '), flush=True)
                if pev.mean_wr > best_wr:
                    best_wr = pev.mean_wr
                    agent.save(ckpt_dir / 'best.pt')
        ev = None
        n_eval = resolve_eval_n(eval_episodes, smoke=smoke)
        if n_eval > 0:
            # Score de checkpoint / HP tune: WR pareado sobre contextos fijos
            # (no -pg_loss: con ventajas normalizadas ronda 0 sin medir calidad).
            ev = evaluate_preview_policy(
                env, _greedy, n=n_eval, baseline=eval_baseline, eval_seed=eval_seed,
            )
            mlog.write('eval', update=int(updates) - 1, episodes=agent.env_episodes,
                       final=True, **ev.as_dict())
            print(ev.lines(), end='', flush=True)
        prov = provenance_meta(env, pool)
        mlog.write('provenance', **prov)
    finally:
        mlog.close()
        if pool is not None:
            pool.close()
    pg = float(last_metrics.get('pg_loss', 0.0) or 0.0)
    meta = {
        'seed': seed,
        'device': device,
        'smoke': smoke,
        'updates': updates,
        'lr': float(cfg.lr),
        'clip_eps': float(cfg.clip_eps),
        'obs_version': OBS_VERSION,
        'obs_dim': OBS_DIM,
        'last_metrics': {k: float(v) for k, v in last_metrics.items()},
        'eval': ev.as_dict() if ev else None,
        'MEAN_WR': ev.mean_wr if ev else None,
        'note': 'MEAN_WR = WR greedy en contextos de evaluación fijos (pareado vs baseline).',
        **prov,
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
        f'obs: v{OBS_VERSION} dim={OBS_DIM}\n'
        f'pg_loss: {pg:.6f}\n'
        + (ev.lines() if ev else 'eval: omitida (sin MEAN_WR)\n')
        + provenance_lines(prov),
        encoding='utf-8',
    )
    final_pt = agent.save(ckpt_dir / 'final.pt')
    meta['final_pt'] = str(final_pt)
    (ckpt_dir / 'train_meta.json').write_text(
        json.dumps(meta, indent=2) + '\n', encoding='utf-8',
    )
    print(f'checkpoint: {final_pt}', flush=True)
    return agent


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description='Team Preview Masked PPO')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cpu', help='cpu | cuda | auto')
    p.add_argument('--smoke', action='store_true', help='N=64 episodes / update')
    p.add_argument('--updates', type=int, default=1)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--clip-eps', type=float, default=None)
    p.add_argument('--checkpoint-dir', default='logs/team_preview_ppo_checkpoints')
    p.add_argument('--canonical', action='store_true',
                   help='90 formaciones canónicas (leads y reserva ordenados) en vez de 360 órdenes')
    p.add_argument('--ent-coef', type=float, default=None, help='Coeficiente de entropía (defecto 0.02)')
    p.add_argument('--obs-abilities', type=int, choices=(0, 1), default=None,
                   help='observación con la corrección de habilidades (P5): 1 = preview v2.1 (defecto), 0 = como v2')
    p.add_argument('--own-team', default='',
                   help='Equipo propio fijo (archivo export; p. ej. showdown/rl/data/ddqn_team.txt)')
    from showdown.rl.preview_live import add_live_cli_args, live_options_from_args
    add_live_cli_args(p)
    add_eval_cli_args(p)
    p.add_argument('--eval-every', type=int, default=5,
                   help='Evaluación pareada cada N updates (curva de aprendizaje; 0 = solo final)')
    p.add_argument('--eval-periodic-episodes', type=int, default=256,
                   help='Contextos de cada evaluación periódica (live: 2 BO1 por contexto)')
    args = p.parse_args(argv)
    if args.own_team:
        import os
        os.environ['PS_PREVIEW_OWN_TEAM'] = args.own_team
        print(f'[preview] equipo propio fijo: {args.own_team}', flush=True)
    train(
        seed=args.seed,
        smoke=args.smoke,
        updates=args.updates,
        device=args.device,
        lr=args.lr,
        clip_eps=args.clip_eps,
        checkpoint_dir=args.checkpoint_dir,
        canonical=bool(args.canonical),
        ent_coef=args.ent_coef,
        obs_abilities=None if args.obs_abilities is None else bool(args.obs_abilities),
        live=live_options_from_args(args),
        eval_episodes=args.eval_episodes,
        eval_baseline=args.eval_baseline,
        eval_seed=args.eval_seed,
        eval_every=args.eval_every,
        eval_periodic_episodes=args.eval_periodic_episodes,
    )
    print('DONE team_preview_ppo', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
