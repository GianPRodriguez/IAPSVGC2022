"""QD-PPGA training loop — docs/qd_ppga_spec.md.

Examples::

  python -m showdown.rl.qd_ppga.train --seed 0 --smoke
  python -m showdown.rl.qd_ppga.train --seed 0 --device cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from showdown.rl.env import MockBattleBackend, VGCEnv
from showdown.rl.ppo.agent import PPOAgent, PPOConfig
from showdown.rl.qd_ppga.arborescence import sample_arborescence_parent
from showdown.rl.qd_ppga.archive import Archive, actor_state_dict, load_actor_into
from showdown.rl.qd_ppga.critic import SharedCritic
from showdown.rl.qd_ppga.evaluate import evaluate_policy
from showdown.rl.qd_ppga.improve import pg_improve
from showdown.rl.qd_ppga.replay import ValueReplay
from showdown.rl.reward import RewardCandidate
from showdown.rl.train_log import TrainHeartbeat
from showdown.rl.train_report import (
    SUITE_FOES,
    TrainReport,
    resolve_suite_games,
    run_suite_ppo_act,
    write_train_report,
)

T_TOTAL = 1_000_000
B_BOOT = 100_000
N_EMIT = 4
T_EMIT = 2048
K_PG = 4
ALPHA = 0.5
P_MUT = 0.0  # contractual: proximal-only


def derived_n_outer(
    *,
    t_total: int = T_TOTAL,
    b_boot: int = B_BOOT,
    n_emit: int = N_EMIT,
    t_emit: int = T_EMIT,
) -> int:
    rem = max(0, int(t_total) - int(b_boot))
    return int(rem // max(1, int(n_emit) * int(t_emit)))


def _make_env(*, mock: bool, seed: int, start_ps: bool) -> VGCEnv:
    if mock or not start_ps:
        backend = MockBattleBackend(max_turns=40)
    else:
        from showdown.rl.pokeenv_backend import PokeEnvBattleBackend
        backend = PokeEnvBattleBackend(
            opponent='random',
            start_ps=start_ps,
            account_prefix='QdPpga',
            combat_family='qd_ppga',
        )
    return VGCEnv(
        reward_candidate=RewardCandidate.DENSE,
        backend=backend,
        max_episode_turns=40,
    )


def _agent_from_actor(cfg: PPOConfig, actor_sd) -> PPOAgent:
    agent = PPOAgent(cfg)
    agent._ensure_torch()
    load_actor_into(agent.net, actor_sd)
    return agent


def run_training(
    *,
    seed: int = 0,
    device: str = 'cpu',
    checkpoint_dir: str | Path = 'logs/qd_ppga_archive',
    t_total: int = T_TOTAL,
    b_boot: int = B_BOOT,
    n_emit: int = N_EMIT,
    t_emit: int = T_EMIT,
    k_pg: int = K_PG,
    alpha: float = ALPHA,
    fitness_games: int = 20,
    g_min: int = 10,
    n_aux: int = 4,
    smoke: bool = False,
    start_ps: bool = False,
    bc_checkpoint: Optional[str] = None,
    log_every: int = 5_000,
    log_every_sec: float = 60.0,
    suite_games: Optional[int] = None,
    lr: Optional[float] = None,
    clip_eps: Optional[float] = None,
) -> Archive:
    assert float(P_MUT) == 0.0, 'QD-PPGA forbids genetic mutation'
    if smoke:
        t_total = min(int(t_total), 6_000)
        b_boot = min(int(b_boot), 1_024)
        n_emit = min(int(n_emit), 2)
        t_emit = min(int(t_emit), 256)
        fitness_games = min(int(fitness_games), 4)
        g_min = min(int(g_min), 2)
        n_aux = min(int(n_aux), 1)
        log_every = min(int(log_every), 256)
        log_every_sec = min(float(log_every_sec), 15.0)

    actor_lr = 3e-4 if lr is None else float(lr)
    actor_clip = 0.2 if clip_eps is None else float(clip_eps)
    cfg = PPOConfig(
        seed=seed, device=device, rollout_steps=t_emit,
        epochs=k_pg, gamma=0.95, gae_lambda=0.95,
        ent_coef=0.01, vf_coef=0.5, lr=actor_lr, clip_eps=actor_clip,
    )
    agent = PPOAgent(cfg)
    if bc_checkpoint:
        agent.load_bc_weights(bc_checkpoint)

    critic = SharedCritic(lr=3e-4, gamma=0.95, vf_coef=0.5, device=device)
    replay = ValueReplay()
    archive = Archive()
    rng = np.random.default_rng(int(seed) + 117)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    env = _make_env(mock=smoke or not start_ps, seed=seed, start_ps=start_ps)
    global_steps = 0
    accepts = 0
    emits = 0
    n_uniform = 0
    n_frontier = 0
    hb = TrainHeartbeat(
        tag='qd_ppga',
        t_total=max(1, int(t_total)),
        log_every=int(log_every),
        log_every_sec=float(log_every_sec),
    )

    try:
        boot_budget = min(int(b_boot), int(t_total) - global_steps)
        print(f'[qd_ppga] bootstrap steps={boot_budget} (of T={t_total})', flush=True)
        while global_steps < boot_budget:
            chunk = min(t_emit, boot_budget - global_steps, t_total - global_steps)
            if chunk <= 0:
                break
            env, taken, _ = pg_improve(
                agent, env, critic, replay,
                steps=chunk, rng=rng, global_steps=global_steps,
                t_total=t_total, seed=seed + global_steps, use_mosaic=True,
                k_pg=k_pg, heartbeat=hb, phase='bootstrap',
            )
            global_steps += taken
            if taken <= 0:
                break

        seed_actor = actor_state_dict(agent.net)

        for aux_i in range(max(1, int(n_aux))):
            load_actor_into(agent.net, seed_actor)
            if aux_i > 0:
                env, taken, _ = pg_improve(
                    agent, env, critic, replay,
                    steps=min(t_emit, max(0, t_total - global_steps)),
                    rng=rng, global_steps=global_steps, t_total=t_total,
                    seed=seed + aux_i, use_mosaic=True, k_pg=k_pg,
                    heartbeat=hb, phase=f'aux={aux_i}',
                )
                global_steps += taken
            ev = evaluate_policy(
                agent, env, rng=rng, n_games=fitness_games, g_min=g_min,
                seed=seed + 1000 + aux_i, use_mosaic=True,
            )
            emits += 1
            if ev.accepted and ev.bd is not None:
                sd = actor_state_dict(agent.net)
                if archive.try_insert(
                    bd1=ev.bd[0], bd2=ev.bd[1], fitness=ev.fitness,
                    state_dict=sd,
                    meta={'stage': 'bootstrap', 'aux': aux_i, 'g': ev.g_eligible},
                ):
                    accepts += 1

        rem = max(0, int(t_total) - global_steps)
        n_outer = int(rem // max(1, n_emit * t_emit))
        hb.log(
            global_steps,
            phase='outer:start',
            extras={
                'N': n_outer,
                'cov': archive.coverage(),
                'best_f': archive.best_fitness() if archive.cells else None,
                'replay': len(replay),
                'alpha': alpha,
            },
            force=True,
        )

        for it in range(n_outer):
            if global_steps >= t_total:
                break
            for em in range(n_emit):
                if global_steps >= t_total:
                    break
                parent_sd, _key, mode = sample_arborescence_parent(
                    archive, rng, alpha=alpha, fallback_state=seed_actor,
                )
                if parent_sd is None:
                    parent_sd = seed_actor
                if mode == 'uniform':
                    n_uniform += 1
                elif mode == 'frontier':
                    n_frontier += 1
                child = _agent_from_actor(cfg, parent_sd)
                env, taken, _metrics = pg_improve(
                    child, env, critic, replay,
                    steps=t_emit, rng=rng, global_steps=global_steps,
                    t_total=t_total, seed=seed + it * 100 + em,
                    use_mosaic=True, k_pg=k_pg,
                    heartbeat=hb, phase=f'iter={it + 1}/{n_outer}',
                )
                global_steps += taken
                ev = evaluate_policy(
                    child, env, rng=rng, n_games=fitness_games, g_min=g_min,
                    seed=seed + 50_000 + it * 100 + em, use_mosaic=True,
                )
                emits += 1
                if ev.accepted and ev.bd is not None:
                    sd = actor_state_dict(child.net)
                    if archive.try_insert(
                        bd1=ev.bd[0], bd2=ev.bd[1], fitness=ev.fitness,
                        state_dict=sd,
                        meta={
                            'iter': it, 'emit': em, 'g': ev.g_eligible,
                            'parent_mode': mode,
                        },
                    ):
                        accepts += 1

            hb.log(
                global_steps,
                phase=f'iter={it + 1}/{n_outer}',
                extras={
                    'cov': archive.coverage(),
                    'qd': archive.qd_score(),
                    'best_f': archive.best_fitness() if archive.cells else None,
                    'accept': f'{accepts}/{emits}',
                    'uni': n_uniform,
                    'fr': n_frontier,
                    'replay': len(replay),
                },
                force=True,
            )
            archive.save(ckpt_dir / 'archive.pt')
            best = archive.best()
            if best is not None:
                load_actor_into(agent.net, best.state_dict)
                agent.save(ckpt_dir / 'best_elite.pt')
            hb.mark_checkpoint(global_steps)

        archive.save(ckpt_dir / 'archive.pt')
        best = archive.best()
        if best is not None:
            load_actor_into(agent.net, best.state_dict)
            agent.save(ckpt_dir / 'best_elite.pt')
        summary = {
            'seed': seed,
            'global_steps': global_steps,
            't_total': t_total,
            'coverage': archive.coverage(),
            'qd_score': archive.qd_score(),
            'best_f': archive.best_fitness() if archive.cells else None,
            'accepts': accepts,
            'emits': emits,
            'n_uniform': n_uniform,
            'n_frontier': n_frontier,
            'n_outer': n_outer,
            'replay_len': len(replay),
            'p_mut': P_MUT,
            'alpha': alpha,
            'smoke': smoke,
        }
        (ckpt_dir / 'summary.json').write_text(
            json.dumps(summary, indent=2), encoding='utf-8',
        )
        n_suite = resolve_suite_games(suite_games=suite_games, smoke=smoke)
        suite_rows, suite_note = [], ''
        if n_suite > 0:
            print(
                f'[qd_ppga] offline suite on π*: {n_suite} BO1 × {len(SUITE_FOES)} foes',
                flush=True,
            )
            suite_rows, suite_note = run_suite_ppo_act(
                agent, env, n_games=n_suite, seed=seed,
            )
        mean_wr = (
            float(sum(s.win_rate for s in suite_rows) / len(suite_rows))
            if suite_rows else None
        )
        report = TrainReport(
            agent='qd_ppga',
            seed=seed,
            device=device,
            smoke=smoke,
            global_steps=global_steps,
            t_total=t_total,
            elapsed_sec=hb.elapsed(),
            steps_per_sec=hb.rate(max(1, global_steps)),
            config={
                'p_mut': P_MUT,
                'alpha': alpha,
                't_emit': t_emit,
                'k_pg': k_pg,
                'n_emit': n_emit,
                'n_outer': n_outer,
            },
            metrics={
                'coverage': summary['coverage'],
                'qd_score': summary['qd_score'],
                'best_f': summary['best_f'],
                'accepts': accepts,
                'emits': emits,
                'accept_rate': float(accepts) / float(max(1, emits)),
                'n_uniform': n_uniform,
                'n_frontier': n_frontier,
                'replay_len': len(replay),
                'MEAN_WR_star': mean_wr,
            },
            artifacts={
                'archive_pt': str(ckpt_dir / 'archive.pt'),
                'best_elite_pt': str(ckpt_dir / 'best_elite.pt'),
                'summary_json': str(ckpt_dir / 'summary.json'),
                'checkpoint_dir': str(ckpt_dir),
            },
            suite_games_per_foe=n_suite,
            suite=suite_rows,
            suite_note=suite_note,
            notes=[
                'Deploy π* = argmax f; arborescence uni/fr counts show parent selection mix.',
                'p_mut=0 by contract (proximal-only).',
                'Live console heartbeats complement this static report.',
            ],
        )
        write_train_report(ckpt_dir, report)
        hb.log(
            global_steps,
            phase='DONE',
            extras={
                'cov': summary['coverage'],
                'qd': summary['qd_score'],
                'best_f': summary['best_f'],
                'accept': f'{accepts}/{emits}',
                'uni': n_uniform,
                'fr': n_frontier,
                'MEAN_WR': mean_wr,
                'reporte': str(ckpt_dir / 'reporte.txt'),
            },
            force=True,
        )
        return archive
    finally:
        env.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description='QD-PPGA (proximal-only MAP-Elites + arborescence)',
    )
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--checkpoint-dir', type=str, default='logs/qd_ppga_archive')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--start-ps', action='store_true')
    p.add_argument('--bc-checkpoint', type=str, default=None)
    p.add_argument('--t-total', type=int, default=T_TOTAL)
    p.add_argument('--b-boot', type=int, default=B_BOOT)
    p.add_argument('--alpha', type=float, default=ALPHA)
    p.add_argument(
        '--suite-games', type=int, default=None,
        help='BO1 per foe on π* (default 20; 0=skip; smoke default 0; use 200 for full suite)',
    )
    p.add_argument('--lr', type=float, default=None, help='Override actor PPOConfig.lr')
    p.add_argument('--clip-eps', type=float, default=None, help='Override actor PPOConfig.clip_eps')
    args = p.parse_args(argv)
    run_training(
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        t_total=args.t_total,
        b_boot=args.b_boot,
        alpha=args.alpha,
        smoke=args.smoke,
        start_ps=args.start_ps,
        bc_checkpoint=args.bc_checkpoint,
        suite_games=args.suite_games,
        lr=args.lr,
        clip_eps=args.clip_eps,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
