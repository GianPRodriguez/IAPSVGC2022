"""QD-PPO training loop — docs/qd_ppo_spec.md.

Examples::

  python -m showdown.rl.qd_ppo.train --seed 0 --smoke
  python -m showdown.rl.qd_ppo.train --seed 0 --device cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from showdown.rl.env import MockBattleBackend, VGCEnv
from showdown.rl.ppo.agent import PPOAgent, PPOConfig
from showdown.rl.qd_ppo.archive import Archive
from showdown.rl.qd_ppo.evaluate import evaluate_policy
from showdown.rl.qd_ppo.improve import ppo_improve
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
ROLLOUT = 2048


def derived_n_outer(
    *,
    t_total: int = T_TOTAL,
    b_boot: int = B_BOOT,
    n_emit: int = N_EMIT,
    rollout: int = ROLLOUT,
) -> int:
    rem = max(0, int(t_total) - int(b_boot))
    return int(rem // max(1, int(n_emit) * int(rollout)))


def _make_env(*, mock: bool, seed: int, start_ps: bool) -> VGCEnv:
    if mock or not start_ps:
        backend = MockBattleBackend(max_turns=40)
    else:
        from showdown.rl.pokeenv_backend import PokeEnvBattleBackend
        backend = PokeEnvBattleBackend(
            opponent='random',
            start_ps=start_ps,
            account_prefix='QdPpo',
            combat_family='qd_ppo',
        )
    return VGCEnv(
        reward_candidate=RewardCandidate.DENSE,
        backend=backend,
        max_episode_turns=40,
    )


def _agent_from_state(cfg: PPOConfig, state_dict) -> PPOAgent:
    agent = PPOAgent(cfg)
    agent._ensure_torch()
    agent.net.load_state_dict(state_dict)
    return agent


def run_training(
    *,
    seed: int = 0,
    device: str = 'cpu',
    checkpoint_dir: str | Path = 'logs/qd_ppo_archive',
    t_total: int = T_TOTAL,
    b_boot: int = B_BOOT,
    n_emit: int = N_EMIT,
    rollout: int = ROLLOUT,
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
    num_envs: int = 1,
    envs_per_server: int = 4,
    base_port: Optional[int] = None,
) -> Archive:
    from showdown.rl.parallel import describe_resources, raise_nofile_limit, resolve_device

    raise_nofile_limit()
    device = resolve_device(device)
    num_envs = max(1, int(num_envs))
    print(f'[qd_ppo] recursos: {describe_resources()} device={device} num_envs={num_envs}',
          flush=True)
    if smoke:
        t_total = min(int(t_total), 6_000)
        b_boot = min(int(b_boot), 2_048)
        n_emit = min(int(n_emit), 2)
        rollout = min(int(rollout), 256)
        fitness_games = min(int(fitness_games), 4)
        g_min = min(int(g_min), 2)
        n_aux = min(int(n_aux), 1)
        log_every = min(int(log_every), 256)
        log_every_sec = min(float(log_every_sec), 15.0)

    cfg_kw = dict(seed=seed, device=device, rollout_steps=rollout)
    if lr is not None:
        cfg_kw['lr'] = float(lr)
    if clip_eps is not None:
        cfg_kw['clip_eps'] = float(clip_eps)
    cfg = PPOConfig(**cfg_kw)
    agent = PPOAgent(cfg)
    if bc_checkpoint:
        agent.load_bc_weights(bc_checkpoint)

    archive = Archive()
    rng = np.random.default_rng(int(seed) + 91)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    from showdown.rl.vec_rollout import is_vec_env, make_qd_env

    mock_env = bool(smoke or not start_ps)
    env = make_qd_env(
        make_single=lambda: _make_env(mock=mock_env, seed=seed, start_ps=start_ps),
        mock=mock_env, num_envs=num_envs, seed=seed, start_ps=start_ps,
        account_prefix='QdPpo', combat_family='qd_ppo',
        envs_per_server=envs_per_server, base_port=base_port,
    )
    global_steps = 0
    accepts = 0
    emits = 0
    hb = TrainHeartbeat(
        tag='qd_ppo',
        t_total=max(1, int(t_total)),
        log_every=int(log_every),
        log_every_sec=float(log_every_sec),
    )

    try:
        # --- Stage 1 bootstrap (Cand. 2) ---
        boot_budget = min(int(b_boot), int(t_total) - global_steps)
        print(f'[qd_ppo] bootstrap steps={boot_budget} (of T={t_total})', flush=True)
        env, taken, metrics = ppo_improve(
            agent, env,
            steps=boot_budget, rng=rng, global_steps=global_steps,
            t_total=t_total, seed=seed, use_mosaic=True,
            heartbeat=hb, phase='bootstrap',
        )
        global_steps += taken
        seed_state = {k: v.detach().cpu().clone() for k, v in agent.net.state_dict().items()}

        # --- Stage 2: evaluate + insert seed (+ aux from same checkpoint) ---
        for aux_i in range(max(1, int(n_aux))):
            if aux_i > 0:
                agent.net.load_state_dict(seed_state)
                env, taken, _ = ppo_improve(
                    agent, env,
                    steps=min(rollout, max(0, t_total - global_steps)),
                    rng=rng, global_steps=global_steps, t_total=t_total,
                    seed=seed + aux_i, use_mosaic=True,
                    heartbeat=hb, phase=f'aux={aux_i}',
                )
                global_steps += taken
            ev = evaluate_policy(
                agent, env, rng=rng, n_games=fitness_games, g_min=g_min,
                seed=seed + 1000 + aux_i, use_mosaic=True,
            )
            emits += 1
            if ev.accepted and ev.bd is not None:
                sd = {k: v.detach().cpu().clone() for k, v in agent.net.state_dict().items()}
                if archive.try_insert(
                    bd1=ev.bd[0], bd2=ev.bd[1], fitness=ev.fitness,
                    state_dict=sd,
                    meta={'stage': 'bootstrap', 'aux': aux_i, 'g': ev.g_eligible},
                ):
                    accepts += 1
            agent.net.load_state_dict(seed_state)

        rem = max(0, int(t_total) - global_steps)
        n_outer = int(rem // max(1, n_emit * rollout))
        hb.log(
            global_steps,
            phase='outer:start',
            extras={
                'N': n_outer,
                'cov': archive.coverage(),
                'best_f': archive.best_fitness() if archive.cells else None,
            },
            force=True,
        )

        for it in range(n_outer):
            if global_steps >= t_total:
                break
            for em in range(n_emit):
                if global_steps >= t_total:
                    break
                parent_sd, _key = archive.sample_parent(rng, fallback_state=seed_state)
                if parent_sd is None:
                    parent_sd = seed_state
                child = _agent_from_state(cfg, parent_sd)
                env, taken, metrics = ppo_improve(
                    child, env,
                    steps=rollout, rng=rng, global_steps=global_steps,
                    t_total=t_total, seed=seed + it * 100 + em, use_mosaic=True,
                    heartbeat=hb, phase=f'iter={it + 1}/{n_outer}',
                )
                global_steps += taken
                ev = evaluate_policy(
                    child, env, rng=rng, n_games=fitness_games, g_min=g_min,
                    seed=seed + 50_000 + it * 100 + em, use_mosaic=True,
                )
                emits += 1
                if ev.accepted and ev.bd is not None:
                    sd = {k: v.detach().cpu().clone() for k, v in child.net.state_dict().items()}
                    if archive.try_insert(
                        bd1=ev.bd[0], bd2=ev.bd[1], fitness=ev.fitness,
                        state_dict=sd,
                        meta={'iter': it, 'emit': em, 'g': ev.g_eligible},
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
                },
                force=True,
            )
            archive.save(ckpt_dir / 'archive.pt')
            best = archive.best()
            if best is not None:
                agent.net.load_state_dict(best.state_dict)
                agent.save(ckpt_dir / 'best_elite.pt')
            hb.mark_checkpoint(global_steps)

        archive.save(ckpt_dir / 'archive.pt')
        best = archive.best()
        if best is not None:
            agent.net.load_state_dict(best.state_dict)
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
            'n_outer': n_outer,
            'smoke': smoke,
        }
        (ckpt_dir / 'summary.json').write_text(
            json.dumps(summary, indent=2), encoding='utf-8',
        )
        n_suite = resolve_suite_games(suite_games=suite_games, smoke=smoke)
        suite_rows, suite_note = [], ''
        if n_suite > 0 and is_vec_env(env):
            # La suite offline usa un entorno simple (BO1 secuenciales por rival).
            env.close()
            env = _make_env(mock=mock_env, seed=seed, start_ps=start_ps)
        if n_suite > 0:
            print(
                f'[qd_ppo] offline suite on π*: {n_suite} BO1 × {len(SUITE_FOES)} foes',
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
            agent='qd_ppo',
            seed=seed,
            device=device,
            smoke=smoke,
            global_steps=global_steps,
            t_total=t_total,
            elapsed_sec=hb.elapsed(),
            steps_per_sec=hb.rate(max(1, global_steps)),
            config={
                'b_boot': b_boot,
                'n_emit': n_emit,
                'rollout': rollout,
                'n_outer': n_outer,
            },
            metrics={
                'coverage': summary['coverage'],
                'qd_score': summary['qd_score'],
                'best_f': summary['best_f'],
                'accepts': accepts,
                'emits': emits,
                'accept_rate': float(accepts) / float(max(1, emits)),
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
                'Deploy policy π* = argmax f (best_elite.pt).',
                'Primary QD metrics: Coverage, QD-score, best_f; primary play: MEAN_WR_star.',
                'Fitness during search used mosaic foes; suite uses fixed control foes.',
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
                'MEAN_WR': mean_wr,
                'reporte': str(ckpt_dir / 'reporte.txt'),
            },
            force=True,
        )
        return archive
    finally:
        env.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='QD-PPO (Grid MAP-Elites + Masked PPO)')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', type=str, default='cpu', help='cpu | cuda | auto')
    p.add_argument('--checkpoint-dir', type=str, default='logs/qd_ppo_archive')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--start-ps', action='store_true')
    p.add_argument('--bc-checkpoint', type=str, default=None)
    p.add_argument('--t-total', type=int, default=T_TOTAL)
    p.add_argument('--b-boot', type=int, default=B_BOOT)
    p.add_argument(
        '--suite-games', type=int, default=None,
        help='BO1 per foe on π* (default 20; 0=skip; smoke default 0; use 200 for full suite)',
    )
    p.add_argument('--lr', type=float, default=None, help='Override PPOConfig.lr')
    p.add_argument('--clip-eps', type=float, default=None, help='Override PPOConfig.clip_eps')
    p.add_argument('--num-envs', type=int, default=1,
                   help='Entornos paralelos (procesos). Servidor: ~24; 0 = automático')
    p.add_argument('--envs-per-server', type=int, default=4,
                   help='Entornos por servidor Showdown')
    p.add_argument('--base-port', type=int, default=None,
                   help='Primer puerto Showdown (default PS_PORT)')
    args = p.parse_args(argv)
    if args.num_envs == 0:
        from showdown.rl.parallel import default_num_envs
        args.num_envs = default_num_envs()
    run_training(
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        t_total=args.t_total,
        b_boot=args.b_boot,
        smoke=args.smoke,
        start_ps=args.start_ps,
        bc_checkpoint=args.bc_checkpoint,
        suite_games=args.suite_games,
        lr=args.lr,
        clip_eps=args.clip_eps,
        num_envs=args.num_envs,
        envs_per_server=args.envs_per_server,
        base_port=args.base_port,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
