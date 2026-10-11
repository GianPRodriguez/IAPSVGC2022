"""QD-PGA-MAP-Elites training loop — docs/qd_pga_map_elites_spec.md.

Examples::

  python -m showdown.rl.qd_pga.train --seed 0 --smoke
  python -m showdown.rl.qd_pga.train --seed 0 --device cpu
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from showdown.rl.env import MockBattleBackend, VGCEnv
from showdown.rl.ppo.agent import PPOAgent, PPOConfig
from showdown.rl.qd_pga.archive import Archive, actor_state_dict, load_actor_into
from showdown.rl.qd_pga.critic import SharedCritic
from showdown.rl.qd_pga.evaluate import evaluate_policy
from showdown.rl.qd_pga.improve import mut_burn_rollout, pg_improve
from showdown.rl.qd_pga.mutate import mutate_actor_state
from showdown.rl.qd_pga.replay import ValueReplay
from showdown.rl.reward import RewardCandidate

T_TOTAL = 1_000_000
B_BOOT = 100_000
N_EMIT = 4
T_EMIT = 1024
P_MUT = 0.5
SIGMA = 0.02
K_PG = 2


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
            account_prefix='QdPga',
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
    checkpoint_dir: str | Path = 'logs/qd_pga_archive',
    t_total: int = T_TOTAL,
    b_boot: int = B_BOOT,
    n_emit: int = N_EMIT,
    t_emit: int = T_EMIT,
    p_mut: float = P_MUT,
    sigma: float = SIGMA,
    k_pg: int = K_PG,
    fitness_games: int = 20,
    g_min: int = 10,
    n_aux: int = 4,
    smoke: bool = False,
    start_ps: bool = False,
    bc_checkpoint: Optional[str] = None,
) -> Archive:
    if smoke:
        t_total = min(int(t_total), 6_000)
        b_boot = min(int(b_boot), 1_024)
        n_emit = min(int(n_emit), 2)
        t_emit = min(int(t_emit), 256)
        fitness_games = min(int(fitness_games), 4)
        g_min = min(int(g_min), 2)
        n_aux = min(int(n_aux), 1)

    cfg = PPOConfig(
        seed=seed, device=device, rollout_steps=t_emit,
        epochs=k_pg, gamma=0.95, gae_lambda=0.95,
        ent_coef=0.01, vf_coef=0.5, lr=3e-4,
    )
    agent = PPOAgent(cfg)
    if bc_checkpoint:
        agent.load_bc_weights(bc_checkpoint)

    critic = SharedCritic(lr=3e-4, gamma=0.95, vf_coef=0.5, device=device)
    replay = ValueReplay()
    archive = Archive()
    rng = np.random.default_rng(int(seed) + 91)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    env = _make_env(mock=smoke or not start_ps, seed=seed, start_ps=start_ps)
    global_steps = 0
    t0 = time.perf_counter()
    accepts = 0
    emits = 0
    n_mut = 0
    n_pg = 0

    try:
        boot_budget = min(int(b_boot), int(t_total) - global_steps)
        print(f'[qd_pga] bootstrap steps={boot_budget} (of T={t_total})')
        # Bootstrap in chunks of t_emit via PG path
        while global_steps < boot_budget:
            chunk = min(t_emit, boot_budget - global_steps, t_total - global_steps)
            if chunk <= 0:
                break
            env, taken, _ = pg_improve(
                agent, env, critic, replay,
                steps=chunk, rng=rng, global_steps=global_steps,
                t_total=t_total, seed=seed + global_steps, use_mosaic=True,
                k_pg=k_pg,
            )
            global_steps += taken
            if taken <= 0:
                break

        seed_actor = actor_state_dict(agent.net)

        for aux_i in range(max(1, int(n_aux))):
            load_actor_into(agent.net, seed_actor)
            use_mut = bool(rng.random() < float(p_mut)) if aux_i > 0 else False
            if aux_i > 0 and use_mut:
                mut_sd = mutate_actor_state(seed_actor, rng, sigma=sigma)
                load_actor_into(agent.net, mut_sd)
                env, taken, _ = mut_burn_rollout(
                    agent, env, critic,
                    steps=min(t_emit, max(0, t_total - global_steps)),
                    rng=rng, global_steps=global_steps, t_total=t_total,
                    seed=seed + aux_i, use_mosaic=True,
                )
                n_mut += 1
            else:
                if aux_i > 0:
                    env, taken, _ = pg_improve(
                        agent, env, critic, replay,
                        steps=min(t_emit, max(0, t_total - global_steps)),
                        rng=rng, global_steps=global_steps, t_total=t_total,
                        seed=seed + aux_i, use_mosaic=True, k_pg=k_pg,
                    )
                    n_pg += 1
                else:
                    taken = 0
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
        print(
            f'[qd_pga] outer N={n_outer} rem_steps={rem} coverage={archive.coverage():.3f} '
            f'best_f={archive.best_fitness()} replay={len(replay)}'
        )

        for it in range(n_outer):
            if global_steps >= t_total:
                break
            for em in range(n_emit):
                if global_steps >= t_total:
                    break
                parent_sd, _key = archive.sample_parent(rng, fallback_state=seed_actor)
                if parent_sd is None:
                    parent_sd = seed_actor
                child = _agent_from_actor(cfg, parent_sd)
                use_mut = bool(rng.random() < float(p_mut))
                if use_mut:
                    mut_sd = mutate_actor_state(parent_sd, rng, sigma=sigma)
                    load_actor_into(child.net, mut_sd)
                    env, taken, metrics = mut_burn_rollout(
                        child, env, critic,
                        steps=t_emit, rng=rng, global_steps=global_steps,
                        t_total=t_total, seed=seed + it * 100 + em, use_mosaic=True,
                    )
                    n_mut += 1
                else:
                    env, taken, metrics = pg_improve(
                        child, env, critic, replay,
                        steps=t_emit, rng=rng, global_steps=global_steps,
                        t_total=t_total, seed=seed + it * 100 + em,
                        use_mosaic=True, k_pg=k_pg,
                    )
                    n_pg += 1
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
                            'branch': 'mut' if use_mut else 'pg',
                        },
                    ):
                        accepts += 1

            if (it + 1) % max(1, n_outer // 5) == 0 or it == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f'[qd_pga] iter={it + 1}/{n_outer} steps={global_steps}/{t_total} '
                    f'cov={archive.coverage():.3f} qd={archive.qd_score():.3f} '
                    f'best_f={archive.best_fitness():.3f} '
                    f'accept={accepts}/{emits} mut={n_mut} pg={n_pg} '
                    f'replay={len(replay)} elapsed={elapsed:.1f}s'
                )
                archive.save(ckpt_dir / 'archive.pt')
                best = archive.best()
                if best is not None:
                    load_actor_into(agent.net, best.state_dict)
                    agent.save(ckpt_dir / 'best_elite.pt')

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
            'n_mut': n_mut,
            'n_pg': n_pg,
            'n_outer': n_outer,
            'replay_len': len(replay),
            'smoke': smoke,
        }
        (ckpt_dir / 'summary.json').write_text(
            json.dumps(summary, indent=2), encoding='utf-8',
        )
        print(f'[qd_pga] DONE {json.dumps(summary)}')
        return archive
    finally:
        env.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='QD-PGA-MAP-Elites (shared Vφ + hybrid mut/PG)')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--checkpoint-dir', type=str, default='logs/qd_pga_archive')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--start-ps', action='store_true')
    p.add_argument('--bc-checkpoint', type=str, default=None)
    p.add_argument('--t-total', type=int, default=T_TOTAL)
    p.add_argument('--b-boot', type=int, default=B_BOOT)
    args = p.parse_args(argv)
    run_training(
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        t_total=args.t_total,
        b_boot=args.b_boot,
        smoke=args.smoke,
        start_ps=args.start_ps,
        bc_checkpoint=args.bc_checkpoint,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
