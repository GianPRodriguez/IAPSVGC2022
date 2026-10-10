"""QD-PPO training loop — docs/qd_ppo_spec.md.

Examples::

  python -m showdown.rl.qd_ppo.train --seed 0 --smoke
  python -m showdown.rl.qd_ppo.train --seed 0 --device cpu
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
from showdown.rl.qd_ppo.archive import Archive
from showdown.rl.qd_ppo.evaluate import evaluate_policy
from showdown.rl.qd_ppo.improve import ppo_improve
from showdown.rl.reward import RewardCandidate

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
) -> Archive:
    if smoke:
        t_total = min(int(t_total), 6_000)
        b_boot = min(int(b_boot), 2_048)
        n_emit = min(int(n_emit), 2)
        rollout = min(int(rollout), 256)
        fitness_games = min(int(fitness_games), 4)
        g_min = min(int(g_min), 2)
        n_aux = min(int(n_aux), 1)

    cfg = PPOConfig(seed=seed, device=device, rollout_steps=rollout)
    agent = PPOAgent(cfg)
    if bc_checkpoint:
        agent.load_bc_weights(bc_checkpoint)

    archive = Archive()
    rng = np.random.default_rng(int(seed) + 91)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    env = _make_env(mock=smoke or not start_ps, seed=seed, start_ps=start_ps)
    global_steps = 0
    t0 = time.perf_counter()
    accepts = 0
    emits = 0

    try:
        # --- Stage 1 bootstrap (Cand. 2) ---
        boot_budget = min(int(b_boot), int(t_total) - global_steps)
        print(f'[qd_ppo] bootstrap steps={boot_budget} (of T={t_total})')
        env, taken, metrics = ppo_improve(
            agent, env,
            steps=boot_budget, rng=rng, global_steps=global_steps,
            t_total=t_total, seed=seed, use_mosaic=True,
        )
        global_steps += taken
        seed_state = {k: v.detach().cpu().clone() for k, v in agent.net.state_dict().items()}

        # --- Stage 2: evaluate + insert seed (+ aux from same checkpoint) ---
        for aux_i in range(max(1, int(n_aux))):
            if aux_i > 0:
                # short improve from seed checkpoint
                agent.net.load_state_dict(seed_state)
                env, taken, _ = ppo_improve(
                    agent, env,
                    steps=min(rollout, max(0, t_total - global_steps)),
                    rng=rng, global_steps=global_steps, t_total=t_total,
                    seed=seed + aux_i, use_mosaic=True,
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
            # restore seed for next aux branch
            agent.net.load_state_dict(seed_state)

        n_outer = derived_n_outer(
            t_total=t_total, b_boot=global_steps, n_emit=n_emit, rollout=rollout,
        )
        # After bootstrap, remaining budget drives N_outer
        rem = max(0, int(t_total) - global_steps)
        n_outer = int(rem // max(1, n_emit * rollout))
        print(
            f'[qd_ppo] outer N={n_outer} rem_steps={rem} coverage={archive.coverage():.3f} '
            f'best_f={archive.best_fitness()}'
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

            if (it + 1) % max(1, n_outer // 5) == 0 or it == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f'[qd_ppo] iter={it + 1}/{n_outer} steps={global_steps}/{t_total} '
                    f'cov={archive.coverage():.3f} qd={archive.qd_score():.3f} '
                    f'best_f={archive.best_fitness():.3f} '
                    f'accept={accepts}/{emits} elapsed={elapsed:.1f}s'
                )
                archive.save(ckpt_dir / 'archive.pt')
                best = archive.best()
                if best is not None:
                    agent.net.load_state_dict(best.state_dict)
                    agent.save(ckpt_dir / 'best_elite.pt')

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
        print(f'[qd_ppo] DONE {json.dumps(summary)}')
        return archive
    finally:
        env.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='QD-PPO (Grid MAP-Elites + Masked PPO)')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--checkpoint-dir', type=str, default='logs/qd_ppo_archive')
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
