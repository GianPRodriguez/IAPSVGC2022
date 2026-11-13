"""Tune de hiperparámetros Masked PPO combate (independiente de DDQN / Preview).

Ejemplos::

  python -m showdown.inference_sim.run_ppo_hp_tune --dry-run
  python -m showdown.inference_sim.run_ppo_hp_tune --seed 0 --max-trials 6 --steps 8000 --suite-games 4
  python -m showdown.inference_sim.run_ppo_hp_tune --seed 0 --max-trials 6 --start-ps --opponent greedy_joint

Knobs: lr, clip_eps. Tracking en logs/ppo_hp_tune_*.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from showdown.inference_sim.paths import REPO_ROOT
from showdown.inference_sim.rl_hp_tune import (
    parse_mean_wr,
    ppo_combat_candidates,
    run_hp_tune_loop,
)

FROZEN = {
    'gamma': 0.95,
    'gae_lambda': 0.95,
    'ent_coef': 0.01,
    'rollout_steps_default': 2048,
    'note': 'Varying: lr, clip_eps',
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='PPO combat HP tune + tracking')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-trials', type=int, default=6)
    ap.add_argument('--steps', type=int, default=8_000)
    ap.add_argument('--opponent', default='mock')
    ap.add_argument('--suite-games', type=int, default=4)
    ap.add_argument('--rollout-steps', type=int, default=256)
    ap.add_argument('--device', default='cpu', help='cpu | cuda | auto')
    ap.add_argument('--parallel-trials', type=int, default=1,
                    help='Trials simultáneos (uno por GPU, puertos separados). Servidor: 3')
    ap.add_argument('--num-envs', type=int, default=1,
                    help='Entornos paralelos por trial (ver showdown.rl.vec_env)')
    ap.add_argument('--start-ps', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-expand-omega', action='store_true')
    args = ap.parse_args(list(argv) if argv is not None else None)

    logs = REPO_ROOT / 'logs'

    def build_cmd(trial, trial_dir: Path):
        cmd = [
            sys.executable, '-m', 'showdown.rl.ppo.train',
            '--seed', str(args.seed),
            '--device', args.device,
            '--steps', str(args.steps),
            '--opponent', args.opponent,
            '--suite-games', str(args.suite_games),
            '--rollout-steps', str(args.rollout_steps),
            '--checkpoint-dir', str(trial_dir),
            '--num-envs', str(args.num_envs),
            '--lr', str(trial.overrides['lr']),
            '--clip-eps', str(trial.overrides['clip_eps']),
            '--log-every', '100000',
            '--log-every-sec', '3600',
        ]
        if args.no_expand_omega:
            cmd.append('--no-expand-omega')
        if args.start_ps:
            cmd.append('--start-ps')
        return cmd

    def parse_score(trial_dir: Path):
        rep = trial_dir / 'reporte.txt'
        if not rep.is_file():
            return None
        return parse_mean_wr(rep.read_text(encoding='utf-8'))

    run_hp_tune_loop(
        agent='ppo',
        candidates=ppo_combat_candidates(),
        build_cmd=build_cmd,
        parse_score_from_dir=parse_score,
        logs_dir=logs,
        max_trials=int(args.max_trials),
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        frozen=FROZEN,
        parallel_trials=int(args.parallel_trials),
        criteria_note='maximize MEAN_WR from offline suite (Cand.1); accept if improves best',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
