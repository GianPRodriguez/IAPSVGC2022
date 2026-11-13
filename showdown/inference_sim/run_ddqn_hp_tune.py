"""Tune de hiperparámetros DDQN combate (independiente de PPO / Preview).

Ejemplos::

  python -m showdown.inference_sim.run_ddqn_hp_tune --dry-run
  python -m showdown.inference_sim.run_ddqn_hp_tune --seed 0 --max-trials 6 --steps 8000 --suite-games 4
  python -m showdown.inference_sim.run_ddqn_hp_tune --seed 0 --max-trials 6 --start-ps --opponent greedy_joint

Knobs: lr, gamma. Tracking en logs/ddqn_hp_tune_*.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from showdown.inference_sim.paths import REPO_ROOT
from showdown.inference_sim.rl_hp_tune import (
    ddqn_candidates,
    parse_mean_wr,
    run_hp_tune_loop,
)

FROZEN = {
    'tau': 0.005,
    'batch_size': 256,
    'replay_capacity': 200_000,
    'eps_end': 0.05,
    'note': 'Varying: lr, gamma',
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='DDQN combat HP tune + tracking')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-trials', type=int, default=6)
    ap.add_argument('--steps', type=int, default=8_000)
    ap.add_argument('--opponent', default='mock')
    ap.add_argument('--suite-games', type=int, default=4)
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
            sys.executable, '-m', 'showdown.rl.ddqn.train',
            '--seed', str(args.seed),
            '--device', args.device,
            '--steps', str(args.steps),
            '--opponent', args.opponent,
            '--suite-games', str(args.suite_games),
            '--checkpoint-dir', str(trial_dir),
            '--num-envs', str(args.num_envs),
            '--lr', str(trial.overrides['lr']),
            '--gamma', str(trial.overrides['gamma']),
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
        agent='ddqn',
        candidates=ddqn_candidates(),
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
