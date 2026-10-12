"""Tune HP Team Preview Nash (independiente de MCTS combate).

Ejemplos::

  python -m showdown.inference_sim.run_team_preview_nash_hp_tune --dry-run
  python -m showdown.inference_sim.run_team_preview_nash_hp_tune --seed 0 --max-trials 3 --smoke --updates 2

Knobs: lr. Score: MEAN_SCORE = -loss.
Tracking: logs/team_preview_nash_hp_tune_* (prev -> new).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from showdown.inference_sim.paths import REPO_ROOT
from showdown.inference_sim.rl_hp_tune import (
    parse_mean_wr,
    preview_nash_candidates,
    run_hp_tune_loop,
)

FROZEN = {
    'grad_clip': 0.5,
    'epochs': 4,
    'loss': 'bce',
    'solver_timeout_s': 0.2,
    'pi_c': 'greedy_joint',
    'note': 'Varying: lr; independent of MCTS combat trains',
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Team Preview Nash HP tune + tracking')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-trials', type=int, default=3)
    ap.add_argument('--updates', type=int, default=2)
    ap.add_argument('--smoke', action='store_true', default=True)
    ap.add_argument('--no-smoke', dest='smoke', action='store_false')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(list(argv) if argv is not None else None)

    logs = REPO_ROOT / 'logs'

    def build_cmd(trial, trial_dir: Path):
        cmd = [
            sys.executable, '-m', 'showdown.rl.team_preview_nash.train',
            '--seed', str(args.seed),
            '--device', args.device,
            '--updates', str(args.updates),
            '--checkpoint-dir', str(trial_dir),
            '--lr', str(trial.overrides['lr']),
        ]
        if args.smoke:
            cmd.append('--smoke')
        return cmd

    def parse_score(trial_dir: Path):
        rep = trial_dir / 'reporte.txt'
        if rep.is_file():
            return parse_mean_wr(rep.read_text(encoding='utf-8'))
        meta = trial_dir / 'train_meta.json'
        if meta.is_file():
            import json
            d = json.loads(meta.read_text(encoding='utf-8'))
            if 'MEAN_SCORE' in d:
                return float(d['MEAN_SCORE'])
        return None

    run_hp_tune_loop(
        agent='team_preview_nash',
        candidates=preview_nash_candidates(),
        build_cmd=build_cmd,
        parse_score_from_dir=parse_score,
        logs_dir=logs,
        max_trials=int(args.max_trials),
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        frozen=FROZEN,
        criteria_note=(
            'maximize MEAN_SCORE (-loss proxy on mock); accept if improves best; '
            'independent of MCTS'
        ),
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
