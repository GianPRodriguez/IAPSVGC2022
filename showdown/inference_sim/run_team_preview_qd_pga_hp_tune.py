"""Tune HP Team Preview QD-PGA (independiente de QD combate).

Ejemplos::

  python -m showdown.inference_sim.run_team_preview_qd_pga_hp_tune --dry-run
  python -m showdown.inference_sim.run_team_preview_qd_pga_hp_tune --seed 0 --max-trials 6 --smoke

Knobs: lr, clip_eps. Score: MEAN_SCORE = best_f.
Tracking: logs/team_preview_qd_pga_hp_tune_* (prev -> new).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from showdown.inference_sim.paths import REPO_ROOT
from showdown.inference_sim.rl_hp_tune import (
    parse_mean_wr,
    preview_qd_pga_candidates,
    run_hp_tune_loop,
)

FROZEN = {
    'p_mut': 0.5,
    'sigma': 0.02,
    'grid': '10x10',
    'pi_c': 'greedy_joint',
    'note': 'Varying: lr, clip_eps; independent of QD-PPO/PGA/PPGA combat trains',
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Team Preview QD-PGA HP tune + tracking')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-trials', type=int, default=6)
    ap.add_argument('--smoke', action='store_true', default=True)
    ap.add_argument('--no-smoke', dest='smoke', action='store_false')
    ap.add_argument('--device', default='cpu', help='cpu | cuda | auto')
    ap.add_argument('--parallel-trials', type=int, default=1,
                    help='Trials simultáneos (uno por GPU, puertos separados). Servidor: 3')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(list(argv) if argv is not None else None)

    logs = REPO_ROOT / 'logs'

    def build_cmd(trial, trial_dir: Path):
        cmd = [
            sys.executable, '-m', 'showdown.rl.team_preview_qd_pga.train',
            '--seed', str(args.seed),
            '--device', args.device,
            '--checkpoint-dir', str(trial_dir),
            '--lr', str(trial.overrides['lr']),
            '--clip-eps', str(trial.overrides['clip_eps']),
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
        agent='team_preview_qd_pga',
        candidates=preview_qd_pga_candidates(),
        build_cmd=build_cmd,
        parse_score_from_dir=parse_score,
        logs_dir=logs,
        max_trials=int(args.max_trials),
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        frozen=FROZEN,
        parallel_trials=int(args.parallel_trials),
        criteria_note=(
            'maximize MEAN_SCORE (best_f elite fitness); accept if improves best; '
            'independent of QD combat'
        ),
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
