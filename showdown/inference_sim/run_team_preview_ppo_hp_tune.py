"""Tune de hiperparámetros Team Preview Masked PPO (independiente de DDQN/PPO combate).

Entrena con pi_c=greedy_joint (mock backend en train actual). Un solo checkpoint
se reutiliza en Conclusión para DDQN y PPO combate.

Ejemplos::

  python -m showdown.inference_sim.run_team_preview_ppo_hp_tune --dry-run
  python -m showdown.inference_sim.run_team_preview_ppo_hp_tune --seed 0 --max-trials 6 --smoke --updates 3

Knobs: lr, clip_eps. Tracking en logs/team_preview_ppo_hp_tune_*.
Score proxy: MEAN_SCORE = -pg_loss (hasta haber suite BO1 dedicada).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from showdown.inference_sim.paths import REPO_ROOT
from showdown.inference_sim.rl_hp_tune import (
    parse_mean_wr,
    preview_ppo_candidates,
    run_hp_tune_loop,
)

FROZEN = {
    'gamma': 1.0,
    'gae_lambda': 1.0,
    'ent_coef': 0.02,
    'pi_c': 'greedy_joint',
    'note': 'Varying: lr, clip_eps; independent of combat DDQN/PPO trains',
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Team Preview PPO HP tune + tracking')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-trials', type=int, default=6)
    ap.add_argument('--updates', type=int, default=3)
    ap.add_argument('--smoke', action='store_true', default=True)
    ap.add_argument('--no-smoke', dest='smoke', action='store_false')
    ap.add_argument('--device', default='cpu', help='cpu | cuda | auto')
    ap.add_argument('--parallel-trials', type=int, default=1,
                    help='Trials simultáneos (uno por GPU, puertos separados). Servidor: 3')
    ap.add_argument('--backend', choices=('mock', 'live'), default='mock',
                    help='mock = proxy; live = BO1 reales con π_c congelado')
    ap.add_argument('--num-workers', type=int, default=0,
                    help='BO1 en paralelo por trial (live); 0 = automático')
    ap.add_argument('--start-ps', action='store_true')
    ap.add_argument('--pilot', default='greedy_joint')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(list(argv) if argv is not None else None)

    logs = REPO_ROOT / 'logs'

    def build_cmd(trial, trial_dir: Path):
        cmd = [
            sys.executable, '-m', 'showdown.rl.team_preview_ppo.train',
            '--seed', str(args.seed),
            '--device', args.device,
            '--updates', str(args.updates),
            '--checkpoint-dir', str(trial_dir),
            '--lr', str(trial.overrides['lr']),
            '--clip-eps', str(trial.overrides['clip_eps']),
        ]
        if args.smoke:
            cmd.append('--smoke')
        cmd.extend(['--backend', args.backend, '--pilot', args.pilot,
                    '--num-workers', str(args.num_workers)])
        if args.start_ps:
            cmd.append('--start-ps')
        return cmd

    def parse_score(trial_dir: Path):
        rep = trial_dir / 'reporte.txt'
        if not rep.is_file():
            meta = trial_dir / 'train_meta.json'
            if meta.is_file():
                import json
                d = json.loads(meta.read_text(encoding='utf-8'))
                if 'MEAN_SCORE' in d:
                    return float(d['MEAN_SCORE'])
            return None
        return parse_mean_wr(rep.read_text(encoding='utf-8'))

    run_hp_tune_loop(
        agent='team_preview_ppo',
        candidates=preview_ppo_candidates(),
        build_cmd=build_cmd,
        parse_score_from_dir=parse_score,
        logs_dir=logs,
        max_trials=int(args.max_trials),
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        frozen=FROZEN,
        parallel_trials=int(args.parallel_trials),
        criteria_note='maximize MEAN_SCORE (-pg_loss proxy on mock); accept if improves best',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
