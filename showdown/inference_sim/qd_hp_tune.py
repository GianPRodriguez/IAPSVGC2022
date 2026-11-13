"""Factory compartida para tune HP de QD-PPO / QD-PGA / QD-PPGA combate.

Knobs mínimos: lr, clip_eps (mismo grid que PPO combate).
Score: MEAN_WR de π* (suite Cand. 1). Independiente de Preview QD-PGA.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from showdown.inference_sim.paths import REPO_ROOT
from showdown.inference_sim.rl_hp_tune import (
    parse_mean_wr,
    qd_combat_candidates,
    run_hp_tune_loop,
)

FROZEN_COMMON = {
    'gamma': 0.95,
    'gae_lambda': 0.95,
    'ent_coef': 0.01,
    'grid': '10x10',
    'critic_lr_shared': 3e-4,
    'note': 'Varying: lr, clip_eps (actor); archive knobs frozen for now',
    'preview': 'independent team_preview_qd_pga (not co-tuned)',
}


def run_qd_agent_hp_tune(
    *,
    agent: str,
    train_module: str,
    argv: Optional[Sequence[str]] = None,
) -> int:
    ap = argparse.ArgumentParser(
        description=f'{agent} combat HP tune + tracking (independent of preview QD-PGA)',
    )
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-trials', type=int, default=6)
    ap.add_argument('--t-total', type=int, default=6_000)
    ap.add_argument('--suite-games', type=int, default=4)
    ap.add_argument('--device', default='cpu', help='cpu | cuda | auto')
    ap.add_argument('--parallel-trials', type=int, default=1,
                    help='Trials simultáneos (uno por GPU, puertos separados). Servidor: 3')
    ap.add_argument('--bc-checkpoint', default=None)
    ap.add_argument('--start-ps', action='store_true')
    ap.add_argument('--smoke', action='store_true', default=True)
    ap.add_argument('--no-smoke', action='store_true', help='Disable smoke caps')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(list(argv) if argv is not None else None)

    smoke = bool(args.smoke) and not bool(args.no_smoke)
    logs = REPO_ROOT / 'logs'
    frozen = dict(FROZEN_COMMON)
    frozen['t_total_per_trial'] = int(args.t_total)
    frozen['smoke'] = smoke

    def build_cmd(trial, trial_dir: Path):
        cmd = [
            sys.executable, '-m', train_module,
            '--seed', str(args.seed),
            '--device', args.device,
            '--t-total', str(args.t_total),
            '--suite-games', str(args.suite_games),
            '--checkpoint-dir', str(trial_dir),
            '--lr', str(trial.overrides['lr']),
            '--clip-eps', str(trial.overrides['clip_eps']),
        ]
        if smoke:
            cmd.append('--smoke')
        if args.start_ps:
            cmd.append('--start-ps')
        if args.bc_checkpoint:
            cmd.extend(['--bc-checkpoint', str(args.bc_checkpoint)])
        return cmd

    def parse_score(trial_dir: Path):
        rep = trial_dir / 'reporte.txt'
        if not rep.is_file():
            return None
        return parse_mean_wr(rep.read_text(encoding='utf-8'))

    run_hp_tune_loop(
        agent=agent,
        candidates=qd_combat_candidates(),
        build_cmd=build_cmd,
        parse_score_from_dir=parse_score,
        logs_dir=logs,
        max_trials=int(args.max_trials),
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        frozen=frozen,
        parallel_trials=int(args.parallel_trials),
        criteria_note=(
            'maximize MEAN_WR of pi* from offline suite (Cand.1); '
            'accept if improves best; QD-score tracked in train report only'
        ),
    )
    return 0
