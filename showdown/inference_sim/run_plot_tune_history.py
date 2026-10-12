"""Regenera figuras desde history/ledger existentes.

Ejemplos::

  python -m showdown.inference_sim.run_plot_tune_history --agent ddqn
  python -m showdown.inference_sim.run_plot_tune_history --all
  python -m showdown.inference_sim.run_plot_tune_history --ledger-only
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from showdown.inference_sim.paths import REPO_ROOT
from showdown.inference_sim.tune_plots import (
    FIGURES_DIR,
    LEDGER_PATH,
    plot_hp_tune_run,
    plot_ledger,
    plot_mcts_sweep_run,
    read_jsonl,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Plot HP-tune / sweep history + ledger')
    ap.add_argument('--agent', default=None, help='e.g. ddqn, ppo, mcts_puct, team_preview_nash')
    ap.add_argument('--all', action='store_true', help='Plot ledger for all agents')
    ap.add_argument('--ledger-only', action='store_true')
    ap.add_argument('--history', type=str, default=None, help='Override history.jsonl path')
    args = ap.parse_args(list(argv) if argv is not None else None)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    logs = REPO_ROOT / 'logs'

    if args.ledger_only or args.all:
        arts = plot_ledger(agent=None if args.all or args.ledger_only else args.agent)
        print(f'ledger={LEDGER_PATH}')
        print(f'figures={arts}')
        if args.ledger_only and not args.agent:
            return 0

    agent = args.agent
    if not agent:
        if args.all:
            # rebuild per-agent ledgers from existing history files
            for hist in sorted(logs.glob('*_hp_tune_history.jsonl')):
                name = hist.name.replace('_hp_tune_history.jsonl', '')
                rows = read_jsonl(hist)
                run_id = 'regen'
                for r in reversed(rows):
                    if r.get('event') == 'tune_done':
                        run_id = str(r.get('run_id') or run_id)
                        break
                best_score = None
                best_ov = None
                for r in reversed(rows):
                    if r.get('event') == 'tune_done':
                        best_score = r.get('best_score')
                        best_ov = r.get('best_overrides')
                        break
                info = plot_hp_tune_run(
                    agent=name,
                    history_path=hist,
                    run_id=run_id,
                    best_score=None if best_score is None else float(best_score),
                    best_overrides=best_ov,
                    update_ledger=False,
                )
                print(f'{name}: {info.get("latest_figures")}')
            for hist in sorted(logs.glob('mcts_sims_sweep_*_history.jsonl')):
                name = hist.name.replace('mcts_sims_sweep_', '').replace('_history.jsonl', '')
                info = plot_mcts_sweep_run(
                    agent=name, history_path=hist, run_id='regen',
                )
                print(f'{name} sweep: {info.get("latest_figures")}')
            print(f'ledger_all={plot_ledger(agent=None)}')
            return 0
        ap.error('pass --agent NAME or --all or --ledger-only')

    if args.history:
        hist = Path(args.history)
    else:
        hist = logs / f'{agent}_hp_tune_history.jsonl'
        if not hist.is_file():
            alt = logs / f'mcts_sims_sweep_{agent}_history.jsonl'
            if alt.is_file():
                hist = alt
    if not hist.is_file():
        print(f'missing history: {hist}')
        return 1

    if 'mcts_sims_sweep' in hist.name:
        info = plot_mcts_sweep_run(agent=agent, history_path=hist, run_id='regen')
    else:
        rows = read_jsonl(hist)
        best_score = None
        best_ov = None
        run_id = 'regen'
        for r in reversed(rows):
            if r.get('event') == 'tune_done':
                best_score = r.get('best_score')
                best_ov = r.get('best_overrides')
                run_id = str(r.get('run_id') or run_id)
                break
        info = plot_hp_tune_run(
            agent=agent,
            history_path=hist,
            run_id=run_id,
            best_score=None if best_score is None else float(best_score),
            best_overrides=best_ov,
            update_ledger=False,
        )
    print(info.get('latest_figures'))
    print(f'ledger_agent={plot_ledger(agent=agent)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
