"""Utilidades compartidas para tune de hiperparámetros RL.

Cubre DDQN / PPO / Preview PPO / QD-PPO / QD-PGA / QD-PPGA.
Tracking: history.jsonl + progress.json + tracking.txt + best_hp.json
(con prev_overrides -> overrides en cada trial).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def parse_mean_wr(reporte_text: str) -> Optional[float]:
    """Lee MEAN_WR / MEAN_SCORE del reporte de train (0–1 o score proxy)."""
    m = re.search(r'MEAN_WR:\s*([+\-]?[\d.]+)', reporte_text)
    if m:
        return float(m.group(1))
    m = re.search(r'MEAN_SCORE:\s*([+\-]?[\d.]+)', reporte_text)
    if m:
        return float(m.group(1))
    return None


@dataclass(frozen=True)
class HpTrial:
    """Un candidato de hiperparámetros."""

    overrides: Dict[str, float]
    label: str = ''

    def key(self) -> str:
        parts = [f'{k}={self.overrides[k]}' for k in sorted(self.overrides)]
        return ','.join(parts)


@dataclass
class TrialOutcome:
    trial_index: int
    overrides: Dict[str, float]
    score: Optional[float]
    accepted: bool
    elapsed_s: float
    return_code: int
    checkpoint_dir: str
    note: str = ''


def ddqn_candidates() -> List[HpTrial]:
    out: List[HpTrial] = []
    for lr in (1e-4, 3e-4, 1e-3):
        for gamma in (0.90, 0.95):
            out.append(HpTrial({'lr': lr, 'gamma': gamma}))
    return out


def ppo_combat_candidates() -> List[HpTrial]:
    out: List[HpTrial] = []
    for lr in (1e-4, 3e-4, 1e-3):
        for clip in (0.10, 0.20):
            out.append(HpTrial({'lr': lr, 'clip_eps': clip}))
    return out


def preview_ppo_candidates() -> List[HpTrial]:
    out: List[HpTrial] = []
    for lr in (1e-4, 3e-4, 1e-3):
        for clip in (0.10, 0.20):
            out.append(HpTrial({'lr': lr, 'clip_eps': clip}))
    return out


def qd_combat_candidates() -> List[HpTrial]:
    """Mismo grid que PPO combate (fase mínima; knobs QD archive después)."""
    return ppo_combat_candidates()


def preview_nash_candidates() -> List[HpTrial]:
    """Solo lr (fase mínima; solver/epochs congelados)."""
    return [HpTrial({'lr': lr}) for lr in (1e-4, 3e-4, 1e-3)]


def preview_qd_pga_candidates() -> List[HpTrial]:
    """Mismo grid que Preview PPO: lr × clip_eps."""
    return preview_ppo_candidates()


def override_deltas(
    prev: Optional[Dict[str, float]],
    cur: Dict[str, float],
) -> Dict[str, Any]:
    """Describe cómo cambian los knobs respecto al trial anterior."""
    if prev is None:
        return {k: {'from': None, 'to': v, 'delta': None} for k, v in cur.items()}
    keys = sorted(set(prev) | set(cur))
    out: Dict[str, Any] = {}
    for k in keys:
        a = prev.get(k)
        b = cur.get(k)
        delta = None
        if a is not None and b is not None:
            delta = float(b) - float(a)
        out[k] = {'from': a, 'to': b, 'delta': delta}
    return out


def pick_best(
    outcomes: Sequence[TrialOutcome],
    *,
    eps: float = 1e-9,
) -> Optional[TrialOutcome]:
    best: Optional[TrialOutcome] = None
    for o in outcomes:
        if o.score is None:
            continue
        if best is None or o.score > (best.score or -1e30) + eps:
            best = o
    return best


def format_tracking_text(
    *,
    agent: str,
    run_id: str,
    outcomes: Sequence[TrialOutcome],
    best: Optional[TrialOutcome],
    frozen: Dict[str, Any],
    criteria_note: str,
) -> str:
    lines = [
        f'=== RL HP tune tracking ({agent}) ===',
        f'run_id: {run_id}',
        f'frozen_defaults: {frozen}',
        f'criteria: {criteria_note}',
        '',
        '--- trial log (prev -> new HP) ---',
    ]
    prev_ov: Optional[Dict[str, float]] = None
    for o in outcomes:
        mark = ' [BEST]' if best is not None and o.trial_index == best.trial_index else ''
        acc = 'ACCEPT' if o.accepted else 'reject'
        sc = '-' if o.score is None else f'{o.score:.4f}'
        deltas = override_deltas(prev_ov, o.overrides)
        delta_bits = []
        for k, d in deltas.items():
            if d['from'] is None:
                delta_bits.append(f"{k}: (start)->{d['to']}")
            else:
                delta_bits.append(f"{k}: {d['from']}->{d['to']} (d={d['delta']})")
        lines.append(
            f'{o.trial_index}. {o.overrides} -> score={sc} {acc} '
            f't={o.elapsed_s:.1f}s rc={o.return_code}{mark}'
        )
        lines.append(f'   param_change: {"; ".join(delta_bits)}')
        if o.note:
            lines.append(f'   note: {o.note}')
        prev_ov = dict(o.overrides)
    lines.append('')
    if best is not None:
        lines.append(f'best_overrides: {best.overrides}')
        lines.append(f'best_score: {best.score}')
    else:
        lines.append('best_overrides: (none)')
    return '\n'.join(lines) + '\n'


@dataclass
class TunePaths:
    root: Path
    history: Path
    progress: Path
    tracking: Path
    summary_json: Path
    summary_txt: Path
    best_hp: Path

    @classmethod
    def for_agent(cls, logs_dir: Path, agent: str) -> 'TunePaths':
        return cls(
            root=logs_dir / f'{agent}_hp_tune',
            history=logs_dir / f'{agent}_hp_tune_history.jsonl',
            progress=logs_dir / f'{agent}_hp_tune_progress.json',
            tracking=logs_dir / f'{agent}_hp_tune_tracking.txt',
            summary_json=logs_dir / f'{agent}_hp_tune.json',
            summary_txt=logs_dir / f'{agent}_hp_tune.txt',
            best_hp=logs_dir / f'{agent}_best_hp.json',
        )


def run_hp_tune_loop(
    *,
    agent: str,
    candidates: Sequence[HpTrial],
    build_cmd,
    parse_score_from_dir,
    logs_dir: Path,
    max_trials: int,
    seed: int,
    dry_run: bool,
    frozen: Dict[str, Any],
    criteria_note: str,
    fake_score_fn=None,
) -> Dict[str, Any]:
    """Ejecuta hasta max_trials candidatos; acepta si mejora el mejor score.

    ``build_cmd(trial, trial_ckpt_dir) -> list[str]``
    ``parse_score_from_dir(trial_ckpt_dir) -> Optional[float]``
    """
    import subprocess
    import time
    import uuid

    from showdown.inference_sim.paths import REPO_ROOT

    run_id = (
        f'{agent}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}_'
        f'{uuid.uuid4().hex[:8]}'
    )
    paths = TunePaths.for_agent(logs_dir, agent)
    if paths.history.is_file():
        paths.history.unlink()
    paths.root.mkdir(parents=True, exist_ok=True)

    trials = list(candidates)[: max(1, int(max_trials))]
    append_jsonl(paths.history, {
        'ts_utc': utc_now(),
        'event': 'tune_start',
        'run_id': run_id,
        'agent': agent,
        'max_trials': max_trials,
        'n_candidates': len(trials),
        'seed': seed,
        'frozen': frozen,
        'dry_run': dry_run,
    })

    outcomes: List[TrialOutcome] = []
    best: Optional[TrialOutcome] = None
    prev_overrides: Optional[Dict[str, float]] = None

    for i, trial in enumerate(trials, start=1):
        trial_dir = paths.root / f'trial_{i:02d}'
        trial_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        score: Optional[float] = None
        rc = 0
        note = ''
        param_change = override_deltas(prev_overrides, dict(trial.overrides))

        if dry_run:
            if fake_score_fn is not None:
                score = float(fake_score_fn(trial))
            else:
                # Prefer values closer to defaults 3e-4 / 0.95 or 0.2
                score = 0.5
                if 'lr' in trial.overrides:
                    score -= abs(float(trial.overrides['lr']) - 3e-4) * 100
                if 'gamma' in trial.overrides:
                    score -= abs(float(trial.overrides['gamma']) - 0.95) * 2
                if 'clip_eps' in trial.overrides:
                    score -= abs(float(trial.overrides['clip_eps']) - 0.2)
            note = 'dry_run_fake_score'
        else:
            cmd = build_cmd(trial, trial_dir)
            print(f'\n>>> trial {i}/{len(trials)} {trial.overrides}\n{" ".join(cmd)}', flush=True)
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True)
            rc = int(proc.returncode)
            score = parse_score_from_dir(trial_dir)
            if score is None:
                note = 'missing_score'

        elapsed = time.perf_counter() - t0
        accepted = False
        if score is not None:
            if best is None or score > (best.score or -1e30):
                accepted = True
                best = TrialOutcome(
                    trial_index=i,
                    overrides=dict(trial.overrides),
                    score=score,
                    accepted=True,
                    elapsed_s=elapsed,
                    return_code=rc,
                    checkpoint_dir=str(trial_dir),
                    note=note,
                )

        outcome = TrialOutcome(
            trial_index=i,
            overrides=dict(trial.overrides),
            score=score,
            accepted=accepted,
            elapsed_s=elapsed,
            return_code=rc,
            checkpoint_dir=str(trial_dir),
            note=note,
        )
        outcomes.append(outcome)

        append_jsonl(paths.history, {
            'ts_utc': utc_now(),
            'event': 'hp_trial',
            'run_id': run_id,
            'agent': agent,
            'trial_index': i,
            'trials_total': len(trials),
            'prev_overrides': prev_overrides,
            'overrides': trial.overrides,
            'param_change': param_change,
            'score': score,
            'accepted': accepted,
            'best_so_far': None if best is None else best.overrides,
            'best_score_so_far': None if best is None else best.score,
            'elapsed_s': elapsed,
            'return_code': rc,
            'checkpoint_dir': str(trial_dir),
            'note': note,
        })
        write_json(paths.progress, {
            'ts_utc': utc_now(),
            'run_id': run_id,
            'agent': agent,
            'status': 'running',
            'trial_index': i,
            'trials_total': len(trials),
            'prev_overrides': prev_overrides,
            'current_overrides': trial.overrides,
            'param_change': param_change,
            'score': score,
            'accepted': accepted,
            'best_overrides': None if best is None else best.overrides,
            'best_score': None if best is None else best.score,
        })
        print(
            f'  TRACK trial {i}/{len(trials)} '
            f'prev={prev_overrides} -> new={trial.overrides} '
            f'score={score} accepted={accepted} best={None if best is None else best.overrides}',
            flush=True,
        )
        prev_overrides = dict(trial.overrides)

    tracking = format_tracking_text(
        agent=agent,
        run_id=run_id,
        outcomes=outcomes,
        best=best,
        frozen=frozen,
        criteria_note=criteria_note,
    )
    paths.tracking.write_text(tracking, encoding='utf-8')
    if best is not None:
        write_json(paths.best_hp, {
            'agent': agent,
            'run_id': run_id,
            'overrides': best.overrides,
            'score': best.score,
            'checkpoint_dir': best.checkpoint_dir,
            'time_utc': utc_now(),
        })

    append_jsonl(paths.history, {
        'ts_utc': utc_now(),
        'event': 'tune_done',
        'run_id': run_id,
        'agent': agent,
        'best_overrides': None if best is None else best.overrides,
        'best_score': None if best is None else best.score,
    })
    write_json(paths.progress, {
        'ts_utc': utc_now(),
        'run_id': run_id,
        'agent': agent,
        'status': 'done',
        'best_overrides': None if best is None else best.overrides,
        'best_score': None if best is None else best.score,
        'artifacts': {
            'history': str(paths.history),
            'tracking': str(paths.tracking),
            'best_hp': str(paths.best_hp),
        },
    })

    payload = {
        'time_utc': utc_now(),
        'run_id': run_id,
        'agent': agent,
        'dry_run': dry_run,
        'seed': seed,
        'frozen': frozen,
        'criteria': criteria_note,
        'outcomes': [asdict(o) for o in outcomes],
        'best_overrides': None if best is None else best.overrides,
        'best_score': None if best is None else best.score,
        'artifacts': {
            'history': str(paths.history),
            'tracking': str(paths.tracking),
            'best_hp': str(paths.best_hp),
            'summary_json': str(paths.summary_json),
        },
    }
    # Figuras reales + ledger histórico (append-only).
    try:
        from showdown.inference_sim.tune_plots import FIGURES_DIR, plot_hp_tune_run
        fig_info = plot_hp_tune_run(
            agent=agent,
            history_path=paths.history,
            run_id=run_id,
            best_score=None if best is None else best.score,
            best_overrides=None if best is None else best.overrides,
            update_ledger=not dry_run,
        )
        payload['figures'] = fig_info
        arts = payload.setdefault('artifacts', {})
        arts['figures_dir'] = str(FIGURES_DIR)
        latest = (fig_info or {}).get('latest_figures') or {}
        if latest.get('svg'):
            arts['figure_svg'] = latest['svg']
        if latest.get('png'):
            arts['figure_png'] = latest['png']
    except Exception as exc:  # pragma: no cover - plotting must not fail tune
        payload['figures_error'] = repr(exc)

    write_json(paths.summary_json, payload)
    paths.summary_txt.write_text(
        f'=== {agent} HP tune ===\nrun_id: {run_id}\n\n' + tracking,
        encoding='utf-8',
    )
    print(tracking, flush=True)
    fig_note = ''
    if payload.get('figures'):
        latest = payload['figures'].get('latest_figures') or {}
        fig_note = f' figure={latest.get("svg") or latest.get("png")}'
    print(
        f'DONE {agent}_hp_tune best={None if best is None else best.overrides} '
        f'score={None if best is None else best.score} tracking={paths.tracking}'
        f'{fig_note}',
        flush=True,
    )
    return payload

