"""Gráficas reales al cerrar tune/sweep/train + ledger histórico acumulado.

- Lee ``*_hp_tune_history.jsonl`` / MCTS sweep history / ``reporte.json``.
- Escribe PNG (si hay matplotlib) y siempre SVG (stdlib).
- Append-only ledger: ``logs/figures/hp_tune_ledger.jsonl`` → figura tesis.

CLI::

  python -m showdown.inference_sim.run_plot_tune_history --agent ddqn
  python -m showdown.inference_sim.run_plot_tune_history --all
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

from showdown.inference_sim.paths import REPO_ROOT

FIGURES_DIR = REPO_ROOT / 'logs' / 'figures'
LEDGER_PATH = FIGURES_DIR / 'hp_tune_ledger.jsonl'


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


@dataclass
class TrialPoint:
    index: int
    score: Optional[float]
    accepted: bool
    label: str
    best_so_far: Optional[float]


def trials_from_hp_history(rows: Sequence[Dict[str, Any]]) -> List[TrialPoint]:
    out: List[TrialPoint] = []
    for r in rows:
        if r.get('event') != 'hp_trial':
            continue
        ov = r.get('overrides') or {}
        label = ','.join(f'{k}={ov[k]}' for k in sorted(ov))
        out.append(TrialPoint(
            index=int(r.get('trial_index') or len(out) + 1),
            score=None if r.get('score') is None else float(r['score']),
            accepted=bool(r.get('accepted')),
            label=label or f'trial{len(out)+1}',
            best_so_far=(
                None if r.get('best_score_so_far') is None
                else float(r['best_score_so_far'])
            ),
        ))
    return out


def trials_from_mcts_history(rows: Sequence[Dict[str, Any]]) -> List[TrialPoint]:
    out: List[TrialPoint] = []
    best: Optional[float] = None
    for r in rows:
        if r.get('event') != 'hyperparam_step':
            continue
        pt = r.get('point') or {}
        wr = pt.get('wr_raw', r.get('wr_raw', r.get('wr', r.get('mean_wr'))))
        # Store as fraction 0–1 if given as percent
        score = None if wr is None else float(wr)
        if score is not None and score > 1.5:
            score = score / 100.0
        if score is not None:
            best = score if best is None else max(best, score)
        hp = r.get('hyperparams_set') or {}
        n = (
            pt.get('n_sims')
            or r.get('current_n_sims')
            or r.get('n_sims')
            or next(iter(hp.values()), None)
        )
        out.append(TrialPoint(
            index=len(out) + 1,
            score=score,
            accepted=True,
            label=f'N={n}',
            best_so_far=best,
        ))
    return out


def trials_from_expectimax_history(rows: Sequence[Dict[str, Any]]) -> List[TrialPoint]:
    out: List[TrialPoint] = []
    best: Optional[float] = None
    for i, r in enumerate(rows, start=1):
        m = r.get('metrics') or {}
        wr = m.get('pair_wr')
        if wr is None:
            wr = m.get('wr_raw')
        score = None if wr is None else float(wr) / 100.0
        acc = bool(r.get('accepted'))
        if score is not None and acc:
            best = score if best is None else max(best, score)
        kind = str(r.get('kind') or f'step{i}')
        out.append(TrialPoint(
            index=i,
            score=score,
            accepted=acc,
            label=kind,
            best_so_far=best,
        ))
    return out


def _svg_polyline_chart(
    *,
    title: str,
    xs: Sequence[float],
    series: Sequence[Tuple[str, Sequence[Optional[float]], str]],
    width: int = 720,
    height: int = 420,
    ylab: str = 'score',
) -> str:
    """Minimal SVG line chart (no external deps)."""
    pad_l, pad_r, pad_t, pad_b = 56, 24, 36, 56
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    ys_all = [v for _, ys, _ in series for v in ys if v is not None]
    if not xs or not ys_all:
        body = (
            f'<text x="{width/2}" y="{height/2}" text-anchor="middle" '
            f'font-family="sans-serif" fill="#666">no data</text>'
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}">{body}</svg>\n'
        )
    ymin = min(ys_all)
    ymax = max(ys_all)
    if abs(ymax - ymin) < 1e-12:
        ymin -= 0.05
        ymax += 0.05
    x0, x1 = min(xs), max(xs)
    if abs(x1 - x0) < 1e-12:
        x0 -= 0.5
        x1 += 0.5

    def sx(x: float) -> float:
        return pad_l + (x - x0) / (x1 - x0) * plot_w

    def sy(y: float) -> float:
        return pad_t + (1.0 - (y - ymin) / (ymax - ymin)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<text x="{width/2}" y="22" text-anchor="middle" '
        f'font-family="sans-serif" font-size="15" fill="#222">{escape(title)}</text>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" '
        f'stroke="#333" stroke-width="1"/>',
        f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" '
        f'y2="{pad_t+plot_h}" stroke="#333" stroke-width="1"/>',
        f'<text x="12" y="{pad_t+plot_h/2}" transform='
        f'"rotate(-90 12,{pad_t+plot_h/2})" font-family="sans-serif" '
        f'font-size="11" fill="#444">{escape(ylab)}</text>',
        f'<text x="{pad_l+plot_w/2}" y="{height-14}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="11" fill="#444">trial / step</text>',
    ]
    # grid
    for i in range(5):
        gy = pad_t + plot_h * i / 4
        val = ymax - (ymax - ymin) * i / 4
        parts.append(
            f'<line x1="{pad_l}" y1="{gy}" x2="{pad_l+plot_w}" y2="{gy}" '
            f'stroke="#ddd" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l-6}" y="{gy+4}" text-anchor="end" '
            f'font-family="sans-serif" font-size="10" fill="#666">'
            f'{val:.3f}</text>'
        )

    legend_y = pad_t + 8
    for name, ys, color in series:
        pts = []
        for x, y in zip(xs, ys):
            if y is None:
                continue
            pts.append(f'{sx(x):.1f},{sy(float(y)):.1f}')
        if pts:
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                f'points="{" ".join(pts)}"/>'
            )
            for x, y in zip(xs, ys):
                if y is None:
                    continue
                parts.append(
                    f'<circle cx="{sx(x):.1f}" cy="{sy(float(y)):.1f}" r="3.5" '
                    f'fill="{color}"/>'
                )
        parts.append(
            f'<text x="{pad_l+plot_w-8}" y="{legend_y}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11" fill="{color}">'
            f'{escape(name)}</text>'
        )
        legend_y += 16

    parts.append('</svg>\n')
    return '\n'.join(parts)


def _try_matplotlib_png(
    path: Path,
    *,
    title: str,
    xs: Sequence[float],
    series: Sequence[Tuple[str, Sequence[Optional[float]], str]],
    ylab: str,
) -> bool:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, ys, color in series:
        yplot = [float('nan') if v is None else float(v) for v in ys]
        ax.plot(xs, yplot, marker='o', label=name, color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('trial / step')
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def write_trial_charts(
    trials: Sequence[TrialPoint],
    *,
    stem: Path,
    title: str,
    ylab: str = 'score',
) -> Dict[str, str]:
    """Write ``stem.svg`` (+ ``stem.png`` if matplotlib). Returns artifact paths."""
    ensure_figures_dir()
    xs = [float(t.index) for t in trials]
    scores = [t.score for t in trials]
    bests = [t.best_so_far for t in trials]
    series = [
        ('score', scores, '#1f77b4'),
        ('best_so_far', bests, '#d62728'),
    ]
    svg_path = Path(str(stem) + '.svg')
    png_path = Path(str(stem) + '.png')
    svg_path.write_text(
        _svg_polyline_chart(title=title, xs=xs, series=series, ylab=ylab),
        encoding='utf-8',
    )
    arts = {'svg': str(svg_path)}
    if _try_matplotlib_png(png_path, title=title, xs=xs, series=series, ylab=ylab):
        arts['png'] = str(png_path)
    return arts


def append_ledger_entry(
    *,
    agent: str,
    run_id: str,
    best_score: Optional[float],
    best_overrides: Optional[Dict[str, Any]],
    n_trials: int,
    source: str,
    figure_stem: Optional[str] = None,
) -> None:
    """Append-only historical row (never truncates)."""
    append_jsonl(LEDGER_PATH, {
        'ts_utc': utc_now(),
        'agent': agent,
        'run_id': run_id,
        'best_score': best_score,
        'best_overrides': best_overrides,
        'n_trials': n_trials,
        'source': source,
        'figure_stem': figure_stem,
    })


def plot_ledger(
    *,
    agent: Optional[str] = None,
    out_stem: Optional[Path] = None,
) -> Dict[str, str]:
    """Cumulative best_score across historical tune runs (thesis figure)."""
    rows = read_jsonl(LEDGER_PATH)
    if agent:
        rows = [r for r in rows if r.get('agent') == agent]
    if out_stem is None:
        tag = agent or 'all'
        out_stem = FIGURES_DIR / f'hp_tune_ledger_{tag}'
    # group by agent for multi-series when agent is None
    if agent:
        xs = list(range(1, len(rows) + 1))
        scores = [
            None if r.get('best_score') is None else float(r['best_score'])
            for r in rows
        ]
        # running max for visual "mejora histórica"
        run_max: List[Optional[float]] = []
        cur: Optional[float] = None
        for s in scores:
            if s is not None:
                cur = s if cur is None else max(cur, s)
            run_max.append(cur)
        series = [
            ('run best', scores, '#1f77b4'),
            ('historical max', run_max, '#d62728'),
        ]
        title = f'Historical HP-tune improvements — {agent}'
    else:
        by_agent: Dict[str, List[Optional[float]]] = {}
        for r in rows:
            a = str(r.get('agent') or '?')
            by_agent.setdefault(a, []).append(
                None if r.get('best_score') is None else float(r['best_score'])
            )
        max_len = max((len(v) for v in by_agent.values()), default=0)
        xs = list(range(1, max_len + 1))
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        ]
        series = []
        for i, (a, vals) in enumerate(sorted(by_agent.items())):
            padded = list(vals) + [None] * (max_len - len(vals))
            # running max per agent
            run_max: List[Optional[float]] = []
            cur = None
            for s in padded:
                if s is not None:
                    cur = s if cur is None else max(cur, s)
                run_max.append(cur)
            series.append((a, run_max, colors[i % len(colors)]))
        title = 'Historical HP-tune improvements — all agents'
    return write_trial_charts(
        [
            TrialPoint(
                index=int(x),
                score=series[0][1][i] if series else None,
                accepted=True,
                label=str(int(x)),
                best_so_far=series[0][1][i] if series else None,
            )
            for i, x in enumerate(xs)
        ] if agent else [
            TrialPoint(index=int(x), score=None, accepted=True, label=str(int(x)), best_so_far=None)
            for x in xs
        ],
        stem=out_stem,
        title=title,
        ylab='best_score',
    ) if agent else _write_multi_series(out_stem, title=title, xs=xs, series=series)


def _write_multi_series(
    stem: Path,
    *,
    title: str,
    xs: Sequence[float],
    series: Sequence[Tuple[str, Sequence[Optional[float]], str]],
) -> Dict[str, str]:
    ensure_figures_dir()
    svg_path = Path(str(stem) + '.svg')
    png_path = Path(str(stem) + '.png')
    svg_path.write_text(
        _svg_polyline_chart(title=title, xs=xs, series=series, ylab='best_score (hist max)'),
        encoding='utf-8',
    )
    arts = {'svg': str(svg_path)}
    if _try_matplotlib_png(
        png_path, title=title, xs=xs, series=series, ylab='best_score (hist max)',
    ):
        arts['png'] = str(png_path)
    return arts


def plot_hp_tune_run(
    *,
    agent: str,
    history_path: Path,
    run_id: str,
    best_score: Optional[float],
    best_overrides: Optional[Dict[str, Any]],
    update_ledger: bool = True,
) -> Dict[str, Any]:
    """End-of-tune: per-run chart + ledger append + ledger figure."""
    ensure_figures_dir()
    rows = read_jsonl(history_path)
    trials = trials_from_hp_history(rows)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    stem = FIGURES_DIR / f'{agent}_hp_tune_{stamp}'
    arts = write_trial_charts(
        trials,
        stem=stem,
        title=f'{agent} HP tune — score vs trial',
        ylab='MEAN_SCORE / MEAN_WR',
    )
    # stable "latest" copies for specs/README
    latest_stem = FIGURES_DIR / f'{agent}_hp_tune_latest'
    latest = write_trial_charts(
        trials,
        stem=latest_stem,
        title=f'{agent} HP tune — score vs trial (latest)',
        ylab='MEAN_SCORE / MEAN_WR',
    )
    if update_ledger:
        append_ledger_entry(
            agent=agent,
            run_id=run_id,
            best_score=best_score,
            best_overrides=best_overrides,
            n_trials=len(trials),
            source=str(history_path),
            figure_stem=str(stem),
        )
        ledger_arts = plot_ledger(agent=agent)
        ledger_all = plot_ledger(agent=None)
    else:
        ledger_arts = {}
        ledger_all = {}
    return {
        'run_figures': arts,
        'latest_figures': latest,
        'ledger_agent': ledger_arts,
        'ledger_all': ledger_all,
        'ledger_path': str(LEDGER_PATH),
    }


def plot_mcts_sweep_run(
    *,
    agent: str,
    history_path: Path,
    run_id: str,
    recommended_score: Optional[float] = None,
    recommended_n: Optional[int] = None,
) -> Dict[str, Any]:
    ensure_figures_dir()
    rows = read_jsonl(history_path)
    trials = trials_from_mcts_history(rows)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    stem = FIGURES_DIR / f'{agent}_sims_sweep_{stamp}'
    arts = write_trial_charts(
        trials, stem=stem, title=f'{agent} N_sim sweep — WR vs step', ylab='WR',
    )
    latest = write_trial_charts(
        trials,
        stem=FIGURES_DIR / f'{agent}_sims_sweep_latest',
        title=f'{agent} N_sim sweep (latest)',
        ylab='WR',
    )
    best_score = recommended_score
    if best_score is None and trials:
        scores = [t.score for t in trials if t.score is not None]
        best_score = max(scores) if scores else None
    append_ledger_entry(
        agent=agent,
        run_id=run_id,
        best_score=best_score,
        best_overrides={'n_sims': recommended_n} if recommended_n is not None else None,
        n_trials=len(trials),
        source=str(history_path),
        figure_stem=str(stem),
    )
    return {
        'run_figures': arts,
        'latest_figures': latest,
        'ledger_agent': plot_ledger(agent=agent),
        'ledger_all': plot_ledger(agent=None),
    }


def plot_expectimax_tune_run(history_path: Path, *, run_id: str) -> Dict[str, Any]:
    ensure_figures_dir()
    rows = read_jsonl(history_path)
    trials = trials_from_expectimax_history(rows)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    stem = FIGURES_DIR / f'expectimax_tune_{stamp}'
    arts = write_trial_charts(
        trials, stem=stem, title='expectimax tune — pair WR vs step', ylab='WR',
    )
    latest = write_trial_charts(
        trials,
        stem=FIGURES_DIR / 'expectimax_tune_latest',
        title='expectimax tune (latest)',
        ylab='WR',
    )
    scores = [t.score for t in trials if t.score is not None and t.accepted]
    best = max(scores) if scores else None
    append_ledger_entry(
        agent='expectimax_bayesian',
        run_id=run_id,
        best_score=best,
        best_overrides=None,
        n_trials=len(trials),
        source=str(history_path),
        figure_stem=str(stem),
    )
    return {
        'run_figures': arts,
        'latest_figures': latest,
        'ledger_agent': plot_ledger(agent='expectimax_bayesian'),
        'ledger_all': plot_ledger(agent=None),
    }


def plot_train_reporte_suite(reporte_json: Path, *, agent: str) -> Dict[str, str]:
    """Bar-like SVG of offline suite WR from train ``reporte.json``."""
    if not reporte_json.is_file():
        return {}
    try:
        data = json.loads(reporte_json.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}
    suite = data.get('suite') or []
    if not suite:
        return {}
    ensure_figures_dir()
    # reuse polyline with discrete points as bars substitute
    xs = list(range(1, len(suite) + 1))
    scores = [float(s.get('win_rate', 0.0)) for s in suite]
    labels = [str(s.get('foe', f'f{i}')) for i, s in enumerate(suite, 1)]
    series = [('WR', scores, '#2ca02c')]
    stem = FIGURES_DIR / f'{agent}_train_suite_latest'
    # annotate title with foe names
    title = f'{agent} train suite — ' + ' | '.join(
        f'{lab}:{sc:.2f}' for lab, sc in zip(labels, scores)
    )
    svg = _svg_polyline_chart(title=title[:120], xs=xs, series=series, ylab='WR')
    svg_path = Path(str(stem) + '.svg')
    svg_path.write_text(svg, encoding='utf-8')
    arts = {'svg': str(svg_path)}
    png_path = Path(str(stem) + '.png')
    if _try_matplotlib_png(png_path, title=title[:80], xs=xs, series=series, ylab='WR'):
        arts['png'] = str(png_path)
    return arts
