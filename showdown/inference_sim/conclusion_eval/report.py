"""Reporte texto/JSON de la eval de Conclusión."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from showdown.inference_sim.conclusion_eval.constants import CROSS_EPS
from showdown.inference_sim.conclusion_eval.pipelines import PipelineSpec
from showdown.inference_sim.conclusion_eval.stats import (
    CrossResult,
    RankRow,
    wilson_ci,
)


def _pct(x: Optional[float]) -> str:
    if x is None:
        return '-'
    return f'{100.0 * x:.1f}%'


def build_report_text(
    *,
    seed: int,
    format_id: str,
    snapshot_n: int,
    n_pairs: int,
    pipelines: Sequence[PipelineSpec],
    crosses: Sequence[CrossResult],
    ranks: Sequence[RankRow],
    elapsed_s: float,
    smoke: bool = False,
) -> str:
    lines = [
        '=== Conclusion Eval Report ===',
        f'time_utc: {datetime.now(timezone.utc).isoformat()}',
        f'protocol: docs/conclusion_eval_protocol.md',
        f'format: {format_id}',
        f'seed: {seed}',
        f'snapshot_n: {snapshot_n}',
        f'n_pairs: {n_pairs} (BO1/cross={2 * n_pairs}: AB+BA swap)',
        f'eps: {CROSS_EPS}',
        f'pipelines: {len(pipelines)}',
        f'smoke: {"YES" if smoke else "NO"}',
        f'elapsed_s: {elapsed_s:.1f}',
        '',
        '--- ranking (points → WR → H2H → vs heuristics → id) ---',
    ]
    for r in ranks:
        dq = ' DQ' if r.dq else ''
        ci = ''
        if r.wilson_lo is not None and r.wilson_hi is not None:
            ci = f' CI95=[{_pct(r.wilson_lo)}, {_pct(r.wilson_hi)}]'
        lines.append(
            f'{r.rank:2d}. {r.pipeline_id:22s} pts={r.points:4.1f} '
            f'WR={_pct(r.wr_global)}{ci} '
            f'vsH={_pct(r.wr_vs_heuristics)} '
            f'valid={r.n_valid} fail={r.n_fails}{dq}'
        )

    lines.append('')
    lines.append('--- crosses ---')
    for cr in crosses:
        pi, pj = cr.points()
        wr, lo, hi = wilson_ci(cr.wins_i, cr.valid)
        wr_ab = (cr.wins_i_ab / cr.valid_ab) if cr.valid_ab else None
        wr_ba = (cr.wins_i_ba / cr.valid_ba) if cr.valid_ba else None
        lines.append(
            f'{cr.id_i} vs {cr.id_j}: '
            f'{cr.wins_i}-{cr.wins_j} (n={cr.valid} fail={cr.fails}) '
            f'WR={_pct(wr)} CI=[{_pct(lo)}, {_pct(hi)}] '
            f'pts={pi}/{pj} '
            f'side_AB={_pct(wr_ab)} side_BA={_pct(wr_ba)}'
        )

    lines.append('')
    lines.append('Player seating: i=p1, j=p2 (i<j canonical); swap teams only.')
    return '\n'.join(lines) + '\n'


def results_payload(
    *,
    seed: int,
    format_id: str,
    snapshot_n: int,
    n_pairs: int,
    pipelines: Sequence[PipelineSpec],
    crosses: Sequence[CrossResult],
    ranks: Sequence[RankRow],
    elapsed_s: float,
    failures: Sequence[Dict[str, Any]],
    smoke: bool = False,
) -> Dict[str, Any]:
    return {
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'seed': seed,
        'format': format_id,
        'snapshot_n': snapshot_n,
        'n_pairs': n_pairs,
        'eps': CROSS_EPS,
        'smoke': smoke,
        'elapsed_s': elapsed_s,
        'pipelines': [
            {
                'id': p.id,
                'combat': p.combat,
                'preview': p.preview,
                'group': p.group,
                'checkpoint': p.checkpoint,
            }
            for p in pipelines
        ],
        'ranking': [
            {
                'rank': r.rank,
                'id': r.pipeline_id,
                'points': r.points,
                'wr_global': r.wr_global,
                'wilson_lo': r.wilson_lo,
                'wilson_hi': r.wilson_hi,
                'wr_vs_heuristics': r.wr_vs_heuristics,
                'n_valid': r.n_valid,
                'n_fails': r.n_fails,
                'dq': r.dq,
            }
            for r in ranks
        ],
        'crosses': [
            {
                'i': cr.i,
                'j': cr.j,
                'id_i': cr.id_i,
                'id_j': cr.id_j,
                'wins_i': cr.wins_i,
                'wins_j': cr.wins_j,
                'valid': cr.valid,
                'fails': cr.fails,
                'wins_i_ab': cr.wins_i_ab,
                'valid_ab': cr.valid_ab,
                'wins_i_ba': cr.wins_i_ba,
                'valid_ba': cr.valid_ba,
                'points': list(cr.points()),
            }
            for cr in crosses
        ],
        'failures': list(failures),
    }
