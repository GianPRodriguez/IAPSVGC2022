"""Lógica pura del barrido N_sim (meseta WR vs tiempo).

Usado por ``run_mcts_sims_sweep`` y tests sin Showdown.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_OPEN_LOOP_SIMS = (50, 100, 200, 400)
DEFAULT_NASH_SIMS = (25, 50, 100, 200)

ENV_SIMS_KEY = {
    'mcts_puct': 'MCTS_SIMS',
    'mcts_puct_nash': 'MCTS_NASH_SIMS',
}


@dataclass(frozen=True)
class SweepPoint:
    """Un punto del barrido (un valor de N_sim)."""

    n_sims: int
    wr_raw: float  # %
    pair_wr: float  # %
    elapsed_s: float
    wins: int = 0
    losses: int = 0
    pair_invalid: int = 0
    need_fix: str = 'NO'
    sync_ok: str = 'YES'
    inf_ok: str = 'YES'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlateauCriteria:
    """Criterio de meseta (parametrizable por CLI)."""

    delta_wr_pp: float = 1.5
    """Mejora mínima de WR_raw (puntos porcentuales) para justificar el siguiente N."""

    time_ratio: float = 1.8
    """Si el siguiente punto tarda ≥ este factor y la mejora de WR es < delta, meseta."""

    prefer_metric: str = 'wr_raw'
    """'wr_raw' o 'pair_wr' — métrica principal al comparar puntos."""


@dataclass(frozen=True)
class PlateauDecision:
    recommended_n_sims: int
    reason: str
    points: Tuple[SweepPoint, ...]
    plateau_index: int  # índice del punto recomendado en `points`


def _metric(p: SweepPoint, prefer: str) -> float:
    if prefer == 'pair_wr':
        return float(p.pair_wr)
    return float(p.wr_raw)


def suggest_plateau(
    points: Sequence[SweepPoint],
    criteria: PlateauCriteria,
) -> PlateauDecision:
    """Elige el menor N_sim en la meseta WR/tiempo.

    Recorre puntos ordenados por ``n_sims`` ascendente. Si el salto al siguiente
    mejora la métrica en menos de ``delta_wr_pp`` **y** el tiempo crece al menos
    ``time_ratio``, se declara meseta en el punto actual. Si ningún salto cumple
    eso, se recomienda el de mejor métrica (desempate: menor N_sim, luego menor
    tiempo).
    """
    if not points:
        raise ValueError('suggest_plateau requiere al menos un punto')

    ordered = tuple(sorted(points, key=lambda p: int(p.n_sims)))
    prefer = str(criteria.prefer_metric or 'wr_raw')
    delta = float(criteria.delta_wr_pp)
    ratio_need = float(criteria.time_ratio)

    for i in range(len(ordered) - 1):
        cur, nxt = ordered[i], ordered[i + 1]
        m_cur = _metric(cur, prefer)
        m_nxt = _metric(nxt, prefer)
        gain = m_nxt - m_cur
        t_cur = max(1e-9, float(cur.elapsed_s))
        t_nxt = max(0.0, float(nxt.elapsed_s))
        tr = t_nxt / t_cur if t_cur > 0 else float('inf')
        if gain < delta and tr >= ratio_need:
            return PlateauDecision(
                recommended_n_sims=int(cur.n_sims),
                reason=(
                    f'meseta: N={cur.n_sims}->{nxt.n_sims} '
                    f'delta_{prefer}={gain:+.2f}pp < {delta}pp y '
                    f'tiempo_x{tr:.2f} >= {ratio_need}'
                ),
                points=ordered,
                plateau_index=i,
            )

    # Sin meseta clara: mejor métrica; empate → menor N; luego menor tiempo.
    best_i = 0
    best = ordered[0]
    for i, p in enumerate(ordered):
        mp = _metric(p, prefer)
        mb = _metric(best, prefer)
        if mp > mb + 1e-9:
            best, best_i = p, i
        elif abs(mp - mb) <= 1e-9:
            if int(p.n_sims) < int(best.n_sims) or (
                int(p.n_sims) == int(best.n_sims)
                and float(p.elapsed_s) < float(best.elapsed_s)
            ):
                best, best_i = p, i

    return PlateauDecision(
        recommended_n_sims=int(best.n_sims),
        reason=(
            f'sin meseta por criterio (delta={delta}pp, time_ratio={ratio_need}); '
            f'mejor {prefer}={_metric(best, prefer):.1f}% en N={best.n_sims}'
        ),
        points=ordered,
        plateau_index=best_i,
    )


def default_sims_grid(agent: str) -> Tuple[int, ...]:
    key = str(agent).strip().lower().replace('-', '_')
    if key == 'mcts_puct_nash':
        return DEFAULT_NASH_SIMS
    if key == 'mcts_puct':
        return DEFAULT_OPEN_LOOP_SIMS
    raise ValueError(f'agente MCTS desconocido: {agent!r}')


def env_key_for_agent(agent: str) -> str:
    key = str(agent).strip().lower().replace('-', '_')
    if key not in ENV_SIMS_KEY:
        raise ValueError(f'agente MCTS desconocido: {agent!r}')
    return ENV_SIMS_KEY[key]


def format_sweep_table(
    points: Sequence[SweepPoint],
    decision: PlateauDecision,
    *,
    prefer_metric: str,
) -> str:
    lines = [
        f'{"N_sim":>8}  {"WR_raw%":>8}  {"pair_WR%":>9}  '
        f'{"elapsed_s":>10}  {"W-L":>9}  note',
        '-' * 64,
    ]
    for p in sorted(points, key=lambda x: x.n_sims):
        mark = (
            ' <-- recommend'
            if int(p.n_sims) == int(decision.recommended_n_sims)
            else ''
        )
        lines.append(
            f'{p.n_sims:8d}  {p.wr_raw:8.1f}  {p.pair_wr:9.1f}  '
            f'{p.elapsed_s:10.1f}  {p.wins:3d}-{p.losses:<3d}  '
            f'{p.need_fix}/{p.sync_ok}/{p.inf_ok}{mark}'
        )
    lines.append('')
    lines.append(f'metric={prefer_metric}')
    lines.append(f'recommended_n_sims={decision.recommended_n_sims}')
    lines.append(f'reason: {decision.reason}')
    return '\n'.join(lines) + '\n'


# Defaults congelados del contrato (no se barren; se registran en el tracking).
FROZEN_SEARCH_HYPERPARAMS = {
    'mcts_puct': {
        'c_puct': 1.5,
        'prior': 'uniform_legal',
        'opp_model': 'uniform_legal',
        'rollout': 'greedy_joint_snapshot',
        'max_turns': 40,
        'discount': 1.0,
        'final_action': 'argmax_N_tie_Q',
    },
    'mcts_puct_nash': {
        'c_puct': 1.5,
        'top_k': 16,
        'fp_inner': 200,
        'fp_root': 500,
        'fp_window': 50,
        'prior': 'uniform_cell',
        'rollout': 'greedy_joint_snapshot',
        'max_turns': 40,
        'discount': 1.0,
        'final_action': 'argmax_sigma',
    },
}


def frozen_hyperparams(agent: str) -> Dict[str, Any]:
    key = str(agent).strip().lower().replace('-', '_')
    return dict(FROZEN_SEARCH_HYPERPARAMS.get(key, {}))


def step_transition(
    prev: Optional[SweepPoint],
    cur: SweepPoint,
    *,
    prefer_metric: str,
) -> Dict[str, Any]:
    """Delta de hiperparámetro / métricas entre dos puntos consecutivos del barrido."""
    row: Dict[str, Any] = {
        'n_sims': int(cur.n_sims),
        'n_sims_prev': int(prev.n_sims) if prev is not None else None,
        'n_sims_delta': (
            int(cur.n_sims) - int(prev.n_sims) if prev is not None else None
        ),
        'wr_raw': float(cur.wr_raw),
        'pair_wr': float(cur.pair_wr),
        'elapsed_s': float(cur.elapsed_s),
        'metric': str(prefer_metric),
        'metric_value': _metric(cur, prefer_metric),
    }
    if prev is None:
        row.update({
            'delta_wr_raw_pp': None,
            'delta_pair_wr_pp': None,
            'delta_metric_pp': None,
            'time_ratio': None,
            'note': 'baseline_first_point',
        })
        return row
    t_prev = max(1e-9, float(prev.elapsed_s))
    row.update({
        'delta_wr_raw_pp': float(cur.wr_raw) - float(prev.wr_raw),
        'delta_pair_wr_pp': float(cur.pair_wr) - float(prev.pair_wr),
        'delta_metric_pp': _metric(cur, prefer_metric) - _metric(prev, prefer_metric),
        'time_ratio': float(cur.elapsed_s) / t_prev,
        'note': 'step',
    })
    return row


def build_tracking_log(
    points: Sequence[SweepPoint],
    decision: PlateauDecision,
    *,
    agent: str,
    criteria: PlateauCriteria,
    player_b: str,
    seed: int,
    teams: int,
    run_id: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Documento de seguimiento: cómo cambió N_sim hasta la meseta."""
    ordered = tuple(sorted(points, key=lambda p: int(p.n_sims)))
    prefer = str(criteria.prefer_metric or 'wr_raw')
    steps: List[Dict[str, Any]] = []
    prev: Optional[SweepPoint] = None
    for p in ordered:
        tr = step_transition(prev, p, prefer_metric=prefer)
        tr['is_recommended'] = int(p.n_sims) == int(decision.recommended_n_sims)
        steps.append(tr)
        prev = p

    return {
        'run_id': run_id,
        'agent': agent,
        'dry_run': bool(dry_run),
        'player_b': player_b,
        'seed': int(seed),
        'teams': int(teams),
        'env_key': env_key_for_agent(agent),
        'varying_hyperparam': 'n_sims',
        'frozen_search_hyperparams': frozen_hyperparams(agent),
        'criteria': asdict(criteria),
        'steps': steps,
        'recommended_n_sims': int(decision.recommended_n_sims),
        'reason': decision.reason,
        'plateau_index': int(decision.plateau_index),
    }


def format_tracking_text(tracking: Dict[str, Any]) -> str:
    """Log humano: evolución N_sim / WR / tiempo hasta la recomendación."""
    lines = [
        f"=== MCTS search-hyperparam tracking ({tracking.get('agent')}) ===",
        f"run_id: {tracking.get('run_id')}",
        f"dry_run: {tracking.get('dry_run')}",
        f"rival: {tracking.get('player_b')}",
        f"seed: {tracking.get('seed')}",
        f"teams: {tracking.get('teams')}",
        f"env_key: {tracking.get('env_key')}",
        f"varying: {tracking.get('varying_hyperparam')} "
        f"(others frozen: {tracking.get('frozen_search_hyperparams')})",
        f"criteria: {tracking.get('criteria')}",
        '',
        '--- step log (N_sim changes) ---',
    ]
    for i, st in enumerate(tracking.get('steps') or [], start=1):
        prev_n = st.get('n_sims_prev')
        arrow = (
            f"N_sim {prev_n} -> {st['n_sims']} (delta={st.get('n_sims_delta')})"
            if prev_n is not None
            else f"N_sim = {st['n_sims']} (first)"
        )
        rec = ' [RECOMMENDED]' if st.get('is_recommended') else ''
        d_m = st.get('delta_metric_pp')
        tr = st.get('time_ratio')
        d_m_s = f'{d_m:+.2f}pp' if d_m is not None else '-'
        tr_s = f'{tr:.2f}x' if tr is not None else '-'
        lines.append(
            f"{i}. {arrow}{rec}\n"
            f"   WR_raw={st['wr_raw']:.1f}%  pair_WR={st['pair_wr']:.1f}%  "
            f"t={st['elapsed_s']:.1f}s\n"
            f"   delta_metric={d_m_s}  time_ratio={tr_s}  "
            f"metric={st.get('metric')}"
        )
    lines.append('')
    lines.append(f"recommended_n_sims: {tracking.get('recommended_n_sims')}")
    lines.append(f"reason: {tracking.get('reason')}")
    return '\n'.join(lines) + '\n'
