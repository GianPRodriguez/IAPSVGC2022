"""WR, Wilson CI, puntos con ε, ranking con desempates."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from showdown.inference_sim.conclusion_eval.constants import (
    CROSS_EPS,
    DQ_FAIL_RATE,
    HEURISTIC_IDS,
    WILSON_Z,
)


def wilson_ci(
    wins: int,
    n: int,
    *,
    z: float = WILSON_Z,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """WR y IC Wilson (aprox. 95%)."""
    if n <= 0:
        return None, None, None
    w = max(0, int(wins))
    n = int(n)
    phat = w / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = phat + z2 / (2.0 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n)
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return phat, max(0.0, lo), min(1.0, hi)


def cross_points(
    wins_i: int,
    n: int,
    *,
    eps: float = CROSS_EPS,
) -> Tuple[float, float]:
    """Puntos (i, j) en un cruce. wins_i = victorias del agente i (p1).

    Empate si |WR − 0.5| ≤ ε. Se compara en margen de victorias
    (|wins − n/2| ≤ ε·n) para evitar falsos negativos por float
    (p.ej. 105/200).
    """
    if n <= 0:
        return 0.0, 0.0
    w = int(wins_i)
    n = int(n)
    # |w/n - 0.5| <= eps  <=>  |2w - n| <= 2*eps*n
    if abs(2 * w - n) <= (2.0 * float(eps) * n) + 1e-9:
        return 0.5, 0.5
    if w * 2 > n:
        return 1.0, 0.0
    return 0.0, 1.0


@dataclass
class CrossResult:
    i: int
    j: int
    id_i: str
    id_j: str
    wins_i: int = 0
    wins_j: int = 0
    valid: int = 0
    fails: int = 0
    wins_i_ab: int = 0  # p1 con A, p2 con B
    valid_ab: int = 0
    wins_i_ba: int = 0  # p1 con B, p2 con A
    valid_ba: int = 0

    @property
    def wr_i(self) -> Optional[float]:
        if self.valid <= 0:
            return None
        return self.wins_i / float(self.valid)

    def points(self, *, eps: float = CROSS_EPS) -> Tuple[float, float]:
        return cross_points(self.wins_i, self.valid, eps=eps)


@dataclass
class RankRow:
    rank: int
    pipeline_id: str
    points: float
    wr_global: Optional[float]
    wr_vs_heuristics: Optional[float]
    dq: bool = False
    n_valid: int = 0
    n_fails: int = 0
    wilson_lo: Optional[float] = None
    wilson_hi: Optional[float] = None


@dataclass
class StandingAgg:
    pipeline_id: str
    points: float = 0.0
    wins: int = 0
    valid: int = 0
    fails: int = 0
    wins_vs_h: int = 0
    valid_vs_h: int = 0
    h2h_wr: Dict[str, Optional[float]] = field(default_factory=dict)
    dq: bool = False

    @property
    def wr_global(self) -> Optional[float]:
        if self.valid <= 0:
            return None
        return self.wins / float(self.valid)

    @property
    def wr_vs_heuristics(self) -> Optional[float]:
        if self.valid_vs_h <= 0:
            return None
        return self.wins_vs_h / float(self.valid_vs_h)

    @property
    def fail_rate(self) -> float:
        tot = self.valid + self.fails
        if tot <= 0:
            return 0.0
        return self.fails / float(tot)


def aggregate_standings(
    pipelines: Sequence[str],
    crosses: Sequence[CrossResult],
    *,
    eps: float = CROSS_EPS,
    heuristic_ids: Sequence[str] = HEURISTIC_IDS,
    dq_fail_rate: float = DQ_FAIL_RATE,
) -> Dict[str, StandingAgg]:
    ids = [str(x) for x in pipelines]
    st: Dict[str, StandingAgg] = {pid: StandingAgg(pipeline_id=pid) for pid in ids}
    hset = {str(x) for x in heuristic_ids}

    for cr in crosses:
        pi, pj = cr.points(eps=eps)
        a, b = st[cr.id_i], st[cr.id_j]
        a.points += pi
        b.points += pj
        a.wins += cr.wins_i
        b.wins += cr.wins_j
        a.valid += cr.valid
        b.valid += cr.valid
        a.fails += cr.fails
        b.fails += cr.fails
        wr_i = cr.wr_i
        a.h2h_wr[cr.id_j] = wr_i
        b.h2h_wr[cr.id_i] = (1.0 - wr_i) if wr_i is not None else None

        if cr.id_j in hset:
            a.wins_vs_h += cr.wins_i
            a.valid_vs_h += cr.valid
        if cr.id_i in hset:
            b.wins_vs_h += cr.wins_j
            b.valid_vs_h += cr.valid

    for s in st.values():
        if s.fail_rate > float(dq_fail_rate):
            s.dq = True
    return st


def _sort_key(a: StandingAgg, b: StandingAgg) -> int:
    """Negativo si a va antes que b (mejor)."""
    if a.dq != b.dq:
        return -1 if (not a.dq and b.dq) else 1
    if a.points != b.points:
        return -1 if a.points > b.points else 1
    wa, wb = a.wr_global, b.wr_global
    if wa is None and wb is not None:
        return 1
    if wa is not None and wb is None:
        return -1
    if wa is not None and wb is not None and wa != wb:
        return -1 if wa > wb else 1
    h2h = a.h2h_wr.get(b.pipeline_id)
    if h2h is not None and abs(h2h - 0.5) > 1e-12:
        return -1 if h2h > 0.5 else 1
    ha, hb = a.wr_vs_heuristics, b.wr_vs_heuristics
    if ha is None and hb is not None:
        return 1
    if ha is not None and hb is None:
        return -1
    if ha is not None and hb is not None and ha != hb:
        return -1 if ha > hb else 1
    if a.pipeline_id < b.pipeline_id:
        return -1
    if a.pipeline_id > b.pipeline_id:
        return 1
    return 0


def rank_pipelines(
    pipelines: Sequence[str],
    crosses: Sequence[CrossResult],
    *,
    eps: float = CROSS_EPS,
    heuristic_ids: Sequence[str] = HEURISTIC_IDS,
    dq_fail_rate: float = DQ_FAIL_RATE,
) -> List[RankRow]:
    st = aggregate_standings(
        pipelines,
        crosses,
        eps=eps,
        heuristic_ids=heuristic_ids,
        dq_fail_rate=dq_fail_rate,
    )
    ordered = list(st.values())
    # Ordenación estable con comparador
    from functools import cmp_to_key

    ordered.sort(key=cmp_to_key(_sort_key))
    rows: List[RankRow] = []
    for r, s in enumerate(ordered, start=1):
        wr, lo, hi = wilson_ci(s.wins, s.valid)
        rows.append(
            RankRow(
                rank=r,
                pipeline_id=s.pipeline_id,
                points=s.points,
                wr_global=wr,
                wr_vs_heuristics=s.wr_vs_heuristics,
                dq=s.dq,
                n_valid=s.valid,
                n_fails=s.fails,
                wilson_lo=lo,
                wilson_hi=hi,
            )
        )
    return rows


def deterministic_cross_order(
    n_pipelines: int,
    *,
    seed: int,
) -> List[Tuple[int, int]]:
    """55 pares (i,j) i<j, orden barajado con seed fija."""
    import random

    pairs = [(i, j) for i in range(n_pipelines) for j in range(i + 1, n_pipelines)]
    rng = random.Random(int(seed) ^ 0xC0FFEE)
    rng.shuffle(pairs)
    return pairs
