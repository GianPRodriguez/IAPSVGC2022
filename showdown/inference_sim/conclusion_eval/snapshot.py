"""Snapshot 250 / split A·B / 100 pares fijos (sin reemplazo)."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from showdown.inference_sim.conclusion_eval.constants import (
    DEFAULT_N_PAIRS,
    DEFAULT_SNAPSHOT_N,
    SNAPSHOT_PATH,
)
from showdown.inference_sim.team_factory import TeamMeta, active_teams


@dataclass(frozen=True)
class TeamRef:
    name: str
    source_url: str = ''

    def key(self) -> str:
        return self.name


@dataclass
class EvalSnapshot:
    seed: int
    snapshot_n: int
    n_pairs: int
    created_utc: str
    pool_a: List[TeamRef]
    pool_b: List[TeamRef]
    pairs: List[Tuple[TeamRef, TeamRef]]  # always len == n_pairs

    def to_dict(self) -> Dict[str, Any]:
        return {
            'seed': int(self.seed),
            'snapshot_n': int(self.snapshot_n),
            'n_pairs': int(self.n_pairs),
            'created_utc': self.created_utc,
            'pool_a': [{'name': t.name, 'source_url': t.source_url} for t in self.pool_a],
            'pool_b': [{'name': t.name, 'source_url': t.source_url} for t in self.pool_b],
            'pairs': [
                {
                    'a': {'name': a.name, 'source_url': a.source_url},
                    'b': {'name': b.name, 'source_url': b.source_url},
                }
                for a, b in self.pairs
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EvalSnapshot':
        def _ref(row: Dict[str, Any]) -> TeamRef:
            return TeamRef(
                name=str(row['name']),
                source_url=str(row.get('source_url') or ''),
            )

        pairs = [
            (_ref(p['a']), _ref(p['b']))
            for p in (d.get('pairs') or [])
        ]
        return cls(
            seed=int(d['seed']),
            snapshot_n=int(d['snapshot_n']),
            n_pairs=int(d['n_pairs']),
            created_utc=str(d.get('created_utc') or ''),
            pool_a=[_ref(x) for x in (d.get('pool_a') or [])],
            pool_b=[_ref(x) for x in (d.get('pool_b') or [])],
            pairs=pairs,
        )


def _team_key(t: TeamMeta) -> str:
    """Clave de dedup: especies ordenadas (+ items) normalizadas."""
    parts = []
    for m in t.mons:
        sp = (m.species or '').strip().lower()
        it = (m.item or '').strip().lower()
        parts.append(f'{sp}|{it}')
    parts.sort()
    return '||'.join(parts)


def dedupe_teams(teams: Sequence[TeamMeta]) -> List[TeamMeta]:
    seen = set()
    out: List[TeamMeta] = []
    for t in teams:
        k = _team_key(t)
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def build_snapshot(
    *,
    seed: int,
    snapshot_n: int = DEFAULT_SNAPSHOT_N,
    n_pairs: int = DEFAULT_N_PAIRS,
    teams: Optional[Sequence[TeamMeta]] = None,
) -> EvalSnapshot:
    """Corte final: N equipos, A/B disjuntos, n_pairs sin reemplazo (misma seed)."""
    pool = dedupe_teams(list(teams) if teams is not None else active_teams())
    if len(pool) < snapshot_n:
        raise RuntimeError(
            f'Snapshot necesita {snapshot_n} equipos únicos; hay {len(pool)}. '
            'Amplía el repo / export antes del corte final.'
        )
    if snapshot_n % 2 != 0:
        raise ValueError('snapshot_n debe ser par')
    if n_pairs < 1:
        raise ValueError('n_pairs debe ser >= 1')

    rng = random.Random(int(seed))
    # Orden estable por nombre antes del sample para reproducibilidad ante
    # cambios de orden en active_teams(), luego sample+shuffle con seed.
    pool_sorted = sorted(pool, key=lambda t: (t.name, t.source_url or ''))
    picked = rng.sample(pool_sorted, int(snapshot_n))
    rng.shuffle(picked)
    half = snapshot_n // 2
    pool_a_m = picked[:half]
    pool_b_m = picked[half:]

    max_pairs = half * half
    if n_pairs > max_pairs:
        raise ValueError(
            f'n_pairs={n_pairs} > |A|×|B|={max_pairs}'
        )

    # Producto cartesiano en orden determinista, luego shuffle + take n_pairs.
    product = [(a, b) for a in pool_a_m for b in pool_b_m]
    rng.shuffle(product)
    chosen = product[: int(n_pairs)]

    def _ref(t: TeamMeta) -> TeamRef:
        return TeamRef(name=t.name, source_url=t.source_url or '')

    return EvalSnapshot(
        seed=int(seed),
        snapshot_n=int(snapshot_n),
        n_pairs=int(n_pairs),
        created_utc=datetime.now(timezone.utc).isoformat(),
        pool_a=[_ref(t) for t in pool_a_m],
        pool_b=[_ref(t) for t in pool_b_m],
        pairs=[(_ref(a), _ref(b)) for a, b in chosen],
    )


def save_snapshot(snap: EvalSnapshot, path: Optional[Path] = None) -> Path:
    p = Path(path) if path is not None else SNAPSHOT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(snap.to_dict(), indent=2), encoding='utf-8')
    return p


def load_snapshot(path: Optional[Path] = None) -> EvalSnapshot:
    p = Path(path) if path is not None else SNAPSHOT_PATH
    return EvalSnapshot.from_dict(json.loads(p.read_text(encoding='utf-8')))


def resolve_teams_by_name(
    names: Sequence[str],
    *,
    pool: Optional[Sequence[TeamMeta]] = None,
) -> Dict[str, TeamMeta]:
    """Mapa name → TeamMeta desde el pool activo (nombres del snapshot)."""
    teams = list(pool) if pool is not None else active_teams()
    by_name: Dict[str, TeamMeta] = {}
    for t in teams:
        by_name.setdefault(t.name, t)
    missing = [n for n in names if n not in by_name]
    if missing:
        raise RuntimeError(
            'Equipos del snapshot no encontrados en active_teams(): '
            + ', '.join(missing[:5])
            + ('...' if len(missing) > 5 else '')
        )
    return by_name


def materialize_pairs(
    snap: EvalSnapshot,
    *,
    pool: Optional[Sequence[TeamMeta]] = None,
) -> List[Tuple[TeamMeta, TeamMeta]]:
    names = [t.name for t in snap.pool_a] + [t.name for t in snap.pool_b]
    by_name = resolve_teams_by_name(names, pool=pool)
    return [(by_name[a.name], by_name[b.name]) for a, b in snap.pairs]
