"""Monotonic opponent-set expansion (Ω) for DDQN training.

Phase t unlocks a new foe; each episode samples o ~ π(Ω_t) with fixed weights
(Expectimax down-weighted). Replay is shared across phases (caller responsibility).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LIVE_OPPONENTS = (
    'random',
    'greedy_solo',
    'greedy_joint',
    'expectimax_bayesian',
)

# Default unlock order for --expand-omega / curriculum
DEFAULT_UNLOCK_ORDER: Tuple[str, ...] = (
    'random',
    'greedy_solo',
    'greedy_joint',
    'expectimax_bayesian',
)

# Sampling weights over current Ω (renormalized each phase)
DEFAULT_OPPONENT_WEIGHTS: Dict[str, float] = {
    'mock': 1.0,
    'random': 1.0,
    'greedy_solo': 1.0,
    'greedy_joint': 1.0,
    # ~10% when Ω = {random, solo, joint, expectimax} with others at 1.0
    'expectimax_bayesian': 0.35,
}

OPPONENT_CODE: Dict[str, int] = {
    'unknown': 0,
    'mock': 1,
    'random': 2,
    'greedy_solo': 3,
    'greedy_joint': 4,
    'expectimax_bayesian': 5,
}
CODE_TO_OPPONENT: Dict[int, str] = {v: k for k, v in OPPONENT_CODE.items()}


@dataclass(frozen=True)
class ExpandPhase:
    """One curriculum stage: unlock ``unlock`` into Ω and train ``steps`` env steps."""

    unlock: str
    steps: int
    omega: Tuple[str, ...]  # cumulative set after unlock


def opponent_code(name: str) -> int:
    key = str(name or 'unknown').strip().lower().replace('-', '_')
    return int(OPPONENT_CODE.get(key, 0))


def parse_weight_overrides(spec: Optional[str]) -> Dict[str, float]:
    """Parse ``random:1,greedy_joint:1.5,expectimax_bayesian:0.35``."""
    if not spec or not str(spec).strip():
        return {}
    out: Dict[str, float] = {}
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if ':' not in part:
            raise ValueError(f'peso inválido {part!r}; usa nombre:float')
        name, val_s = part.split(':', 1)
        key = name.strip().lower().replace('-', '_')
        out[key] = float(val_s)
    return out


def merge_weights(overrides: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
    w = dict(DEFAULT_OPPONENT_WEIGHTS)
    if overrides:
        for k, v in overrides.items():
            key = str(k).strip().lower().replace('-', '_')
            w[key] = float(v)
    return w


def build_expand_phases(
    unlock_order: Sequence[str],
    phase_steps: Sequence[int],
) -> List[ExpandPhase]:
    """Build monotonic Ω schedule from unlock list + per-phase step counts."""
    order = [
        str(x).strip().lower().replace('-', '_')
        for x in unlock_order
        if str(x).strip()
    ]
    if not order:
        raise ValueError('unlock_order vacío')
    steps_list = [int(x) for x in phase_steps]
    if len(steps_list) == 1 and len(order) > 1:
        steps_list = steps_list * len(order)
    if len(steps_list) != len(order):
        raise ValueError(
            f'phase_steps ({len(steps_list)}) debe coincidir con unlocks ({len(order)}) '
            f'o ser un único valor'
        )
    omega: List[str] = []
    out: List[ExpandPhase] = []
    for unlock, n in zip(order, steps_list):
        if unlock not in omega:
            omega.append(unlock)
        out.append(ExpandPhase(unlock=unlock, steps=int(n), omega=tuple(omega)))
    return out


def default_expand_curriculum(
    phase_steps: int = 500_000,
    *,
    include_expectimax: bool = True,
) -> List[ExpandPhase]:
    order = list(DEFAULT_UNLOCK_ORDER)
    if not include_expectimax:
        order = [o for o in order if o != 'expectimax_bayesian']
    n = int(phase_steps)
    return build_expand_phases(order, [n] * len(order))


def expand_phases_from_exclusive(schedule: Sequence) -> List[ExpandPhase]:
    """
    Convert exclusive TrainPhase list (opponent, steps) into monotonic Ω.

    Order of first appearance defines unlock order; Ω accumulates.
    """
    omega: List[str] = []
    out: List[ExpandPhase] = []
    for ph in schedule:
        unlock = str(getattr(ph, 'opponent', ph)).strip().lower().replace('-', '_')
        steps = int(getattr(ph, 'steps', 0) or 0)
        if unlock not in omega:
            omega.append(unlock)
        out.append(ExpandPhase(unlock=unlock, steps=steps, omega=tuple(omega)))
    return out


def sample_from_omega(
    omega: Sequence[str],
    rng: np.random.Generator,
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> str:
    """Sample one opponent from Ω with renormalized weights."""
    names = [str(o).strip().lower().replace('-', '_') for o in omega]
    if not names:
        raise ValueError('Ω vacío')
    wmap = merge_weights(weights)
    w = np.asarray([max(0.0, float(wmap.get(n, 1.0))) for n in names], dtype=np.float64)
    if not np.any(w > 0):
        w = np.ones(len(names), dtype=np.float64)
    w = w / w.sum()
    idx = int(rng.choice(len(names), p=w))
    return names[idx]


def omega_sampling_probs(
    omega: Sequence[str],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    names = [str(o).strip().lower().replace('-', '_') for o in omega]
    wmap = merge_weights(weights)
    w = np.asarray([max(0.0, float(wmap.get(n, 1.0))) for n in names], dtype=np.float64)
    if not np.any(w > 0):
        w = np.ones(len(names), dtype=np.float64)
    w = w / w.sum()
    return {n: float(p) for n, p in zip(names, w)}
