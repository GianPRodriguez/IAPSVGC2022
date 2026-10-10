"""Grid MAP-Elites archive (docs/qd_ppo_spec.md)."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

GRID_N1 = 10
GRID_N2 = 10


def bin_bd(bd1: float, bd2: float, n1: int = GRID_N1, n2: int = GRID_N2) -> Tuple[int, int]:
    i = min(int(n1) - 1, int(np.floor(float(bd1) * float(n1))))
    j = min(int(n2) - 1, int(np.floor(float(bd2) * float(n2))))
    return max(0, i), max(0, j)


@dataclass
class Elite:
    fitness: float
    bd: Tuple[float, float]
    state_dict: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)


class Archive:
    """N1 x N2 grid; at most one elite per cell."""

    def __init__(self, n1: int = GRID_N1, n2: int = GRID_N2):
        self.n1 = int(n1)
        self.n2 = int(n2)
        self.cells: Dict[Tuple[int, int], Elite] = {}

    @property
    def n_cells(self) -> int:
        return self.n1 * self.n2

    def filled_keys(self) -> List[Tuple[int, int]]:
        return sorted(self.cells.keys())

    def coverage(self) -> float:
        return float(len(self.cells)) / float(max(1, self.n_cells))

    def qd_score(self) -> float:
        return float(sum(max(0.0, e.fitness) for e in self.cells.values()))

    def best(self) -> Optional[Elite]:
        if not self.cells:
            return None
        return max(self.cells.values(), key=lambda e: e.fitness)

    def best_fitness(self) -> float:
        e = self.best()
        return float(e.fitness) if e is not None else float('-inf')

    def try_insert(
        self,
        *,
        bd1: float,
        bd2: float,
        fitness: float,
        state_dict: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Insert or strict-replace. Returns True if archive changed."""
        key = bin_bd(bd1, bd2, self.n1, self.n2)
        elite = Elite(
            fitness=float(fitness),
            bd=(float(bd1), float(bd2)),
            state_dict=copy.deepcopy(state_dict),
            meta=dict(meta or {}),
        )
        cur = self.cells.get(key)
        if cur is None or elite.fitness > cur.fitness:
            self.cells[key] = elite
            return True
        return False

    def sample_parent(
        self,
        rng: np.random.Generator,
        fallback_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[int, int]]]:
        """Uniform over filled cells; if empty, return fallback_state."""
        keys = self.filled_keys()
        if not keys:
            return (copy.deepcopy(fallback_state) if fallback_state else None), None
        key = keys[int(rng.integers(0, len(keys)))]
        return copy.deepcopy(self.cells[key].state_dict), key

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Torch tensors in state_dict — save via torch if available
        try:
            import torch
            payload = {
                'n1': self.n1,
                'n2': self.n2,
                'cells': {
                    f'{i},{j}': {
                        'fitness': e.fitness,
                        'bd': list(e.bd),
                        'state_dict': e.state_dict,
                        'meta': e.meta,
                    }
                    for (i, j), e in self.cells.items()
                },
            }
            torch.save(payload, path)
        except ImportError:
            meta_only = {
                'n1': self.n1,
                'n2': self.n2,
                'cells': {
                    f'{i},{j}': {'fitness': e.fitness, 'bd': list(e.bd), 'meta': e.meta}
                    for (i, j), e in self.cells.items()
                },
            }
            path.write_text(json.dumps(meta_only, indent=2), encoding='utf-8')
        return path

    def load(self, path: Path | str) -> None:
        path = Path(path)
        import torch
        try:
            payload = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location='cpu')
        self.n1 = int(payload.get('n1', GRID_N1))
        self.n2 = int(payload.get('n2', GRID_N2))
        self.cells = {}
        for k, v in (payload.get('cells') or {}).items():
            i_s, j_s = str(k).split(',')
            self.cells[(int(i_s), int(j_s))] = Elite(
                fitness=float(v['fitness']),
                bd=(float(v['bd'][0]), float(v['bd'][1])),
                state_dict=v['state_dict'],
                meta=dict(v.get('meta') or {}),
            )
