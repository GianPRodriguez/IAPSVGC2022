"""Uniform self-play pool: last K checkpoints ∪ best_vs_greedy_joint."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Set

import numpy as np


class SelfPlayPool:
    """Uniform pool: last K checkpoints union best_vs_greedy_joint (never evicted)."""

    def __init__(self, *, k: int = 10, root: Path | str = 'logs/ppo_checkpoints/pool'):
        self.k = int(k)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._recent: Deque[Path] = deque(maxlen=self.k)
        self.best_path: Optional[Path] = None
        self.best_wr: float = -1.0
        self._seq = 0

    def __len__(self) -> int:
        return len(self.paths())

    def paths(self) -> List[Path]:
        seen: Set[str] = set()
        out: List[Path] = []
        for p in list(self._recent):
            key = str(p.resolve()) if p.exists() else str(p)
            if key not in seen and p.is_file():
                seen.add(key)
                out.append(p)
        if self.best_path is not None and self.best_path.is_file():
            key = str(self.best_path.resolve())
            if key not in seen:
                out.append(self.best_path)
        return out

    def push(self, agent, *, tag: str = 'ckpt') -> Path:
        """Save agent weights into the recent-K queue."""
        self._seq += 1
        path = self.root / f'{tag}_{self._seq:06d}.pt'
        agent.save(path)
        self._recent.append(path)
        return path

    def maybe_update_best(self, agent, wr: float, *, min_wr: float = 0.0) -> bool:
        """Update immortal best_vs_greedy_joint if WR improves."""
        if float(wr) < float(min_wr):
            return False
        if float(wr) <= self.best_wr:
            return False
        path = self.root / 'best_vs_greedy_joint.pt'
        agent.save(path)
        self.best_path = path
        self.best_wr = float(wr)
        return True

    def sample(self, rng: np.random.Generator) -> Optional[Path]:
        """Uniform over deduplicated pool; None if empty."""
        ps = self.paths()
        if not ps:
            return None
        return ps[int(rng.integers(0, len(ps)))]
