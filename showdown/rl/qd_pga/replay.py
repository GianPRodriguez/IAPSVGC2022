"""Shared value replay — capacity 200_000 (docs/qd_pga_map_elites_spec.md)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from showdown.rl.constants import OBS_DIM

REPLAY_CAPACITY = 200_000


class ValueReplay:
    """Stores (s, r, s', terminated) for 1-step TD(0) on shared V_phi."""

    def __init__(self, capacity: int = REPLAY_CAPACITY, obs_dim: int = OBS_DIM):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self._s = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._r = np.zeros((self.capacity,), dtype=np.float32)
        self._s2 = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._term = np.zeros((self.capacity,), dtype=np.float32)
        self._n = 0
        self._i = 0

    def __len__(self) -> int:
        return int(self._n)

    def add(
        self,
        *,
        s: np.ndarray,
        r: float,
        s_next: np.ndarray,
        terminated: bool,
        skip_buffer: bool = False,
        trunc_reason: Optional[str] = None,
    ) -> bool:
        if skip_buffer or trunc_reason == 'crash':
            return False
        i = self._i
        self._s[i] = np.asarray(s, dtype=np.float32).reshape(self.obs_dim)
        self._r[i] = float(r)
        self._s2[i] = np.asarray(s_next, dtype=np.float32).reshape(self.obs_dim)
        # turn_cap: not terminal for TD bootstrap
        self._term[i] = 1.0 if terminated else 0.0
        self._i = (i + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)
        return True

    def sample(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._n < batch_size:
            raise ValueError(f'replay too small: {self._n} < {batch_size}')
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self._n, size=int(batch_size))
        return self._s[idx], self._r[idx], self._s2[idx], self._term[idx]
