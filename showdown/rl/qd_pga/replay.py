"""Shared value replay — capacity 200_000 (docs/qd_pga_map_elites_spec.md)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from showdown.rl.constants import OBS_DIM

REPLAY_CAPACITY = 200_000


class ValueReplay:
    """Stores (s, r, s', terminated) for 1-step TD(0) on shared V_phi.

    Obs v2 (≈6k dims): float16 y reserva perezosa (ver ddqn/replay.py).
    """

    def __init__(self, capacity: int = REPLAY_CAPACITY, obs_dim: int = OBS_DIM):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self._allocated = False
        self._n = 0
        self._i = 0

    _FIELDS = ('_s', '_r', '_s2', '_term')

    def _alloc(self) -> None:
        """Reserva perezosa y creciente (×2 hasta ``capacity``)."""
        if not self._allocated:
            self._size = min(self.capacity, 4096)
            self._s = np.zeros((self._size, self.obs_dim), dtype=np.float16)
            self._r = np.zeros((self._size,), dtype=np.float32)
            self._s2 = np.zeros((self._size, self.obs_dim), dtype=np.float16)
            self._term = np.zeros((self._size,), dtype=np.float32)
            self._allocated = True
            return
        if self._i >= self._size and self._size < self.capacity:
            new = min(self.capacity, 2 * self._size)
            for name in self._FIELDS:
                old = getattr(self, name)
                arr = np.zeros((new,) + old.shape[1:], dtype=old.dtype)
                arr[: self._size] = old
                setattr(self, name, arr)
            self._size = new

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
        self._alloc()
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
        return (self._s[idx].astype(np.float32), self._r[idx],
                self._s2[idx].astype(np.float32), self._term[idx])
