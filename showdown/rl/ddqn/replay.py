"""Replay buffer — capacity 200_000; drops skip_buffer / truncated.

Tuple (spec): e_t = (s_t, a_t, r_t, s', terminated, m_next[, opponent_code]).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from showdown.rl.constants import JOINT_ACTION_N, OBS_DIM
from showdown.rl.ddqn.curriculum import CODE_TO_OPPONENT


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s_next: np.ndarray
    terminated: bool
    m_next: np.ndarray
    opponent_code: int = 0


class ReplayBuffer:
    """Memoria circular. Con obs v2 (≈6k dims) se guarda en float16 y las
    máscaras en uint8 (≈4,9 GB para 200k transiciones en vez de ≈9,7 GB) y se
    reserva al primer ``add`` (los jugadores de inferencia no la usan)."""

    def __init__(self, capacity: int = 200_000, obs_dim: int = OBS_DIM):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self._allocated = False
        self._n = 0
        self._i = 0

    _FIELDS = ('_s', '_a', '_r', '_s2', '_term', '_m2', '_opp')

    def _alloc(self) -> None:
        """Reserva perezosa y creciente (×2 hasta ``capacity``): una corrida corta
        no reserva los ≈5 GB de 200k transiciones con obs v2."""
        if not self._allocated:
            self._size = min(self.capacity, 4096)
            self._s = np.zeros((self._size, self.obs_dim), dtype=np.float16)
            self._a = np.zeros((self._size,), dtype=np.int64)
            self._r = np.zeros((self._size,), dtype=np.float32)
            self._s2 = np.zeros((self._size, self.obs_dim), dtype=np.float16)
            self._term = np.zeros((self._size,), dtype=np.float32)
            self._m2 = np.zeros((self._size, JOINT_ACTION_N), dtype=np.uint8)
            self._opp = np.zeros((self._size,), dtype=np.int16)
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
        return self._n

    def add(
        self,
        *,
        s: np.ndarray,
        a: int,
        r: float,
        s_next: np.ndarray,
        terminated: bool,
        m_next: np.ndarray,
        skip_buffer: bool = False,
        truncated: bool = False,
        opponent_code: int = 0,
    ) -> bool:
        """Insert transition. Returns False if dropped (L2/L3)."""
        if skip_buffer or truncated:
            return False
        self._alloc()
        i = self._i
        self._s[i] = np.asarray(s, dtype=np.float32).reshape(self.obs_dim)
        self._a[i] = int(a)
        self._r[i] = float(r)
        self._s2[i] = np.asarray(s_next, dtype=np.float32).reshape(self.obs_dim)
        self._term[i] = 1.0 if terminated else 0.0
        self._m2[i] = (np.asarray(m_next, dtype=np.float32).reshape(JOINT_ACTION_N) > 0.5)
        self._opp[i] = int(opponent_code)
        self._i = (i + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)
        return True

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None):
        if self._n < batch_size:
            raise ValueError(f'replay too small: {self._n} < {batch_size}')
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self._n, size=int(batch_size))
        return (
            self._s[idx].astype(np.float32),
            self._a[idx],
            self._r[idx],
            self._s2[idx].astype(np.float32),
            self._term[idx],
            self._m2[idx].astype(np.float32),
            self._opp[idx],
        )

    def origin_fractions(self) -> Dict[str, float]:
        """Fraction of stored transitions by opponent tag."""
        if self._n <= 0 or not self._allocated:
            return {}
        codes = self._opp[: self._n]
        ctr = Counter(int(c) for c in codes.tolist())
        total = float(self._n)
        out: Dict[str, float] = {}
        for code, n in ctr.items():
            name = CODE_TO_OPPONENT.get(code, f'code_{code}')
            out[name] = float(n) / total
        return out

    @staticmethod
    def batch_origin_fractions(opp_codes: np.ndarray) -> Dict[str, float]:
        codes = np.asarray(opp_codes).reshape(-1)
        if codes.size == 0:
            return {}
        ctr = Counter(int(c) for c in codes.tolist())
        total = float(codes.size)
        return {
            CODE_TO_OPPONENT.get(code, f'code_{code}'): float(n) / total
            for code, n in ctr.items()
        }
