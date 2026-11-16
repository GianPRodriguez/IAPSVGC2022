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
    reserva al primer ``add`` (los jugadores de inferencia no la usan).

    ``device='cuda'`` guarda obs y máscaras en la GPU (ddqn_spec 1.13): el muestreo y la
    aumentación por simetría se hacen allí, en vez de copiar ~25 MB por lote desde la CPU.
    Los índices se siguen sorteando con el generador NumPy del agente, así que los lotes
    son idénticos a los de la versión en CPU. Acción, recompensa, fin y rival quedan en CPU
    (son chicos). Las fracciones por rival se llevan con contadores incrementales en vez de
    recorrer todo el replay en cada paso de gradiente (~10 ms con 200k transiciones).
    """

    _FIELDS = ('_s', '_a', '_r', '_s2', '_term', '_m2', '_opp')
    _GPU_FIELDS = ('_s', '_s2', '_m2')

    def __init__(self, capacity: int = 200_000, obs_dim: int = OBS_DIM, device: str = 'cpu'):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.device = str(device or 'cpu')
        self._allocated = False
        self._n = 0
        self._i = 0
        self._opp_counts: Counter = Counter()

    @property
    def on_gpu(self) -> bool:
        return self.device.startswith('cuda')

    def _spec(self, name: str):
        d = self.obs_dim
        return {
            '_s': ((d,), np.float16), '_s2': ((d,), np.float16),
            '_a': ((), np.int64), '_r': ((), np.float32), '_term': ((), np.float32),
            '_m2': ((JOINT_ACTION_N,), np.uint8), '_opp': ((), np.int16),
        }[name]

    def _zeros(self, name: str, rows: int):
        tail, dtype = self._spec(name)
        if self.on_gpu and name in self._GPU_FIELDS:
            import torch
            tdt = torch.float16 if dtype is np.float16 else torch.uint8
            return torch.zeros((rows,) + tail, dtype=tdt, device=self.device)
        return np.zeros((rows,) + tail, dtype=dtype)

    def _alloc(self) -> None:
        """Reserva perezosa y creciente (×2 hasta ``capacity``): una corrida corta
        no reserva los ≈5 GB de 200k transiciones con obs v2."""
        if not self._allocated:
            self._size = min(self.capacity, 4096)
            for name in self._FIELDS:
                setattr(self, name, self._zeros(name, self._size))
            self._allocated = True
            return
        if self._i >= self._size and self._size < self.capacity:
            new = min(self.capacity, 2 * self._size)
            for name in self._FIELDS:
                old = getattr(self, name)
                arr = self._zeros(name, new)
                arr[: self._size] = old
                setattr(self, name, arr)
                del old
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
        s16 = np.asarray(s, dtype=np.float32).reshape(self.obs_dim).astype(np.float16)
        s2_16 = np.asarray(s_next, dtype=np.float32).reshape(self.obs_dim).astype(np.float16)
        m8 = (np.asarray(m_next, dtype=np.float32).reshape(JOINT_ACTION_N) > 0.5).astype(np.uint8)
        if self.on_gpu:
            import torch
            self._s[i] = torch.from_numpy(s16).to(self.device)
            self._s2[i] = torch.from_numpy(s2_16).to(self.device)
            self._m2[i] = torch.from_numpy(m8).to(self.device)
        else:
            self._s[i] = s16
            self._s2[i] = s2_16
            self._m2[i] = m8
        self._a[i] = int(a)
        self._r[i] = float(r)
        self._term[i] = 1.0 if terminated else 0.0
        if self._n >= self.capacity:  # se sobrescribe la transición más vieja
            self._opp_counts[int(self._opp[i])] -= 1
        self._opp[i] = int(opponent_code)
        self._opp_counts[int(opponent_code)] += 1
        self._i = (i + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)
        return True

    def _sample_idx(self, batch_size: int, rng: Optional[np.random.Generator]) -> np.ndarray:
        if self._n < batch_size:
            raise ValueError(f'replay too small: {self._n} < {batch_size}')
        rng = rng or np.random.default_rng()
        return rng.integers(0, self._n, size=int(batch_size))

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None):
        """Lote en NumPy (float32). En GPU copia el lote a la CPU (tests, diagnóstico)."""
        idx = self._sample_idx(batch_size, rng)
        if self.on_gpu:
            import torch
            it = torch.as_tensor(idx, device=self.device)
            s = self._s[it].float().cpu().numpy()
            s2 = self._s2[it].float().cpu().numpy()
            m2 = self._m2[it].float().cpu().numpy()
        else:
            s = self._s[idx].astype(np.float32)
            s2 = self._s2[idx].astype(np.float32)
            m2 = self._m2[idx].astype(np.float32)
        return (s, self._a[idx], self._r[idx], s2, self._term[idx], m2, self._opp[idx])

    def sample_torch(self, batch_size: int, rng: Optional[np.random.Generator] = None,
                     device: Optional[str] = None):
        """Lote como tensores float32 en ``device`` (mismo sorteo de índices que ``sample``)."""
        import torch
        idx = self._sample_idx(batch_size, rng)
        dev = device or (self.device if self.on_gpu else 'cpu')
        if self.on_gpu:
            it = torch.as_tensor(idx, device=self.device)
            s, s2, m2 = self._s[it].float(), self._s2[it].float(), self._m2[it].float()
        else:
            s = torch.as_tensor(self._s[idx].astype(np.float32), device=dev)
            s2 = torch.as_tensor(self._s2[idx].astype(np.float32), device=dev)
            m2 = torch.as_tensor(self._m2[idx].astype(np.float32), device=dev)
        return (s, torch.as_tensor(self._a[idx], device=dev),
                torch.as_tensor(self._r[idx], device=dev), s2,
                torch.as_tensor(self._term[idx], device=dev), m2, self._opp[idx])

    def origin_fractions(self) -> Dict[str, float]:
        """Fracción de transiciones guardadas por rival (contadores incrementales, O(1))."""
        if self._n <= 0 or not self._allocated:
            return {}
        total = float(self._n)
        return {CODE_TO_OPPONENT.get(code, f'code_{code}'): float(n) / total
                for code, n in self._opp_counts.items() if n > 0}

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
