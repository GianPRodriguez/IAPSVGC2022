"""On-policy rollout buffer — drops skip_buffer / crash (docs/ppo_spec.md)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from showdown.rl.constants import OBS_DIM, SLOT_ACTION_N


class RolloutBuffer:
    """Fixed-capacity trajectory store for PPO + GAE."""

    def __init__(self, capacity: int, *, obs_dim: int = OBS_DIM):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._a1 = np.zeros(self.capacity, dtype=np.int64)
        self._a2 = np.zeros(self.capacity, dtype=np.int64)
        self._logp = np.zeros(self.capacity, dtype=np.float32)
        self._rew = np.zeros(self.capacity, dtype=np.float32)
        self._val = np.zeros(self.capacity, dtype=np.float32)
        self._term = np.zeros(self.capacity, dtype=np.float32)
        self._eoe = np.zeros(self.capacity, dtype=np.float32)
        self._boot = np.zeros(self.capacity, dtype=np.float32)
        self._m1 = np.zeros((self.capacity, SLOT_ACTION_N), dtype=np.float32)
        self._m2 = np.zeros((self.capacity, SLOT_ACTION_N), dtype=np.float32)
        self._adv = np.zeros(self.capacity, dtype=np.float32)
        self._ret = np.zeros(self.capacity, dtype=np.float32)
        self._pos = 0

    def __len__(self) -> int:
        return int(self._pos)

    def clear(self) -> None:
        self._pos = 0

    def add(
        self,
        *,
        obs: np.ndarray,
        a1: int,
        a2: int,
        logp: float,
        reward: float,
        value: float,
        terminated: bool,
        m1: np.ndarray,
        m2: np.ndarray,
        skip_buffer: bool = False,
        trunc_reason: Optional[str] = None,
        bootstrap_value: float = 0.0,
    ) -> bool:
        """
        Store one transition.

        Rejects L2 (`skip_buffer`) and L3 crash. Turn-cap is stored with
        `terminated=False`, `eoe=True`, and `bootstrap_value=V(s')`.
        """
        if skip_buffer or trunc_reason == 'crash':
            return False
        if self._pos >= self.capacity:
            return False
        i = self._pos
        eoe = bool(terminated) or trunc_reason == 'turn_cap'
        boot = 0.0
        if trunc_reason == 'turn_cap' and not terminated:
            boot = float(bootstrap_value)
        self._obs[i] = np.asarray(obs, dtype=np.float32).reshape(self.obs_dim)
        self._a1[i] = int(a1)
        self._a2[i] = int(a2)
        self._logp[i] = float(logp)
        self._rew[i] = float(reward)
        self._val[i] = float(value)
        self._term[i] = 1.0 if terminated else 0.0
        self._eoe[i] = 1.0 if eoe else 0.0
        self._boot[i] = boot
        self._m1[i] = np.asarray(m1, dtype=np.float32).reshape(SLOT_ACTION_N)
        self._m2[i] = np.asarray(m2, dtype=np.float32).reshape(SLOT_ACTION_N)
        self._pos += 1
        return True

    def compute_gae(
        self,
        *,
        last_value: float = 0.0,
        gamma: float = 0.95,
        lam: float = 0.95,
        normalize_adv: bool = True,
    ) -> None:
        """Fill advantages / returns for stored prefix `[0, len)`."""
        n = len(self)
        if n == 0:
            return
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            if self._eoe[t] > 0.5:
                next_val = 0.0 if self._term[t] > 0.5 else float(self._boot[t])
                nonterminal = 0.0 if self._term[t] > 0.5 else 1.0
                delta = (
                    float(self._rew[t])
                    + float(gamma) * next_val * nonterminal
                    - float(self._val[t])
                )
                # Cut λ-chain from a later episode when walking backward.
                gae = delta
            else:
                if t + 1 < n:
                    next_val = float(self._val[t + 1])
                else:
                    next_val = float(last_value)
                delta = (
                    float(self._rew[t])
                    + float(gamma) * next_val
                    - float(self._val[t])
                )
                gae = delta + float(gamma) * float(lam) * gae
            adv[t] = gae
        self._adv[:n] = adv
        self._ret[:n] = adv + self._val[:n]
        if normalize_adv and n > 1:
            mu = float(adv.mean())
            std = float(adv.std())
            if std < 1e-8:
                std = 1.0
            self._adv[:n] = (adv - mu) / std

    def get(self) -> Dict[str, np.ndarray]:
        n = len(self)
        return {
            'obs': self._obs[:n].copy(),
            'a1': self._a1[:n].copy(),
            'a2': self._a2[:n].copy(),
            'logp': self._logp[:n].copy(),
            'adv': self._adv[:n].copy(),
            'ret': self._ret[:n].copy(),
            'val': self._val[:n].copy(),
            'm1': self._m1[:n].copy(),
            'm2': self._m2[:n].copy(),
        }

    def minibatches(self, batch_size: int, rng: np.random.Generator):
        data = self.get()
        n = len(self)
        if n == 0:
            return
        idx = rng.permutation(n)
        bs = max(1, int(batch_size))
        for start in range(0, n, bs):
            sl = idx[start:start + bs]
            yield {k: v[sl] for k, v in data.items()}


class VecRolloutBuffer:
    """Un ``RolloutBuffer`` por entorno paralelo.

    El GAE se calcula *por entorno* (cada sub-buffer es una secuencia temporal
    contigua de un mismo entorno; mezclar entornos rompería la cadena lambda) y
    la normalización de ventajas es global sobre todo el lote, igual que en el
    caso de un solo entorno. ``len()`` es el total de transiciones guardadas.
    """

    def __init__(self, num_envs: int, capacity: int, *, obs_dim: int = OBS_DIM):
        self.num_envs = max(1, int(num_envs))
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.buffers = [
            RolloutBuffer(capacity=self.capacity, obs_dim=self.obs_dim)
            for _ in range(self.num_envs)
        ]

    def __len__(self) -> int:
        return int(sum(len(b) for b in self.buffers))

    def clear(self) -> None:
        for b in self.buffers:
            b.clear()

    def add(self, *, env_index: int = 0, **kwargs) -> bool:
        return self.buffers[int(env_index)].add(**kwargs)

    def any_full(self) -> bool:
        return any(len(b) >= b.capacity for b in self.buffers)

    def compute_gae(
        self,
        *,
        last_value=0.0,
        gamma: float = 0.95,
        lam: float = 0.95,
        normalize_adv: bool = True,
    ) -> None:
        lv = np.broadcast_to(
            np.asarray(last_value, dtype=np.float32).reshape(-1), (self.num_envs,)
        )
        for i, b in enumerate(self.buffers):
            b.compute_gae(
                last_value=float(lv[i]), gamma=gamma, lam=lam, normalize_adv=False,
            )
        if not normalize_adv:
            return
        n = len(self)
        if n <= 1:
            return
        adv = np.concatenate([b._adv[:len(b)] for b in self.buffers])
        mu = float(adv.mean())
        std = float(adv.std())
        if std < 1e-8:
            std = 1.0
        for b in self.buffers:
            k = len(b)
            if k:
                b._adv[:k] = (b._adv[:k] - mu) / std

    def get(self) -> Dict[str, np.ndarray]:
        parts = [b.get() for b in self.buffers if len(b)]
        if not parts:
            return RolloutBuffer(capacity=1, obs_dim=self.obs_dim).get()
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def minibatches(self, batch_size: int, rng: np.random.Generator):
        data = self.get()
        n = len(self)
        if n == 0:
            return
        idx = rng.permutation(n)
        bs = max(1, int(batch_size))
        for start in range(0, n, bs):
            sl = idx[start:start + bs]
            yield {k: v[sl] for k, v in data.items()}
