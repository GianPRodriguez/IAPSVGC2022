r"""PPGA fiel (agente K 2.0, docs/qd_ppga_spec.md §K 2.0; Batra et al., ICLR 2024, arXiv 2305.13795).

Piezas del Algoritmo 1 del paper:

* ``XNES``: distribución de los coeficientes \(c \in \mathbb R^{k+1}\) (fitness + k descriptores);
  Glasmachers et al. (2010), utilidades por rango, \(\eta_\mu = 1\), \(\eta_\sigma = \eta_B =
  (9 + 3\ln d)/(5d\sqrt d)\).
* ``SoftArchive``: umbrales CMA-MAE por celda (Algoritmo 2): si \(f > t\), \(\Delta = f - t\) y
  \(t \leftarrow (1-\alpha)t + \alpha f\); si no, \(\Delta = 0\).
* ``ppo_iterations``: PPO con una recompensa a medida (\(r_f\), un proxy \(\delta_j\) o la combinación
  del paseo), sobre el entorno vectorizado.
* ``param_delta`` / ``apply_step``: los "gradientes" de VPPO son diferencias de parámetros
  \(\theta'_o - \theta_\mu\) (solo el actor), normalizados.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- xNES
class XNES:
    """xNES (Glasmachers et al., 2010) para la distribución de coeficientes de PPGA."""

    def __init__(self, dim: int, sigma0: float = 1.0, seed: int = 0):
        self.d = int(dim)
        self.sigma0 = float(sigma0)
        self.rng = np.random.default_rng(seed)
        self.eta_mu = 1.0
        self.eta_sigma = self.eta_b = (9.0 + 3.0 * math.log(self.d)) / (5.0 * self.d * math.sqrt(self.d))
        self.reset()

    def reset(self) -> None:
        self.mu = np.zeros(self.d)
        self.sigma = self.sigma0
        self.B = np.eye(self.d)
        self._z: Optional[np.ndarray] = None

    def ask(self, lam: int) -> np.ndarray:
        self._z = self.rng.standard_normal((int(lam), self.d))
        return self.mu + self.sigma * self._z @ self.B.T

    @staticmethod
    def utilities(lam: int) -> np.ndarray:
        k = np.arange(1, lam + 1)
        u = np.maximum(0.0, math.log(lam / 2.0 + 1.0) - np.log(k))
        return u / u.sum() - 1.0 / lam

    def tell(self, scores: Sequence[float]) -> None:
        """Actualiza con las puntuaciones de las muestras de ``ask`` (mayor = mejor; empates al azar)."""
        z = self._z
        lam = z.shape[0]
        noise = self.rng.random(lam) * 1e-12
        order = np.argsort(-(np.asarray(scores, dtype=np.float64) + noise))
        u = np.zeros(lam)
        u[order] = self.utilities(lam)
        g_delta = (u[:, None] * z).sum(axis=0)
        g_m = sum(u[i] * (np.outer(z[i], z[i]) - np.eye(self.d)) for i in range(lam))
        g_sigma = float(np.trace(g_m)) / self.d
        g_b = g_m - g_sigma * np.eye(self.d)
        self.mu = self.mu + self.eta_mu * self.sigma * self.B @ g_delta
        self.sigma = self.sigma * math.exp(0.5 * self.eta_sigma * g_sigma)
        w, v = np.linalg.eigh(0.5 * self.eta_b * (g_b + g_b.T) / 2.0)
        self.B = self.B @ (v @ np.diag(np.exp(w)) @ v.T)


# --------------------------------------------------------------------------- archivo suave
class SoftArchive:
    """Umbrales CMA-MAE por celda (solo para ordenar ramas; el archivo de resultados va aparte)."""

    def __init__(self, alpha: float = 0.1, t0: float = -1.0):
        self.alpha, self.t0 = float(alpha), float(t0)
        self.t: Dict[Tuple[int, int], float] = {}

    def update(self, key: Tuple[int, int], f: float) -> float:
        t = self.t.get(key, self.t0)
        if f > t:
            self.t[key] = (1.0 - self.alpha) * t + self.alpha * float(f)
            return float(f) - t
        return 0.0


# --------------------------------------------------------------------------- parámetros
def actor_keys(state: Dict[str, Any]) -> List[str]:
    import torch
    return [k for k, v in state.items() if not k.startswith('value.') and torch.is_floating_point(v)]


def param_delta(new: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (new[k] - base[k]) for k in actor_keys(base)}


def delta_norm(d: Dict[str, Any]) -> float:
    return float(math.sqrt(sum(float((v.double() ** 2).sum()) for v in d.values())))


def apply_step(base: Dict[str, Any], grads: Sequence[Dict[str, Any]], coef: Sequence[float],
               scale: float) -> Dict[str, Any]:
    """θ' = θ_μ + scale · Σ_j c_j · ĝ_j (ĝ_j ya normalizados)."""
    out = {k: v.clone() for k, v in base.items()}
    for c, g in zip(coef, grads):
        for k, v in g.items():
            out[k] = out[k] + float(scale) * float(c) * v
    return out


def normalized(d: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    n = delta_norm(d)
    if n <= 0.0:
        return d, 0.0
    return {k: v / n for k, v in d.items()}, n


# --------------------------------------------------------------------------- PPO con recompensa a medida
def objective_reward(kind: str, coef: Optional[Sequence[float]] = None) -> Callable[[Dict[str, Any]], float]:
    """Recompensa por paso: ``f`` = Cand. 2 (densa), ``m1``/``m2`` = proxies δ1/δ2, ``mix`` = Σ c_j·r_j."""
    def _d(r, j):
        b = (r.get('info') or {}).get('bd_step')
        return float(b[j]) if b else 0.5

    if kind == 'f':
        return lambda r: float(r['reward'])
    if kind == 'm1':
        return lambda r: _d(r, 0)
    if kind == 'm2':
        return lambda r: _d(r, 1)
    c = [float(x) for x in ((1.0, 0.0, 0.0) if coef is None else coef)]
    return lambda r: c[0] * float(r['reward']) + c[1] * _d(r, 0) + c[2] * _d(r, 1)


def ppo_iterations(agent, venv, *, n_iters: int, rollout: int, rng, sampler, reward_fn,
                   heartbeat=None, hb_offset: int = 0, phase: str = '') -> int:
    """``n_iters`` iteraciones de PPO (``rollout`` pasos cada una) con ``reward_fn(resultado)``."""
    from showdown.rl.reward import RewardCandidate
    from showdown.rl.vec_rollout import run_vec_rollout

    total = int(n_iters) * int(rollout)
    if total <= 0:
        return 0
    agent.cfg.rollout_steps = int(rollout)
    agent.use_vec_buffer(int(venv.num_envs))
    venv.set_reward(RewardCandidate.DENSE.value)

    def act_fn(obs, m1, idx):
        a1, a2, logp, value, m2 = agent.act_batch(obs, m1, lambda a1s: venv.mask2_subset(idx, a1s),
                                                  greedy=False)
        return {'a1': a1, 'a2': a2, 'logp': logp, 'value': value, 'm2': m2}

    def on_step(i, rec):
        r = rec['result']
        agent.observe(obs=rec['obs'], a1=rec['a1'], a2=rec['a2'], logp=rec['logp'],
                      reward=float(reward_fn(r)), value=rec['value'], terminated=bool(r['terminated']),
                      m1=rec['m1'], m2=rec['m2'], skip_buffer=rec['skip'],
                      trunc_reason=r['info']['trunc_reason'], bootstrap_value=rec['boot_v'], env_index=i)

    def flush(obs, last_skip):
        if len(agent.buffer) == 0:
            return
        last_v = agent.values_batch(obs)
        for i, sk in enumerate(last_skip):
            if sk:
                last_v[i] = 0.0
        agent.update(last_value=last_v)

    out = run_vec_rollout(venv, steps=total, act_fn=act_fn, on_step=on_step, boot_fn=agent.values_batch,
                          foe_sampler=lambda: sampler(rng), should_flush=agent.is_rollout_full,
                          flush_fn=flush, final_flush=True, heartbeat=heartbeat, hb_offset=int(hb_offset),
                          phase=phase or 'ppo')
    return int(out['taken'])
