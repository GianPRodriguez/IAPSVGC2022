"""Fictitious Play + Nash value (docs/mcts_puct_nash_spec.md)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

EPS_OPT = 1e-6
FP_WINDOW = 50


def fictitious_play(
    A: np.ndarray,
    *,
    T: int = 200,
    W: int = FP_WINDOW,
    eps: float = EPS_OPT,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Zero-sum Fictitious Play on payoff matrix A (row maximizer).

    Returns (σ*, τ*, v*) where σ*/τ* are averages of the last W empirical
    frequency vectors. Tied best responses use uniform (max-entropy) mixes.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f'payoff must be 2D, got shape {A.shape}')
    n, m = int(A.shape[0]), int(A.shape[1])
    if n == 0 or m == 0:
        return np.zeros(n), np.zeros(m), 0.0
    if n == 1 and m == 1:
        return np.array([1.0]), np.array([1.0]), float(A[0, 0])

    row_counts = np.zeros(n, dtype=np.float64)
    col_counts = np.zeros(m, dtype=np.float64)
    sigma = np.ones(n, dtype=np.float64) / float(n)
    tau = np.ones(m, dtype=np.float64) / float(m)

    hist_sigma: list[np.ndarray] = []
    hist_tau: list[np.ndarray] = []
    T = max(1, int(T))
    W = max(1, int(W))

    for _ in range(T):
        row_payoffs = A @ tau
        best_row = float(np.max(row_payoffs))
        br_rows = np.flatnonzero(row_payoffs >= best_row - eps)
        br_mix_row = np.zeros(n, dtype=np.float64)
        br_mix_row[br_rows] = 1.0 / float(br_rows.size)

        col_payoffs = sigma @ A
        best_col = float(np.min(col_payoffs))
        br_cols = np.flatnonzero(col_payoffs <= best_col + eps)
        br_mix_col = np.zeros(m, dtype=np.float64)
        br_mix_col[br_cols] = 1.0 / float(br_cols.size)

        row_counts += br_mix_row
        col_counts += br_mix_col
        sigma = row_counts / row_counts.sum()
        tau = col_counts / col_counts.sum()
        hist_sigma.append(sigma.copy())
        hist_tau.append(tau.copy())

    w = min(W, len(hist_sigma))
    sigma_avg = np.mean(hist_sigma[-w:], axis=0)
    tau_avg = np.mean(hist_tau[-w:], axis=0)
    sigma_avg = sigma_avg / sigma_avg.sum()
    tau_avg = tau_avg / tau_avg.sum()
    v = float(sigma_avg @ A @ tau_avg)
    return sigma_avg, tau_avg, v


class RMPlusState:
    """Regrets acumulados y sumas de estrategia (ponderadas por iteración) de un nodo."""

    __slots__ = ('Rr', 'Rc', 'Sr', 'Sc', 't')

    def __init__(self, n: int, m: int):
        self.Rr = np.zeros(n)
        self.Rc = np.zeros(m)
        self.Sr = np.zeros(n)
        self.Sc = np.zeros(m)
        self.t = 0


def _rm_strategy(R: np.ndarray) -> np.ndarray:
    s = float(R.sum())
    return R / s if s > 0 else np.full(R.shape[0], 1.0 / R.shape[0])


def regret_matching_plus(
    A: np.ndarray,
    *,
    T: int = 10,
    state: Optional[RMPlusState] = None,
) -> Tuple[np.ndarray, np.ndarray, float, RMPlusState]:
    r"""Regret matching+ de suma cero (fila maximiza) con arranque en caliente (H 2.1, brazo RM+).

    ``state`` conserva regrets y promedios entre llamadas: en un nodo de SM-MCTS la matriz cambia poco
    entre simulaciones, así que bastan pocas iteraciones por backup (``T``) en lugar de un FP completo.
    Promedio lineal (peso \(t\)), como CFR+. Devuelve \((\bar\sigma, \bar\tau, \bar\sigma^\top A \bar\tau, state)\).
    """
    A = np.asarray(A, dtype=np.float64)
    n, m = int(A.shape[0]), int(A.shape[1])
    if n == 0 or m == 0:
        return np.zeros(n), np.zeros(m), 0.0, state or RMPlusState(n, m)
    if state is None or state.Rr.shape[0] != n or state.Rc.shape[0] != m:
        state = RMPlusState(n, m)
    for _ in range(max(1, int(T))):
        sig = _rm_strategy(state.Rr)
        tau = _rm_strategy(state.Rc)
        u_row = A @ tau
        u_col = sig @ A
        ev = float(sig @ u_row)
        state.Rr = np.maximum(state.Rr + u_row - ev, 0.0)
        state.Rc = np.maximum(state.Rc + ev - u_col, 0.0)
        state.t += 1
        state.Sr += state.t * sig
        state.Sc += state.t * tau
    sigma = state.Sr / state.Sr.sum()
    tau = state.Sc / state.Sc.sum()
    return sigma, tau, float(sigma @ A @ tau), state


def nash_value(A: np.ndarray, *, T: int = 200) -> float:
    """Scalar v* = σ*^T A τ* via Fictitious Play."""
    _s, _t, v = fictitious_play(A, T=T)
    return float(v)
