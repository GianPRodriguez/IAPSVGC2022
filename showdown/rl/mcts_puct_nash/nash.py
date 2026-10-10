"""Fictitious Play + Nash value (docs/mcts_puct_nash_spec.md)."""
from __future__ import annotations

from typing import Tuple

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


def nash_value(A: np.ndarray, *, T: int = 200) -> float:
    """Scalar v* = σ*^T A τ* via Fictitious Play."""
    _s, _t, v = fictitious_play(A, T=T)
    return float(v)
