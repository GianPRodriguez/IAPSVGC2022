"""Nash LP (HiGHS) + FP fallback for Team Preview (spec §4)."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from showdown.rl.mcts_puct_nash.nash import fictitious_play

SUPPORT_EPS = 1e-4
DOM_EPS = 1e-6
DEFAULT_TIMEOUT_S = 0.2


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if s <= 0:
        return np.ones_like(p) / float(p.size)
    return p / s


def _row_mean_softmax(A: np.ndarray, tau: float = 1.0) -> np.ndarray:
    row_mean = A.mean(axis=1)
    z = row_mean / max(float(tau), 1e-8)
    z = z - z.max()
    e = np.exp(z)
    return _normalize(e)


def solve_nash_lp(
    A: np.ndarray,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Zero-sum LP for row maximizer on payoff A.
    Uses positivity shift A' = A - min(A) + eps.
    Returns (p*, q*, v*) on original A, or None on failure.
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        return None

    A = np.asarray(A, dtype=np.float64)
    n, m = A.shape
    if n == 0 or m == 0:
        return None

    shift = float(A.min()) - 1e-3
    Ap = A - shift  # strictly positive enough for the classic form

    # Row player: min 1^T x  s.t. Ap^T x >= 1, x >= 0
    # p* = x / sum(x), v'_Ap = 1/sum(x); v_A = v'_Ap + shift
    c = np.ones(n, dtype=np.float64)
    bounds = [(0.0, None)] * n
    # Ap.T @ x >= 1  →  -Ap.T @ x <= -1
    A_ub = -Ap.T
    b_ub = -np.ones(m, dtype=np.float64)

    t0 = time.perf_counter()
    try:
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={'time_limit': float(timeout_s)},
        )
    except Exception:
        return None
    if time.perf_counter() - t0 > timeout_s * 1.5 and not res.success:
        return None
    if not res.success or res.x is None:
        return None

    x = np.asarray(res.x, dtype=np.float64)
    s = float(x.sum())
    if s <= 1e-12:
        return None
    p = _normalize(x)
    v_ap = 1.0 / s
    v = v_ap + shift

    # Column player (minimizer) via LP on -A^T (row max on -A.T)
    Am = -A.T
    shift_c = float(Am.min()) - 1e-3
    Amp = Am - shift_c
    c2 = np.ones(m, dtype=np.float64)
    bounds2 = [(0.0, None)] * m
    A_ub2 = -Amp.T
    b_ub2 = -np.ones(n, dtype=np.float64)
    try:
        res2 = linprog(
            c2,
            A_ub=A_ub2,
            b_ub=b_ub2,
            bounds=bounds2,
            method='highs',
            options={'time_limit': float(timeout_s)},
        )
    except Exception:
        # Derive q from best response to p
        col_pay = p @ A
        best = float(col_pay.min())
        q = np.zeros(m, dtype=np.float64)
        q[np.flatnonzero(col_pay <= best + DOM_EPS)] = 1.0
        q = _normalize(q)
        return p, q, float(p @ A @ q)

    if not res2.success or res2.x is None:
        col_pay = p @ A
        best = float(col_pay.min())
        q = np.zeros(m, dtype=np.float64)
        q[np.flatnonzero(col_pay <= best + DOM_EPS)] = 1.0
        q = _normalize(q)
        return p, q, float(p @ A @ q)

    q = _normalize(np.asarray(res2.x, dtype=np.float64))
    v = float(p @ A @ q)
    return p, q, v


def solve_mixed_strategy(
    A: np.ndarray,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    fp_iters: int = 200,
) -> Tuple[np.ndarray, np.ndarray, float, str]:
    """
    Primary LP → FP fallback → row-mean softmax.
    Returns (p*, q*, v*, method).
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] == 0:
        n = int(A.shape[0]) if A.ndim == 2 else 0
        p = np.ones(max(n, 1)) / max(n, 1)
        return p, p.copy(), 0.0, 'empty'

    out = solve_nash_lp(A, timeout_s=timeout_s)
    if out is not None:
        p, q, v = out
        return _normalize(p), _normalize(q), float(v), 'lp'

    try:
        p, q, v = fictitious_play(A, T=int(fp_iters))
        return _normalize(p), _normalize(q), float(v), 'fp'
    except Exception:
        pass

    p = _row_mean_softmax(A)
    q = _row_mean_softmax(-A.T)
    v = float(p @ A @ q)
    return p, q, v, 'softmax'


def support_size(p: np.ndarray, eps: float = SUPPORT_EPS) -> int:
    return int(np.sum(np.asarray(p) > eps))


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())
