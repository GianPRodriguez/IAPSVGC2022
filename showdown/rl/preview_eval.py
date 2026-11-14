"""Evaluación de un preview sobre contextos fijos frente a una línea base.

Sustituye al proxy ``-pg_loss`` (que con ventajas normalizadas ronda 0 sin
importar la calidad de la política) como criterio de checkpoint y de HP tune.

Protocolo (pareado):

1. ``env.reseed(eval_seed)`` + ``reset_batch(n)`` → n contextos idénticos para
   todos los trials (misma semilla de evaluación).
2. La política elige en modo greedy (argmax enmascarado).
3. La línea base juega los mismos contextos:
   - ``fixed``: ``/team 1234`` (orden de exportación, fallback del contrato);
   - ``pilot``: preview propio de π_c (solo backend live);
   - ``random``: formación uniforme legal.
4. WR = (victorias + 0.5·empates) / n. ``DELTA_WR`` = WR_política − WR_base, con
   su error estándar pareado.

En live cada contexto cuesta dos BO1 (política y base).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np

DEFAULT_EVAL_SEED = 2_000_003
BASELINES = ('fixed', 'pilot', 'random')


def wr_from_rewards(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    return np.where(r > 0, 1.0, np.where(r < 0, 0.0, 0.5))


@dataclass
class PreviewEval:
    n: int
    baseline: str
    eval_seed: int
    mean_wr: float
    base_wr: float
    delta_wr: float
    delta_se: float

    def as_dict(self) -> dict:
        return asdict(self)

    def lines(self) -> str:
        return (
            f'eval_contexts: {self.n} (seed {self.eval_seed}, baseline {self.baseline})\n'
            f'MEAN_WR: {self.mean_wr:.6f}\n'
            f'BASE_WR: {self.base_wr:.6f}\n'
            f'DELTA_WR: {self.delta_wr:+.6f} (SE {self.delta_se:.6f})\n'
        )


def evaluate_preview_policy(
    env,
    choose: Callable[[np.ndarray, np.ndarray], int],
    *,
    n: int,
    baseline: str = 'fixed',
    eval_seed: int = DEFAULT_EVAL_SEED,
) -> PreviewEval:
    """``choose(obs, mask) -> acción`` (greedy). Devuelve WR pareado vs base."""
    if baseline not in BASELINES:
        raise ValueError(f'baseline inválida {baseline!r}; opciones: {", ".join(BASELINES)}')
    n = int(n)
    if n <= 0:
        raise ValueError('n de evaluación debe ser > 0')
    # Restaurar el RNG de contextos al final: la evaluación no debe alterar la
    # secuencia de contextos de entrenamiento (ni solaparla con la de evaluación).
    saved_rng = copy.deepcopy(getattr(env, '_rng', None))
    try:
        return _evaluate(env, choose, n=n, baseline=baseline, eval_seed=eval_seed)
    finally:
        if saved_rng is not None:
            env._rng = saved_rng


def _evaluate(env, choose, *, n: int, baseline: str, eval_seed: int) -> PreviewEval:
    env.reseed(int(eval_seed))
    obs, infos = env.reset_batch(n)
    masks = [
        (info.get('action_mask') if info.get('action_mask') is not None else env.action_masks())
        for info in infos
    ]
    acts = [int(choose(obs[k], masks[k])) for k in range(n)]
    r_pol = env.step_batch(acts)
    if baseline == 'fixed':
        r_base = env.step_batch([0] * n)  # acción 0 ↔ /team 1234
    elif baseline == 'pilot':
        r_base = env.step_batch_pilot()
    else:
        rng = np.random.default_rng(int(eval_seed) + 1)
        r_base = env.step_batch([int(rng.choice(np.flatnonzero(np.asarray(m) > 0.5))) for m in masks])
    w_pol = wr_from_rewards(r_pol)
    w_base = wr_from_rewards(r_base)
    diff = w_pol - w_base
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float('nan')
    return PreviewEval(
        n=n, baseline=baseline, eval_seed=int(eval_seed),
        mean_wr=float(w_pol.mean()), base_wr=float(w_base.mean()),
        delta_wr=float(diff.mean()), delta_se=se,
    )


def add_eval_cli_args(parser, *, default_n: int = 512) -> None:
    parser.add_argument('--eval-episodes', type=int, default=default_n,
                        help='Contextos de evaluación pareada al final (0 = omitir). '
                             'En live cuesta 2 BO1 por contexto')
    parser.add_argument('--eval-baseline', choices=BASELINES, default='fixed',
                        help='Línea base: fixed=/team 1234, pilot=preview de π_c (live), random')
    parser.add_argument('--eval-seed', type=int, default=DEFAULT_EVAL_SEED,
                        help='Semilla de contextos de evaluación (igual en todos los trials)')


def resolve_eval_n(eval_episodes: int, *, smoke: bool, smoke_n: int = 64) -> int:
    n = int(eval_episodes)
    if smoke and n > smoke_n:
        return smoke_n
    return max(0, n)


__all__: Sequence[str] = (
    'BASELINES', 'DEFAULT_EVAL_SEED', 'PreviewEval', 'add_eval_cli_args',
    'evaluate_preview_policy', 'resolve_eval_n', 'wr_from_rewards',
)
