"""RewardSpec Candidates 1 & 2 — docs/rl_env_spec.md v1.1."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from showdown.rl.constants import (
    C2_TIE_PENALTY,
    R_CLIP_HI,
    R_CLIP_LO,
    W_FAINT,
    W_HP,
    W_LOSS,
    W_WIN,
)
from showdown.rl.snapshot import TeamMaterial


class RewardCandidate(str, Enum):
    SPARSE = 'candidate_1'
    DENSE = 'candidate_2'


@dataclass(frozen=True)
class RewardResult:
    reward: float
    skip_buffer: bool = False
    components: Optional[dict] = None


def _hp_self_weight() -> float:
    import os
    try:
        return max(0.0, float(os.environ.get('RL_REWARD_HP_SELF', '0') or 0.0))
    except ValueError:
        return 0.0


def _clip(r: float) -> float:
    return max(R_CLIP_LO, min(R_CLIP_HI, float(r)))


def compute_reward(
    *,
    candidate: RewardCandidate,
    prev_our: TeamMaterial,
    prev_foe: TeamMaterial,
    cur_our: TeamMaterial,
    cur_foe: TeamMaterial,
    finished: bool,
    won: Optional[bool],
    tied: bool,
    skip_buffer: bool = False,
    truncated: bool = False,
) -> RewardResult:
    """
    Compute R_t from Spec.

    - L2 skip_buffer / L3 truncated → R=0 (caller may still pass finished=False).
    - Candidate 1: sparse terminal only; tie=0.
    - Candidate 2: dense deltas + tie=-0.25; clip to [-1.30, +1.40].
    """
    if skip_buffer or truncated:
        return RewardResult(reward=0.0, skip_buffer=skip_buffer, components={'reason': 'skip_or_trunc'})

    d_faint_opp = float(cur_foe.faint_count - prev_foe.faint_count)
    d_faint_self = float(cur_our.faint_count - prev_our.faint_count)
    # HP% drop on foe team is positive reward progress (opp lost HP).
    d_hp_opp = float(prev_foe.hp_pct_total - cur_foe.hp_pct_total) / 100.0
    # Variante simétrica (diagnóstico 16-11-2026): la recompensa densa no castigaba los PS
    # perdidos por NUESTRO equipo (solo los KO propios), así que dañar al propio aliado salía
    # gratis. RL_REWARD_HP_SELF=<k> resta k·W_HP por cada 100 % de PS propios perdidos.
    # 0 (por defecto) = recompensa del contrato actual.
    d_hp_self = float(prev_our.hp_pct_total - cur_our.hp_pct_total) / 100.0
    hp_self_k = _hp_self_weight()

    d_win = 0.0
    d_loss = 0.0
    if finished:
        if tied or won is None:
            pass
        elif won:
            d_win = 1.0
        else:
            d_loss = 1.0

    if candidate == RewardCandidate.SPARSE:
        if not finished:
            r = 0.0
        elif tied or won is None:
            r = 0.0
        elif won:
            r = 1.0
        else:
            r = -1.0
        return RewardResult(
            reward=r,
            components={
                'd_win': d_win,
                'd_loss': d_loss,
                'candidate': candidate.value,
            },
        )

    # Candidate 2
    r = (
        W_WIN * d_win
        - W_LOSS * d_loss
        + W_FAINT * d_faint_opp
        - W_FAINT * d_faint_self
        + W_HP * d_hp_opp
        - hp_self_k * W_HP * d_hp_self
    )
    if finished and (tied or won is None) and d_win == 0.0 and d_loss == 0.0:
        r += C2_TIE_PENALTY
    r = _clip(r)
    return RewardResult(
        reward=r,
        components={
            'd_win': d_win,
            'd_loss': d_loss,
            'd_faint_opp': d_faint_opp,
            'd_faint_self': d_faint_self,
            'd_hp_opp': d_hp_opp,
            'd_hp_self': d_hp_self,
            'candidate': candidate.value,
        },
    )


def flip_reward_for_opponent(r: float) -> float:
    """Self-play symmetry: R_P2 = -R_P1 for the same transition."""
    return -float(r)
