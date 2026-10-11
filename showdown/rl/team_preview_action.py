"""Flat-360 Team Preview action codec (Lehmer / factorial number system).

Contract: docs/team_preview_masked_ppo_spec.md §2.
Gym / logits: a in {0..359}. Showdown: 1-based /team string.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

N_POOL = 6
N_PICK = 4
N_ACTIONS = 360  # P(6,4)
_FACTORS = (60, 12, 3, 1)  # 5*4*3, 4*3, 3, 1


def encode_preview_action(indices_0based: Sequence[int]) -> int:
    """(i1,i2,i3,i4) distinct in 0..5 → a in 0..359."""
    if len(indices_0based) != N_PICK:
        raise ValueError(f'need {N_PICK} indices, got {len(indices_0based)}')
    idxs = [int(x) for x in indices_0based]
    if len(set(idxs)) != N_PICK or any(i < 0 or i >= N_POOL for i in idxs):
        raise ValueError(f'invalid injection {idxs}')
    chosen: List[int] = []
    a = 0
    for k, ik in enumerate(idxs):
        c = ik - sum(1 for prev in chosen if ik > prev)
        a += _FACTORS[k] * c
        chosen.append(ik)
    return int(a)


def decode_preview_action(a: int) -> Tuple[int, int, int, int]:
    """a in 0..359 → (i1,i2,i3,i4) 0-based distinct."""
    a = int(a)
    if a < 0 or a >= N_ACTIONS:
        raise ValueError(f'action {a} out of range [0,{N_ACTIONS})')
    remaining = list(range(N_POOL))
    out: List[int] = []
    rem = a
    for f in _FACTORS:
        c, rem = divmod(rem, f)
        if c >= len(remaining):
            raise ValueError(f'corrupt action {a}')
        out.append(remaining.pop(c))
    return out[0], out[1], out[2], out[3]


def to_showdown_team_string(indices_0based: Sequence[int]) -> str:
    """0-based injection → e.g. '1234'."""
    return ''.join(str(int(i) + 1) for i in indices_0based)


def action_to_team_string(a: int) -> str:
    return to_showdown_team_string(decode_preview_action(a))


def legal_mask(pool_size: int = N_POOL) -> List[float]:
    """
    Mask length 360; 1=legal.
    Full pool of 6 → all ones. Smaller pool → only injections using ids < pool_size.
    """
    pool_size = int(pool_size)
    if pool_size >= N_POOL:
        return [1.0] * N_ACTIONS
    if pool_size < N_PICK:
        return [0.0] * N_ACTIONS
    m = [0.0] * N_ACTIONS
    for a in range(N_ACTIONS):
        tup = decode_preview_action(a)
        if all(i < pool_size for i in tup):
            m[a] = 1.0
    return m
