"""Acciones podadas, prior y modelo de rival para MCTS+PUCT v2 (docs/mcts_puct_spec.md §4–§7).

v1 expandía la máscara del snapshot (``legal_joint_mask``): los 4 objetivos para cada
movimiento, ``pass`` siempre y la mega duplicada → ~660 joints en la raíz, explorados
en orden de id. Aquí cada movimiento solo recibe los objetivos con sentido según su
campo ``target`` de ``moves.json``, ``pass`` solo cuando no hay otra cosa y, dentro del
árbol, sin variantes mega (S2 no distingue mega de no mega).

Todo trabaja desde la perspectiva ``our`` del snapshot; para el rival se usa
:func:`flip_view` (vista barata, sin copia profunda).
"""
from __future__ import annotations

import dataclasses
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from showdown.rl.action_space import (
    decode_joint_action,
    decode_slot_action,
    encode_joint_action,
    encode_slot_action,
    legal_joint_mask,
)
from showdown.rl.constants import MEGA_BASE, PASS_ID, SLOT_ACTION_N, SWITCH_BASE
from showdown.rl.damage_heuristic import estimate_damage_pct, move_entry
from showdown.rl.snapshot import BattleSnapshot, MonSnapshot

# Objetivos Spec: 0 rival a, 1 rival b, 2 aliado, 3 sin objetivo / self / campo.
_SINGLE_FOE = {'normal', 'adjacentfoe', 'any', 'randomnormal'}
_ALLY = {'adjacentally'}


def _norm(s: str) -> str:
    return ''.join(ch for ch in str(s or '').lower() if ch.isalnum())


@lru_cache(maxsize=4096)
def move_target_class(move_id: str) -> str:
    """'foe' (un rival a elegir), 'ally' o 'none' (área, self, campo)."""
    entry = move_entry(move_id)
    if not entry:
        return 'foe'
    tgt = _norm(entry.get('target', 'normal'))
    if tgt in _SINGLE_FOE:
        return 'foe'
    if tgt in _ALLY:
        return 'ally'
    return 'none'


def _alive(m: Optional[MonSnapshot]) -> bool:
    return m is not None and not m.fainted and float(m.hp_pct) > 0.0


def flip_view(snap: BattleSnapshot) -> BattleSnapshot:
    """Vista con los lados intercambiados SIN copiar Pokémon (solo lectura)."""
    f = snap.field
    field = dataclasses.replace(
        f,
        our_tailwind=f.opp_tailwind, opp_tailwind=f.our_tailwind,
        our_tailwind_turns=f.opp_tailwind_turns, opp_tailwind_turns=f.our_tailwind_turns,
        our_reflect=f.opp_reflect, opp_reflect=f.our_reflect,
        our_lightscreen=f.opp_lightscreen, opp_lightscreen=f.our_lightscreen,
        our_auroraveil=f.opp_auroraveil, opp_auroraveil=f.our_auroraveil,
    )
    won = None if snap.won is None else (not snap.won)
    return dataclasses.replace(
        snap, our=snap.foe, foe=snap.our, field=field, won=won,
        force_switch=(False, False), avail_moves=None, avail_switches=None,
    )


def slot_candidates(snap: BattleSnapshot, slot: int) -> List[int]:
    """Acciones con sentido de un hueco (sin mega, sin ``pass`` salvo que no quede nada)."""
    out: List[int] = []
    side = snap.our
    active = side.active[slot] if slot < len(side.active) else None
    if _alive(active):
        foes = [t for t in (0, 1) if t < len(snap.foe.active) and _alive(snap.foe.active[t])]
        if not foes:
            foes = [0]
        ally = side.active[1 - slot] if (1 - slot) < len(side.active) else None
        for mi, mv in enumerate(active.moves[:4]):
            if not mv:
                continue
            cls = move_target_class(mv)
            if cls == 'foe':
                targets = foes
            elif cls == 'ally':
                if not _alive(ally):
                    continue
                targets = [2]
            else:
                targets = [3]
            for t in targets:
                out.append(encode_slot_action(kind='move', move_index=mi, target=t))
    for bi, mon in enumerate(side.bench):
        if _alive(mon):
            out.append(SWITCH_BASE + bi)
    if not out:
        out.append(PASS_ID)
    return out


def pruned_joints(snap: BattleSnapshot) -> List[int]:
    """Joints podados desde la perspectiva ``our`` (para el rival pasar ``flip_view``)."""
    if any(snap.force_switch):
        mask = legal_joint_mask(snap)
        return [int(j) for j in np.flatnonzero(mask > 0.5)]
    c0 = slot_candidates(snap, 0)
    c1 = slot_candidates(snap, 1)
    out: List[int] = []
    for a1 in c0:
        for a2 in c1:
            if a1 >= SWITCH_BASE and a1 == a2 and a1 != PASS_ID:
                continue  # los dos huecos no pueden cambiar al mismo Pokémon
            out.append(encode_joint_action(a1, a2))
    return out


def prefer_mega(joints: Sequence[int]) -> List[int]:
    """Quita un joint si existe en el conjunto el mismo joint con mega en un hueco más.

    S2 no modela la mega (mismos tipos y daño), así que mega y no mega empatan y la
    decisión sería al azar. Regla: si se puede megaevolucionar con ese mismo movimiento,
    se megaevoluciona (cambios y joints con la mega ya usada se conservan).
    """
    js = {int(j) for j in joints}
    keep: List[int] = []
    for j in joints:
        a1, a2 = decode_joint_action(int(j))
        twins = []
        if a1 < MEGA_BASE:
            twins.append(encode_joint_action(a1 + MEGA_BASE, a2))
        if a2 < MEGA_BASE:
            twins.append(encode_joint_action(a1, a2 + MEGA_BASE))
        if any(t in js for t in twins):
            continue
        keep.append(int(j))
    return keep or [int(j) for j in joints]


def _strip_mega(aid: int) -> int:
    return aid - MEGA_BASE if MEGA_BASE <= aid < MEGA_BASE + 16 else aid


def slot_damage_table(snap: BattleSnapshot, slot: int) -> Dict[int, float]:
    """Daño medio estimado (S2, % HP) de cada acción de movimiento del hueco."""
    out: Dict[int, float] = {}
    side, opp = snap.our, snap.foe
    atk = side.active[slot] if slot < len(side.active) else None
    if not _alive(atk):
        return out
    weather = str(getattr(snap.field, 'weather', '') or '')
    for mi, mv in enumerate(atk.moves[:4]):
        if not mv:
            continue
        for t in (0, 1):
            dfn = opp.active[t] if t < len(opp.active) else None
            if not _alive(dfn):
                continue
            lo, hi = estimate_damage_pct(
                mv, attacker_types=atk.types, defender_types=dfn.types, weather=weather,
            )
            out[mi * 4 + t] = 0.5 * (float(lo) + float(hi))
        # Área / sin objetivo (S2 lo resuelve sobre el rival b): se puntúa igual que v1.
        if move_target_class(mv) == 'none':
            dfn = opp.active[1] if len(opp.active) > 1 and _alive(opp.active[1]) else None
            if dfn is not None:
                lo, hi = estimate_damage_pct(
                    mv, attacker_types=atk.types, defender_types=dfn.types, weather=weather,
                )
                out[mi * 4 + 3] = 0.5 * (float(lo) + float(hi))
    return out


def joint_scores(snap: BattleSnapshot, joints: Sequence[int]) -> np.ndarray:
    """Daño conjunto estimado de cada joint (misma idea que ``score_joint`` de v1)."""
    d0 = slot_damage_table(snap, 0)
    d1 = slot_damage_table(snap, 1)
    out = np.zeros(len(joints), dtype=np.float64)
    for k, j in enumerate(joints):
        a1, a2 = decode_joint_action(int(j))
        out[k] = d0.get(_strip_mega(a1), 0.0) + d1.get(_strip_mega(a2), 0.0)
    return out


def softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    t = max(1e-6, float(temperature))
    z = (np.asarray(x, dtype=np.float64) - np.max(x)) / t
    e = np.exp(z)
    return e / e.sum()


def prior_from_scores(scores: np.ndarray, temperature: float, eps: float) -> np.ndarray:
    """P = (1−ε)·softmax(daño/τ) + ε·uniforme: Protect/cambios conservan masa."""
    n = len(scores)
    if n == 0:
        return np.zeros(0)
    u = np.full(n, 1.0 / n)
    return (1.0 - eps) * softmax(scores, temperature) + eps * u


def top_k(joints: Sequence[int], scores: np.ndarray, k: int) -> Tuple[List[int], np.ndarray]:
    """Recorta a los ``k`` de mayor daño (estable: empate → id menor)."""
    if k <= 0 or len(joints) <= k:
        return list(joints), np.asarray(scores)
    order = sorted(range(len(joints)), key=lambda i: (-scores[i], joints[i]))[:k]
    order.sort()
    return [joints[i] for i in order], np.asarray([scores[i] for i in order])


def sample_opponent_joint(
    snap: BattleSnapshot,
    rng: np.random.Generator,
    *,
    temperature: float,
    eps: float,
) -> int:
    """Rival del árbol: softmax del daño con prob. 1−ε, uniforme (podado) con prob. ε."""
    view = flip_view(snap)
    joints = pruned_joints(view)
    if not joints:
        return encode_joint_action(PASS_ID, PASS_ID)
    if len(joints) == 1:
        return int(joints[0])
    if rng.random() < eps:
        return int(joints[int(rng.integers(len(joints)))])
    p = softmax(joint_scores(view, joints), temperature)
    return int(joints[int(rng.choice(len(joints), p=p))])


def greedy_joint(snap: BattleSnapshot) -> int:
    """Argmax de daño sobre joints podados (rollout de nuestro lado)."""
    joints = pruned_joints(snap)
    if not joints:
        return encode_joint_action(PASS_ID, PASS_ID)
    s = joint_scores(snap, joints)
    return int(joints[int(np.argmax(s))])


def joint_still_valid(snap: BattleSnapshot, joint: int) -> bool:
    """¿El joint sigue siendo jugable en este estado? (open-loop: estados distintos por sim)."""
    a1, a2 = decode_joint_action(int(joint))
    for slot, aid in ((0, a1), (1, a2)):
        act = decode_slot_action(int(aid))
        if act.kind == 'pass':
            continue
        if act.kind == 'switch':
            b = snap.our.bench
            if act.switch_index >= len(b) or not _alive(b[act.switch_index]):
                return False
            continue
        m = snap.our.active[slot] if slot < len(snap.our.active) else None
        if not _alive(m) or act.move_index >= len(m.moves):
            return False
    if a1 == a2 and SWITCH_BASE <= a1 < SWITCH_BASE + 4:
        return False
    return True


__all__ = [
    'flip_view', 'slot_candidates', 'pruned_joints', 'prefer_mega', 'joint_scores',
    'prior_from_scores', 'top_k', 'sample_opponent_joint', 'greedy_joint',
    'joint_still_valid', 'move_target_class', 'SLOT_ACTION_N',
]
