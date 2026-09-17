"""Utilidades VGC 1-ply para expectimax: Knock Off item, Fake Out deny, mega foe.

Greedy-joint NO importa esto (debe quedarse greedy puro).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from showdown.battle_bots.move_mechanics.charge import norm_id
from showdown.battle_bots.move_mechanics.entry_abilities import _WEATHER_SETTERS, _norm_weather

# Knock Off: valor de *quitar* item (turnos futuros). El +50% BP lo da el calc.
KNOCK_OFF_ITEM_BONUS = 55.0
# Floor mínimo si hay amenaza positiva (prioridad/utilidad).
FAKE_OUT_BASE = 25.0


def _item_id(item: Optional[str]) -> str:
    return norm_id(item)


def is_mega_stone(item: Optional[str]) -> bool:
    iid = _item_id(item)
    if not iid or iid == 'eviolite':
        return False
    # charizarditey / salamencite / …
    return 'ite' in iid


def is_knock_off_removable(item: Optional[str]) -> bool:
    """False si no hay item o no se puede Knock Off (mega piedra, etc.)."""
    iid = _item_id(item)
    if not iid:
        return False
    if is_mega_stone(item):
        return False
    if iid in (
        'redorb', 'blueorb',
        'rustedsword', 'rustedshield',
        'boosterenergy',
    ):
        return False
    if iid.endswith('iumz') or iid.endswith('iz'):
        return False
    # Memories / plates / drives: Knock Off no los quita en PS genéricas.
    if iid.endswith('memory') or iid.endswith('plate') or iid.endswith('drive'):
        return False
    return True


def knock_off_item_bonus(defender: Optional[Any]) -> float:
    """Bonus por quitar item (no es el +50% BP; eso ya está en el rango de daño)."""
    if defender is None:
        return 0.0
    item = getattr(defender, 'item', None)
    if not is_knock_off_removable(item):
        return 0.0
    return float(KNOCK_OFF_ITEM_BONUS)


def fake_out_deny_from_threat(
    *,
    attacker: Optional[Any],
    threat_denied: float,
    fails: bool = False,
) -> float:
    """
    Deny = amenaza real del foe flincheado este turno.

    - fails (Psychic Terrain / no first-turn / Ghost…): 0
    - amenaza baja → deny bajo → full attack gana
    - amenaza alta (Heat Wave KO-ish) → deny ~ soft_KO
    """
    if fails:
        return 0.0
    if attacker is not None and not bool(getattr(attacker, 'just_switched', False)):
        return 0.0
    t = max(0.0, float(threat_denied))
    if t <= 1e-9:
        return 0.0
    return max(float(FAKE_OUT_BASE), t)


def best_fake_out_target_by_threat(
    attacker: Optional[Any],
    foes: Sequence[Tuple[int, Any]],
    threat_by_tid: Dict[int, float],
    *,
    can_affect=None,
    fails_vs=None,
) -> Tuple[int, float]:
    """Elige tid con mayor amenaza denegable."""
    best_tid = 0
    best = -1.0
    for tid, foe in foes:
        if can_affect is not None and not can_affect(attacker, foe):
            continue
        if fails_vs is not None and fails_vs(foe):
            continue
        sc = fake_out_deny_from_threat(
            attacker=attacker,
            threat_denied=float(threat_by_tid.get(int(tid), 0.0)),
        )
        if sc > best:
            best = sc
            best_tid = int(tid)
    if best_tid == 0 and foes:
        best_tid = int(foes[0][0])
        best = 0.0
    return best_tid, max(0.0, float(best))


def project_foe_active_forme(foe: Any) -> Any:
    """Si lleva megapiedra, proyecta forme mega (stats + ability) para este turno."""
    if foe is None:
        return foe
    item = getattr(foe, 'item', None)
    if not is_mega_stone(item):
        return foe
    species = str(getattr(foe, 'species', '') or '')
    if 'mega' in species.lower():
        return foe
    try:
        from showdown.battle_bots.greedy_solo.greedy import apply_pending_mega
        return apply_pending_mega(foe)
    except Exception:
        return foe


def weather_from_projected_foes(
    foes: Sequence[Any],
    *,
    current_weather: Optional[str] = None,
) -> Optional[str]:
    """Si un foe proyectado setea clima (Drought…), úsalo en calcs del turno."""
    cur = _norm_weather(current_weather) if current_weather else ''
    for foe in foes:
        if foe is None:
            continue
        abi = norm_id(getattr(foe, 'ability', None))
        w = _WEATHER_SETTERS.get(abi)
        if not w:
            continue
        w = _norm_weather(w)
        if w and w != cur:
            return w
    return current_weather


def pair_vgc_utility_extra(
    left: Any,
    right: Any,
    *,
    left_atk: Optional[Any],
    right_atk: Optional[Any],
    foes: Sequence[Tuple[int, Any]],
    threat_by_tid: Optional[Dict[int, float]] = None,
    fakeout_fails_by_tid: Optional[Dict[int, bool]] = None,
) -> float:
    """Utilidad Knock Off / Fake Out del par (solo expectimax V)."""
    threat_by_tid = threat_by_tid or {}
    fails = fakeout_fails_by_tid or {}
    foe_by_tid = {int(tid): f for tid, f in foes}
    total = 0.0
    for atom, atk in ((left, left_atk), (right, right_atk)):
        if atom is None:
            continue
        mid = norm_id(getattr(atom, 'move_name', None))
        tid = int(getattr(atom, 'target', 0) or 0)
        foe = foe_by_tid.get(tid)
        if mid == 'knockoff':
            total += knock_off_item_bonus(foe)
        elif mid == 'fakeout' or getattr(atom, 'utility', '') == 'fakeout':
            if foe is None and foes:
                tid = int(getattr(atom, 'target', 0) or foes[0][0])
                foe = foe_by_tid.get(tid) or foes[0][1]
            total += fake_out_deny_from_threat(
                attacker=atk,
                threat_denied=float(threat_by_tid.get(tid, 0.0)),
                fails=bool(fails.get(tid, False)),
            )
    return float(total)


# Compat tests / exports antiguos
def fake_out_deny_bonus(
    *,
    attacker: Optional[Any],
    defender: Optional[Any] = None,
    our_side: Sequence[Any] = (),
    foe_known_moves: Sequence[str] = (),
    threat_denied: float = 0.0,
    fails: bool = False,
) -> float:
    """API legacy: preferir threat_denied; sin amenaza → 0 (salvo first-turn floor no)."""
    del defender, our_side, foe_known_moves  # señales heurísticas retiradas
    return fake_out_deny_from_threat(
        attacker=attacker, threat_denied=threat_denied, fails=fails,
    )


def best_fake_out_target(
    attacker: Optional[Any],
    foes: Sequence[Tuple[int, Any]],
    *,
    our_side: Sequence[Any] = (),
    foe_known_moves: Optional[dict] = None,
    can_affect=None,
    threat_by_tid: Optional[Dict[int, float]] = None,
) -> Tuple[int, float]:
    del our_side, foe_known_moves
    return best_fake_out_target_by_threat(
        attacker, foes, threat_by_tid or {}, can_affect=can_affect,
    )
