"""Status / secundarios para scoring 1-ply (Sleep Powder, Dire Claw, …).

No simula RNG completo: E[valor] = p_hit * Σ p_effect * VALUE[effect],
con gates de inmunidad (powder, Limber, ya statused, Substitute, terrain…).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from showdown.battle_bots.greedy_solo.type_chart import normalize_type
from showdown.battle_bots.move_mechanics.accuracy import move_accuracy
from showdown.battle_bots.move_mechanics.charge import norm_id

# Valores en “unidades de %HP” del score greedy/joint (mismo orden de magnitud).
# Valor de score (chip/%HP) al aplicar el efecto.
STATUS_VALUE: Dict[str, float] = {
    'slp': 55.0,
    'par': 32.0,
    'brn': 22.0,
    'tox': 20.0,
    'psn': 10.0,
    'frz': 40.0,
    'flinch': 28.0,
}

# Alivio de amenaza en V(S'): cuánto baja la presión del rival si queda statused.
# Misma magnitud que threat/~soft_ko parcial (no 1000).
THREAT_RELIEF: Dict[str, float] = {
    'slp': 78.0,   # slot casi apagado el turno siguiente
    'par': 42.0,
    'frz': 70.0,
    'flinch': 34.0,
    'brn': 18.0,
    'tox': 16.0,
    'psn': 9.0,
}

# Drop de stats secundario (Shadow Ball -SpD, etc.): por stage negativo al rival.
BOOST_DROP_UNIT = 6.0

_MAJOR = frozenset({'slp', 'par', 'brn', 'psn', 'tox', 'frz'})

_POWDER_IMMUNE_ABILITIES = frozenset({
    'overcoat', 'immunity', 'pastelveil',  # Pastel Veil: ally psn; overcoat powder
})
# Overcoat / Grass / Safety Goggles vs powder — Pastel Veil no bloquea powder.

_STATUS_IMMUNE_ABILITY: Dict[str, frozenset] = {
    'par': frozenset({'limber', 'comatose'}),
    'slp': frozenset({'insomnia', 'vitalspirit', 'sweetveil', 'comatose'}),
    'brn': frozenset({'waterbubble', 'waterveil', 'comatose'}),
    'psn': frozenset({'immunity', 'pastelveil', 'comatose'}),
    'tox': frozenset({'immunity', 'pastelveil', 'comatose'}),
    'frz': frozenset({'magmaarmor', 'comatose'}),
}

# Moves especiales no bien tipados en moves.json / Champions.
_MANUAL_EFFECTS: Dict[str, List[Dict[str, Any]]] = {
    # Dire Claw: 50% de aplicar uno de {psn, par, slp} al azar.
    'direclaw': [
        {'chance': 50 / 3, 'status': 'psn'},
        {'chance': 50 / 3, 'status': 'par'},
        {'chance': 50 / 3, 'status': 'slp'},
    ],
}


def _moves_json() -> Dict[str, Any]:
    try:
        from showdown.battle_modifier import _moves_json as _load
        return _load() or {}
    except Exception:
        return {}


@lru_cache(maxsize=512)
def move_entry(move_name: Optional[str]) -> Dict[str, Any]:
    mid = norm_id(move_name)
    if not mid:
        return {}
    return dict(_moves_json().get(mid) or {})


def _poke_types(poke) -> Tuple[str, ...]:
    if poke is None:
        return ()
    if bool(getattr(poke, 'is_terastallized', False)) and getattr(poke, 'tera_type', None):
        t = normalize_type(getattr(poke, 'tera_type'))
        return (t,) if t else ()
    out = []
    for t in tuple(getattr(poke, 'types', None) or ()):
        nt = normalize_type(t)
        if nt:
            out.append(nt)
    return tuple(out)


def _abi(poke) -> str:
    return norm_id(getattr(poke, 'ability', None) if poke is not None else None)


def _item(poke) -> str:
    return norm_id(getattr(poke, 'item', None) if poke is not None else None)


def _current_status(poke) -> str:
    if poke is None:
        return ''
    raw = str(getattr(poke, 'status', '') or '').strip().lower()
    if not raw or raw in ('none', 'empty'):
        return ''
    # poke-env enums / 'par' / 'paralysis'
    for key in _MAJOR:
        if raw == key or raw.startswith(key):
            return key
    if 'paral' in raw:
        return 'par'
    if 'sleep' in raw or raw == 'slp':
        return 'slp'
    if 'burn' in raw:
        return 'brn'
    if 'poison' in raw or raw == 'tox':
        return 'tox' if 'tox' in raw or raw == 'tox' else 'psn'
    if 'freez' in raw or 'frost' in raw:
        return 'frz'
    return raw[:3] if raw else ''


def _is_powder_move(move_name: Optional[str], entry: Optional[Dict[str, Any]] = None) -> bool:
    mid = norm_id(move_name)
    if mid in ('ragepowder', 'sleeppowder', 'stunspore', 'spore', 'poisonpowder', 'cottonspore'):
        return True
    flags = (entry or move_entry(move_name)).get('flags') or {}
    return bool(flags.get('powder'))


def status_can_affect(
    effect: str,
    *,
    attacker=None,
    defender=None,
    terrain: Optional[str] = None,
    misty_terrain_blocks: bool = True,
) -> bool:
    """True si el efecto de status/volatile puede aplicar al defensor."""
    eff = norm_id(effect)
    if not eff or defender is None:
        return False

    if bool(getattr(defender, 'has_substitute', False)) and eff in _MAJOR:
        # Status vía powder/reflectable suele fallar vs Sub; contact secondaries también.
        return False

    if eff in _MAJOR and _current_status(defender):
        return False

    types = _poke_types(defender)
    abi = _abi(defender)
    item = _item(defender)

    if eff == 'par' and 'Electric' in types:
        return False
    if eff in ('psn', 'tox') and ('Poison' in types or 'Steel' in types):
        return False
    if eff == 'brn' and 'Fire' in types:
        return False
    if eff == 'frz' and 'Ice' in types:
        return False

    immune_abis = _STATUS_IMMUNE_ABILITY.get(eff, frozenset())
    if abi in immune_abis:
        return False
    if eff == 'slp' and abi in ('insomnia', 'vitalspirit', 'comatose'):
        return False

    terr = norm_id(terrain)
    if misty_terrain_blocks and terr == 'mistyterrain' and eff in _MAJOR:
        # Aprox.: grounded (sin full Magnet Rise). Flying / balloon / levitate escapan.
        if 'Flying' not in types and item != 'airballoon' and abi != 'levitate':
            return False
    if terr == 'electricterrain' and eff == 'slp':
        if 'Flying' not in types and item != 'airballoon' and abi != 'levitate':
            return False

    return True


def powder_can_affect(defender, move_name: Optional[str] = None) -> bool:
    if defender is None:
        return False
    if 'Grass' in _poke_types(defender):
        return False
    if _abi(defender) == 'overcoat':
        return False
    if _item(defender) in ('safetygoggles',):
        return False
    return True


def _effect_specs(move_name: Optional[str]) -> List[Dict[str, Any]]:
    mid = norm_id(move_name)
    if mid in _MANUAL_EFFECTS:
        return list(_MANUAL_EFFECTS[mid])
    entry = move_entry(move_name)
    specs: List[Dict[str, Any]] = []
    primary = entry.get('status')
    if primary:
        # Status move puro: chance 100% si acierta.
        specs.append({'chance': 100, 'status': str(primary)})
    sec = entry.get('secondary')
    if isinstance(sec, dict) and sec:
        specs.append(dict(sec))
    secs = entry.get('secondaries')
    if isinstance(secs, list):
        for s in secs:
            if isinstance(s, dict):
                specs.append(dict(s))
    return specs


def _value_for_spec(
    spec: Dict[str, Any],
    *,
    attacker=None,
    defender=None,
    terrain: Optional[str] = None,
    we_outspeed: Optional[bool] = None,
    value_table: Optional[Dict[str, float]] = None,
) -> float:
    chance = float(spec.get('chance') or 100) / 100.0
    if chance <= 0:
        return 0.0
    table = value_table or STATUS_VALUE

    total = 0.0
    st = spec.get('status')
    if st:
        st_id = norm_id(st)
        if status_can_affect(st_id, attacker=attacker, defender=defender, terrain=terrain):
            total += table.get(st_id, 0.0)

    vol = spec.get('volatileStatus') or spec.get('volatile')
    if vol and norm_id(vol) == 'flinch':
        if we_outspeed is False:
            flinch_p = 0.0
        elif we_outspeed is True:
            flinch_p = 1.0
        else:
            flinch_p = 0.45  # incertidumbre de Spe
        total += table.get('flinch', 0.0) * flinch_p

    boosts = spec.get('boosts') or {}
    if isinstance(boosts, dict) and value_table is None:
        for _stat, stage in boosts.items():
            try:
                iv = int(stage)
            except (TypeError, ValueError):
                continue
            if iv < 0:
                total += BOOST_DROP_UNIT * abs(iv)

    return chance * total


@dataclass(frozen=True)
class EffectScore:
    move_id: str
    score: float
    accuracy: float
    is_primary_status: bool


def expected_effect_score(
    move_name: Optional[str],
    *,
    attacker=None,
    defender=None,
    terrain: Optional[str] = None,
    we_outspeed: Optional[bool] = None,
    skip_if_fakeout_flinch: bool = True,
) -> EffectScore:
    """E[secundarios/status] * accuracy. 0 si no aplica."""
    mid = norm_id(move_name)
    acc = move_accuracy(move_name)
    if not mid or defender is None:
        return EffectScore(mid, 0.0, acc, False)

    entry = move_entry(move_name)
    if _is_powder_move(move_name, entry) and not powder_can_affect(defender, move_name):
        return EffectScore(mid, 0.0, acc, False)

    # Fake Out ya tiene bonus dedicado en joint; no doble-contar flinch.
    if skip_if_fakeout_flinch and mid == 'fakeout':
        return EffectScore(mid, 0.0, acc, False)

    specs = _effect_specs(move_name)
    if not specs:
        return EffectScore(mid, 0.0, acc, bool(entry.get('status')))

    raw = 0.0
    for spec in specs:
        raw += _value_for_spec(
            spec,
            attacker=attacker,
            defender=defender,
            terrain=terrain,
            we_outspeed=we_outspeed,
        )
    return EffectScore(
        mid,
        float(raw) * acc,
        acc,
        is_primary_status=bool(entry.get('status')) and not entry.get('basePower'),
    )


def expected_threat_relief(
    move_name: Optional[str],
    *,
    attacker=None,
    defender=None,
    terrain: Optional[str] = None,
    we_outspeed: Optional[bool] = None,
) -> float:
    """E[alivio de amenaza en V] si el move aplica status/flinch al defensor."""
    mid = norm_id(move_name)
    acc = move_accuracy(move_name)
    if not mid or defender is None:
        return 0.0
    entry = move_entry(move_name)
    if _is_powder_move(move_name, entry) and not powder_can_affect(defender, move_name):
        return 0.0
    # Fake Out: el flinch ya se valora como utilidad dedicada; no doblar en threat relief.
    if mid == 'fakeout':
        return 0.0
    raw = 0.0
    for spec in _effect_specs(move_name):
        raw += _value_for_spec(
            spec,
            attacker=attacker,
            defender=defender,
            terrain=terrain,
            we_outspeed=we_outspeed,
            value_table=THREAT_RELIEF,
        )
    return float(raw) * acc


def best_status_target_score(
    move_name: Optional[str],
    *,
    attacker=None,
    foes: Sequence[Tuple[int, Any]] = (),
    terrain: Optional[str] = None,
) -> Tuple[int, float]:
    """Mejor (tid, score) para un status/secundario single-target."""
    best_tid = 0
    best = 0.0
    for tid, foe in foes:
        sc = expected_effect_score(
            move_name, attacker=attacker, defender=foe, terrain=terrain,
        ).score
        if sc > best:
            best = sc
            best_tid = int(tid)
    return best_tid, best
