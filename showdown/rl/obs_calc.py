"""Pedidos a la calculadora real de daño para la observación (P5 completo; rl_env_spec v1.33, borrador).

Decisión del usuario (memorizacion_equipos.md §18.1): el daño de la observación de DDQN y PPO sale de la
misma calculadora que usan los heurísticos (``@smogon/calc`` Champions vía ``damage_calc_bridge``), con
los efectos de campo como en la realidad. Este módulo solo ARMA los pedidos y convierte las respuestas;
cómo y cuándo se llaman (lote por observación, caché) depende de la medición de velocidad.

Equivalencias con la aproximación anterior (``observation.move_damage_frac``):
- ``mode='mid'``: stats medios de cada lado → (mínimo + máximo de la tirada) / 2 × precisión.
- ``mode='max'``: atacante con su stat máximo, defensor con el mínimo → tirada máxima.
- ``mode='min'``: atacante mínimo, defensor máximo → tirada mínima.
Nuestros stats son exactos (smin = smax); los del rival son un rango (SP ocultos en la hoja abierta).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from showdown.rl import team_preview_obs as P
from showdown.rl.snapshot import BattleSnapshot, FieldSnapshot, MonSnapshot

STAT_KEYS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
_RUIN = {'swordofruin': 'isSwordOfRuin', 'beadsofruin': 'isBeadsOfRuin',
         'tabletsofruin': 'isTabletsOfRuin', 'vesselofruin': 'isVesselOfRuin'}


def calc_species_name(mon: MonSnapshot, prof) -> str:
    """Nombre de la forma efectiva (la Mega si lleva su piedra, igual que el perfil de la obs)."""
    ent = P.species_entry(mon.ots_species or mon.species)
    if ent is None:
        return mon.species
    sid, sp = ent
    item = P.to_id(getattr(prof, 'item', '') or mon.item)
    mid = P.mega_species_id(sid, item) if item else None
    if mid and mid in P._dex()['species']:
        return P._dex()['species'][mid].get('name') or mon.species
    return sp.get('name') or mon.species


def _stats(prof, mode: str, *, attacker: bool) -> Dict[str, int]:
    out = {}
    for k in STAT_KEYS:
        lo, hi = float(prof.smin[k]), float(prof.smax[k])
        if mode == 'mid':
            v = 0.5 * (lo + hi)
        else:
            high = (mode == 'max') == attacker  # max: atacante alto / defensor bajo
            v = hi if high else lo
        out[k] = max(1, int(round(v)))
    return out


def mon_payload(mon: MonSnapshot, prof, mode: str, *, attacker: bool, allies_fainted: int = 0) -> Dict[str, Any]:
    from showdown.stat_inference import _poke_payload
    stats = _stats(prof, mode, attacker=attacker)
    cur = None
    if mon.hp_pct is not None and float(mon.hp_pct) < 100.0:
        cur = max(1, int(round(stats['hp'] * float(mon.hp_pct) / 100.0)))
    status = str(mon.status or '').lower()
    return _poke_payload(
        calc_species_name(mon, prof), getattr(prof, 'nature', None) or mon.nature or None, stats,
        item=getattr(prof, 'item', '') or mon.item or None, ability=getattr(prof, 'ability', '') or mon.ability or None,
        status=status if status in ('brn', 'par', 'psn', 'tox', 'slp', 'frz') else '',
        boosts=dict(mon.boosts or {}), cur_hp=cur, types=list(prof.types) if prof.types else None,
        allies_fainted=int(allies_fainted),
    )


def _abilities_on_field(actives: Sequence[Tuple[Optional[MonSnapshot], Any]]) -> set:
    return {getattr(p, 'ability', '') for m, p in actives if m is not None and p is not None and not m.fainted}


def field_payload(field: FieldSnapshot, *, attacker_ours: bool, att_ally_prof, def_ally_prof,
                  on_field: set, is_spread: bool) -> Dict[str, Any]:
    from showdown.stat_inference import _terrain_for_calc, _weather_for_calc
    atk_side: Dict[str, bool] = {}
    ally_ab = getattr(att_ally_prof, 'ability', '') if att_ally_prof is not None else ''
    if ally_ab == 'battery':
        atk_side['isBattery'] = True
    if ally_ab == 'powerspot':
        atk_side['isPowerSpot'] = True
    if ally_ab == 'steelyspirit':
        atk_side['isSteelySpirit'] = True
    def_side: Dict[str, bool] = {}
    if attacker_ours:  # el defensor es el rival
        veil, refl, ls = field.opp_auroraveil, field.opp_reflect, field.opp_lightscreen
    else:
        veil, refl, ls = field.our_auroraveil, field.our_reflect, field.our_lightscreen
    if veil:
        def_side['isAuroraVeil'] = True
    else:
        if refl:
            def_side['isReflect'] = True
        if ls:
            def_side['isLightScreen'] = True
    if def_ally_prof is not None and getattr(def_ally_prof, 'ability', '') == 'friendguard':
        def_side['isFriendGuard'] = True
    out = {
        'gameType': 'Doubles',
        'weather': _weather_for_calc(field.weather),
        'terrain': _terrain_for_calc(field.terrain),
        'isGravity': bool(getattr(field, 'gravity', False)),
        'isSpread': bool(is_spread),
        'attackerSide': atk_side,
        'defenderSide': def_side,
        # Auras de campo como en la realidad: cualquiera de los 4 en campo (memorizacion §18.1).
        'isFairyAura': 'fairyaura' in on_field,
        'isDarkAura': 'darkaura' in on_field,
        'isAuraBreak': 'aurabreak' in on_field,
    }
    for ab, key in _RUIN.items():
        out[key] = ab in on_field
    return out


def build_request(att: MonSnapshot, att_p, dfn: MonSnapshot, def_p, move_id: str, field: FieldSnapshot, *,
                  attacker_ours: bool, att_ally_prof, def_ally_prof, on_field: set, is_spread: bool,
                  mode: str, allies_fainted: int = 0) -> Dict[str, Any]:
    m = P._dex()['moves'].get(move_id, {}) or {}
    return {
        'attacker': mon_payload(att, att_p, mode, attacker=True, allies_fainted=allies_fainted),
        'defender': mon_payload(dfn, def_p, mode, attacker=False),
        'move': m.get('name') or move_id,
        'field': field_payload(field, attacker_ours=attacker_ours, att_ally_prof=att_ally_prof,
                               def_ally_prof=def_ally_prof, on_field=on_field, is_spread=is_spread),
    }


def damage_frac(rng: Optional[Tuple[int, int]], def_p, mode: str, accuracy: float) -> Optional[float]:
    """(mín, máx) de la calculadora → fracción del HP máximo del defensor, como ``move_damage_frac``."""
    if rng is None:
        return None
    lo, hi = float(rng[0]), float(rng[1])
    hp = max(1.0, float(_stats(def_p, mode, attacker=False)['hp']))
    if mode == 'mid':
        return 0.5 * (lo + hi) * float(accuracy) / hp
    return (hi if mode == 'max' else lo) / hp


def fainted_count(side) -> int:
    mons = list(side.active) + list(side.bench) + list(getattr(side, 'reserve', []) or [])
    seen, n = set(), 0
    for m in mons:
        if m is None or id(m) in seen:
            continue
        seen.add(id(m))
        n += 1 if m.fainted else 0
    return n


def requests_for_snapshot(snap: BattleSnapshot, profile_fn, our_slots: Sequence[Optional[MonSnapshot]],
                          foe_slots: Sequence[Optional[MonSnapshot]]) -> Dict[tuple, tuple]:
    """Todos los cálculos que usa una observación, sin repetir: ``clave → (pedido, perfil defensor, modo, precisión)``.

    Clave: ``(id(atacante), id(defensor), movimiento, modo)``, la misma que consulta
    ``observation.move_damage_frac``. Cubre: nuestros movimientos contra cada rival activo (3 modos) y
    contra el aliado (área); los del rival activo contra cada uno de los nuestros (medio y máximo); y la
    matriz de pares (``_pair_block``: 4 nuestros × 6 rivales en ambos sentidos, modo medio).
    ``profile_fn(mon)`` es ``observation._profile``.
    """
    out: Dict[tuple, tuple] = {}
    actives_ours = list(snap.our.active[:2])
    actives_foe = list(snap.foe.active[:2])
    on_field = _abilities_on_field([(m, profile_fn(m)) for m in actives_ours + actives_foe])
    fainted = {True: fainted_count(snap.our), False: fainted_count(snap.foe)}
    table = P._dex()['moves']

    def ally_of(mon, actives):
        if mon is None:
            return None
        for i, m in enumerate(actives):
            if m is mon:
                other = actives[1 - i]
                return profile_fn(other) if other is not None and not other.fainted else None
        return None

    def add(att, dfn, mv, mode, att_ours, *, spread_target=None):
        ap, dp = profile_fn(att), profile_fn(dfn)
        if att is None or dfn is None or ap is None or dp is None or att.fainted or dfn.fainted:
            return
        key = (id(att), id(dfn), mv.mid, mode)
        if key in out:
            return
        m = table.get(mv.mid) or {}
        spread = m.get('target') in ('allAdjacentFoes', 'allAdjacent') if spread_target is None else spread_target
        a_act = actives_ours if att_ours else actives_foe
        d_act = actives_foe if att_ours else actives_ours
        req = build_request(att, ap, dfn, dp, mv.mid, snap.field, attacker_ours=att_ours,
                            att_ally_prof=ally_of(att, a_act), def_ally_prof=ally_of(dfn, d_act),
                            on_field=on_field, is_spread=bool(spread), mode=mode,
                            allies_fainted=fainted[att_ours])
        out[key] = (req, dp, mode, float(getattr(mv, 'accuracy', 1.0) or 1.0))

    # bloques de movimientos (activos)
    for s_, mon in enumerate(actives_ours):
        prof = profile_fn(mon)
        if mon is None or prof is None or mon.fainted:
            continue
        for mv in prof.damaging:
            for dfn in actives_foe:
                for mode in ('mid', 'max', 'min'):
                    add(mon, dfn, mv, mode, True)
            ally = actives_ours[1 - s_]
            if (table.get(mv.mid) or {}).get('target') == 'allAdjacent' and ally is not None:
                ap = profile_fn(ally)
                if ap is not None and not ally.fainted:
                    key = (id(mon), id(ally), mv.mid, 'mid')
                    if key not in out:
                        req = build_request(mon, prof, ally, ap, mv.mid, snap.field, attacker_ours=False,
                                            att_ally_prof=None, def_ally_prof=None, on_field=on_field,
                                            is_spread=True, mode='mid', allies_fainted=fainted[True])
                        out[key] = (req, ap, 'mid', float(getattr(mv, 'accuracy', 1.0) or 1.0))
    for mon in actives_foe:
        prof = profile_fn(mon)
        if mon is None or prof is None or mon.fainted:
            continue
        for mv in prof.damaging:
            for dfn in actives_ours:
                for mode in ('mid', 'max'):
                    add(mon, dfn, mv, mode, False)
    # matriz de pares (también banca)
    for a in our_slots:
        pa = profile_fn(a)
        for b in foe_slots:
            pb = profile_fn(b)
            if pa is not None:
                for mv in pa.damaging:
                    add(a, b, mv, 'mid', True)
            if pb is not None:
                for mv in pb.damaging:
                    add(b, a, mv, 'mid', False)
    return out


def damage_table(reqs: Dict[tuple, tuple], ranges_fn=None) -> Tuple[Dict[tuple, float], int]:
    """Resuelve los pedidos en UN lote. Devuelve (clave → fracción del HP, nº de pedidos sin respuesta)."""
    if not reqs:
        return {}, 0
    if ranges_fn is None:
        from showdown.damage_calc_bridge import damage_ranges_batch as ranges_fn
    keys = list(reqs)
    res = ranges_fn([reqs[k][0] for k in keys])
    table: Dict[tuple, float] = {}
    missing = 0
    for k, r in zip(keys, res):
        _req, dp, mode, acc = reqs[k]
        v = damage_frac(r, dp, mode, acc)
        if v is None:
            missing += 1
        else:
            table[k] = v
    return table, missing


_real_damage_table = damage_table  # referencia estable (tests)
