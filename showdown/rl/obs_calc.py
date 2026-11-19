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


def requests_for_snapshot(snap: BattleSnapshot, profile_fn) -> Tuple[List[Dict[str, Any]], List[tuple]]:
    """Todos los pedidos de una observación y su clave (lado, hueco, movimiento, objetivo, modo).

    ``profile_fn(mon)`` devuelve el perfil (``observation._profile``). Nuestros movimientos contra cada
    rival en los 3 modos y contra el aliado (área); los del rival contra cada uno de los nuestros (mid,
    max). Es el conjunto que hoy llena la observación.
    """
    reqs: List[Dict[str, Any]] = []
    keys: List[tuple] = []
    ours = list(snap.our.active[:2])
    foes = list(snap.foe.active[:2])
    actives = [(m, profile_fn(m)) for m in ours + foes]
    on_field = _abilities_on_field(actives)
    our_f, foe_f = fainted_count(snap.our), fainted_count(snap.foe)
    table = P._dex()['moves']
    for side, atk_list, def_list, att_ours in (('our', ours, foes, True), ('foe', foes, ours, False)):
        for s, mon in enumerate(atk_list):
            prof = profile_fn(mon)
            if mon is None or prof is None or mon.fainted:
                continue
            ally = atk_list[1 - s]
            ally_p = profile_fn(ally) if ally is not None else None
            for raw in list(mon.moves)[:4]:
                mid = P.to_id(raw)
                m = table.get(mid) or {}
                if m.get('category') not in ('Physical', 'Special'):
                    continue
                spread = m.get('target') in ('allAdjacentFoes', 'allAdjacent')
                modes = ('mid', 'max', 'min') if att_ours else ('mid', 'max')
                for ti, dfn in enumerate(def_list):
                    dp = profile_fn(dfn)
                    if dfn is None or dp is None or dfn.fainted:
                        continue
                    d_ally = def_list[1 - ti]
                    for mode in modes:
                        reqs.append(build_request(
                            mon, prof, dfn, dp, mid, snap.field, attacker_ours=att_ours, att_ally_prof=ally_p,
                            def_ally_prof=profile_fn(d_ally) if d_ally is not None else None, on_field=on_field,
                            is_spread=spread, mode=mode, allies_fainted=our_f if att_ours else foe_f))
                        keys.append((side, s, mid, ti, mode))
                if att_ours and m.get('target') == 'allAdjacent' and ally is not None and ally_p is not None \
                        and not ally.fainted:
                    reqs.append(build_request(
                        mon, prof, ally, ally_p, mid, snap.field, attacker_ours=False, att_ally_prof=None,
                        def_ally_prof=None, on_field=on_field, is_spread=True, mode='mid', allies_fainted=our_f))
                    keys.append((side, s, mid, 'ally', 'mid'))
    return reqs, keys
