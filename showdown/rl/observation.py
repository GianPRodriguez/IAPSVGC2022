"""ObsSpec v2 — observación de combate con OTS completo + estado (docs/rl_env_spec.md v1.18).

Principio: la red recibe **toda** la información disponible al decidir, en forma
aprendible (one-hot / magnitudes normalizadas), sin hashes ni segmentos en cero.

Layout (``constants.OBS_DIM``)::

  10 bloques de Pokémon            nuestros a, b, banca0, banca1 (orden = índice de cambio)
                                   rival a, b, reserva0..3 (vistos primero, luego por especie)
      cada bloque = OTS (464, mismo encoder que el Team Preview) + estado (40)
  8 × 72  movimientos de nuestros activos, en el orden de ``available_moves``
          (= índice de movimiento de la acción): semántica, disponibilidad, PP,
          daño esperado vs rival a / rival b / aliado, KO, efectividad
  8 × 39  movimientos de la hoja de los activos rivales: tipo, categoría, poder,
          prioridad, spread, utilidades clave, daño y KO sobre nuestros activos
  72      matrices nuestros(4) × rival(6): daño us→foe y foe→us (fracción del HP
          actual) y velocidad efectiva (Tailwind, Trick Room, parálisis, boosts,
          Scarf, habilidades de clima)
  48      campo: clima, terreno, Trick Room, Tailwind, pantallas (+ turnos),
          gravedad, turno, megas usadas, Pokémon vivos/vistos, HP total, cambios forzados

Daño: fórmula nivel 50 con stats (exactos propios; rango medio del rival), STAB,
tipos, inmunidades por habilidad, items, boosts, clima, terreno, pantallas y
quemadura. Es una aproximación (sin críticos ni cálculos de habilidades raras).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from showdown.rl.constants import (
    FIELD_DIM,
    FOE_MOVE_DIM,
    MON_BLOCK_DIM,
    MON_DYN_DIM,
    MOVE_SLOT_DIM,
    N_FOE_MONS,
    N_OUR_MONS,
    OBS_DIM,
    OTS_MON_DIM,
    PAIR_DIM,
)
from showdown.rl.snapshot import BattleSnapshot, FieldSnapshot, MonSnapshot, SideSnapshot
from showdown.rl import team_preview_obs as P

_STATUS_IDS = ('brn', 'par', 'psn', 'tox', 'slp', 'frz')
_VOL_FLAGS = ('substitute', 'confusion', 'taunt', 'encore', 'disable', 'yawn', 'leechseed',
              'saltcure', 'healblock', 'throatchop', 'torment', 'flashfire', 'focusenergy')
_TARGET_CLASSES = ('normal', 'self', 'adjacentAlly', 'allySide', 'foeSide', 'all',
                   'allAdjacent', 'allAdjacentFoes')
_FOE_UTIL = ('fakeout', 'protect', 'tailwind', 'trickroom', 'redirect', 'speed_drop', 'sleep',
             'wideguard', 'helpinghand', 'offense_drop')
_UIDX = {k: i for i, k in enumerate(P.MOVE_UTILITY_FLAGS)}


# --------------------------------------------------------------------------- #
# Pokémon: OTS estático + estado
# --------------------------------------------------------------------------- #

def _feat(mon: MonSnapshot) -> P.PreviewMonFeat:
    return P.PreviewMonFeat(
        species=mon.display_species() or mon.species,
        item=mon.item or None,
        ability=mon.ability or None,
        nature=mon.nature or None,
        moves=tuple(mon.moves or ()),
        stats=dict(mon.stats) if mon.stats else None,
        sp_bounds=dict(mon.sp_bounds) if (mon.sp_bounds and not mon.stats) else None,
    )


def _profile(mon: Optional[MonSnapshot]):
    if mon is None:
        return None
    prof = P.mon_profile(_feat(mon))
    return prof if prof is not None and prof.present else None


def _dyn_block(mon: Optional[MonSnapshot], *, can_mega: bool) -> np.ndarray:
    v = np.zeros(MON_DYN_DIM, dtype=np.float32)
    if mon is None:
        return v
    v[0] = 1.0
    v[1] = 1.0 if mon.is_active else 0.0
    v[2] = 1.0 if (mon.fainted or mon.hp_pct <= 0.0) else 0.0
    v[3] = 1.0 if mon.seen else 0.0
    v[4] = float(mon.hp_pct) / 100.0
    st = str(mon.status or '').lower()
    for i, s in enumerate(_STATUS_IDS):
        if st == s:
            v[5 + i] = 1.0
    v[11] = min(1.0, float(mon.status_counter) / 3.0)
    boosts = mon.boosts or {}
    for i, k in enumerate(('atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion')):
        v[12 + i] = max(-1.0, min(1.0, float(boosts.get(k, 0)) / 6.0))
    vol = {str(k).lower(): x for k, x in (mon.volatiles or {}).items()}
    for i, k in enumerate(_VOL_FLAGS):
        if any(key.startswith(k) for key in vol):
            v[19 + i] = 1.0
    perish = [int(k[len('perish'):]) for k in vol if k.startswith('perish') and k[len('perish'):].isdigit()]
    v[32] = (min(perish) / 3.0) if perish else 0.0
    v[33] = min(1.0, float(mon.protect_counter) / 2.0)
    v[34] = 1.0 if mon.just_switched else 0.0
    v[35] = 1.0 if mon.mega_evolved else 0.0
    v[36] = 1.0 if mon.item else 0.0
    v[37] = 1.0 if can_mega else 0.0
    v[38] = 1.0 if 'protect' in vol else 0.0
    v[39] = 1.0 if mon.revealed else 0.0
    # Incertidumbre de SP por stat: 0 = conocido (propio o resuelto por la
    # inferencia), 1 = rango completo [0, SP_MAX].
    for i, k in enumerate(P.STAT_KEYS):
        if mon.stats:
            v[40 + i] = 0.0
        elif mon.sp_bounds and k in mon.sp_bounds:
            lo, hi = mon.sp_bounds[k]
            v[40 + i] = max(0.0, min(1.0, (float(hi) - float(lo)) / float(P.SP_MAX)))
        else:
            v[40 + i] = 1.0
    # Historial (protocolo del combate) y pertenencia a los 4 traídos.
    v[46] = min(1.0, float(mon.turns_active) / 5.0)
    v[47] = (1.0 / (1.0 + float(mon.last_move_ago))) if mon.last_move_ago >= 0 else 0.0
    last = P._dex()['moves'].get(P.to_id(mon.last_move)) if mon.last_move else None
    v[48] = 1.0 if (last and last.get('stallingMove')) else 0.0
    v[49] = 1.0 if (last and last.get('category') == 'Status') else 0.0
    v[50] = 1.0 if (last and str(last.get('target')) in ('allAdjacentFoes', 'allAdjacent')) else 0.0
    v[51] = 1.0 if mon.locked_move else 0.0
    v[52] = min(1.0, float(mon.hp_lost_last_turn) / 100.0)
    v[53] = min(1.0, float(mon.times_switched_in) / 3.0)
    v[54] = min(1.0, float(mon.moves_since_switch) / 3.0)
    known = {P.to_id(m) for m in mon.moves}
    v[55] = 1.0 if (mon.is_active and mon.moves_since_switch == 0
                    and known & {'fakeout', 'firstimpression'}) else 0.0
    v[56] = float(mon.p_brought)
    v[57] = 1.0 if mon.excluded else 0.0
    return v


def _mon_block(mon: Optional[MonSnapshot], *, can_mega: bool = False) -> np.ndarray:
    out = np.zeros(MON_BLOCK_DIM, dtype=np.float32)
    if mon is None:
        return out
    out[:OTS_MON_DIM] = P._encode_mon(_feat(mon))
    out[OTS_MON_DIM:] = _dyn_block(mon, can_mega=can_mega)
    return out


def _our_slots(side: SideSnapshot) -> List[Optional[MonSnapshot]]:
    rest = list(side.reserve) if side.reserve else [m for m in side.bench if m is not None]
    slots = list(side.active[:2]) + rest[:N_OUR_MONS - 2]
    return (slots + [None] * N_OUR_MONS)[:N_OUR_MONS]


def _foe_slots(side: SideSnapshot) -> List[Optional[MonSnapshot]]:
    """Activos, luego el resto de su hoja en orden fijo por especie.

    Orden estable (spec combate v2.1): el resto se ordena solo por especie, así que un
    Pokémon no cambia de bloque cuando se revela otro (antes iban los vistos primero) ni
    cuando se descarta uno. Los confirmados como NO traídos (ya se vieron los 4) quedan
    como bloque en cero EN SU LUGAR y no entran en las matrices. "Visto" sigue como
    feature del bloque. Lo único que mueve bloques es un cambio de activos, inherente a
    tener los dos activos en los bloques 0 y 1.
    """
    rest = list(side.reserve) if side.reserve else [m for m in side.bench if m is not None]
    rest = sorted(rest, key=lambda m: P.to_id(m.display_species()))
    rest = [None if m.excluded else m for m in rest]
    slots = list(side.active[:2]) + rest[:N_FOE_MONS - 2]
    return (slots + [None] * N_FOE_MONS)[:N_FOE_MONS]


# --------------------------------------------------------------------------- #
# Daño y velocidad en contexto
# --------------------------------------------------------------------------- #

def _stage_mult(stage: int) -> float:
    s = max(-6, min(6, int(stage)))
    return (2.0 + s) / 2.0 if s >= 0 else 2.0 / (2.0 - s)


def _grounded(prof, mon: MonSnapshot) -> bool:
    if 'Flying' in prof.types or prof.ability == 'levitate':
        return False
    return P.to_id(mon.item) != 'airballoon'


def _weather(field: FieldSnapshot) -> str:
    w = str(field.weather or '').lower()
    if 'rain' in w:
        return 'rain'
    if 'sun' in w:
        return 'sun'
    if 'sand' in w:
        return 'sand'
    if 'snow' in w or 'hail' in w:
        return 'snow'
    return ''


def _screens(field: FieldSnapshot, defender_is_ours: bool) -> Tuple[bool, bool]:
    if defender_is_ours:
        return (field.our_reflect or field.our_auroraveil, field.our_lightscreen or field.our_auroraveil)
    return (field.opp_reflect or field.opp_auroraveil, field.opp_lightscreen or field.opp_auroraveil)


def _pick(prof, k: str, mode: str, *, attacker: bool) -> float:
    if mode == 'mid':
        return P._mid(prof.smin, prof.smax, k)
    high = (mode == 'max') == attacker  # max: atacante alto / defensor bajo
    return prof.smax[k] if high else prof.smin[k]


def move_damage_frac(att_p, att: MonSnapshot, def_p, dfn: MonSnapshot, mv, field: FieldSnapshot,
                     *, defender_is_ours: bool, mode: str = 'mid') -> float:
    """Daño de ``mv`` como fracción del HP MÁXIMO del defensor.

    ``mode``: 'mid' = esperado (stats medios, tirada media, × precisión);
    'max' = mejor caso (atacante máx., defensor mín., tirada 1.00);
    'min' = peor caso (atacante mín., defensor máx., tirada 0.85).
    """
    immune_type = P.IMMUNITY_ABILITIES.get(def_p.ability)
    mtype = P.normalize_type(mv.type)
    if immune_type and mtype == immune_type and att_p.ability != 'moldbreaker':
        return 0.0
    if mtype == 'Ground' and P.to_id(dfn.item) == 'airballoon':
        return 0.0
    eff = P.type_effectiveness(mv.type, def_p.types)
    if eff <= 0:
        return 0.0
    hp = max(1.0, _pick(def_p, 'hp', mode, attacker=False))
    acc = mv.accuracy if mode == 'mid' else 1.0
    if mv.fixed_damage is not None:
        return float(mv.fixed_damage) * acc / hp
    ab, db = att.boosts or {}, dfn.boosts or {}
    if mv.use_target_atk:
        a = _pick(def_p, 'atk', mode, attacker=True) * _stage_mult(db.get('atk', 0))
    else:
        a = _pick(att_p, mv.offensive_stat, mode, attacker=True) * _stage_mult(ab.get(mv.offensive_stat, 0))
        if mv.offensive_stat == 'atk' and att_p.ability in ('hugepower', 'purepower'):
            a *= 2.0
        if mv.category == 'Physical' and att_p.item == 'choiceband':
            a *= 1.5
        if mv.category == 'Special' and att_p.item == 'choicespecs':
            a *= 1.5
    d = max(1.0, _pick(def_p, mv.defensive_stat, mode, attacker=False) * _stage_mult(db.get(mv.defensive_stat, 0)))
    w = _weather(field)
    if mv.defensive_stat == 'spd':
        if def_p.item == 'assaultvest':
            d *= 1.5
        if w == 'sand' and 'Rock' in def_p.types:
            d *= 1.5
    if mv.defensive_stat == 'def' and w == 'snow' and 'Ice' in def_p.types:
        d *= 1.5
    base = math.floor(math.floor(22.0 * mv.power * a / d) / 50.0) + 2.0
    mod = {'mid': 0.925, 'max': 1.0, 'min': 0.85}[mode]
    if mv.spread:
        mod *= 0.75
    if mv.type in att_p.types:
        mod *= 2.0 if att_p.ability == 'adaptability' else 1.5
    mod *= eff
    if w == 'rain':
        mod *= 1.5 if mtype == 'Water' else (0.5 if mtype == 'Fire' else 1.0)
    elif w == 'sun':
        mod *= 1.5 if mtype == 'Fire' else (0.5 if mtype == 'Water' else 1.0)
    terrain = str(field.terrain or '').lower()
    if terrain and _grounded(att_p, att):
        if ('electric' in terrain and mtype == 'Electric') or ('grassy' in terrain and mtype == 'Grass') \
                or ('psychic' in terrain and mtype == 'Psychic'):
            mod *= 1.3
    if 'misty' in terrain and mtype == 'Dragon' and _grounded(def_p, dfn):
        mod *= 0.5
    refl, ls = _screens(field, defender_is_ours)
    if (mv.category == 'Physical' and refl) or (mv.category == 'Special' and ls):
        mod *= 2.0 / 3.0
    if mv.category == 'Physical' and str(att.status or '').lower() == 'brn' and att_p.ability != 'guts':
        mod *= 0.5
    if att_p.item == 'lifeorb':
        mod *= 1.3
    elif att_p.item == 'expertbelt' and eff > 1.0:
        mod *= 1.2
    elif P.TYPE_BOOST_ITEMS.get(att_p.item) == mtype:
        mod *= 1.2
    if eff > 1.0 and P.RESIST_BERRIES.get(def_p.item) == mtype:
        mod *= 0.5
    if def_p.ability in ('multiscale', 'shadowshield') and dfn.hp_pct >= 99.9:
        mod *= 0.5
    elif def_p.ability in ('filter', 'solidrock', 'prismarmor') and eff > 1.0:
        mod *= 0.75
    return float(base * mod * acc / hp)


def _cur_frac(dmg_max_frac: float, dfn: MonSnapshot) -> float:
    """Daño respecto del HP ACTUAL (≥1 = KO esperado), recortado a 2."""
    cur = max(0.01, float(dfn.hp_pct) / 100.0)
    return float(min(2.0, dmg_max_frac / cur))


def best_damage(att_p, att, def_p, dfn, field, *, defender_is_ours: bool) -> float:
    best = 0.0
    for mv in att_p.damaging:
        best = max(best, move_damage_frac(att_p, att, def_p, dfn, mv, field,
                                          defender_is_ours=defender_is_ours))
    return _cur_frac(best, dfn)


def effective_speed(prof, mon: MonSnapshot, field: FieldSnapshot, *, ours: bool) -> Tuple[float, float]:
    lo, hi = prof.smin['spe'], prof.smax['spe']
    m = _stage_mult((mon.boosts or {}).get('spe', 0))
    if prof.item == 'choicescarf':
        m *= 1.5
    if (field.our_tailwind if ours else field.opp_tailwind):
        m *= 2.0
    if str(mon.status or '').lower() == 'par' and prof.ability != 'quickfeet':
        m *= 0.5
    w = _weather(field)
    if (w == 'rain' and prof.ability == 'swiftswim') or (w == 'sun' and prof.ability == 'chlorophyll') \
            or (w == 'sand' and prof.ability == 'sandrush') or (w == 'snow' and prof.ability == 'slushrush'):
        m *= 2.0
    if prof.ability == 'unburden' and not mon.item:
        m *= 2.0
    return lo * m, hi * m


def speed_relation(ap, a, bp, b, field) -> float:
    alo, ahi = effective_speed(ap, a, field, ours=True)
    blo, bhi = effective_speed(bp, b, field, ours=False)
    if alo > bhi:
        r = 1.0
    elif ahi < blo:
        r = -1.0
    else:
        am, bm = 0.5 * (alo + ahi), 0.5 * (blo + bhi)
        r = 0.5 * (am - bm) / max(am, bm, 1.0)
    return -r if field.trick_room else r


# --------------------------------------------------------------------------- #
# Bloques de movimientos
# --------------------------------------------------------------------------- #

def _priority_blocked(prio: int, mv_cat: str, att_p, def_p, dfn: MonSnapshot,
                      foe_side_profiles, field: FieldSnapshot) -> bool:
    """Prioridad anulada: Psychic Terrain sobre objetivo en tierra, Armor Tail /
    Dazzling / Queenly Majesty en el bando rival, Prankster de estado vs Siniestro."""
    if prio <= 0:
        return False
    if 'psychic' in str(field.terrain or '').lower() and _grounded(def_p, dfn):
        return True
    if any(p is not None and p.ability in P._PRIORITY_BLOCK for p in foe_side_profiles):
        return True
    if mv_cat == 'Status' and att_p.ability == 'prankster' and 'Dark' in def_p.types:
        return True
    return False


def _redirected(mtype: str, target: str, foe_side_profiles) -> bool:
    """Movimiento a un objetivo atraído por Storm Drain / Lightning Rod rival."""
    if target not in ('normal', 'any', 'adjacentFoe'):
        return False
    nt = P.normalize_type(mtype)
    for p in foe_side_profiles:
        if p is None:
            continue
        if (p.ability == 'stormdrain' and nt == 'Water') or (p.ability == 'lightningrod' and nt == 'Electric'):
            return True
    return False


def _eff_code(mv, def_p) -> float:
    if def_p is None:
        return 0.0
    eff = P.type_effectiveness(mv.type, def_p.types)
    if eff <= 0:
        return -1.0
    return max(-1.0, min(1.0, math.log2(eff) / 2.0))


def _our_move_block(snap: BattleSnapshot, ours: List[Optional[MonSnapshot]],
                    foes: List[Optional[MonSnapshot]]) -> np.ndarray:
    out = np.zeros(2 * 4 * MOVE_SLOT_DIM, dtype=np.float32)
    table = P._dex()['moves']
    field = snap.field
    for s in range(2):
        mon = ours[s]
        prof = _profile(mon)
        if mon is None or prof is None or mon.fainted:
            continue
        ally, ally_p = ours[1 - s], _profile(ours[1 - s])
        if snap.avail_moves is not None:
            moves = list(snap.avail_moves[s])
        else:
            moves = list(mon.moves)
        known = [P.to_id(m) for m in mon.moves]
        for mi, raw in enumerate(moves[:4]):
            mid = P.to_id(raw)
            m = table.get(mid)
            if not m:
                continue
            o = (s * 4 + mi) * MOVE_SLOT_DIM
            dmg, agg, util = P._move_infos([mid], prof.ability, prof.types)
            v = out[o:o + MOVE_SLOT_DIM]
            v[0] = 1.0
            if mid in known and known.index(mid) < len(mon.move_pps):
                v[1] = min(1.0, float(mon.move_pps[known.index(mid)]) / 40.0)
            v[2:24] = agg
            v[24:52] = util
            cat = m.get('category')
            v[52] = 1.0 if cat == 'Physical' else 0.0
            v[53] = 1.0 if cat == 'Special' else 0.0
            v[54] = 1.0 if cat == 'Status' else 0.0
            acc = m.get('accuracy')
            v[55] = 1.0 if (acc is None or acc >= 100) else float(acc) / 100.0
            v[56] = float(m.get('priority') or 0) / 5.0
            tgt = str(m.get('target') or 'normal')
            tgt = 'normal' if tgt in ('any', 'randomNormal', 'adjacentFoe', 'adjacentAllyOrSelf') else tgt
            tgt = 'allySide' if tgt == 'allies' else tgt
            if tgt in _TARGET_CLASSES:
                v[57 + _TARGET_CLASSES.index(tgt)] = 1.0
            if dmg:
                mv = dmg[0]
                for fi in range(2):
                    f, fp = foes[fi], _profile(foes[fi])
                    if f is None or fp is None or f.fainted:
                        continue
                    frac = move_damage_frac(prof, mon, fp, f, mv, field, defender_is_ours=False)
                    v[65 + fi] = min(2.0, frac) / 2.0
                    v[68 + fi] = 1.0 if _cur_frac(frac, f) >= 1.0 else 0.0
                    v[70 + fi] = _eff_code(mv, fp)
                if ally is not None and ally_p is not None and not ally.fainted and \
                        str(m.get('target')) in ('allAdjacent', 'adjacentAlly', 'adjacentAllyOrSelf'):
                    v[67] = min(2.0, move_damage_frac(prof, mon, ally_p, ally, mv, field,
                                                      defender_is_ours=True)) / 2.0
                for fi in range(2):
                    f, fp = foes[fi], _profile(foes[fi])
                    if f is None or fp is None or f.fainted:
                        continue
                    cur = max(0.01, float(f.hp_pct) / 100.0)
                    hi = move_damage_frac(prof, mon, fp, f, mv, field, defender_is_ours=False, mode='max')
                    lo = move_damage_frac(prof, mon, fp, f, mv, field, defender_is_ours=False, mode='min')
                    v[77 + fi] = 1.0 if hi >= cur else 0.0  # KO posible
                    v[79 + fi] = 1.0 if lo >= cur else 0.0  # KO garantizado (si acierta)
            # Historial / mecánicas del formato
            v[72] = 1.0 if (mon.last_move and P.to_id(mon.last_move) == mid) else 0.0
            v[73] = 1.0 if (mon.locked_move and P.to_id(mon.locked_move) == mid) else 0.0
            foe_profiles = [_profile(f) if (f is not None and not f.fainted) else None for f in foes[:2]]
            prio = int(m.get('priority') or 0) + (1 if prof.ability == 'prankster' and cat == 'Status' else 0)
            for fi in range(2):
                f, fp = foes[fi], foe_profiles[fi]
                if f is not None and fp is not None:
                    v[74 + fi] = 1.0 if _priority_blocked(prio, cat, prof, fp, f, foe_profiles, field) else 0.0
            v[76] = 1.0 if _redirected(dmg[0].type if dmg else str(m.get('type') or ''),
                                       str(m.get('target') or 'normal'), foe_profiles) else 0.0
    return out


def _foe_move_block(snap: BattleSnapshot, ours: List[Optional[MonSnapshot]],
                    foes: List[Optional[MonSnapshot]]) -> np.ndarray:
    out = np.zeros(2 * 4 * FOE_MOVE_DIM, dtype=np.float32)
    table = P._dex()['moves']
    field = snap.field
    for s in range(2):
        mon = foes[s]
        prof = _profile(mon)
        if mon is None or prof is None or mon.fainted:
            continue
        moves = sorted({P.to_id(m) for m in mon.moves if P.to_id(m) in table})
        for mi, mid in enumerate(moves[:4]):
            m = table[mid]
            o = (s * 4 + mi) * FOE_MOVE_DIM
            v = out[o:o + FOE_MOVE_DIM]
            dmg, _agg, util = P._move_infos([mid], prof.ability, prof.types)
            v[0] = 1.0
            mtype = dmg[0].type if dmg else str(m.get('type') or '')
            nt = P.normalize_type(mtype)
            if nt in P.TYPES:
                v[1 + P.TYPES.index(nt)] = 1.0
            cat = m.get('category')
            v[19] = 1.0 if cat == 'Physical' else 0.0
            v[20] = 1.0 if cat == 'Special' else 0.0
            v[21] = 1.0 if cat == 'Status' else 0.0
            v[22] = (dmg[0].power / 250.0) if dmg else 0.0
            prio = int(m.get('priority') or 0) + (1 if prof.ability == 'prankster' and cat == 'Status' else 0)
            v[23] = prio / 5.0
            v[24] = 1.0 if str(m.get('target')) in ('allAdjacentFoes', 'allAdjacent') else 0.0
            for ui, k in enumerate(_FOE_UTIL):
                v[25 + ui] = util[_UIDX[k]]
            if dmg:
                for ti in range(2):
                    t, tp = ours[ti], _profile(ours[ti])
                    if t is None or tp is None or t.fainted:
                        continue
                    frac = move_damage_frac(prof, mon, tp, t, dmg[0], field, defender_is_ours=True)
                    v[35 + ti] = min(2.0, frac) / 2.0
                    v[37 + ti] = 1.0 if _cur_frac(frac, t) >= 1.0 else 0.0
                    hi = move_damage_frac(prof, mon, tp, t, dmg[0], field, defender_is_ours=True, mode='max')
                    v[41 + ti] = 1.0 if hi >= max(0.01, float(t.hp_pct) / 100.0) else 0.0
            v[39] = 1.0 if (mon.last_move and P.to_id(mon.last_move) == mid) else 0.0
            v[40] = 1.0 if (mon.locked_move and P.to_id(mon.locked_move) == mid) else 0.0
    return out


def _pair_block(snap: BattleSnapshot, ours, foes) -> np.ndarray:
    d_of = np.zeros((N_OUR_MONS, N_FOE_MONS), dtype=np.float32)
    d_fo = np.zeros_like(d_of)
    spd = np.zeros_like(d_of)
    po = [_profile(m) for m in ours]
    pf = [_profile(m) for m in foes]
    for i in range(N_OUR_MONS):
        a, ap = ours[i], po[i]
        if a is None or ap is None or a.fainted:
            continue
        for j in range(N_FOE_MONS):
            b, bp = foes[j], pf[j]
            if b is None or bp is None or b.fainted:
                continue
            d_of[i, j] = best_damage(ap, a, bp, b, snap.field, defender_is_ours=False) / 2.0
            d_fo[i, j] = best_damage(bp, b, ap, a, snap.field, defender_is_ours=True) / 2.0
            spd[i, j] = speed_relation(ap, a, bp, b, snap.field)
    out = np.concatenate([d_of.reshape(-1), d_fo.reshape(-1), spd.reshape(-1)])
    assert out.shape[0] == PAIR_DIM
    return out


def _field_block(snap: BattleSnapshot, ours, foes) -> np.ndarray:
    f = snap.field
    v = np.zeros(FIELD_DIM, dtype=np.float32)
    w = _weather(f)
    for i, k in enumerate(('', 'rain', 'sun', 'sand', 'snow')):
        v[i] = 1.0 if w == k else 0.0
    v[5] = float(f.weather_turns) / 8.0
    terrain = str(f.terrain or '').lower()
    for i, k in enumerate(('electric', 'grassy', 'misty', 'psychic')):
        v[6 + i] = 1.0 if k in terrain else 0.0
    v[10] = 1.0 if not terrain else 0.0
    v[11] = float(f.terrain_turns) / 8.0
    v[12] = 1.0 if f.trick_room else 0.0
    v[13] = float(f.trick_room_turns) / 5.0
    v[14] = 1.0 if f.our_tailwind else 0.0
    v[15] = float(f.our_tailwind_turns) / 4.0
    v[16] = 1.0 if f.opp_tailwind else 0.0
    v[17] = float(f.opp_tailwind_turns) / 4.0
    for i, flag in enumerate((f.our_reflect, f.our_lightscreen, f.our_auroraveil,
                              f.opp_reflect, f.opp_lightscreen, f.opp_auroraveil)):
        v[18 + i] = 1.0 if flag else 0.0
    v[24] = 1.0 if f.gravity else 0.0
    v[25] = min(1.0, float(snap.turn) / 40.0)
    v[26] = 1.0 if snap.our.mega_used else 0.0
    v[27] = 1.0 if snap.foe.mega_used else 0.0
    our_alive = [m for m in ours if m is not None and not m.fainted and m.hp_pct > 0]
    v[28] = len(our_alive) / 4.0
    v[29] = sum(m.hp_pct for m in our_alive) / 400.0
    v[30] = (4 - snap.foe.faint_count()) / 4.0
    v[31] = snap.foe.team_hp_pct_total() / 400.0
    seen = [m for m in foes if m is not None and m.seen]
    v[32] = min(1.0, len(seen) / 4.0)
    v[33] = 1.0 if snap.force_switch[0] else 0.0
    v[34] = 1.0 if snap.force_switch[1] else 0.0
    lt = snap.last_turn or {}
    for i, k in enumerate(('our_protect', 'our_switch', 'our_faint', 'our_fakeout', 'our_spread',
                           'foe_protect', 'foe_switch', 'foe_faint', 'foe_fakeout', 'foe_spread')):
        v[35 + i] = min(1.0, float(lt.get(k, 0.0)) / 2.0)
    foe_all = [m for m in list(snap.foe.active) + list(snap.foe.reserve) if m is not None]
    v[45] = 1.0 if sum(1 for m in foe_all if m.seen) >= 4 else 0.0
    return v


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

_LAST: Tuple[Optional[BattleSnapshot], Optional[np.ndarray]] = (None, None)


def build_observation(battle: BattleSnapshot) -> np.ndarray:
    """float32[OBS_DIM]. Memo del último snapshot (env y entrenador piden la misma obs)."""
    global _LAST
    if _LAST[0] is battle and _LAST[1] is not None:
        return _LAST[1].copy()
    ours = _our_slots(battle.our)
    foes = _foe_slots(battle.foe)
    can_mega = [bool(m.can_mega) if m is not None else False for m in battle.our.active[:2]]
    blocks = [
        _mon_block(m, can_mega=(i < 2 and can_mega[i] if i < len(can_mega) else False))
        for i, m in enumerate(ours)
    ]
    blocks += [_mon_block(m) for m in foes]
    obs = np.concatenate(blocks + [
        _our_move_block(battle, ours, foes),
        _foe_move_block(battle, ours, foes),
        _pair_block(battle, ours, foes),
        _field_block(battle, ours, foes),
    ]).astype(np.float32)
    if obs.shape[0] != OBS_DIM:  # pragma: no cover - garantía de contrato
        raise AssertionError(f'obs {obs.shape[0]} != OBS_DIM {OBS_DIM}')
    _LAST = (battle, obs)
    return obs.copy()


def build_observation_ddqn(battle: BattleSnapshot) -> np.ndarray:
    """v2: igual para todos los agentes (ya no hay segmento belief que anular)."""
    return build_observation(battle)


def build_observation_ppo(battle: BattleSnapshot) -> np.ndarray:
    return build_observation(battle)


def obs_layout() -> Dict[str, Tuple[int, int]]:
    """Rangos [inicio, fin) de cada sección (diagnóstico / tests / tesis)."""
    out: Dict[str, Tuple[int, int]] = {}
    o = 0
    for i in range(N_OUR_MONS):
        out[f'our_{i}'] = (o, o + MON_BLOCK_DIM)
        o += MON_BLOCK_DIM
    for j in range(N_FOE_MONS):
        out[f'foe_{j}'] = (o, o + MON_BLOCK_DIM)
        o += MON_BLOCK_DIM
    out['our_moves'] = (o, o + 8 * MOVE_SLOT_DIM)
    o += 8 * MOVE_SLOT_DIM
    out['foe_moves'] = (o, o + 8 * FOE_MOVE_DIM)
    o += 8 * FOE_MOVE_DIM
    out['pairs'] = (o, o + PAIR_DIM)
    o += PAIR_DIM
    out['field'] = (o, o + FIELD_DIM)
    o += FIELD_DIM
    assert o == OBS_DIM
    return out
