"""Planes clásicos de VGC para el modo arquetipo **nivel 3** de greedy_joint (27-11-2026).

Motivo (protocolo §4.17): D1 mostró que los heurísticos casi nunca ejecutan sus planes (Trick Room en el 0,1 % de
las partidas en que pueden, Tailwind en el 1,6 %) y las personas ganaron con lluvia + Trick Room y sol + Tailwind.
Los niveles 1–2 solo fuerzan Tailwind / Trick Room; el nivel 3 añade clima, campos y los núcleos clásicos:

- **Clima**: lluvia (Drizzle / Rain Dance → Swift Swim, Electro Shot en un turno, Thunder / Hurricane, Agua), sol
  (Drought, también Mega-Charizard Y → Chlorophyll, Solar Power, Protosynthesis, Solar Beam, Fuego), arena
  (Sand Stream, también Mega-Tyranitar → Sand Rush, Sand Force, Roca), nieve (Snow Warning / Snowscape / Chilly
  Reception → Slush Rush, Blizzard, Aurora Veil, Hielo). Con dos climas en el equipo (sol + lluvia) se elige el que
  mejor encaja contra el equipo rival.
- **Campos**: Grassy / Psychic / Electric / Misty (Surge o movimiento) → semillas (con Unburden, p. ej. Sneasler),
  Grassy Glide, Expanding Force, Rising Voltage, Surge Surfer / Quark Drive.
- **Velocidad**: Trick Room (abre el que lo pone con apoyo de redirección / Fake Out, atacantes lentos detrás) y
  Tailwind.

El plan decide el preview (abre quien pone el plan + quien lo aprovecha o lo protege), los bonos de jugada (poner el
clima / campo si no está; Protect o redirección del compañero mientras se pone) y los cambios forzados (vuelve quien
pone el clima / campo si se perdió; si está, entra quien lo aprovecha). greedy_joint no hace cambios voluntarios: un
clima de habilidad que se pierde solo vuelve con un cambio forzado o con el movimiento.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _id(x) -> str:
    return ''.join(ch for ch in str(x or '').lower() if ch.isalnum())


@lru_cache(maxsize=1)
def _dex():
    from showdown.rl import team_preview_obs as P
    return P._dex()


@dataclass(frozen=True)
class MonInfo:
    index: int                     # 1-based (orden del preview / del equipo)
    species: str
    ability: str = ''              # id, ya resuelto a la Mega si lleva su piedra
    item: str = ''
    moves: frozenset = frozenset()
    types: Tuple[str, ...] = ()
    base_spe: int = 80


def mon_info(index: int, species: str, ability: str = '', item: str = '', moves: Iterable = ()) -> MonInfo:
    """``MonInfo`` desde datos crudos: resuelve la Mega por la piedra y completa tipos / velocidad base del dex."""
    dex = _dex()
    sp_all = dex.get('species') or {}
    it = _id(item)
    sid = _id(species)
    rec = sp_all.get(sid) or {}
    ab = _id(ability)
    mega = ((dex.get('items') or {}).get(it) or {}).get('megaStone') or {}
    for base, mega_name in mega.items():
        if _id(base) == _id(rec.get('baseSpecies') or species) or _id(base) == sid:
            mrec = sp_all.get(_id(mega_name))
            if mrec:
                rec = mrec
                ab = _id((mrec.get('abilities') or {}).get('0', ab))
    if not ab:
        abilities = [a for a in (rec.get('abilities') or {}).values() if a]
        if len(abilities) == 1:
            ab = _id(abilities[0])
    types = tuple(str(t) for t in (rec.get('types') or ()))
    spe = int((rec.get('baseStats') or {}).get('spe', 80) or 80)
    return MonInfo(int(index), str(species), ab, it, frozenset(_id(m) for m in moves if m), types, spe)


@dataclass(frozen=True)
class Plan:
    name: str
    kind: str                                   # 'weather' | 'terrain' | 'speed'
    fields: frozenset = frozenset()             # cadenas de clima / campo activos (como los da el combate)
    setter_abilities: frozenset = frozenset()
    setter_moves: frozenset = frozenset()
    abuser_abilities: frozenset = frozenset()
    abuser_moves: frozenset = frozenset()
    seed: str = ''                              # semilla del campo (con Unburden el portador se acelera)
    boosted_types: frozenset = frozenset()
    support_moves: frozenset = frozenset()      # Trick Room / Tailwind: redirección, Fake Out…


def _fs(*xs) -> frozenset:
    return frozenset(_id(x) for x in xs)


PLANS: Dict[str, Plan] = {p.name: p for p in (
    Plan('lluvia', 'weather', _fs('raindance', 'primordialsea'), _fs('drizzle', 'primordialsea'), _fs('raindance'),
         _fs('swiftswim', 'raindish', 'dryskin', 'hydration'),
         _fs('electroshot', 'thunder', 'hurricane', 'weatherball', 'wavecrash', 'waterspout'),
         boosted_types=frozenset({'Water'})),
    Plan('sol', 'weather', _fs('sunnyday', 'desolateland'), _fs('drought', 'desolateland', 'orichalcumpulse'),
         _fs('sunnyday'), _fs('chlorophyll', 'solarpower', 'protosynthesis', 'flowergift', 'harvest', 'leafguard'),
         _fs('solarbeam', 'solarblade', 'weatherball', 'growth', 'eruption', 'heatwave'),
         boosted_types=frozenset({'Fire'})),
    Plan('arena', 'weather', _fs('sandstorm'), _fs('sandstream', 'sandspit'), _fs('sandstorm'),
         _fs('sandrush', 'sandforce', 'sandveil'), _fs('weatherball', 'shoreup'),
         boosted_types=frozenset({'Rock'})),
    Plan('nieve', 'weather', _fs('snow', 'snowscape', 'hail'), _fs('snowwarning'),
         _fs('snowscape', 'chillyreception', 'hail'), _fs('slushrush', 'icebody', 'snowcloak'),
         _fs('blizzard', 'auroraveil', 'weatherball'), boosted_types=frozenset({'Ice'})),
    Plan('campo_hierba', 'terrain', _fs('grassyterrain'), _fs('grassysurge', 'seedsower'), _fs('grassyterrain'),
         _fs('grasspelt'), _fs('grassyglide'), seed='grassyseed', boosted_types=frozenset({'Grass'})),
    Plan('campo_psiquico', 'terrain', _fs('psychicterrain'), _fs('psychicsurge'), _fs('psychicterrain'),
         frozenset(), _fs('expandingforce'), seed='psychicseed', boosted_types=frozenset({'Psychic'})),
    Plan('campo_electrico', 'terrain', _fs('electricterrain'), _fs('electricsurge', 'hadronengine'),
         _fs('electricterrain'), _fs('surgesurfer', 'quarkdrive'), _fs('risingvoltage'), seed='electricseed',
         boosted_types=frozenset({'Electric'})),
    Plan('campo_niebla', 'terrain', _fs('mistyterrain'), _fs('mistysurge'), _fs('mistyterrain'),
         frozenset(), _fs('mistyexplosion'), seed='mistyseed'),
    Plan('trick_room', 'speed', _fs('trickroom'), frozenset(), _fs('trickroom'),
         support_moves=_fs('followme', 'ragepowder', 'fakeout', 'allyswitch')),
    Plan('tailwind', 'speed', frozenset(), frozenset(), _fs('tailwind'),
         support_moves=_fs('fakeout', 'followme', 'ragepowder')),
)}

TR_SLOW_SPE = 60   # velocidad base de un atacante de Trick Room


def _damaging_types(mon: MonInfo) -> set:
    mv = _dex().get('moves') or {}
    out = set()
    for m in mon.moves:
        d = mv.get(m) or {}
        if d.get('category') in ('Physical', 'Special'):
            out.add(str(d.get('type')))
    return out


def is_setter(mon: MonInfo, plan: Plan) -> bool:
    return mon.ability in plan.setter_abilities or bool(mon.moves & plan.setter_moves)


def is_auto_setter(mon: MonInfo, plan: Plan) -> bool:
    """Pone el plan al entrar (habilidad): clima y campo."""
    return mon.ability in plan.setter_abilities


def abuser_score(mon: MonInfo, plan: Plan) -> float:
    """Cuánto aprovecha ``mon`` el plan (0 = nada)."""
    if plan.name == 'trick_room':
        if mon.base_spe <= TR_SLOW_SPE and _damaging_types(mon) and 'trickroom' not in mon.moves:
            return 2.0 + (TR_SLOW_SPE - mon.base_spe) / 30.0
        return 0.0
    if plan.name == 'tailwind':
        return 1.0 if (_damaging_types(mon) and 50 <= mon.base_spe <= 100 and 'tailwind' not in mon.moves) else 0.0
    s = 0.0
    if mon.ability in plan.abuser_abilities:
        s += 3.0
    if plan.seed and mon.item == plan.seed:
        s += 3.0 if mon.ability == 'unburden' else 1.5
    if mon.moves & plan.abuser_moves:
        s += 2.0
    if plan.boosted_types & _damaging_types(mon) & set(mon.types):
        s += 1.0
    return s


def support_score(mon: MonInfo, plan: Plan) -> float:
    if not plan.support_moves:
        return 0.0
    s = 0.0
    if mon.moves & _fs('followme', 'ragepowder'):
        s += 2.0
    if 'fakeout' in mon.moves:
        s += 1.5
    if mon.ability == 'intimidate':
        s += 0.5
    return s


@dataclass
class PlanFit:
    plan: Plan
    setters: List[int]
    abusers: List[Tuple[int, float]]          # (índice, puntaje), de mayor a menor
    support: List[Tuple[int, float]] = field(default_factory=list)
    score: float = 0.0


def detect_plans(team: Sequence[MonInfo]) -> List[PlanFit]:
    """Planes que el equipo puede ejecutar: alguien lo pone y alguien lo aprovecha (de mayor a menor puntaje)."""
    out: List[PlanFit] = []
    for plan in PLANS.values():
        setters = [m.index for m in team if is_setter(m, plan)]
        if not setters:
            continue
        ab = sorted(((m.index, abuser_score(m, plan)) for m in team if m.index not in setters[:1]),
                    key=lambda t: -t[1])
        ab = [t for t in ab if t[1] > 0]
        sup = sorted(((m.index, support_score(m, plan)) for m in team if m.index not in setters[:1]),
                     key=lambda t: -t[1])
        sup = [t for t in sup if t[1] > 0]
        if not ab:
            continue
        auto = any(is_auto_setter(m, plan) for m in team if m.index in setters)
        score = 1.0 + 0.5 * auto + sum(s for _i, s in ab[:3]) + 0.5 * sum(s for _i, s in sup[:1])
        out.append(PlanFit(plan, setters, ab, sup, score))
    out.sort(key=lambda f: -f.score)
    return out


def _effectiveness(atk_type: str, def_types: Sequence[str]) -> float:
    from showdown.battle_bots.greedy_solo.type_chart import type_effectiveness
    return float(type_effectiveness(atk_type, tuple(def_types)))


def _eff(atk_type: str, def_types: Sequence[str]) -> float:
    try:
        return _effectiveness(atk_type, def_types)
    except Exception:  # pragma: no cover - sin tabla de tipos
        return 1.0


def matchup_bonus(fit: PlanFit, foes: Sequence[MonInfo]) -> float:
    """Ajuste por el equipo rival: tipos potenciados que pegan / son resistidos; velocidad rival para TR / TW."""
    p = fit.plan
    if p.kind in ('weather', 'terrain') and p.boosted_types:
        b = 0.0
        for f in foes:
            e = max(_eff(t, f.types) for t in p.boosted_types) if f.types else 1.0
            b += 0.5 if e > 1.0 else (-0.5 if e < 1.0 else 0.0)
        return b
    if p.name == 'trick_room':
        return 0.4 * sum(1 for f in foes if f.base_spe >= 90) - 0.4 * sum(1 for f in foes if f.base_spe <= 50)
    if p.name == 'tailwind':
        return 0.3 * sum(1 for f in foes if 70 <= f.base_spe <= 110)
    return 0.0


def choose_plan(team: Sequence[MonInfo], foes: Sequence[MonInfo] = ()) -> Optional[PlanFit]:
    """Plan principal contra ``foes`` (doble clima: el que mejor encaja con el rival)."""
    fits = detect_plans(team)
    if not fits:
        return None
    return max(fits, key=lambda f: f.score + matchup_bonus(f, foes))


def preview_order(team: Sequence[MonInfo], foes: Sequence[MonInfo], base_order: Sequence[int]
                  ) -> Tuple[Tuple[int, ...], Optional[PlanFit]]:
    """Orden de 4 (los dos primeros abren): quien pone el plan + quien lo aprovecha / protege, luego el resto."""
    fit = choose_plan(team, foes)
    base = [int(i) for i in base_order]
    if fit is None:
        return tuple(base[:4]), None
    by = {m.index: m for m in team}
    setters = sorted(fit.setters, key=lambda i: (not is_auto_setter(by[i], fit.plan),
                                                 -support_score(by[i], fit.plan)))
    setter = setters[0]
    partner = None
    if fit.plan.kind == 'speed' and fit.support:
        partner = next((i for i, _s in fit.support if i != setter), None)
    if partner is None:
        partner = next((i for i, _s in fit.abusers if i != setter), None)
    order: List[int] = [setter] + ([partner] if partner else [])
    for i, _s in fit.abusers:
        if len(order) >= 4:
            break
        if i not in order:
            order.append(i)
    for i in base + [m.index for m in team]:
        if len(order) >= 4:
            break
        if i not in order:
            order.append(i)
    return tuple(order[:4]), fit


def plan_field_active(fit: Optional[PlanFit], weather: Optional[str], terrain: Optional[str]) -> bool:
    if fit is None or fit.plan.kind == 'speed':
        return False
    cur = _id(weather) if fit.plan.kind == 'weather' else _id(terrain)
    return cur in fit.plan.fields


def sets_plan_field(move_id: str, fit: Optional[PlanFit]) -> bool:
    """¿El movimiento pone el clima / campo del plan?"""
    return bool(fit is not None and fit.plan.kind != 'speed' and _id(move_id) in fit.plan.setter_moves)
