"""S3 — modelo de turno para MCTS+PUCT v3, fase 3a (docs/mcts_puct_spec.md v3.0).

Sustituye a S2 (``simulator.py``) solo para G; MCTS–Nash sigue con S2.

Qué modela (fase 3a):

- **Daño real**: al inicio de cada decisión se pide en UN lote a la calculadora real
  (``@smogon/calc`` Champions vía ``damage_calc_bridge``, los mismos pedidos que P5:
  ``obs_calc.build_request``) el rango de daño de cada atacante × defensor × movimiento,
  para los 4 Pokémon de cada lado (activos y banca), en variante individual y de área
  (×0,75) y el fuego amigo de ``allAdjacent``. Las simulaciones solo consultan la tabla
  y sortean la tirada dentro del rango.
- **Stats**: los propios exactos; los del rival con el SP preferido por el prior meta
  (leave-one-out) de sus creencias, como greedy/expectimax. Sin creencias, el punto
  medio del rango de la hoja.
- **Orden del turno**: cambios primero; luego movimientos por prioridad (Prankster,
  Grassy Glide…) y velocidad efectiva (boosts, Scarf, parálisis, Tailwind; Trick Room
  invierte), con desempate al azar. Un Pokémon debilitado antes de actuar no actúa.
- **Objetivos**: un ataque individual se redirige al otro rival si su objetivo cayó;
  área sobre los rivales vivos (y el aliado en ``allAdjacent``) con ×0,75 si golpea a
  más de uno; precisión por movimiento.

Fase 3b (``phase='3b'``, MCTS+PUCT v4): Protect y variantes (falla con prob. creciente si
se repite: 1/3^n), Wide Guard (bloquea área en su lado), Fake Out y First Impression (solo
en el primer turno en campo; Fake Out hace retroceder salvo Inner Focus / Covert Cloak),
Follow Me / Rage Powder (redirigen los golpes individuales del rival; Rage Powder no
afecta a tipo Planta, Overcoat ni Safety Goggles) y Helping Hand (×1,5 al aliado). Feint
rompe la protección. La puntuación (prior, rival, rollout) valora estas jugadas con la
amenaza que evitan.

Fase 3c (``phase='3c'``, MCTS+PUCT v5; incluye 3b): Tailwind (4 turnos) y Trick Room
(5 turnos, se cancela si se repite) como acciones; orden del turno recalculado tras cada
acción (velocidad dinámica, Gen 9); estados (sueño 1–3 turnos, parálisis ×0,5 velocidad y
25 % de no actuar, quemadura ×0,5 daño físico, veneno; daño residual 1/16 y 1/8) con
inmunidades (tipo, Safety Goggles/polvo, ya con estado); boosts dentro del árbol
(movimientos de estado, secundarios, ``selfBoosts`` como Close Combat / Draco Meteor,
Parting Shot), con el daño de la tabla ajustado por la razón de etapas respecto a la raíz;
Intimidate al entrar (Clear Body y similares lo bloquean; Defiant / Competitive);
clima por movimiento o habilidad al entrar (Fuego/Agua ×1,5 / ×0,5 respecto al clima de la
raíz; Swift Swim, Chlorophyll, Sand Rush, Slush Rush; daño de arena); cambios de pivote
(U-turn, Volt Switch, Flip Turn, Parting Shot).

Fase 3d (``phase='3d'``, MCTS+PUCT v6; incluye 3c): terreno. Al entrar por habilidad
(Grassy / Psychic / Electric / Misty Surge, Hadron Engine), 5 turnos. Daño ajustado respecto al
terreno de la raíz (Grass / Electric / Psychic ×1,3 del atacante en el suelo; Dragon ×0,5 contra
un defensor en el suelo con Misty; Earthquake / Bulldoze ×0,5 con Grassy). Prioridad de Grassy
Glide según el terreno actual. Psychic Terrain bloquea los movimientos con prioridad contra
rivales en el suelo (Fake Out, Sucker Punch, Aqua Jet…). Expanding Force golpea en área y ×1,5
en Psychic Terrain. Grassy cura 1/16; Misty bloquea estados y Electric el sueño a quien está
en el suelo. "En el suelo" = no Volador, sin Levitate ni Air Balloon.

Qué NO modela: críticos, Quick Guard, Spotlight, Yawn, objetos consumibles (Terrain Extender,
semillas), habilidades raras.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from showdown.rl.action_space import decode_joint_action, decode_slot_action
from showdown.rl.constants import MEGA_BASE
from showdown.rl.damage_heuristic import estimate_damage_pct, move_entry
from showdown.rl.snapshot import BattleSnapshot, MonSnapshot, SideSnapshot

_SINGLE = {'normal', 'adjacentfoe', 'any', 'randomnormal'}
_SPREAD_FOES = 'alladjacentfoes'
_SPREAD_ALL = 'alladjacent'
KO_BONUS = 40.0  # puntos de % por KO esperado en la puntuación (prior / rival / rollout)
# Fase 3b: peso de la amenaza evitada en la puntuación de jugadas de apoyo.
PROTECT_W = 0.5     # Protect: fracción de la amenaza entrante que "vale" protegerse
FLINCH_W = 0.5      # Fake Out: fracción del daño del objetivo que se le niega
REDIRECT_W = 0.4    # Follow Me / Rage Powder: fracción de la amenaza al aliado absorbida
HELP_W = 0.5        # Helping Hand: fracción del daño del aliado que se suma (×1,5 → +50 %)
WIDEGUARD_W = 0.5   # Wide Guard: fracción del daño en área evitado
_PROTECT_VOL = {'protect', 'kingsshield', 'spikyshield', 'banefulbunker', 'silktrap',
                'obstruct', 'burningbulwark', 'maxguard'}
_FIRST_TURN = {'fakeout', 'firstimpression'}
# Fase 3c
TAILWIND_W = 25.0   # valor de Tailwind / Trick Room por fracción de duelos de velocidad ganados
SLEEP_W = 0.8       # fracción del daño del objetivo que se niega al dormirlo
PARA_W = 0.35
BURN_W = 0.5        # solo si su mejor ataque es físico
POISON_V = 8.0
BOOST_W = 0.15      # por etapa de atk/spa propia: fracción del propio mejor daño
SPEEDCTL_V = 5.0    # por rival al que se le baja la velocidad
_INTIM_IMMUNE = {'clearbody', 'whitesmoke', 'fullmetalbody', 'hypercutter', 'innerfocus', 'oblivious',
                 'owntempo', 'scrappy', 'guarddog'}
_DROP_IMMUNE = {'clearbody', 'whitesmoke', 'fullmetalbody'}
_WEATHER_ABIL = {'drought': 'sunnyday', 'drizzle': 'raindance', 'sandstream': 'sandstorm',
                 'snowwarning': 'snow', 'orichalcumpulse': 'sunnyday'}
_WEATHER_SPEED = {'swiftswim': ('raindance', 'primordialsea'), 'chlorophyll': ('sunnyday', 'desolateland'),
                  'sandrush': ('sandstorm',), 'slushrush': ('snow', 'snowscape', 'hail')}
_EXTRA_TARGET_BOOSTS = {'partingshot': {'atk': -1, 'spa': -1}}
# Fase 3d: terreno
_TERRAIN_ABIL = {'grassysurge': 'grassy', 'psychicsurge': 'psychic', 'electricsurge': 'electric',
                 'mistysurge': 'misty', 'hadronengine': 'electric'}
_TERRAIN_BOOST = {'grassy': 'grass', 'electric': 'electric', 'psychic': 'psychic'}
_GRASSY_HALVED = {'earthquake', 'bulldoze', 'magnitude'}


def _norm(s) -> str:
    return ''.join(ch for ch in str(s or '').lower() if ch.isalnum())


def _alive(m: Optional[MonSnapshot]) -> bool:
    return m is not None and not m.fainted and float(m.hp_pct) > 0.0


def dex_move(move_id: str) -> dict:
    """Entrada del dex de Champions (el de P5 / team_preview_obs); respaldo: moves.json antiguo.

    El dex antiguo (``API/JSONs/moves.json``, el de S2) no tiene 21 de los 212 movimientos del
    meta (Make It Rain, Kowtow Cleave, Glaive Rush, Electro Shot…): con S2 esos ataques no
    hacían nada.
    """
    from showdown.rl import team_preview_obs as P
    e = P._dex()['moves'].get(_norm(move_id))
    return e if e else (move_entry(move_id) or {})


def move_target(move_id: str) -> str:
    return _norm(dex_move(move_id).get('target', 'normal')) or 'normal'


def move_accuracy(move_id: str) -> float:
    acc = dex_move(move_id).get('accuracy', True)
    if acc is True or acc is None:
        return 1.0
    try:
        return max(0.0, min(1.0, float(acc) / 100.0))  # 101 = no falla (dex de Champions)
    except (TypeError, ValueError):
        return 1.0


def target_class(move_id: str) -> str:
    """'foe', 'ally' o 'none' (área / self / campo) para la poda de objetivos de v3."""
    tgt = move_target(move_id)
    if tgt in _SINGLE:
        return 'foe'
    if tgt == 'adjacentally':
        return 'ally'
    return 'none'


def move_effect(move_id: str) -> str:
    """Efecto de fase 3b: protect | wideguard | redirect | redirect_powder | helpinghand | ''."""
    e = dex_move(move_id)
    vol = _norm(e.get('volatileStatus'))
    if e.get('stallingMove') and vol in _PROTECT_VOL:
        return 'protect'
    if _norm(e.get('sideCondition')) == 'wideguard':
        return 'wideguard'
    if vol == 'followme':
        return 'redirect'
    if vol == 'ragepowder':
        return 'redirect_powder'
    if vol == 'helpinghand':
        return 'helpinghand'
    return ''


def _flags(move_id: str) -> set:
    f = dex_move(move_id).get('flags') or []
    return set(f) if isinstance(f, (list, tuple, set)) else {k for k, v in dict(f).items() if v}


def blockable(move_id: str) -> bool:
    """¿Protect lo bloquea? (flag ``protect`` y sin ``breaksProtect``)."""
    return 'protect' in _flags(move_id) and not dex_move(move_id).get('breaksProtect')


def breaks_protect(move_id: str) -> bool:
    return bool(dex_move(move_id).get('breaksProtect'))


def _powder_immune(m: MonSnapshot) -> bool:
    types = {_norm(t) for t in (m.types or ())}
    return 'grass' in types or _norm(m.ability) == 'overcoat' or _norm(m.item) == 'safetygoggles'


def _flinch_immune(m: MonSnapshot) -> bool:
    return _norm(m.ability) in ('innerfocus', 'shielddust') or _norm(m.item) == 'covertcloak'


def stage_mult(st: int) -> float:
    st = int(max(-6, min(6, st)))
    return (2.0 + st) / 2.0 if st >= 0 else 2.0 / (2.0 - st)


def _wnorm(w) -> str:
    w = _norm(w)
    return {'snowscape': 'snow', 'hail': 'snow', 'sun': 'sunnyday', 'rain': 'raindance', 'sand': 'sandstorm'}.get(w, w)


def _weather_factor(mtype: str, weather: str) -> float:
    t, w = _norm(mtype), _wnorm(weather)
    if w in ('sunnyday', 'desolateland'):
        return 1.5 if t == 'fire' else (0.5 if t == 'water' else 1.0)
    if w in ('raindance', 'primordialsea'):
        return 1.5 if t == 'water' else (0.5 if t == 'fire' else 1.0)
    return 1.0


def _tnorm(t) -> str:
    t = _norm(t)
    for k in ('grassy', 'psychic', 'electric', 'misty'):
        if t.startswith(k):
            return k
    return ''


def grounded(m: MonSnapshot) -> bool:
    types = {_norm(t) for t in (m.types or ())}
    return 'flying' not in types and _norm(m.ability) != 'levitate' and _norm(m.item) != 'airballoon'


def _terrain_factor(att: MonSnapshot, dfn: MonSnapshot, mid: str, terrain: str) -> float:
    t = _tnorm(terrain)
    if not t:
        return 1.0
    mtype = _norm(dex_move(mid).get('type', ''))
    f = 1.0
    if _TERRAIN_BOOST.get(t) == mtype and grounded(att):
        f *= 1.3
    if t == 'misty' and mtype == 'dragon' and grounded(dfn):
        f *= 0.5
    if t == 'grassy' and _norm(mid) in _GRASSY_HALVED and grounded(dfn):
        f *= 0.5
    return f


def move_effect3(move_id: str) -> str:
    """Efecto de fase 3c para movimientos de estado: tailwind | trickroom | weather | status | boost | ''."""
    e = dex_move(move_id)
    if _norm(e.get('sideCondition')) == 'tailwind':
        return 'tailwind'
    if _norm(e.get('pseudoWeather')) == 'trickroom':
        return 'trickroom'
    if e.get('weather'):
        return 'weather'
    if str(e.get('category') or '').lower() == 'status':
        if e.get('status'):
            return 'status'
        if e.get('boosts') or _norm(move_id) in _EXTRA_TARGET_BOOSTS:
            return 'boost'
    return ''


def _status_immune(m: MonSnapshot, status: str, move_id: str, terrain: str = '') -> bool:
    if m.status:
        return True
    t = _tnorm(terrain)
    if t == 'misty' and grounded(m):
        return True
    if t == 'electric' and status == 'slp' and grounded(m):
        return True
    types = {_norm(t) for t in (m.types or ())}
    if 'powder' in _flags(move_id) and _powder_immune(m):
        return True
    if status == 'par':
        return 'electric' in types or (_norm(move_id) == 'thunderwave' and 'ground' in types)
    if status == 'brn':
        return 'fire' in types
    if status in ('psn', 'tox'):
        return 'poison' in types or 'steel' in types
    if status == 'slp':
        return _norm(m.ability) in ('insomnia', 'vitalspirit', 'sweetveil')
    return False


def is_damaging(move_id: str) -> bool:
    entry = dex_move(move_id)
    if str(entry.get('category') or '').lower() == 'status':
        return False
    return float(entry.get('basePower') or 0) > 0 or bool(entry.get('damage'))


def fast_clone(snap: BattleSnapshot) -> BattleSnapshot:
    """Copia suficiente para una transición: solo cambian hp/fainted/is_active y las listas."""
    def side(s: SideSnapshot) -> SideSnapshot:
        return SideSnapshot(
            active=[copy.copy(m) if m is not None else None for m in s.active],
            bench=[copy.copy(m) if m is not None else None for m in s.bench],
            reserve=s.reserve,
            mega_used=s.mega_used,
        )
    out = copy.copy(snap)
    out.our = side(snap.our)
    out.foe = side(snap.foe)
    out.field = copy.copy(snap.field)  # fase 3c cambia Tailwind / Trick Room / clima
    return out


def _key(m: MonSnapshot) -> Tuple[str, str]:
    return getattr(m, '_s3', None) or ('?', _norm(m.display_species()))


@dataclass
class S3Stats:
    requests: int = 0
    missing: int = 0
    build_sec: float = 0.0
    error: str = ''
    foe_no_moves: int = 0
    foe_point_stats: int = 0


@dataclass
class S3Model:
    # (atacante, defensor, movimiento, área) → (mín %, máx %) del HP máximo del defensor
    table: Dict[tuple, Tuple[float, float]] = field(default_factory=dict)
    speed: Dict[Tuple[str, str], int] = field(default_factory=dict)
    prio: Dict[Tuple[Tuple[str, str], str], int] = field(default_factory=dict)
    trick_room: bool = False
    phase: str = '3a'
    # Fase 3c: estado de la raíz (para ajustar la tabla) y velocidad base por Pokémon.
    root_boosts: Dict[Tuple[str, str], Dict[str, int]] = field(default_factory=dict)
    root_status: Dict[Tuple[str, str], str] = field(default_factory=dict)
    spe_raw: Dict[Tuple[str, str], int] = field(default_factory=dict)
    root_weather: str = ''
    root_terrain: str = ''
    stats: S3Stats = field(default_factory=S3Stats)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        root: BattleSnapshot,
        *,
        stat_beliefs=None,
        ranges_fn: Optional[Callable] = None,
        phase: str = '3a',
    ) -> 'S3Model':
        """Etiqueta los Pokémon de ``root`` (se modifica in situ: es la copia preparada)
        y resuelve la tabla de daño en un lote."""
        from showdown.rl import obs_calc as C
        from showdown.rl.observation import _profile
        from showdown.stat_inference import effective_speed, move_priority

        t0 = time.perf_counter()
        model = cls(trick_room=bool(getattr(root.field, 'trick_room', False)), phase=str(phase))
        st = model.stats

        our = [m for m in list(root.our.active) + list(root.our.bench) if m is not None]
        foe = [m for m in list(root.foe.active) + list(root.foe.bench) if m is not None]
        for m in our:
            m._s3 = ('o', _norm(m.display_species()))
        for m in foe:
            m._s3 = ('f', _norm(m.display_species()))
            if stat_beliefs is not None and not m.stats:
                pts = _point_sp_bounds(stat_beliefs, m.display_species())
                if pts:
                    m.sp_bounds = pts
                    st.foe_point_stats += 1
        st.foe_no_moves = sum(1 for m in root.foe.active if m is not None and not m.moves)

        profs = {id(m): _profile(m) for m in our + foe}
        on_field = {getattr(profs.get(id(m)), 'ability', '') for m in list(root.our.active) + list(root.foe.active)
                    if m is not None and profs.get(id(m)) is not None and not m.fainted}
        terrain = str(getattr(root.field, 'terrain', '') or '')
        fainted = {True: C.fainted_count(root.our), False: C.fainted_count(root.foe)}

        def ally_prof(m, side_active):
            for i, x in enumerate(side_active):
                if x is m:
                    o = side_active[1 - i] if len(side_active) > 1 else None
                    return profs.get(id(o)) if o is not None and not o.fainted else None
            return None

        reqs: Dict[tuple, tuple] = {}

        def add(att, dfn, mid, spread, att_ours, same_side=False):
            ap, dp = profs.get(id(att)), profs.get(id(dfn))
            if ap is None or dp is None:
                return
            k = (_key(att), _key(dfn), mid, bool(spread))
            if k in reqs:
                return
            a_act = root.our.active if att_ours else root.foe.active
            d_act = a_act if same_side else (root.foe.active if att_ours else root.our.active)
            req = C.build_request(att, ap, dfn, dp, mid, root.field, attacker_ours=att_ours,
                                  att_ally_prof=ally_prof(att, a_act), def_ally_prof=ally_prof(dfn, d_act),
                                  on_field=on_field, is_spread=bool(spread), mode='mid',
                                  allies_fainted=fainted[att_ours])
            hp = float(C._stats(dp, 'mid', attacker=False)['hp'])
            reqs[k] = (req, max(1.0, hp))

        for att_ours, atts, dfns in ((True, our, foe), (False, foe, our)):
            mates = our if att_ours else foe
            for att in atts:
                for mv in att.moves[:4]:
                    mid = _norm(mv)
                    if not mid or not is_damaging(mid):
                        continue
                    tgt = move_target(mid)
                    for dfn in dfns:
                        add(att, dfn, mid, False, att_ours)
                        if tgt in (_SPREAD_FOES, _SPREAD_ALL):
                            add(att, dfn, mid, True, att_ours)
                    if tgt == _SPREAD_ALL:
                        for mate in mates:
                            if mate is not att:
                                add(att, mate, mid, True, att_ours, same_side=True)

        if ranges_fn is None:
            import os
            if os.environ.get('MCTS_S3_CALC', '1').strip().lower() in ('0', 'false', 'no', 'off'):
                ranges_fn = lambda rs: [None] * len(rs)  # sin Node (tests): respaldo S2 por pedido
            else:
                from showdown.damage_calc_bridge import damage_ranges_batch

                # Sin caché en disco: la clave lleva el HP actual y casi no se repite; con disco
                # la caché creció a 360 GB en un día y llenó el disco del servidor (20-11-2026).
                def ranges_fn(rs):
                    return damage_ranges_batch(rs, persist=False)
        keys = list(reqs)
        st.requests = len(keys)
        try:
            res = ranges_fn([reqs[k][0] for k in keys]) if keys else []
        except Exception as exc:  # calculadora caída → respaldo S2 por pedido
            res = [None] * len(keys)
            st.error = repr(exc)[:200]
        for k, r in zip(keys, res):
            if r is None:
                st.missing += 1
                continue
            hp = reqs[k][1]
            model.table[k] = (100.0 * float(r[0]) / hp, 100.0 * float(r[1]) / hp)

        for m in our + foe:
            p = profs.get(id(m))
            if m.stats and m.stats.get('spe'):
                spe = int(m.stats['spe'])
            elif p is not None:
                spe = int(round(0.5 * (p.smin['spe'] + p.smax['spe'])))
            else:
                spe = 100
            status = str(m.status or '').lower()
            model.spe_raw[_key(m)] = int(spe)
            model.root_boosts[_key(m)] = dict(m.boosts or {})
            model.root_status[_key(m)] = status
            model.speed[_key(m)] = int(effective_speed(
                spe, boost=int((m.boosts or {}).get('spe') or 0), item=m.item or None,
                paralyzed=status.startswith('par'),
            ))
            for mv in m.moves[:4]:
                mid = _norm(mv)
                if mid:
                    pr = move_priority(mid, m.ability or None, terrain or None)
                    model.prio[(_key(m), mid)] = int(pr) if pr is not None else 0
        model.root_weather = _wnorm(getattr(root.field, 'weather', ''))
        model.root_terrain = _tnorm(getattr(root.field, 'terrain', ''))
        if model.phase in ('3c', '3d'):  # sueño en la raíz: status_counter = turnos ya dormidos -> restantes
            for m in our + foe:
                if str(m.status or '').startswith('slp'):
                    m.status_counter = max(1, 3 - int(getattr(m, 'status_counter', 0) or 0))
        st.build_sec = time.perf_counter() - t0
        return model

    # --------------------------------------------------------------- consultas
    def damage_range(self, att: MonSnapshot, dfn: MonSnapshot, mid: str, spread: bool,
                     weather: str = '') -> Tuple[float, float]:
        r = self.table.get((_key(att), _key(dfn), mid, bool(spread)))
        if r is not None:
            return r
        # Respaldo (pedido sin respuesta): heurística de S2, como antes.
        lo, hi = estimate_damage_pct(mid, attacker_types=att.types, defender_types=dfn.types,
                                     weather=weather)
        f = 0.75 if spread else 1.0
        return lo * f, hi * f

    def _targets(self, side: SideSnapshot, opp: SideSnapshot, slot: int, act, mid: str
                 ) -> Tuple[List[MonSnapshot], bool]:
        """Defensores efectivos y si el golpe es de área (×0,75)."""
        tgt = move_target(mid)
        foes = [m for m in opp.active if _alive(m)]
        ally = side.active[1 - slot] if len(side.active) > 1 else None
        if tgt == _SPREAD_FOES:
            return foes, len(foes) > 1
        if tgt == _SPREAD_ALL:
            hit = foes + ([ally] if _alive(ally) else [])
            return hit, len(hit) > 1
        if tgt in _SINGLE:
            if act.target == 2:
                return ([ally] if _alive(ally) else []), False
            t = int(min(max(act.target, 0), 1))
            want = opp.active[t] if t < len(opp.active) else None
            if _alive(want):
                return [want], False
            return foes[:1], False  # redirección al otro rival
        return [], False  # estado / self / campo: sin efecto en 3a

    # --------------------------------------------------------- fase 3c: estado
    def cur_speed(self, s: BattleSnapshot, m: MonSnapshot, ours: bool) -> float:
        from showdown.stat_inference import effective_speed
        k = _key(m)
        spe = float(effective_speed(
            int(self.spe_raw.get(k, 100)), boost=int((m.boosts or {}).get('spe') or 0), item=m.item or None,
            paralyzed=str(m.status or '').startswith('par'),
            tailwind=bool(s.field.our_tailwind if ours else s.field.opp_tailwind),
        ))
        w = _wnorm(s.field.weather)
        if w and w in _WEATHER_SPEED.get(_norm(m.ability), ()):
            spe *= 2.0
        return -spe if s.field.trick_room else spe

    def _mods(self, att: MonSnapshot, dfn: MonSnapshot, mid: str, weather: str, terrain: str = '') -> float:
        """Ajuste de la tabla (calculada en la raíz) por etapas, quemadura y clima actuales."""
        e = dex_move(mid)
        cat = str(e.get('category') or '').lower()
        a_st, d_st = ('atk', 'def') if cat == 'physical' else ('spa', 'spd')
        ka, kd = _key(att), _key(dfn)
        ra, rd = self.root_boosts.get(ka, {}), self.root_boosts.get(kd, {})
        f = (stage_mult((att.boosts or {}).get(a_st, 0)) / stage_mult(ra.get(a_st, 0))
             * stage_mult(rd.get(d_st, 0)) / stage_mult((dfn.boosts or {}).get(d_st, 0)))
        if cat == 'physical' and _norm(att.ability) != 'guts':
            now_b = str(att.status or '').startswith('brn')
            was_b = self.root_status.get(ka, '').startswith('brn')
            if now_b and not was_b:
                f *= 0.5
            elif was_b and not now_b:
                f *= 2.0
        f *= _weather_factor(e.get('type', ''), weather) / _weather_factor(e.get('type', ''), self.root_weather)
        if self.phase == '3d':
            f *= (_terrain_factor(att, dfn, mid, terrain)
                  / _terrain_factor(att, dfn, mid, self.root_terrain))
        return f

    @staticmethod
    def _boost(m: MonSnapshot, boosts: Dict[str, int], *, by_foe: bool) -> None:
        if not boosts or not _alive(m):
            return
        ab = _norm(m.ability)
        b = dict(m.boosts or {})
        dropped = False
        for k, d in boosts.items():
            d = int(d)
            if d < 0 and by_foe and ab in _DROP_IMMUNE:
                continue
            b[k] = int(max(-6, min(6, int(b.get(k, 0)) + d)))
            dropped = dropped or d < 0
        if dropped and by_foe:
            if ab == 'defiant':
                b['atk'] = int(min(6, int(b.get('atk', 0)) + 2))
            elif ab == 'competitive':
                b['spa'] = int(min(6, int(b.get('spa', 0)) + 2))
        m.boosts = b

    def _set_weather(self, s: BattleSnapshot, w: str) -> None:
        s.field.weather = _wnorm(w)
        s.field.weather_turns = 5.0

    def on_entry(self, s: BattleSnapshot, m: MonSnapshot, ours: bool) -> None:
        """Habilidades al entrar (fase 3c): Intimidate y clima."""
        if self.phase not in ('3c', '3d') or not _alive(m):
            return
        ab = _norm(m.ability)
        if ab == 'intimidate':
            opp = s.foe if ours else s.our
            for f in opp.active:
                if _alive(f) and _norm(f.ability) not in _INTIM_IMMUNE:
                    self._boost(f, {'atk': -1}, by_foe=True)
                elif _alive(f) and _norm(f.ability) == 'guarddog':
                    self._boost(f, {'atk': 1}, by_foe=False)
        if ab in _WEATHER_ABIL:
            self._set_weather(s, _WEATHER_ABIL[ab])
        if self.phase == '3d' and ab in _TERRAIN_ABIL:
            s.field.terrain = _TERRAIN_ABIL[ab] + 'terrain'
            s.field.terrain_turns = 5.0

    def _pivot(self, s: BattleSnapshot, side: SideSnapshot, actor: MonSnapshot, ours: bool) -> None:
        """U-turn / Volt Switch / Parting Shot: entra el de banca con más HP."""
        slot = next((i for i, x in enumerate(side.active) if x is actor), None)
        if slot is None:
            return
        best = None
        for bi, b in enumerate(side.bench):
            if _alive(b) and (best is None or b.hp_pct > side.bench[best].hp_pct):
                best = bi
        if best is None:
            return
        inc = side.bench[best]
        side.bench[best], side.active[slot] = actor, inc
        inc.is_active, actor.is_active = True, False
        inc.turns_active = 0
        inc.protect_counter = 0
        self.on_entry(s, inc, ours)

    def replace_fainted(self, s: BattleSnapshot) -> None:
        """Reemplazo tras KO (como ``openloop.auto_replace``) + efectos al entrar en 3c."""
        before = {(o, i): (s.our if o else s.foe).active[i] for o in (True, False) for i in range(2)}
        from showdown.rl.mcts_puct.openloop import auto_replace
        auto_replace(s)
        if self.phase not in ('3c', '3d'):
            return
        for (o, i), old in before.items():
            side = s.our if o else s.foe
            new = side.active[i] if i < len(side.active) else None
            if new is not None and new is not old:
                self.on_entry(s, new, o)

    def _end_of_turn(self, s: BattleSnapshot) -> None:
        w = _wnorm(s.field.weather)
        for m in list(s.our.active) + list(s.foe.active):
            if not _alive(m):
                continue
            st = str(m.status or '')
            chip = 0.0
            if st.startswith('brn'):
                chip += 6.25
            elif st.startswith('psn') or st.startswith('tox'):
                chip += 12.5
            types = {_norm(t) for t in (m.types or ())}
            if w == 'sandstorm' and not types & {'rock', 'ground', 'steel'} and _norm(m.ability) not in (
                    'sandveil', 'sandrush', 'sandforce', 'overcoat', 'magicguard'):
                chip += 6.25
            if self.phase == '3d' and _tnorm(s.field.terrain) == 'grassy' and grounded(m):
                chip -= 6.25
            if chip:
                m.hp_pct = min(100.0, max(0.0, float(m.hp_pct) - chip))
                if m.hp_pct <= 0.0:
                    m.fainted, m.hp_pct = True, 0.0
        f = s.field
        for flag, turns in (('our_tailwind', 'our_tailwind_turns'), ('opp_tailwind', 'opp_tailwind_turns'),
                            ('trick_room', 'trick_room_turns')):
            if getattr(f, flag):
                t = float(getattr(f, turns) or 0.0)
                t = (t if t > 0 else 3.0) - 1.0  # sin dato de la raíz: quedan ~2
                setattr(f, turns, t)
                if t <= 0:
                    setattr(f, flag, False)
        if self.phase == '3d' and f.terrain:
            tt = float(f.terrain_turns or 0.0)
            tt = (tt if tt > 0 else 4.0) - 1.0  # sin dato de la raíz: quedan ~3
            f.terrain_turns = tt
            if tt <= 0:
                f.terrain = ''
        if f.weather:
            t = float(f.weather_turns or 0.0)
            if t > 0:
                f.weather_turns = t - 1.0
                if f.weather_turns <= 0:
                    f.weather = ''

    # ------------------------------------------------------------- transición
    def step(self, snap: BattleSnapshot, joint_self: int, joint_opp: int,
             rng: np.random.Generator, *, max_turns: int = 40) -> BattleSnapshot:
        s = fast_clone(snap)
        if s.finished:
            return s
        weather = str(getattr(s.field, 'weather', '') or '')
        plan = []  # (lado propio?, hueco, acción)
        for ours, joint in ((True, joint_self), (False, joint_opp)):
            a1, a2 = decode_joint_action(int(joint))
            plan.append((ours, 0, decode_slot_action(int(a1))))
            plan.append((ours, 1, decode_slot_action(int(a2))))

        # 1) cambios (antes que cualquier movimiento)
        for ours, slot, act in plan:
            if act.kind != 'switch':
                continue
            side = s.our if ours else s.foe
            bi = int(act.switch_index)
            if bi < len(side.bench) and _alive(side.bench[bi]) and slot < len(side.active):
                out_m, in_m = side.active[slot], side.bench[bi]
                side.active[slot], side.bench[bi] = in_m, out_m
                in_m.is_active = True
                in_m.turns_active = 0
                in_m.protect_counter = 0
                if out_m is not None:
                    out_m.is_active = False
                self.on_entry(s, in_m, ours)

        # 2) movimientos por prioridad y velocidad
        queue = []
        for ours, slot, act in plan:
            if act.kind not in ('move', 'mega', 'tera'):
                continue
            side = s.our if ours else s.foe
            actor = side.active[slot] if slot < len(side.active) else None
            if not _alive(actor) or act.move_index >= len(actor.moves):
                continue
            mid = _norm(actor.moves[act.move_index])
            if not mid:
                continue
            k = _key(actor)
            spe = float(self.speed.get(k, 100))
            tw = s.field.our_tailwind if ours else s.field.opp_tailwind
            if tw:
                spe *= 2.0
            if self.trick_room:
                spe = -spe
            pr = self.prio.get((k, mid), 0)
            if self.phase == '3d' and mid == 'grassyglide':
                pr = 1 if (_tnorm(s.field.terrain) == 'grassy' and grounded(actor)) else 0
            queue.append((pr, spe, float(rng.random()), ours, slot, act, actor, mid))
        queue.sort(key=lambda q: (q[0], q[1], q[2]), reverse=True)

        c3 = self.phase in ('3c', '3d')
        b3 = self.phase in ('3b', '3c', '3d')
        d3 = self.phase == '3d'
        protected: set = set()           # id() de los protegidos este turno
        wide_guard = {True: False, False: False}
        redirect: Dict[bool, Optional[Tuple[MonSnapshot, bool]]] = {True: None, False: None}
        helped: set = set()              # id() de quien recibe Helping Hand
        flinched: set = set()            # id() de quien retrocede

        pending = list(queue)
        while pending:
            if c3:  # velocidad dinámica (Gen 9): el siguiente se elige con la velocidad actual
                ib = max(range(len(pending)), key=lambda i: (
                    pending[i][0],
                    self.cur_speed(s, pending[i][6], pending[i][3]) if _alive(pending[i][6]) else -1e9,
                    pending[i][2]))
                _p, _sp, _tb, ours, slot, act, actor, mid = pending.pop(ib)
            else:
                _p, _sp, _tb, ours, slot, act, actor, mid = pending.pop(0)
            if not _alive(actor):
                continue  # cayó antes de actuar
            side, opp = (s.our, s.foe) if ours else (s.foe, s.our)
            slot_now = next((i for i, x in enumerate(side.active) if x is actor), None)
            if slot_now is None:
                continue
            if b3:
                if id(actor) in flinched:
                    actor.protect_counter = 0
                    continue
                eff = move_effect(mid)
                if eff == 'protect':
                    n = int(getattr(actor, 'protect_counter', 0) or 0)
                    if rng.random() < (1.0 / 3.0) ** n:
                        protected.add(id(actor))
                        actor.protect_counter = n + 1
                    else:
                        actor.protect_counter = 0
                    continue
                actor.protect_counter = 0
                if eff == 'wideguard':
                    wide_guard[ours] = True
                    continue
                if eff in ('redirect', 'redirect_powder'):
                    redirect[ours] = (actor, eff == 'redirect_powder')
                    continue
                if eff == 'helpinghand':
                    ally = side.active[1 - slot_now] if len(side.active) > 1 else None
                    if _alive(ally):
                        helped.add(id(ally))
                    continue
                if mid in _FIRST_TURN and int(getattr(actor, 'turns_active', 0) or 0) > 0:
                    continue  # Fake Out / First Impression fallan fuera del primer turno
            if c3:
                st = str(actor.status or '')
                if st.startswith('slp'):
                    c = int(getattr(actor, 'status_counter', 0) or 0)
                    if c > 0:
                        actor.status_counter = c - 1
                        continue
                    actor.status = ''  # despierta y actúa
                elif st.startswith('par') and rng.random() < 0.25:
                    continue
                e3 = move_effect3(mid)
                if e3 == 'tailwind':
                    if ours:
                        s.field.our_tailwind, s.field.our_tailwind_turns = True, 4.0
                    else:
                        s.field.opp_tailwind, s.field.opp_tailwind_turns = True, 4.0
                    continue
                if e3 == 'trickroom':
                    on = not s.field.trick_room
                    s.field.trick_room, s.field.trick_room_turns = on, (5.0 if on else 0.0)
                    continue
                if e3 == 'weather':
                    self._set_weather(s, dex_move(mid).get('weather', ''))
                    continue
                if e3 in ('status', 'boost'):
                    e = dex_move(mid)
                    if move_target(mid) == 'self':
                        self._boost(actor, e.get('boosts') or {}, by_foe=False)
                    else:
                        tg, _spr = self._targets(side, opp, slot_now, act, mid)
                        if tg and move_target(mid) in _SINGLE and act.target != 2:
                            rd = redirect[not ours]
                            if rd is not None and _alive(rd[0]) and not (rd[1] and _powder_immune(actor)):
                                tg = [rd[0]]
                        acc3 = move_accuracy(mid)
                        for dfn in tg:
                            if id(dfn) in protected and blockable(mid):
                                continue
                            if d3 and _p > 0 and _tnorm(s.field.terrain) == 'psychic' and grounded(dfn) \
                                    and not any(dfn is x for x in side.active):
                                continue  # Psychic Terrain: prioridad bloqueada (Prankster…)
                            if acc3 < 1.0 and rng.random() > acc3:
                                continue
                            if e3 == 'status':
                                stt = str(e.get('status') or '')
                                if not _status_immune(dfn, stt, mid, s.field.terrain if d3 else ''):
                                    dfn.status = stt
                                    if stt == 'slp':
                                        dfn.status_counter = int(rng.integers(1, 4))
                            else:
                                self._boost(dfn, e.get('boosts') or _EXTRA_TARGET_BOOSTS.get(mid, {}), by_foe=True)
                    if e.get('selfSwitch'):
                        self._pivot(s, side, actor, ours)
                    continue
            targets, spread = self._targets(side, opp, slot_now, act, mid)
            if b3 and targets and move_target(mid) in _SINGLE and act.target != 2:
                rd = redirect[not ours]
                if rd is not None and _alive(rd[0]) and not (rd[1] and _powder_immune(actor)):
                    targets = [rd[0]]
            ef = (d3 and mid == 'expandingforce' and _tnorm(s.field.terrain) == 'psychic' and grounded(actor))
            if ef:
                targets = [m for m in opp.active if _alive(m)]
                spread = len(targets) > 1
            if not targets or not is_damaging(mid):
                continue
            acc = move_accuracy(mid)
            hit_any = False
            for dfn in targets:
                if b3:
                    dfn_ours = any(dfn is x for x in s.our.active)
                    if spread and wide_guard[dfn_ours]:
                        continue
                    if id(dfn) in protected:
                        if breaks_protect(mid):
                            protected.discard(id(dfn))
                        elif blockable(mid):
                            continue
                if d3 and _p > 0 and _tnorm(s.field.terrain) == 'psychic' and grounded(dfn) \
                        and not any(dfn is x for x in side.active):
                    continue  # Psychic Terrain: Fake Out, Sucker Punch, Aqua Jet… no golpean
                if acc < 1.0 and rng.random() > acc:
                    continue
                if ef:  # la tabla solo tiene Expanding Force individual
                    lo, hi = self.damage_range(actor, dfn, mid, False, weather)
                    k_ef = 1.5 * (0.75 if spread else 1.0)
                    lo, hi = lo * k_ef, hi * k_ef
                else:
                    lo, hi = self.damage_range(actor, dfn, mid, spread, weather)
                dmg = float(rng.uniform(lo, hi)) if hi > lo else float(lo)
                if b3 and id(actor) in helped:
                    dmg *= 1.5
                if c3:
                    dmg *= self._mods(actor, dfn, mid, s.field.weather, s.field.terrain if d3 else '')
                hit_any = True
                dfn.hp_pct = max(0.0, float(dfn.hp_pct) - dmg)
                if dfn.hp_pct <= 0.0:
                    dfn.fainted = True
                    dfn.hp_pct = 0.0
                    continue
                if b3 and mid == 'fakeout' and not _flinch_immune(dfn):
                    flinched.add(id(dfn))
                if c3:  # secundarios sobre el objetivo
                    for sec in dex_move(mid).get('secondaries') or []:
                        if rng.random() * 100.0 >= float(sec.get('chance', 100) or 100):
                            continue
                        if sec.get('status') and not _status_immune(dfn, str(sec['status']), mid,
                                                                    s.field.terrain if d3 else ''):
                            dfn.status = str(sec['status'])
                            if dfn.status == 'slp':
                                dfn.status_counter = int(rng.integers(1, 4))
                        if sec.get('boosts'):
                            self._boost(dfn, sec['boosts'], by_foe=True)
            if c3 and hit_any and _alive(actor):
                e = dex_move(mid)
                if e.get('selfBoosts'):
                    self._boost(actor, e['selfBoosts'], by_foe=False)
                if e.get('selfSwitch'):
                    self._pivot(s, side, actor, ours)

        if c3:
            self._end_of_turn(s)
        if b3:
            for m in list(s.our.active) + list(s.foe.active):
                if _alive(m):
                    m.turns_active = int(getattr(m, 'turns_active', 0) or 0) + 1
        s.turn = int(s.turn) + 1
        our_alive = any(_alive(m) for m in s.our.all_mons())
        foe_alive = any(_alive(m) for m in s.foe.all_mons())
        if not foe_alive:
            s.finished, s.won, s.tied = True, True, False
        elif not our_alive:
            s.finished, s.won, s.tied = True, False, False
        elif s.turn > int(max_turns):
            s.finished, s.won, s.tied = True, None, False
        return s

    # ------------------------------------------------------------- puntuación
    def slot_scores(self, view: BattleSnapshot, slot: int) -> Dict[int, float]:
        """Valor inmediato de cada acción de movimiento del hueco (perspectiva ``view.our``)."""
        out: Dict[int, float] = {}
        side, opp = view.our, view.foe
        actor = side.active[slot] if slot < len(side.active) else None
        if not _alive(actor):
            return out
        weather = str(getattr(view.field, 'weather', '') or '')
        from showdown.rl.snapshot import SlotAction
        for mi, mv in enumerate(actor.moves[:4]):
            mid = _norm(mv)
            if not mid or not is_damaging(mid):
                continue
            acc = move_accuracy(mid)
            for t in range(4):
                targets, spread = self._targets(side, opp, slot, SlotAction(kind='move', move_index=mi, target=t), mid)
                v = 0.0
                for dfn in targets:
                    lo, hi = self.damage_range(actor, dfn, mid, spread, weather)
                    hp = float(dfn.hp_pct)
                    exp = min(0.5 * (lo + hi), hp)
                    p_ko = 1.0 if lo >= hp else (0.0 if hi < hp else (hi - hp) / max(1e-6, hi - lo))
                    sign = -1.0 if any(dfn is x for x in side.active) else 1.0  # golpear al aliado resta
                    v += sign * acc * (exp + KO_BONUS * p_ko)
                out[mi * 4 + t] = v
            if self.phase in ('3b', '3c', '3d') and mid in _FIRST_TURN:
                first = int(getattr(actor, 'turns_active', 0) or 0) == 0
                for t in range(4):
                    if (mi * 4 + t) not in out:
                        continue
                    if not first:
                        out[mi * 4 + t] = 0.0  # falla
                    elif mid == 'fakeout' and t in (0, 1):
                        tg = opp.active[t] if t < len(opp.active) else None
                        if _alive(tg) and not _flinch_immune(tg):
                            out[mi * 4 + t] += FLINCH_W * self.best_output(view, tg, flip=True)
        if self.phase in ('3b', '3c', '3d'):
            self._support_scores(view, slot, actor, out)
        if self.phase in ('3c', '3d'):
            self._scores_3c(view, slot, actor, out)
        return out

    # --------------------------------------------------- puntuación (fase 3c)
    def _speed_share(self, view: BattleSnapshot, *, tailwind: bool = False, trick: Optional[bool] = None) -> float:
        """Fracción de duelos (nuestro activo, rival activo) que ganamos en velocidad."""
        f = view.field
        saved = (f.our_tailwind, f.trick_room)
        if tailwind:
            f.our_tailwind = True
        if trick is not None:
            f.trick_room = trick
        try:
            n = w = 0
            for a in view.our.active:
                for b in view.foe.active:
                    if _alive(a) and _alive(b):
                        n += 1
                        w += self.cur_speed(view, a, True) > self.cur_speed(view, b, False)
            return (w / n) if n else 0.0
        finally:
            f.our_tailwind, f.trick_room = saved

    def _scores_3c(self, view: BattleSnapshot, slot: int, actor: MonSnapshot, out: Dict[int, float]) -> None:
        side, opp = view.our, view.foe
        from showdown.rl.snapshot import SlotAction
        for mi, mv in enumerate(actor.moves[:4]):
            mid = _norm(mv)
            if not mid:
                continue
            e = dex_move(mid)
            e3 = move_effect3(mid)
            if e3 == 'tailwind':
                base = self._speed_share(view)
                gain = self._speed_share(view, tailwind=True) - base
                out[mi * 4 + 3] = 0.0 if view.field.our_tailwind else TAILWIND_W * gain
            elif e3 == 'trickroom':
                base = self._speed_share(view)
                out[mi * 4 + 3] = TAILWIND_W * (self._speed_share(view, trick=not view.field.trick_room) - base)
            elif e3 == 'status':
                stt = str(e.get('status') or '')
                for t in (0, 1):
                    tgts, _ = self._targets(side, opp, slot, SlotAction(kind='move', move_index=mi, target=t), mid)
                    v = 0.0
                    for d in tgts:
                        if _status_immune(d, stt, mid, view.field.terrain if self.phase == '3d' else ''):
                            continue
                        outp = self.best_output(view, d, flip=True)
                        if stt == 'slp':
                            v += SLEEP_W * outp
                        elif stt == 'par':
                            v += PARA_W * outp
                        elif stt == 'brn':
                            phys = any(str(dex_move(_norm(x)).get('category', '')).lower() == 'physical'
                                       for x in d.moves[:4] if x)
                            v += (BURN_W if phys else 0.1) * outp
                        else:
                            v += POISON_V
                    out[mi * 4 + t] = move_accuracy(mid) * v
            elif e3 == 'boost' and move_target(mid) == 'self':
                b = e.get('boosts') or {}
                own = self.best_output(view, actor)
                v = BOOST_W * own * (max(0, int(b.get('atk', 0))) + max(0, int(b.get('spa', 0))))
                v += SPEEDCTL_V * max(0, int(b.get('spe', 0)))
                out[mi * 4 + 3] = v
            elif e3 == 'boost':
                for t in (0, 1):
                    tgts, _ = self._targets(side, opp, slot, SlotAction(kind='move', move_index=mi, target=t), mid)
                    out[mi * 4 + t] = sum(0.2 * self.best_output(view, d, flip=True) for d in tgts)
            if is_damaging(mid):  # control de velocidad por secundarios seguros (Icy Wind, Electroweb)
                for sec in e.get('secondaries') or []:
                    if float(sec.get('chance', 0) or 0) >= 100 and (sec.get('boosts') or {}).get('spe', 0) < 0:
                        for t in range(4):
                            if (mi * 4 + t) in out:
                                out[mi * 4 + t] += SPEEDCTL_V * sum(1 for d in opp.active if _alive(d))

    # --------------------------------------------------- amenazas (fase 3b)
    def best_output(self, view: BattleSnapshot, att: MonSnapshot, *, flip: bool = False) -> float:
        """Mejor daño esperado (acotado al HP) de ``att`` contra un activo del otro lado.

        ``flip=True``: ``att`` está en ``view.foe`` y golpea a ``view.our``.
        """
        if not _alive(att):
            return 0.0
        weather = str(getattr(view.field, 'weather', '') or '')
        dfns = view.our.active if flip else view.foe.active
        best = 0.0
        for mv in att.moves[:4]:
            mid = _norm(mv)
            if not mid or not is_damaging(mid):
                continue
            acc = move_accuracy(mid)
            spread = move_target(mid) in (_SPREAD_FOES, _SPREAD_ALL)
            tot = 0.0
            for d in dfns:
                if not _alive(d):
                    continue
                lo, hi = self.damage_range(att, d, mid, spread, weather)
                v = acc * min(0.5 * (lo + hi), float(d.hp_pct))
                tot = (tot + v) if spread else max(tot, v)
            best = max(best, tot)
        return best

    def threat_to(self, view: BattleSnapshot, mon: MonSnapshot, *, spread_only: bool = False,
                  single_only: bool = False) -> float:
        """Daño esperado que ``mon`` (lado ``view.our``) recibiría si cada rival lo eligiera."""
        if not _alive(mon):
            return 0.0
        weather = str(getattr(view.field, 'weather', '') or '')
        tot = 0.0
        for f in view.foe.active:
            if not _alive(f):
                continue
            best = 0.0
            for mv in f.moves[:4]:
                mid = _norm(mv)
                if not mid or not is_damaging(mid):
                    continue
                spread = move_target(mid) in (_SPREAD_FOES, _SPREAD_ALL)
                if (spread_only and not spread) or (single_only and spread):
                    continue
                lo, hi = self.damage_range(f, mon, mid, spread, weather)
                best = max(best, move_accuracy(mid) * 0.5 * (lo + hi))
            tot += best
        return min(tot, float(mon.hp_pct))

    def _support_scores(self, view: BattleSnapshot, slot: int, actor: MonSnapshot,
                        out: Dict[int, float]) -> None:
        side = view.our
        ally = side.active[1 - slot] if len(side.active) > 1 else None
        for mi, mv in enumerate(actor.moves[:4]):
            mid = _norm(mv)
            eff = move_effect(mid) if mid else ''
            if not eff:
                continue
            if eff == 'protect':
                n = int(getattr(actor, 'protect_counter', 0) or 0)
                out[mi * 4 + 3] = PROTECT_W * ((1.0 / 3.0) ** n) * self.threat_to(view, actor)
            elif eff == 'wideguard':
                v = sum(self.threat_to(view, m, spread_only=True) for m in side.active if _alive(m))
                out[mi * 4 + 3] = WIDEGUARD_W * v
            elif eff in ('redirect', 'redirect_powder'):
                out[mi * 4 + 3] = REDIRECT_W * self.threat_to(view, ally, single_only=True) if _alive(ally) else 0.0
            elif eff == 'helpinghand':
                out[mi * 4 + 2] = HELP_W * self.best_output(view, ally) if _alive(ally) else 0.0

    def joint_scores(self, view: BattleSnapshot, joints: Sequence[int]) -> np.ndarray:
        d0 = self.slot_scores(view, 0)
        d1 = self.slot_scores(view, 1)
        out = np.zeros(len(joints), dtype=np.float64)
        for k, j in enumerate(joints):
            a1, a2 = decode_joint_action(int(j))
            out[k] = d0.get(_strip_mega(a1), 0.0) + d1.get(_strip_mega(a2), 0.0)
        return out


def _strip_mega(aid: int) -> int:
    return aid - MEGA_BASE if MEGA_BASE <= aid < MEGA_BASE + 16 else aid


def _point_sp_bounds(stat_beliefs, species: str) -> Optional[Dict[str, Tuple[int, int]]]:
    """SP preferido (prior meta leave-one-out) de cada stat como rango degenerado."""
    try:
        from showdown.stat_inference import STAT_KEYS, normalize_species
        belief = (getattr(stat_beliefs, 'beliefs', None) or {}).get(normalize_species(species))
        if belief is None:
            return None
        out: Dict[str, Tuple[int, int]] = {}
        for stat in STAT_KEYS:
            sp = belief.preferred_sp(stat)
            if sp is not None:
                out[stat] = (int(sp), int(sp))
        return out or None
    except Exception:
        return None


__all__ = ['S3Model', 'S3Stats', 'fast_clone', 'move_target', 'move_accuracy', 'is_damaging']
