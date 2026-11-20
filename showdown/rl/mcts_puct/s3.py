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

Qué NO modela todavía (fases 3b/3c): Protect, Fake Out, redirección, Helping Hand,
Tailwind/Trick Room como acciones, Intimidate, estados, clima/terreno cambiantes,
boosts dentro del árbol, críticos. Los movimientos de estado no hacen nada.
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
    stats: S3Stats = field(default_factory=S3Stats)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        root: BattleSnapshot,
        *,
        stat_beliefs=None,
        ranges_fn: Optional[Callable] = None,
    ) -> 'S3Model':
        """Etiqueta los Pokémon de ``root`` (se modifica in situ: es la copia preparada)
        y resuelve la tabla de daño en un lote."""
        from showdown.rl import obs_calc as C
        from showdown.rl.observation import _profile
        from showdown.stat_inference import effective_speed, move_priority

        t0 = time.perf_counter()
        model = cls(trick_room=bool(getattr(root.field, 'trick_room', False)))
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
            model.speed[_key(m)] = int(effective_speed(
                spe, boost=int((m.boosts or {}).get('spe') or 0), item=m.item or None,
                paralyzed=status.startswith('par'),
            ))
            for mv in m.moves[:4]:
                mid = _norm(mv)
                if mid:
                    pr = move_priority(mid, m.ability or None, terrain or None)
                    model.prio[(_key(m), mid)] = int(pr) if pr is not None else 0
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
                if out_m is not None:
                    out_m.is_active = False

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
            queue.append((self.prio.get((k, mid), 0), spe, float(rng.random()), ours, slot, act, actor, mid))
        queue.sort(key=lambda q: (q[0], q[1], q[2]), reverse=True)

        for _p, _sp, _tb, ours, slot, act, actor, mid in queue:
            if not _alive(actor):
                continue  # cayó antes de actuar
            side, opp = (s.our, s.foe) if ours else (s.foe, s.our)
            slot_now = next((i for i, x in enumerate(side.active) if x is actor), None)
            if slot_now is None:
                continue
            targets, spread = self._targets(side, opp, slot_now, act, mid)
            if not targets or not is_damaging(mid):
                continue
            acc = move_accuracy(mid)
            for dfn in targets:
                if acc < 1.0 and rng.random() > acc:
                    continue
                lo, hi = self.damage_range(actor, dfn, mid, spread, weather)
                dmg = float(rng.uniform(lo, hi)) if hi > lo else float(lo)
                dfn.hp_pct = max(0.0, float(dfn.hp_pct) - dmg)
                if dfn.hp_pct <= 0.0:
                    dfn.fainted = True
                    dfn.hp_pct = 0.0

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
        return out

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
