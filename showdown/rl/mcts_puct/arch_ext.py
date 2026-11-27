"""Extensiones opcionales de G (v8) y H (2.3), protocolo §4.19 (27-11-2026). Apagadas por defecto.

Motivo: D1 (§5.15) y D3 (§4.17) muestran que los baselines casi nunca ejecutan planes (Trick Room, Tailwind, clima),
y el modelo del rival dentro del árbol de G y H tiene el mismo sesgo: el rival se muestrea con el softmax del
puntaje S3, que valora Trick Room / Tailwind por velocidad ganada pero da **0** a los movimientos de clima, sin
mirar el plan del equipo. H reutiliza de G el modelo S3, la hoja y el preview por búsqueda, así que las tres
extensiones valen para los dos agentes:

- ``MCTS_ARCH=1`` (A, arquetipos): bono en el puntaje S3 de cada lado (también el del rival, que usa la vista
  invertida) por poner el clima / campo de su plan si no está activo y por Trick Room / Tailwind si su plan es ese
  (``greedy_joint/archetypes.py``, sobre la hoja abierta). El preview por búsqueda suma, a las formaciones greedy,
  la del plan de cada lado (quien lo pone + quien lo aprovecha o lo protege).
- ``MCTS_SERIES=1`` (S, serie BO3): en las partidas 2–3 el preview simula solo los 4 que el rival trajo antes (sus 6
  parejas de leads), y en el árbol los movimientos que el rival ya usó en la serie suman ``SERIES_MOVE_W``.
- ``MCTS_LEAF_FIELD=1`` (F, hoja): la hoja suma el valor del control de velocidad vigente (fracción de duelos de
  velocidad ganados con el campo actual menos sin Trick Room ni Tailwind) × ``LEAF_FIELD_W``.

Los campos (terreno) solo se bonifican en fases de S3 que los simulan (3d / 3e); G 5.2 y H 2.2 usan la 3c.
"""
from __future__ import annotations

import contextvars
import os
from typing import Dict, List, Optional, Sequence

ARCH_FIELD_W = 30.0      # mismo orden que un buen ataque (puntaje S3 ≈ % de daño)
ARCH_SPEED_W = 15.0      # extra para Trick Room / Tailwind cuando son el plan del equipo
SERIES_MOVE_W = 5.0
LEAF_FIELD_W = 0.15      # la hoja material está en [−1, 1]

# Movimientos que el rival usó en partidas anteriores de la serie: {especie normalizada: {movimientos}}.
SERIES_MOVES: 'contextvars.ContextVar[Optional[Dict[str, set]]]' = contextvars.ContextVar(
    'mcts_series_moves', default=None)


def _flag(name: str) -> bool:
    return os.environ.get(name, '').strip().lower() in ('1', 'true', 'yes', 'on')


def arch_on() -> bool:
    return _flag('MCTS_ARCH')


def series_on() -> bool:
    return _flag('MCTS_SERIES')


def leaf_field_on() -> bool:
    return _flag('MCTS_LEAF_FIELD')


def _id(x) -> str:
    return ''.join(ch for ch in str(x or '').lower() if ch.isalnum())


def _species(m) -> str:
    try:
        return str(m.display_species())
    except Exception:
        return str(getattr(m, 'species', '') or '')


def mon_infos(mons: Sequence) -> List:
    from showdown.battle_bots.greedy_joint import archetypes as A
    return [A.mon_info(i + 1, _species(m), getattr(m, 'ability', '') or '', getattr(m, 'item', '') or '',
                       getattr(m, 'moves', ()) or ()) for i, m in enumerate(mons)]


_PLAN_CACHE: Dict[tuple, object] = {}


def side_plans(side) -> list:
    """Planes del bando (activos, banca y reserva de la hoja abierta): todos los detectados, porque un equipo puede
    correr varios a la vez (p. ej. Trick Room y campo psíquico con Indeedee + Hatterene)."""
    from showdown.battle_bots.greedy_joint import archetypes as A
    mons = [m for m in list(side.active) + list(side.bench) + list(getattr(side, 'reserve', None) or [])
            if m is not None]
    key = tuple(sorted((_id(_species(m)), _id(getattr(m, 'ability', '')), _id(getattr(m, 'item', '')),
                        tuple(_id(x) for x in (getattr(m, 'moves', ()) or ()))) for m in mons))
    if key not in _PLAN_CACHE:
        if len(_PLAN_CACHE) > 4096:
            _PLAN_CACHE.clear()
        _PLAN_CACHE[key] = A.detect_plans(mon_infos(mons))
    return _PLAN_CACHE[key]


def arch_bonus(view, mid: str, phase: str) -> float:
    """A: bono por poner un plan del bando ``view.our`` (perspectiva del que actúa); el mayor si hay varios."""
    from showdown.battle_bots.greedy_joint import archetypes as A
    m = _id(mid)
    f = view.field
    best = 0.0
    for fit in side_plans(view.our):
        b = 0.0
        if fit.plan.kind == 'weather':
            if A.sets_plan_field(m, fit) and not A.plan_field_active(fit, f.weather, None):
                b = ARCH_FIELD_W
        elif fit.plan.kind == 'terrain' and phase in ('3d', '3e'):
            if A.sets_plan_field(m, fit) and not A.plan_field_active(fit, None, f.terrain):
                b = ARCH_FIELD_W
        elif fit.plan.name == 'trick_room' and m == 'trickroom' and not f.trick_room:
            b = ARCH_SPEED_W
        elif fit.plan.name == 'tailwind' and m == 'tailwind' and not f.our_tailwind:
            b = ARCH_SPEED_W
        best = max(best, b)
    return best


def series_bonus(actor, mid: str) -> float:
    """S: el rival repite movimientos que ya usó en la serie (solo actores del bando rival)."""
    mem = SERIES_MOVES.get()
    tag = getattr(actor, '_s3', None)
    if not mem or not tag or tag[0] != 'f':
        return 0.0
    return SERIES_MOVE_W if _id(mid) in mem.get(_id(tag[1]), ()) else 0.0


def apply_scores(model, view, slot: int, actor, out: Dict[int, float]) -> None:
    """A y S sobre el puntaje S3 de ``actor`` (se llama al final de ``S3Model.slot_scores``)."""
    a, s = arch_on(), series_on()
    if not (a or s):
        return
    for mi, mv in enumerate(getattr(actor, 'moves', ())[:4]):
        mid = _id(mv)
        if not mid:
            continue
        if a:
            b = arch_bonus(view, mid, getattr(model, 'phase', ''))
            if b:
                out[mi * 4 + 3] = out.get(mi * 4 + 3, 0.0) + b
        if s:
            b = series_bonus(actor, mid)
            if b:
                for t in range(4):
                    if (mi * 4 + t) in out:
                        out[mi * 4 + t] += b


def field_advantage(model, s) -> float:
    """F: duelos de velocidad ganados con el campo actual menos sin Trick Room ni Tailwind, en [−1, 1]."""
    f = s.field
    try:
        cur = model._speed_share(s)
        saved = (f.trick_room, f.our_tailwind, f.opp_tailwind)
        f.trick_room, f.our_tailwind, f.opp_tailwind = False, False, False
        try:
            neutral = model._speed_share(s)
        finally:
            f.trick_room, f.our_tailwind, f.opp_tailwind = saved
        return float(cur - neutral)
    except Exception:
        return 0.0


def plan_formation(team: Sequence, foes: Sequence) -> Optional[tuple]:
    """A: formación (lead1, lead2, banca1, banca2) del plan del equipo, o None."""
    from showdown.battle_bots.greedy_joint import archetypes as A
    infos, finfos = mon_infos(team), mon_infos(foes)
    order, fit = A.preview_order(infos, finfos, ())
    if fit is None or len(order) < 4:
        return None
    return tuple(int(i) for i in order[:4])


def seen_formations(team: Sequence, brought: Sequence[str]) -> List[tuple]:
    """S: las 6 formaciones de los 4 que el rival trajo antes (por especie), o [] si no se reconocen."""
    from itertools import permutations
    idx = {}
    for i, m in enumerate(team, start=1):
        idx.setdefault(_id(_species(m)), i)
    combo = [idx.get(_id(sp)) for sp in brought]
    if len(combo) != 4 or any(c is None for c in combo):
        return []
    out = []
    for a, b in permutations(combo, 2):
        if a < b:
            rest = tuple(c for c in combo if c not in (a, b))
            out.append((a, b) + rest)
    return out
