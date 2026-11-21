"""MCTS+PUCT v2 — búsqueda open-loop de verdad (docs/mcts_puct_spec.md v2.0).

Diferencias con v1 (``search.py``, que sigue intacto porque MCTS–Nash lo reutiliza):

1. **Open-loop real.** Los nodos guardan solo estadísticas de NUESTRAS secuencias de
   acciones. Cada simulación parte de la raíz y vuelve a muestrear la acción rival y el
   azar en cada ply. En v1 el hijo de una acción se creaba una sola vez (un rival y una
   tirada fijos para siempre).
2. **Acciones podadas + prior.** Raíz = joints legales reales (``valid_orders`` vía
   ``root_joints``) o podados del snapshot; dentro del árbol, joints podados y Top-K por
   daño. Prior P = mezcla softmax(daño)/uniforme; FPU = media del nodo padre. Ya no se
   expande "la primera no probada" en orden de id.
3. **Rival del árbol** = softmax del daño estimado (+ ε uniforme), no uniforme sobre
   ~660 joints que incluían ``pass`` y golpes al aliado.
4. **Hoja** = horizonte corto (``rollout_turns`` turnos greedy) + evaluación heurística
   de material en [−1, 1] (terminal = ±1), en vez de rollout greedy hasta el turno 40.
5. **Higiene de estado.** Banca rival = vistos no excluidos + no vistos más probables
   (hasta 4 traídos); movimientos propios alineados con ``available_moves``; reemplazo
   automático tras KO; movimientos de estado sin el chip falso de S2.

S2 (``simulator.step_sim``) sigue siendo el modelo de transición: velocidad, Protect,
área y stats reales quedan para la siguiente fase (spec §12).
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from showdown.rl.action_space import encode_joint_action
from showdown.rl.constants import PASS_ID
from showdown.rl.mcts_puct import actions as A
from showdown.rl.mcts_puct.search import make_sim_rng
from showdown.rl.mcts_puct.simulator import step_sim
from showdown.rl.snapshot import BROUGHT_N, BattleSnapshot, MonSnapshot, SideSnapshot


@dataclass
class OpenLoopConfig:
    n_sims: int = 200
    c_puct: float = 1.5
    max_turns: int = 40
    seed: int = 0
    # Prior propio: (1−ε)·softmax(daño/τ) + ε·uniforme.
    prior_temperature: float = 20.0
    prior_eps: float = 0.25
    # Top-K de acciones propias dentro del árbol (la raíz no se recorta).
    inner_top_k: int = 12
    # Rival del árbol y de los rollouts.
    opp_temperature: float = 10.0
    opp_eps: float = 0.10
    # Hoja: turnos de rollout greedy antes de evaluar (0 = evaluar directamente).
    rollout_turns: int = 2
    # Profundidad máxima del árbol (plies propios).
    max_depth: int = 6
    prefer_mega: bool = True
    # Modelo de transición: 's2' (v2, simulator.step_sim), 's3' (v3: daño real + orden) o
    # 's3b' (v4: + Protect, Wide Guard, Fake Out, redirección, Helping Hand) o 's3c' (v5: + Tailwind,
    # Trick Room, estados, boosts en el árbol, Intimidate, clima, pivotes).
    model: str = 's2'


@dataclass
class OpenLoopTelemetry:
    sims: int = 0
    nodes: int = 0
    failures: int = 0
    avg_depth: float = 0.0
    root_actions: int = 0
    root_max_n: int = 0
    root_visited: int = 0
    elapsed_sec: float = 0.0
    trivial: bool = False
    model: str = 's2'
    calc_requests: int = 0
    calc_missing: int = 0
    calc_sec: float = 0.0
    calc_error: str = ''
    foe_no_moves: int = 0
    foe_point_stats: int = 0


@dataclass
class Edge:
    N: int = 0
    W: float = 0.0
    P: float = 0.0


@dataclass
class OLNode:
    edges: Dict[int, Edge] = field(default_factory=dict)
    children: Dict[int, 'OLNode'] = field(default_factory=dict)
    N_sum: int = 0
    W_sum: float = 0.0
    expanded: bool = False


# --------------------------------------------------------------------------- estado


def fast_clone(snap: BattleSnapshot) -> BattleSnapshot:
    """Copia suficiente para S2: S2 solo muta hp/fainted/is_active y las listas de lado."""
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


def _alive(m: Optional[MonSnapshot]) -> bool:
    return m is not None and not m.fainted and float(m.hp_pct) > 0.0


def _placeholder(m: MonSnapshot) -> MonSnapshot:
    c = copy.copy(m)
    c.hp_pct = 100.0 if not c.fainted else 0.0
    return c


def prepare_root(snap: BattleSnapshot) -> BattleSnapshot:
    """Copia de la raíz con banca rival creíble y movimientos alineados."""
    s = copy.deepcopy(snap)
    # Movimientos propios en el orden de available_moves (= índice de la acción real).
    if s.avail_moves:
        for slot in (0, 1):
            m = s.our.active[slot] if slot < len(s.our.active) else None
            am = s.avail_moves[slot] if slot < len(s.avail_moves) else ()
            if m is not None and am:
                m.moves = tuple(am)
    # Banca rival: con ``reserve`` (poke-env) la banca v1 era reserve[:2], que podía
    # incluir Pokémon de la hoja que no se trajeron. Ahora: vistos no excluidos y, si
    # faltan para completar 4 traídos, los no vistos con mayor p_brought.
    foe = s.foe
    if foe.reserve:
        n_active = sum(1 for m in foe.active if m is not None)
        seen = [m for m in foe.reserve if m.seen and not m.excluded]
        unseen = sorted(
            (m for m in foe.reserve if not m.seen and not m.excluded),
            key=lambda m: -float(m.p_brought),
        )
        need = max(0, BROUGHT_N - n_active - len(seen))
        bench = seen + [_placeholder(m) for m in unseen[:need]]
        foe.bench = bench[: max(0, BROUGHT_N - n_active)]
    return s


def auto_replace(s: BattleSnapshot) -> None:
    """Tras un KO entra el Pokémon de banca vivo con más HP (S2 no lo hacía)."""
    for side in (s.our, s.foe):
        for slot in range(len(side.active)):
            m = side.active[slot]
            if m is not None and not m.fainted and float(m.hp_pct) > 0:
                continue
            best = None
            for bi, b in enumerate(side.bench):
                if _alive(b) and (best is None or b.hp_pct > side.bench[best].hp_pct):
                    best = bi
            if best is None:
                continue
            incoming = side.bench[best]
            side.bench[best] = m
            side.active[slot] = incoming
            incoming.is_active = True
            incoming.turns_active = 0      # S3 fase 3b: primer turno (Fake Out) y Protect
            incoming.protect_counter = 0
            if m is not None:
                m.is_active = False


def material_value(s: BattleSnapshot) -> float:
    """Valor heurístico en [−1, 1] desde nuestra perspectiva; terminal = ±1 / 0."""
    if s.finished:
        if s.won is True:
            return 1.0
        if s.won is False:
            return -1.0
        return 0.0

    def hp(side: SideSnapshot) -> float:
        tot = 0.0
        n = 0
        for m in list(side.active) + list(side.bench):
            if m is None:
                continue
            n += 1
            if not m.fainted:
                tot += max(0.0, min(100.0, float(m.hp_pct)))
        return tot + 100.0 * max(0, BROUGHT_N - n)

    v = (hp(s.our) - hp(s.foe)) / (100.0 * BROUGHT_N)
    return float(max(-1.0, min(1.0, v)))


# --------------------------------------------------------------------------- árbol


class _Ctx:
    """Modelo de transición y puntuación de una búsqueda (S2 o S3)."""

    def __init__(self, cfg: OpenLoopConfig, model=None):
        self.cfg = cfg
        self.model = model
        self.scorer = model.joint_scores if model is not None else None
        if model is not None:
            from showdown.rl.mcts_puct.s3 import target_class
            self.target_fn = target_class
        else:
            self.target_fn = None


def _step(s: BattleSnapshot, a_self: int, a_opp: int, rng, ctx: '_Ctx') -> BattleSnapshot:
    cfg = ctx.cfg
    if ctx.model is not None:
        out = ctx.model.step(s, a_self, a_opp, rng, max_turns=cfg.max_turns)
    else:
        out = step_sim(s, a_self, a_opp, rng, max_turns=cfg.max_turns,
                       status_chip=False, clone_fn=fast_clone)
    if not out.finished:
        if ctx.model is not None:
            ctx.model.replace_fainted(out)  # S3 3c: + Intimidate / clima al entrar
        else:
            auto_replace(out)
    out.force_switch = (False, False)
    out.avail_moves = None
    return out


def _expand(node: OLNode, s: BattleSnapshot, ctx: '_Ctx',
            joints: Optional[Sequence[int]] = None, *, root: bool = False) -> None:
    cfg = ctx.cfg
    if joints is None:
        joints = A.pruned_joints(s, ctx.target_fn)
    joints = [int(j) for j in joints]
    if cfg.prefer_mega:
        joints = A.prefer_mega(joints)
    scores = (ctx.scorer or A.joint_scores)(s, joints)
    if not root:
        joints, scores = A.top_k(joints, scores, cfg.inner_top_k)
    pri = A.prior_from_scores(scores, cfg.prior_temperature, cfg.prior_eps)
    for j, p in zip(joints, pri):
        node.edges[int(j)] = Edge(P=float(p))
    node.expanded = True


def _select(node: OLNode, s: BattleSnapshot, cfg: OpenLoopConfig, *, root: bool) -> Optional[int]:
    sqrt_n = math.sqrt(max(1, node.N_sum))
    fpu = node.W_sum / node.N_sum if node.N_sum > 0 else 0.0
    best_a, best_u = None, -1e30
    for a, e in node.edges.items():
        if not root and not A.joint_still_valid(s, a):
            continue
        q = e.W / e.N if e.N > 0 else fpu
        u = q + cfg.c_puct * e.P * sqrt_n / (1.0 + e.N)
        if u > best_u or (u == best_u and best_a is not None and a < best_a):
            best_u, best_a = u, a
    return best_a


def _opp(s: BattleSnapshot, rng, ctx: '_Ctx') -> int:
    return A.sample_opponent_joint(s, rng, temperature=ctx.cfg.opp_temperature,
                                   eps=ctx.cfg.opp_eps, scorer=ctx.scorer, target_fn=ctx.target_fn)


def _rollout_value(s: BattleSnapshot, rng, ctx: '_Ctx') -> float:
    for _ in range(max(0, int(ctx.cfg.rollout_turns))):
        if s.finished:
            break
        a_self = A.greedy_joint(s, scorer=ctx.scorer, target_fn=ctx.target_fn)
        s = _step(s, a_self, _opp(s, rng, ctx), rng, ctx)
    return material_value(s)


def _simulate(root: OLNode, root_state: BattleSnapshot, sim_id: int,
              ctx: '_Ctx', tel: OpenLoopTelemetry, *, root_force: bool) -> None:
    cfg = ctx.cfg
    rng = make_sim_rng(cfg.seed, int(root_state.turn), int(sim_id))
    node, s = root, root_state
    path: List[Tuple[OLNode, int]] = []
    depth = 0
    try:
        while True:
            if s.finished:
                z = material_value(s)
                break
            if not node.expanded:
                _expand(node, s, ctx)
                tel.nodes += 1
                z = _rollout_value(s, rng, ctx) if path else material_value(s)
                break
            a = _select(node, s, cfg, root=(node is root))
            if a is None or depth >= cfg.max_depth:
                z = _rollout_value(s, rng, ctx)
                break
            if node is root and root_force:
                # Fase de cambio forzado: el rival no actúa en este ply.
                a_opp = encode_joint_action(PASS_ID, PASS_ID)
            else:
                a_opp = _opp(s, rng, ctx)
            path.append((node, a))
            depth += 1
            s = _step(s, a, a_opp, rng, ctx)
            node = node.children.setdefault(a, OLNode())
    except Exception:
        tel.failures += 1
        z = 0.0
    for n, a in path:
        e = n.edges[a]
        e.N += 1
        e.W += z
        n.N_sum += 1
        n.W_sum += z
    tel.sims += 1
    tel.avg_depth += (depth - tel.avg_depth) / tel.sims


def search_openloop(
    root_state: BattleSnapshot,
    cfg: Optional[OpenLoopConfig] = None,
    *,
    root_joints: Optional[Sequence[int]] = None,
    stat_beliefs=None,
    ranges_fn=None,
) -> Tuple[int, OLNode, OpenLoopTelemetry]:
    """N_sim simulaciones open-loop; devuelve el joint más visitado (desempate Q, id).

    ``cfg.model='s3'``: tabla de daño real (un lote a la calculadora por decisión) con
    ``stat_beliefs`` del rival (prior meta) y orden por velocidad/prioridad.
    """
    cfg = cfg or OpenLoopConfig()
    t0 = time.perf_counter()
    tel = OpenLoopTelemetry(model=str(cfg.model))
    s0 = prepare_root(root_state)
    root_force = bool(any(root_state.force_switch))
    model = None
    if str(cfg.model).lower() in ('s3', 's3b', 's3c'):
        from showdown.rl.mcts_puct.s3 import S3Model
        phase = {'s3': '3a', 's3b': '3b', 's3c': '3c'}[str(cfg.model).lower()]
        model = S3Model.build(s0, stat_beliefs=stat_beliefs, ranges_fn=ranges_fn, phase=phase)
        ms = model.stats
        tel.calc_requests, tel.calc_missing, tel.calc_sec = ms.requests, ms.missing, ms.build_sec
        tel.calc_error, tel.foe_no_moves, tel.foe_point_stats = ms.error, ms.foe_no_moves, ms.foe_point_stats
    ctx = _Ctx(cfg, model)
    root = OLNode()
    _expand(root, s0, ctx, joints=root_joints, root=True)
    tel.root_actions = len(root.edges)
    if not root.edges:
        tel.trivial = True
        tel.elapsed_sec = time.perf_counter() - t0
        return encode_joint_action(PASS_ID, PASS_ID), root, tel
    if len(root.edges) == 1:
        tel.trivial = True
        tel.elapsed_sec = time.perf_counter() - t0
        return int(next(iter(root.edges))), root, tel
    for k in range(int(cfg.n_sims)):
        _simulate(root, s0, k, ctx, tel, root_force=root_force)

    def key(item):
        a, e = item
        q = e.W / e.N if e.N > 0 else -2.0
        return (e.N, q, -a)

    best_a, best_e = max(root.edges.items(), key=key)
    tel.root_max_n = int(best_e.N)
    tel.root_visited = sum(1 for e in root.edges.values() if e.N > 0)
    tel.elapsed_sec = time.perf_counter() - t0
    return int(best_a), root, tel


__all__ = [
    'OpenLoopConfig', 'OpenLoopTelemetry', 'search_openloop', 'prepare_root',
    'auto_replace', 'material_value', 'fast_clone',
]
