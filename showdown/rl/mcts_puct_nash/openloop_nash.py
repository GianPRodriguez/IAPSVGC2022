r"""MCTS–PUCT/Nash v2 (agente H 2.0): SM-MCTS open-loop con el modelo S3 de G (docs/mcts_puct_nash_spec.md v2.0).

Lo que H conserva de v1: cada nodo es un **juego simultáneo de suma cero** entre nuestros joints
\(I(s)\) y los del rival \(J(s)\); la matriz \(\hat A_{ij}\) se llena con simulaciones y el valor que
sube al padre es el valor de Nash \(v^*(\hat A)\) (Fictitious Play), no la media. La acción final es
\(\arg\max \sigma^*\) en la raíz.

Lo que se porta de G (diagnóstico de G §2–§8):

1. **Modelo S3** (``s3.py``, fase 3c por defecto = combate de G v5 confirmado): daño real de la
   calculadora, orden por velocidad/prioridad, Protect, Fake Out, redirección, Tailwind, Trick Room,
   estados, boosts, Intimidate, clima, pivotes. v1 usaba S2.
2. **Open-loop.** Los nodos guardan solo estadísticas de pares de acciones; cada simulación vuelve a
   muestrear el azar desde la raíz. En v1 el hijo de una celda se creaba una vez (una tirada fija).
3. **Acciones podadas + Top-K por S3** para ambos lados (raíz propia = joints legales reales de
   ``valid_orders``) y **prior** \(P_{ij} = p_i q_j\) (softmax del daño estimado). v1 probaba primero
   las celdas no visitadas en orden de id, así que con 50 sims solo veía las primeras filas.
4. **Selección desacoplada contra la mezcla actual.** La fila maximiza
   \(\sum_j \tau_j \hat A_{ij} + c\,p_i\sqrt{N}/(1+n_i)\) y la columna minimiza
   \(\sum_i \sigma_i \hat A_{ij} - c\,q_j\sqrt{N}/(1+n_j)\), con \((\sigma, \tau)\) del último FP del nodo.
   v1 maximizaba \(Q + U\) sobre celdas: optimista para ambos lados (el rival "colaboraba").
5. **Hoja** = 2 turnos greedy + material en [−1, 1], como G (v1: rollout S2 hasta el turno 40).
   Celdas no visitadas entran en \(\hat A\) con la media del nodo (FPU), no con 0.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from showdown.rl.action_space import encode_joint_action
from showdown.rl.constants import PASS_ID
from showdown.rl.mcts_puct import actions as A
from showdown.rl.mcts_puct.openloop import (
    OpenLoopConfig,
    _Ctx,
    _rollout_value,
    _step,
    material_value,
    prepare_root,
)
from showdown.rl.mcts_puct.search import make_sim_rng
from showdown.rl.mcts_puct_nash.nash import fictitious_play
from showdown.rl.snapshot import BattleSnapshot

_PASS2 = encode_joint_action(PASS_ID, PASS_ID)


@dataclass
class NashOLConfig:
    n_sims: int = 200
    c_puct: float = 1.5
    max_turns: int = 40
    seed: int = 0
    model: str = 's3c'          # combate de G v5 (confirmado)
    k_self_root: int = 12       # filas en la raíz (de los joints legales reales)
    k_self: int = 8             # filas dentro del árbol
    k_opp: int = 8              # columnas (rival)
    prior_temperature: float = 20.0
    prior_eps: float = 0.25
    opp_temperature: float = 10.0
    opp_eps: float = 0.10
    rollout_turns: int = 2
    max_depth: int = 4
    prefer_mega: bool = True
    t_fp_internal: int = 100
    t_fp_root: int = 500

    def openloop(self) -> OpenLoopConfig:
        """Config de G para el modelo de transición y los rollouts compartidos."""
        return OpenLoopConfig(
            n_sims=self.n_sims, c_puct=self.c_puct, max_turns=self.max_turns, seed=self.seed,
            prior_temperature=self.prior_temperature, prior_eps=self.prior_eps,
            opp_temperature=self.opp_temperature, opp_eps=self.opp_eps,
            rollout_turns=self.rollout_turns, max_depth=self.max_depth,
            prefer_mega=self.prefer_mega, model=self.model,
        )


@dataclass
class NashOLTelemetry:
    sims: int = 0
    nodes: int = 0
    failures: int = 0
    avg_depth: float = 0.0
    root_rows: int = 0
    root_cols: int = 0
    root_cells_visited: int = 0
    root_value: float = 0.0
    root_sigma_max: float = 0.0
    elapsed_sec: float = 0.0
    trivial: bool = False
    model: str = 's3c'
    calc_requests: int = 0
    calc_missing: int = 0
    calc_sec: float = 0.0
    calc_error: str = ''
    foe_no_moves: int = 0
    foe_point_stats: int = 0


@dataclass
class NashNode:
    I: List[int] = field(default_factory=list)
    J: List[int] = field(default_factory=list)
    p: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    W: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    sigma: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tau: np.ndarray = field(default_factory=lambda: np.zeros(0))
    children: Dict[Tuple[int, int], 'NashNode'] = field(default_factory=dict)
    N: int = 0
    W_sum: float = 0.0
    expanded: bool = False

    def payoff(self) -> np.ndarray:
        """\\(\\hat A\\): media por celda; las no visitadas con la media del nodo (FPU)."""
        fpu = self.W_sum / self.N if self.N > 0 else 0.0
        with np.errstate(invalid='ignore', divide='ignore'):
            q = np.where(self.n > 0, self.W / np.maximum(self.n, 1), fpu)
        return q


def _ranked(view: BattleSnapshot, joints: Sequence[int], ctx: _Ctx, k: int,
            temperature: float, eps: float) -> Tuple[List[int], np.ndarray]:
    joints = [int(j) for j in joints]
    if ctx.cfg.prefer_mega:
        joints = A.prefer_mega(joints)
    if not joints:
        return [_PASS2], np.ones(1)
    scores = ctx.scorer(view, joints)
    joints, scores = A.top_k(joints, scores, k)
    return joints, A.prior_from_scores(scores, temperature, eps)


def _expand(node: NashNode, s: BattleSnapshot, ctx: _Ctx, cfg: NashOLConfig, *,
            root_joints: Optional[Sequence[int]] = None, force_pass_opp: bool = False) -> None:
    rows = root_joints if root_joints is not None else A.pruned_joints(s, ctx.target_fn)
    k_rows = cfg.k_self_root if root_joints is not None else cfg.k_self
    node.I, node.p = _ranked(s, rows, ctx, k_rows, cfg.prior_temperature, cfg.prior_eps)
    if force_pass_opp:
        node.J, node.q = [_PASS2], np.ones(1)
    else:
        view = A.flip_view(s)
        node.J, node.q = _ranked(view, A.pruned_joints(view, ctx.target_fn), ctx, cfg.k_opp,
                                 cfg.opp_temperature, cfg.opp_eps)
    ni, nj = len(node.I), len(node.J)
    node.n = np.zeros((ni, nj))
    node.W = np.zeros((ni, nj))
    node.sigma = np.asarray(node.p, dtype=np.float64).copy()
    node.tau = np.asarray(node.q, dtype=np.float64).copy()
    node.expanded = True


def _select(node: NashNode, s: BattleSnapshot, c: float, *, check_rows: bool) -> Tuple[int, int]:
    Ahat = node.payoff()
    sq = math.sqrt(max(1, node.N))
    n_row = node.n.sum(axis=1)
    n_col = node.n.sum(axis=0)
    u_row = Ahat @ node.tau + c * node.p * sq / (1.0 + n_row)
    if check_rows:  # open-loop: una fila puede no ser jugable en este estado muestreado
        for i, a in enumerate(node.I):
            if not A.joint_still_valid(s, a):
                u_row[i] = -np.inf
    u_col = node.sigma @ Ahat - c * node.q * sq / (1.0 + n_col)
    i = int(np.argmax(u_row))
    j = int(np.argmin(u_col))
    return i, j


def _backup(path: List[Tuple[NashNode, int, int]], z: float, t_fp: int) -> None:
    for node, i, j in reversed(path):
        node.n[i, j] += 1
        node.W[i, j] += z
        node.N += 1
        node.W_sum += z
        sigma, tau, v = fictitious_play(node.payoff(), T=t_fp)
        node.sigma, node.tau = sigma, tau
        z = float(v)


def _simulate(root: NashNode, s0: BattleSnapshot, sim_id: int, ctx: _Ctx, cfg: NashOLConfig,
              tel: NashOLTelemetry) -> None:
    rng = make_sim_rng(cfg.seed, int(s0.turn), int(sim_id))
    node, s = root, s0
    path: List[Tuple[NashNode, int, int]] = []
    depth = 0
    try:
        while True:
            if s.finished:
                z = material_value(s)
                break
            if not node.expanded:
                _expand(node, s, ctx, cfg)
                tel.nodes += 1
                z = _rollout_value(s, rng, ctx)
                break
            if depth >= cfg.max_depth:
                z = _rollout_value(s, rng, ctx)
                break
            i, j = _select(node, s, cfg.c_puct, check_rows=node is not root)
            path.append((node, i, j))
            depth += 1
            s = _step(s, node.I[i], node.J[j], rng, ctx)
            node = node.children.setdefault((i, j), NashNode())
    except Exception:
        tel.failures += 1
        z = 0.0
    _backup(path, z, cfg.t_fp_internal)
    tel.sims += 1
    tel.avg_depth += (depth - tel.avg_depth) / tel.sims


def search_nash_openloop(
    root_state: BattleSnapshot,
    cfg: Optional[NashOLConfig] = None,
    *,
    root_joints: Optional[Sequence[int]] = None,
    stat_beliefs=None,
    ranges_fn=None,
) -> Tuple[int, NashNode, NashOLTelemetry]:
    """N_sim simulaciones SM-MCTS open-loop con S3; devuelve argmax σ* de la raíz."""
    cfg = cfg or NashOLConfig()
    t0 = time.perf_counter()
    tel = NashOLTelemetry(model=str(cfg.model))
    s0 = prepare_root(root_state)
    from showdown.rl.mcts_puct.s3 import S3Model
    phase = {'s3': '3a', 's3b': '3b', 's3c': '3c', 's3d': '3d', 's3e': '3e'}[str(cfg.model).lower()]
    model = S3Model.build(s0, stat_beliefs=stat_beliefs, ranges_fn=ranges_fn, phase=phase)
    ms = model.stats
    tel.calc_requests, tel.calc_missing, tel.calc_sec = ms.requests, ms.missing, ms.build_sec
    tel.calc_error, tel.foe_no_moves, tel.foe_point_stats = ms.error, ms.foe_no_moves, ms.foe_point_stats
    ctx = _Ctx(cfg.openloop(), model)
    root = NashNode()
    force = bool(any(root_state.force_switch))
    rj = list(root_joints) if root_joints else A.pruned_joints(s0, ctx.target_fn)
    _expand(root, s0, ctx, cfg, root_joints=rj, force_pass_opp=force)
    tel.root_rows, tel.root_cols = len(root.I), len(root.J)
    if len(root.I) <= 1:
        tel.trivial = True
        tel.elapsed_sec = time.perf_counter() - t0
        return int(root.I[0]) if root.I else _PASS2, root, tel
    for k in range(int(cfg.n_sims)):
        _simulate(root, s0, k, ctx, cfg, tel)
    sigma, _tau, v = fictitious_play(root.payoff(), T=int(cfg.t_fp_root))
    visits = root.n.sum(axis=1)
    smax = float(np.max(sigma))
    cands = [i for i in range(len(root.I)) if sigma[i] >= smax - 1e-9]
    best = max(cands, key=lambda i: (visits[i], -root.I[i]))  # desempate: más visitas, menor id
    tel.root_value = float(v)
    tel.root_sigma_max = smax
    tel.root_cells_visited = int((root.n > 0).sum())
    tel.elapsed_sec = time.perf_counter() - t0
    return int(root.I[best]), root, tel


__all__ = ['NashOLConfig', 'NashOLTelemetry', 'NashNode', 'search_nash_openloop']
