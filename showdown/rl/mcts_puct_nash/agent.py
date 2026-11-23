"""MCTS–PUCT/Nash agent (docs/mcts_puct_nash_spec.md).

``version=2`` (default desde el 22-11-2026, candidata H 2.0): SM-MCTS open-loop con el modelo S3 de G
(fase 3c), Top-K podado por S3, selección desacoplada contra la mezcla de Nash y valor de Nash al padre
(``openloop_nash.py``).
``version=1``: SM-MCTS B1 original (S2, árbol cerrado; ``search.py``), para reproducibilidad.
En el player: ``MCTS_NASH_VERSION``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from showdown.rl.action_space import decode_joint_action
from showdown.rl.mcts_puct_nash.search import SearchConfig, search
from showdown.rl.mcts_puct_nash.topk import DEFAULT_K
from showdown.rl.snapshot import BattleSnapshot


@dataclass
class MCTSPuctNashConfig:
    n_sims: int = 50
    c_puct: float = 1.5
    k: int = DEFAULT_K
    max_turns: int = 40
    seed: int = 0
    t_fp_internal: int = 200
    t_fp_root: int = 500
    version: int = 1   # el player usa 2 por defecto (MCTS_NASH_VERSION); 1 aquí: smoke con backend mock sin calculadora
    model: str = 's3c'
    # H 2.1 (protocolo §4.7): los defaults reproducen H 2.0.
    k_self_root: int = 12
    k_self: int = 8
    k_opp: int = 8
    decision: str = 'sigma'
    solver: str = 'fp'


class MCTSPuctNashAgent:
    def __init__(self, cfg: Optional[MCTSPuctNashConfig] = None):
        self.cfg = cfg or MCTSPuctNashConfig()
        self.last_telemetry = None
        self.last_joint: int = 0
        self.last_sigma: Optional[np.ndarray] = None

    def act(self, snap: BattleSnapshot, root_joints: Optional[Sequence[int]] = None,
            stat_beliefs=None) -> int:
        """v2: argmax σ* tras N_sim sims open-loop con S3. v1: SM-MCTS B1 original."""
        if int(self.cfg.version) >= 2:
            from showdown.rl.mcts_puct_nash.openloop_nash import NashOLConfig, search_nash_openloop
            cfg = NashOLConfig(n_sims=int(self.cfg.n_sims), c_puct=float(self.cfg.c_puct),
                               max_turns=int(self.cfg.max_turns), seed=int(self.cfg.seed),
                               model=str(self.cfg.model), t_fp_root=int(self.cfg.t_fp_root),
                               k_self_root=int(self.cfg.k_self_root), k_self=int(self.cfg.k_self),
                               k_opp=int(self.cfg.k_opp), decision=str(self.cfg.decision),
                               solver=str(self.cfg.solver))
            joint, root, tel = search_nash_openloop(snap, cfg, root_joints=root_joints,
                                                    stat_beliefs=stat_beliefs)
            self.last_sigma = root.sigma
            from showdown.rl.mcts_puct_nash import diag
            if diag.enabled():  # paso 0 de H 2.1; apagado por defecto, no cambia la decisión
                diag.record(snap, root_joints, stat_beliefs, joint, root, tel, cfg, int(self.cfg.seed))
        else:
            sc = SearchConfig(
                n_sims=int(self.cfg.n_sims),
                c_puct=float(self.cfg.c_puct),
                k=int(self.cfg.k),
                max_turns=int(self.cfg.max_turns),
                seed=int(self.cfg.seed),
                t_fp_internal=int(self.cfg.t_fp_internal),
                t_fp_root=int(self.cfg.t_fp_root),
            )
            joint, _root, tel, sigma = search(snap, sc)
            self.last_sigma = sigma
        self.last_telemetry = tel
        self.last_joint = int(joint)
        return int(joint)

    def act_slots(self, snap: BattleSnapshot) -> Tuple[int, int]:
        j = self.act(snap)
        return decode_joint_action(j)
