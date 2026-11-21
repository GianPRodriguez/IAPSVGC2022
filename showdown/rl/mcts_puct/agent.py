"""MCTS+PUCT agent (docs/mcts_puct_spec.md).

``version=4`` (default desde el 21-11-2026, confirmada: protocolo §5.3): S3 fase 3b
(+ Protect, Wide Guard, Fake Out, redirección, Helping Hand).
``version=3``: open-loop con el modelo S3 fase 3a (``s3.py``: daño real de la calculadora,
orden por velocidad/prioridad, área, precisión).
``version=2``: open-loop con S2 (``simulator.py``).
``version=1``: búsqueda original de ``search.py``.
En el player: ``MCTS_PUCT_VERSION`` (reproducibilidad / ablación).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from showdown.rl.action_space import decode_joint_action
from showdown.rl.mcts_puct.openloop import OpenLoopConfig, search_openloop
from showdown.rl.mcts_puct.search import SearchConfig, search
from showdown.rl.snapshot import BattleSnapshot


@dataclass
class MCTSPuctConfig:
    n_sims: int = 200
    c_puct: float = 1.5
    max_turns: int = 40
    seed: int = 0
    version: int = 4
    # --- solo v2 (ver OpenLoopConfig) ---
    prior_temperature: float = 20.0
    prior_eps: float = 0.25
    inner_top_k: int = 12
    opp_temperature: float = 10.0
    opp_eps: float = 0.10
    rollout_turns: int = 2
    max_depth: int = 6
    prefer_mega: bool = True

    def openloop(self) -> OpenLoopConfig:
        return OpenLoopConfig(
            n_sims=int(self.n_sims), c_puct=float(self.c_puct),
            max_turns=int(self.max_turns), seed=int(self.seed),
            prior_temperature=float(self.prior_temperature), prior_eps=float(self.prior_eps),
            inner_top_k=int(self.inner_top_k), opp_temperature=float(self.opp_temperature),
            opp_eps=float(self.opp_eps), rollout_turns=int(self.rollout_turns),
            max_depth=int(self.max_depth), prefer_mega=bool(self.prefer_mega),
            model='s3b' if int(self.version) >= 4 else ('s3' if int(self.version) == 3 else 's2'),
        )


class MCTSPuctAgent:
    def __init__(self, cfg: Optional[MCTSPuctConfig] = None):
        self.cfg = cfg or MCTSPuctConfig()
        self.last_telemetry = None
        self.last_joint: int = 0

    def act(self, snap: BattleSnapshot, root_joints: Optional[Sequence[int]] = None,
            stat_beliefs=None) -> int:
        """Joint más visitado tras N_sim simulaciones.

        ``root_joints``: joints legales reales (poke-env ``valid_orders``); v2/v3.
        ``stat_beliefs``: creencias de SP del rival (prior meta); solo v3.
        """
        if int(self.cfg.version) == 1:
            sc = SearchConfig(
                n_sims=int(self.cfg.n_sims),
                c_puct=float(self.cfg.c_puct),
                max_turns=int(self.cfg.max_turns),
                seed=int(self.cfg.seed),
            )
            joint, _root, tel = search(snap, sc)
        else:
            joint, _root, tel = search_openloop(snap, self.cfg.openloop(), root_joints=root_joints,
                                                stat_beliefs=stat_beliefs)
        self.last_telemetry = tel
        self.last_joint = int(joint)
        return int(joint)

    def act_slots(self, snap: BattleSnapshot) -> Tuple[int, int]:
        j = self.act(snap)
        return decode_joint_action(j)
