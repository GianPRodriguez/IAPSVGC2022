"""MCTS+PUCT agent (docs/mcts_puct_spec.md).

``version=2`` (default desde spec v2.0): búsqueda open-loop de ``openloop.py``.
``version=1``: búsqueda original de ``search.py`` (reproducibilidad / ablación;
``MCTS_PUCT_VERSION=1`` en el player).
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
    version: int = 2
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
        )


class MCTSPuctAgent:
    def __init__(self, cfg: Optional[MCTSPuctConfig] = None):
        self.cfg = cfg or MCTSPuctConfig()
        self.last_telemetry = None
        self.last_joint: int = 0

    def act(self, snap: BattleSnapshot, root_joints: Optional[Sequence[int]] = None) -> int:
        """Joint más visitado tras N_sim simulaciones.

        ``root_joints``: joints legales reales (poke-env ``valid_orders``); solo v2.
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
            joint, _root, tel = search_openloop(snap, self.cfg.openloop(), root_joints=root_joints)
        self.last_telemetry = tel
        self.last_joint = int(joint)
        return int(joint)

    def act_slots(self, snap: BattleSnapshot) -> Tuple[int, int]:
        j = self.act(snap)
        return decode_joint_action(j)
