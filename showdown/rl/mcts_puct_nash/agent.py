"""MCTS–PUCT/Nash agent (docs/mcts_puct_nash_spec.md)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from showdown.rl.action_space import decode_joint_action
from showdown.rl.mcts_puct_nash.search import SearchConfig, SearchTelemetry, search
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


class MCTSPuctNashAgent:
    def __init__(self, cfg: Optional[MCTSPuctNashConfig] = None):
        self.cfg = cfg or MCTSPuctNashConfig()
        self.last_telemetry: Optional[SearchTelemetry] = None
        self.last_joint: int = 0
        self.last_sigma: Optional[np.ndarray] = None

    def act(self, snap: BattleSnapshot) -> int:
        """Return joint action via argmax σ* after N_sim SM-MCTS sims."""
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
        self.last_telemetry = tel
        self.last_joint = int(joint)
        self.last_sigma = sigma
        return int(joint)

    def act_slots(self, snap: BattleSnapshot) -> Tuple[int, int]:
        j = self.act(snap)
        return decode_joint_action(j)
