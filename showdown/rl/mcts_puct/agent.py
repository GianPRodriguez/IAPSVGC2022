"""MCTS+PUCT agent (docs/mcts_puct_spec.md)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from showdown.rl.action_space import decode_joint_action
from showdown.rl.mcts_puct.search import SearchConfig, SearchTelemetry, search
from showdown.rl.snapshot import BattleSnapshot


@dataclass
class MCTSPuctConfig:
    n_sims: int = 200
    c_puct: float = 1.5
    max_turns: int = 40
    seed: int = 0


class MCTSPuctAgent:
    def __init__(self, cfg: Optional[MCTSPuctConfig] = None):
        self.cfg = cfg or MCTSPuctConfig()
        self.last_telemetry: Optional[SearchTelemetry] = None
        self.last_joint: int = 0

    def act(self, snap: BattleSnapshot) -> int:
        """Return joint action id via visit-greedy after N_sim sims."""
        sc = SearchConfig(
            n_sims=int(self.cfg.n_sims),
            c_puct=float(self.cfg.c_puct),
            max_turns=int(self.cfg.max_turns),
            seed=int(self.cfg.seed),
        )
        joint, _root, tel = search(snap, sc)
        self.last_telemetry = tel
        self.last_joint = int(joint)
        return int(joint)

    def act_slots(self, snap: BattleSnapshot) -> Tuple[int, int]:
        j = self.act(snap)
        return decode_joint_action(j)
