"""QD-PPO agent — deploy π* (best elite by f)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from showdown.rl.ppo.agent import PPOAgent, PPOConfig
from showdown.rl.qd_ppo.archive import Archive


class QDPPOAgent:
    """Thin wrapper: Masked PPO act API loaded from best elite / archive."""

    def __init__(
        self,
        cfg: Optional[PPOConfig] = None,
        *,
        checkpoint: Optional[str] = None,
        archive_path: Optional[str] = None,
    ):
        self.cfg = cfg or PPOConfig(
            device=os.environ.get('QD_PPO_DEVICE', os.environ.get('PPO_DEVICE', 'cpu')),
        )
        self.ppo = PPOAgent(self.cfg)
        self.archive: Optional[Archive] = None
        ckpt = checkpoint or os.environ.get('QD_PPO_CHECKPOINT') or os.environ.get('PPO_CHECKPOINT')
        arch = archive_path or os.environ.get('QD_PPO_ARCHIVE')
        if arch and Path(arch).is_file():
            self.archive = Archive()
            self.archive.load(arch)
            best = self.archive.best()
            if best is not None:
                self.ppo._ensure_torch()
                self.ppo.net.load_state_dict(best.state_dict)
        elif ckpt and Path(ckpt).is_file():
            self.ppo.load(ckpt)

    def load_best_from_archive(self, archive: Archive) -> bool:
        self.archive = archive
        best = archive.best()
        if best is None:
            return False
        self.ppo._ensure_torch()
        self.ppo.net.load_state_dict(best.state_dict)
        return True

    def act(
        self,
        obs: np.ndarray,
        m1: np.ndarray,
        m2_fn,
        *,
        greedy: bool = True,
    ) -> Tuple[int, int, float, float]:
        return self.ppo.act(obs, m1, m2_fn, greedy=greedy)
