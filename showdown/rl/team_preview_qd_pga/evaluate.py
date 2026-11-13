"""Evaluate preview actor: Cand.1 fitness + mean BD."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from showdown.rl.team_preview_env import TeamPreviewEnv
from showdown.rl.team_preview_qd_pga.bd import preview_bd_from_action


def evaluate_actor(
    env: TeamPreviewEnv,
    act_fn,
    *,
    G: int = 20,
    g_min: int = 10,
    greedy: bool = True,
) -> Tuple[float, Tuple[float, float], int, Dict]:
    """
    act_fn(obs, mask) -> action int.
    Returns (f, (b1,b2), g, meta). If g < g_min, caller must not insert.
    """
    rewards = []
    b1s = []
    b2s = []
    n = int(G)
    if n > 0:
        # Por lotes: los G BO1 de la evaluación se juegan en paralelo si live.
        obs, infos = env.reset_batch(n)
        acts = []
        for k in range(n):
            mask = infos[k].get('action_mask')
            if mask is None:
                mask = env.action_masks()
            a = int(act_fn(obs[k], mask, greedy=greedy))
            b1, b2 = preview_bd_from_action(a, infos[k].get('our_species') or [])
            acts.append(a)
            b1s.append(b1)
            b2s.append(b2)
        rewards = [float(r) for r in env.step_batch(acts)]
    g = len(rewards)
    f = float(np.mean(rewards)) if g else 0.0
    bd = (float(np.mean(b1s)), float(np.mean(b2s))) if g else (0.0, 0.0)
    meta = {'g': g, 'g_min': int(g_min), 'eligible': g >= int(g_min)}
    return f, bd, g, meta
