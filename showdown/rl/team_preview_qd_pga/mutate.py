"""Gaussian actor mutation σ=0.02."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from showdown.rl.team_preview_qd_pga.archive import is_actor_key

SIGMA_DEFAULT = 0.02


def mutate_actor_state(
    actor_sd: Dict[str, Any],
    rng: np.random.Generator,
    *,
    sigma: float = SIGMA_DEFAULT,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    sig = float(sigma)
    for k, v in actor_sd.items():
        if not is_actor_key(k):
            continue
        if hasattr(v, 'detach'):
            arr = v.detach().cpu().numpy().astype(np.float32)
            noise = rng.normal(0.0, sig, size=arr.shape).astype(np.float32)
            import torch
            out[k] = torch.as_tensor(arr + noise)
        else:
            arr = np.asarray(v, dtype=np.float32)
            noise = rng.normal(0.0, sig, size=arr.shape).astype(np.float32)
            out[k] = arr + noise
    return out
