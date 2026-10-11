"""Actor-only archive helpers — docs/qd_pga_map_elites_spec.md."""
from __future__ import annotations

from typing import Any, Dict

from showdown.rl.qd_ppo.archive import Archive, Elite, bin_bd  # noqa: F401

__all__ = ['Archive', 'Elite', 'bin_bd', 'actor_state_dict', 'is_actor_key', 'load_actor_into']


def is_actor_key(key: str) -> bool:
    """Elite payload excludes the local value head (`value.*`)."""
    return not str(key).startswith('value.')


def actor_state_dict(net) -> Dict[str, Any]:
    return {
        k: v.detach().cpu().clone()
        for k, v in net.state_dict().items()
        if is_actor_key(k)
    }


def load_actor_into(net, actor_sd: Dict[str, Any]) -> None:
    """Load actor keys into an ActorCritic; leave value head untouched."""
    full = net.state_dict()
    merged = {}
    for k, v in full.items():
        if k in actor_sd and is_actor_key(k):
            src = actor_sd[k]
            if hasattr(src, 'to'):
                merged[k] = src.to(device=v.device, dtype=v.dtype)
            else:
                import torch
                merged[k] = torch.as_tensor(src, dtype=v.dtype, device=v.device)
        else:
            merged[k] = v
    net.load_state_dict(merged)
