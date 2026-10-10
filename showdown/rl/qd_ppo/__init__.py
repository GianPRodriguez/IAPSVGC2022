"""QD-PPO — docs/qd_ppo_spec.md v1.0 (Grid MAP-Elites + Masked PPO)."""

from showdown.rl.qd_ppo.archive import Archive, bin_bd
from showdown.rl.qd_ppo.agent import QDPPOAgent

__all__ = [
    'Archive',
    'bin_bd',
    'QDPPOAgent',
    'derived_n_outer',
    'run_training',
]


def __getattr__(name: str):
    if name in ('derived_n_outer', 'run_training'):
        from showdown.rl.qd_ppo import train as _train
        return getattr(_train, name)
    raise AttributeError(name)
