"""QD-PGA-MAP-Elites — docs/qd_pga_map_elites_spec.md v1.0."""

from showdown.rl.qd_pga.archive import Archive, bin_bd, actor_state_dict
from showdown.rl.qd_pga.agent import QDPGAAgent

__all__ = [
    'Archive',
    'bin_bd',
    'actor_state_dict',
    'QDPGAAgent',
    'derived_n_outer',
    'run_training',
]


def __getattr__(name: str):
    if name in ('derived_n_outer', 'run_training'):
        from showdown.rl.qd_pga import train as _train
        return getattr(_train, name)
    raise AttributeError(name)
