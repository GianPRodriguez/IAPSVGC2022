"""QD-PPGA — docs/qd_ppga_spec.md v1.0 (proximal-only + arborescence)."""

from showdown.rl.qd_ppga.archive import Archive, bin_bd, actor_state_dict
from showdown.rl.qd_ppga.agent import QDPPGAAgent

__all__ = [
    'Archive',
    'bin_bd',
    'actor_state_dict',
    'QDPPGAAgent',
    'derived_n_outer',
    'run_training',
]


def __getattr__(name: str):
    if name in ('derived_n_outer', 'run_training'):
        from showdown.rl.qd_ppga import train as _train
        return getattr(_train, name)
    raise AttributeError(name)
