"""Team Preview QD-PGA — docs/team_preview_qd_pga_spec.md v1.0."""
from __future__ import annotations

__all__ = ['PreviewQDPGAAgent', 'PreviewQDPGAConfig', 'Archive', 'train']


def __getattr__(name: str):
    if name in ('PreviewQDPGAAgent', 'PreviewQDPGAConfig'):
        from showdown.rl.team_preview_qd_pga.agent import (
            PreviewQDPGAAgent,
            PreviewQDPGAConfig,
        )
        return PreviewQDPGAAgent if name == 'PreviewQDPGAAgent' else PreviewQDPGAConfig
    if name == 'Archive':
        from showdown.rl.team_preview_qd_pga.archive import Archive
        return Archive
    if name == 'train':
        from showdown.rl.team_preview_qd_pga.train import train
        return train
    raise AttributeError(name)
