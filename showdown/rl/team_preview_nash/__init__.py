"""Team Preview Deep Payoff Nash — docs/team_preview_nash_spec.md v1.0."""
from __future__ import annotations

# Lazy: importing agent pulls torch. Public surface for tests/train.

__all__ = [
    'PreviewNashAgent',
    'PreviewNashConfig',
    'solve_mixed_strategy',
    'LowRankPayoffNet',
]


def __getattr__(name: str):
    if name in ('PreviewNashAgent', 'PreviewNashConfig'):
        from showdown.rl.team_preview_nash.agent import PreviewNashAgent, PreviewNashConfig
        return PreviewNashAgent if name == 'PreviewNashAgent' else PreviewNashConfig
    if name == 'solve_mixed_strategy':
        from showdown.rl.team_preview_nash.nash_solver import solve_mixed_strategy
        return solve_mixed_strategy
    if name == 'LowRankPayoffNet':
        from showdown.rl.team_preview_nash.network import LowRankPayoffNet
        return LowRankPayoffNet
    raise AttributeError(name)
