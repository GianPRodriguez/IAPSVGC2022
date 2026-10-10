"""Joint Double DQN — docs/ddqn_spec.md v1.0."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    'DDQNAgent',
    'DDQNConfig',
    'QNetwork',
    'soft_update',
    'default_expand_curriculum',
    'expand_phases_from_exclusive',
    'sample_from_omega',
]


def __getattr__(name: str):
    if name in ('DDQNAgent', 'DDQNConfig'):
        from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
        return {'DDQNAgent': DDQNAgent, 'DDQNConfig': DDQNConfig}[name]
    if name in ('QNetwork', 'soft_update'):
        from showdown.rl.ddqn.network import QNetwork, soft_update
        return {'QNetwork': QNetwork, 'soft_update': soft_update}[name]
    if name in (
        'default_expand_curriculum',
        'expand_phases_from_exclusive',
        'sample_from_omega',
    ):
        from showdown.rl.ddqn import curriculum as cur
        return getattr(cur, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


if TYPE_CHECKING:
    from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
    from showdown.rl.ddqn.curriculum import (
        default_expand_curriculum,
        expand_phases_from_exclusive,
        sample_from_omega,
    )
    from showdown.rl.ddqn.network import QNetwork, soft_update
