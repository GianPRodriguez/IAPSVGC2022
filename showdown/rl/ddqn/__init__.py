"""Joint Double DQN — docs/ddqn_spec.md v1.0."""

from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
from showdown.rl.ddqn.curriculum import (
    default_expand_curriculum,
    expand_phases_from_exclusive,
    sample_from_omega,
)
from showdown.rl.ddqn.network import QNetwork, soft_update

__all__ = [
    'DDQNAgent',
    'DDQNConfig',
    'QNetwork',
    'soft_update',
    'default_expand_curriculum',
    'expand_phases_from_exclusive',
    'sample_from_omega',
]
