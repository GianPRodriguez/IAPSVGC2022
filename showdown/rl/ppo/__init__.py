"""Masked PPO — docs/ppo_spec.md v1.0."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    'PPOAgent',
    'PPOConfig',
    'ActorCritic',
    'BCTrainer',
    'BCConfig',
    'SelfPlayPool',
]


def __getattr__(name: str):
    if name in ('PPOAgent', 'PPOConfig'):
        from showdown.rl.ppo.agent import PPOAgent, PPOConfig
        return {'PPOAgent': PPOAgent, 'PPOConfig': PPOConfig}[name]
    if name == 'ActorCritic':
        from showdown.rl.ppo.network import ActorCritic
        return ActorCritic
    if name in ('BCTrainer', 'BCConfig'):
        from showdown.rl.ppo.bc import BCTrainer, BCConfig
        return {'BCTrainer': BCTrainer, 'BCConfig': BCConfig}[name]
    if name == 'SelfPlayPool':
        from showdown.rl.ppo.selfplay import SelfPlayPool
        return SelfPlayPool
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


if TYPE_CHECKING:
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    from showdown.rl.ppo.bc import BCConfig, BCTrainer
    from showdown.rl.ppo.network import ActorCritic
    from showdown.rl.ppo.selfplay import SelfPlayPool
