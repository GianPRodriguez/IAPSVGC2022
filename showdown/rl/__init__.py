"""VGC Gen 9 RL package — implements docs/rl_env_spec.md v1.3 (frozen)."""

from showdown.rl.constants import JOINT_ACTION_N, OBS_DIM, SPEC_VERSION
from showdown.rl.env import VGCEnv, MockBattleBackend, BattleBackend
from showdown.rl.reward import RewardCandidate, compute_reward
from showdown.rl.behavioral import BehavioralTracker
from showdown.rl.action_space import (
    SLOT_ACTION_N,
    apply_logit_mask,
    decode_joint_action,
    decode_slot_action,
    encode_joint_action,
    encode_slot_action,
    legal_joint_mask,
)

__all__ = [
    'OBS_DIM',
    'SPEC_VERSION',
    'JOINT_ACTION_N',
    'VGCEnv',
    'BattleBackend',
    'MockBattleBackend',
    'RewardCandidate',
    'compute_reward',
    'BehavioralTracker',
    'SLOT_ACTION_N',
    'apply_logit_mask',
    'decode_slot_action',
    'encode_slot_action',
    'encode_joint_action',
    'decode_joint_action',
    'legal_joint_mask',
]


def make_pokeenv_backend(**kwargs):
    """Lazy import live backend (requires poke-env + local PS)."""
    from showdown.rl.pokeenv_backend import PokeEnvBattleBackend
    return PokeEnvBattleBackend(**kwargs)
