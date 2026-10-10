"""VGCEnv — Gymnasium env implementing docs/rl_env_spec.md v1.3."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym  # type: ignore
    from gym import spaces  # type: ignore

from showdown.rl.action_space import (
    SLOT_ACTION_N,
    build_autoregressive_masks,
    decode_slot_action,
    resolve_action_or_fallback,
)
from showdown.rl.behavioral import BehavioralTracker
from showdown.rl.constants import OBS_DIM, SPEC_VERSION
from showdown.rl.observation import build_observation
from showdown.rl.reward import RewardCandidate, compute_reward, flip_reward_for_opponent
from showdown.rl.snapshot import (
    BattleSnapshot,
    FieldSnapshot,
    MonSnapshot,
    SideSnapshot,
    TeamMaterial,
    material_from_side,
)


class BattleBackend:
    """Minimal backend interface (Mock offline / PokeEnv live)."""

    def reset(self, *, seed: Optional[int] = None) -> BattleSnapshot:
        raise NotImplementedError

    def step(
        self,
        action: Tuple[int, int],
    ) -> Tuple[BattleSnapshot, Dict[str, Any]]:
        """Apply joint action; return new snapshot + engine meta."""
        raise NotImplementedError

    def action_masks(
        self,
        *,
        slot1_action_id: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Optional live masks; None → derive from BattleSnapshot."""
        return None

    def close(self) -> None:
        return None


class MockBattleBackend(BattleBackend):
    """
    Deterministic offline backend for unit tests / CI (no PS / poke-env).

    Simulates a short BO1: dealing damage / optional faint, then terminal win.
    """

    def __init__(self, max_turns: int = 8):
        self.max_turns = int(max_turns)
        self._snap: Optional[BattleSnapshot] = None
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed: Optional[int] = None) -> BattleSnapshot:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        our = SideSnapshot(
            active=[
                MonSnapshot(
                    species='Incineroar', ots_species='Incineroar', hp_pct=100.0,
                    types=('Fire', 'Dark'), moves=('flareblitz', 'fakeout', 'knockoff', 'partingshot'),
                    move_pps=(24, 16, 32, 32), is_active=True, revealed=True,
                    ability='Intimidate', ability_known='confirmed',
                ),
                MonSnapshot(
                    species='Rillaboom', ots_species='Rillaboom', hp_pct=100.0,
                    types=('Grass',), moves=('grassyglide', 'woodhammer', 'uturn', 'fakeout'),
                    move_pps=(32, 24, 32, 16), is_active=True, revealed=True,
                    ability='Grassy Surge', ability_known='confirmed',
                ),
            ],
            bench=[
                MonSnapshot(species='Flutter Mane', ots_species='Flutter Mane', hp_pct=100.0,
                            types=('Ghost', 'Fairy'), moves=('moonblast', 'shadowball', 'mysticalfire', 'protect'),
                            is_active=False, revealed=True),
                MonSnapshot(species='Chien-Pao', ots_species='Chien-Pao', hp_pct=100.0,
                            types=('Dark', 'Ice'), moves=('iciclecrash', 'suckerpunch', 'sacredsword', 'protect'),
                            is_active=False, revealed=True),
            ],
        )
        foe = SideSnapshot(
            active=[
                MonSnapshot(
                    species='Landorus-Therian', ots_species='Landorus-Therian', hp_pct=100.0,
                    types=('Ground', 'Flying'), moves=('earthquake', 'rockslide', 'uturn', 'protect'),
                    is_active=True, revealed=True, ability_known='confirmed', ability='Intimidate',
                ),
                MonSnapshot(
                    species='Amoonguss', ots_species='Amoonguss', hp_pct=100.0,
                    types=('Grass', 'Poison'), moves=('spore', 'ragepowder', 'pollenpuff', 'protect'),
                    is_active=True, revealed=True,
                ),
            ],
            bench=[
                MonSnapshot(species='Heatran', ots_species='Heatran', hp_pct=100.0,
                            types=('Fire', 'Steel'), moves=('heatwave', 'earthpower', 'protect', 'flashcannon'),
                            is_active=False, revealed=True),
                MonSnapshot(species='Pelipper', ots_species='Pelipper', hp_pct=100.0,
                            types=('Water', 'Flying'), moves=('hurricane', 'scald', 'protect', 'tailwind'),
                            is_active=False, revealed=True),
            ],
        )
        self._snap = BattleSnapshot(
            turn=1,
            our=our,
            foe=foe,
            field=FieldSnapshot(),
            finished=False,
            won=None,
            tied=False,
            force_switch=(False, False),
            player_id='p1',
        )
        return self._snap

    def step(self, action: Tuple[int, int]) -> Tuple[BattleSnapshot, Dict[str, Any]]:
        assert self._snap is not None
        snap = self._snap
        meta: Dict[str, Any] = {'engine': 'mock'}
        a0 = decode_slot_action(int(action[0]))
        a1 = decode_slot_action(int(action[1]))

        # Apply crude damage if offensive moves chosen
        for act in (a0, a1):
            if act.kind in ('move', 'mega', 'tera') and snap.foe.active:
                tgt = snap.foe.active[min(act.target, 1)]
                if tgt is not None and not tgt.fainted:
                    # Real HP only (Spec: Sub excluded — mock has no Sub)
                    dmg = 18.0 + float(self._rng.integers(0, 8))
                    tgt.hp_pct = max(0.0, tgt.hp_pct - dmg)
                    if tgt.hp_pct <= 0:
                        tgt.fainted = True
                        tgt.hp_pct = 0.0

        # Hard switch: swap active0 with chosen bench
        for slot, act in ((0, a0), (1, a1)):
            if act.kind != 'switch':
                continue
            bi = act.switch_index
            if bi < len(snap.our.bench) and snap.our.bench[bi] is not None:
                outgoing = snap.our.active[slot]
                incoming = snap.our.bench[bi]
                snap.our.active[slot] = incoming
                snap.our.bench[bi] = outgoing
                if incoming is not None:
                    incoming.is_active = True
                    incoming.just_switched = True
                if outgoing is not None:
                    outgoing.is_active = False

        snap.turn += 1
        foe_alive = any(m is not None and not m.fainted for m in snap.foe.all_mons())
        our_alive = any(m is not None and not m.fainted for m in snap.our.all_mons())

        if not foe_alive:
            snap.finished = True
            snap.won = True
            snap.tied = False
        elif not our_alive:
            snap.finished = True
            snap.won = False
            snap.tied = False
        elif snap.turn > self.max_turns:
            # Turn-cap: not a game tie — VGCEnv maps this to truncated + trunc_reason
            snap.finished = True
            snap.tied = False
            snap.won = None
            meta['turn_cap'] = True

        self._snap = snap
        return snap, meta


class VGCEnv(gym.Env):
    """
    Gymnasium environment for VGC doubles RL (Spec v1.3).

    observation: float32[1024]
    action: MultiDiscrete([24, 24])  — Slot1 then Slot2 (auto-regressive masks in info)
    info['trunc_reason']: None | 'turn_cap' | 'crash'
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        *,
        reward_candidate: RewardCandidate | str = RewardCandidate.DENSE,
        backend: Optional[BattleBackend] = None,
        opponent_perspective: bool = False,
        max_episode_turns: int = 40,
    ):
        super().__init__()
        if isinstance(reward_candidate, str):
            reward_candidate = RewardCandidate(reward_candidate)
        self.reward_candidate = reward_candidate
        self.backend = backend or MockBattleBackend(max_turns=max_episode_turns)
        self.opponent_perspective = bool(opponent_perspective)
        self.max_episode_turns = int(max_episode_turns)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([SLOT_ACTION_N, SLOT_ACTION_N])

        self._snap: Optional[BattleSnapshot] = None
        self._prev_our: Optional[TeamMaterial] = None
        self._prev_foe: Optional[TeamMaterial] = None
        self._bd = BehavioralTracker()
        self._illegal_fallbacks = 0
        self._episode_return = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._bd = BehavioralTracker()
        self._illegal_fallbacks = 0
        self._episode_return = 0.0
        self._snap = self.backend.reset(seed=seed)
        self._prev_our = material_from_side(self._snap.our)
        self._prev_foe = material_from_side(self._snap.foe)
        obs = build_observation(self._snap)
        info = self._info(skip_buffer=False, truncated=False, trunc_reason=None)
        return obs, info

    def step(self, action):
        assert self._snap is not None and self._prev_our is not None and self._prev_foe is not None
        a = np.asarray(action, dtype=np.int64).reshape(-1)
        if a.shape[0] != 2:
            raise ValueError(f'action must be length-2 MultiDiscrete, got {action!r}')

        skip_buffer = False
        truncated = False
        terminated = False
        trunc_reason: Optional[str] = None
        reward = 0.0

        try:
            m1, _ = self._masks(slot1_action_id=None)
            a0, fb0 = resolve_action_or_fallback(int(a[0]), m1)
            _, m2 = self._masks(slot1_action_id=a0)
            a1, fb1 = resolve_action_or_fallback(int(a[1]), m2)
            if fb0 or fb1:
                skip_buffer = True
                self._illegal_fallbacks += 1

            new_snap, engine_meta = self.backend.step((a0, a1))
        except Exception as exc:
            # L3 crash — drop from buffers; no value bootstrap
            obs = build_observation(self._snap)
            info = self._info(
                skip_buffer=True, truncated=True, trunc_reason='crash',
            )
            info['error'] = repr(exc)
            info['spec_version'] = SPEC_VERSION
            return obs, 0.0, False, True, info

        # BD tracking
        s0 = decode_slot_action(a0)
        s1 = decode_slot_action(a1)
        move_names = []
        for slot, act in ((0, s0), (1, s1)):
            mon = self._snap.our.active[slot] if slot < len(self._snap.our.active) else None
            if act.kind in ('move', 'mega', 'tera') and mon and act.move_index < len(mon.moves):
                move_names.append(mon.moves[act.move_index])
            else:
                move_names.append('')
        self._bd.on_turn_actions(
            [s0, s1],
            move_names,
            force_switch=self._snap.force_switch,
        )

        cur_our = material_from_side(new_snap.our)
        cur_foe = material_from_side(new_snap.foe)
        finished = bool(new_snap.finished)
        won = new_snap.won
        tied = bool(new_snap.tied)
        is_turn_cap = bool(
            isinstance(engine_meta, dict) and engine_meta.get('turn_cap')
        )

        if is_turn_cap:
            truncated = True
            terminated = False
            trunc_reason = 'turn_cap'
            rr = compute_reward(
                candidate=self.reward_candidate,
                prev_our=self._prev_our,
                prev_foe=self._prev_foe,
                cur_our=cur_our,
                cur_foe=cur_foe,
                finished=False,
                won=None,
                tied=False,
                skip_buffer=skip_buffer,
                truncated=True,
            )
        else:
            if finished:
                terminated = True
            rr = compute_reward(
                candidate=self.reward_candidate,
                prev_our=self._prev_our,
                prev_foe=self._prev_foe,
                cur_our=cur_our,
                cur_foe=cur_foe,
                finished=finished,
                won=won,
                tied=tied,
                skip_buffer=skip_buffer,
                truncated=False,
            )

        reward = rr.reward
        if self.opponent_perspective:
            reward = flip_reward_for_opponent(reward)

        self._episode_return += float(reward)
        self._snap = new_snap
        self._prev_our = cur_our
        self._prev_foe = cur_foe

        obs = build_observation(new_snap)
        info = self._info(
            skip_buffer=rr.skip_buffer or skip_buffer,
            truncated=truncated,
            trunc_reason=trunc_reason,
        )
        info['reward_components'] = rr.components
        info['engine'] = engine_meta
        return obs, float(reward), terminated, truncated, info

    def _masks(self, *, slot1_action_id: Optional[int] = None):
        live = self.backend.action_masks(slot1_action_id=slot1_action_id)
        if live is not None:
            return live
        assert self._snap is not None
        return build_autoregressive_masks(self._snap, slot1_action_id=slot1_action_id)

    def _info(
        self,
        *,
        skip_buffer: bool,
        truncated: bool,
        trunc_reason: Optional[str] = None,
    ) -> dict:
        metrics = self._bd.metrics()
        metrics['illegal_action_fallbacks'] = self._illegal_fallbacks
        m1, m2 = (np.ones(SLOT_ACTION_N), np.ones(SLOT_ACTION_N))
        if self._snap is not None:
            m1, m2 = self._masks(slot1_action_id=None)
        return {
            'spec_version': SPEC_VERSION,
            'skip_buffer': bool(skip_buffer),
            'truncated_flag': bool(truncated),
            'trunc_reason': trunc_reason,
            'action_mask_slot1': np.asarray(m1, dtype=np.float32),
            'action_mask_slot2': np.asarray(m2, dtype=np.float32),
            'bd': self._bd.bd_vector() if self._bd.archive_eligible() else None,
            'bd_raw': self._bd.bd_vector(),
            'telemetry': metrics,
            'episode_return': self._episode_return,
            'reward_candidate': self.reward_candidate.value,
            'turn': int(getattr(self._snap, 'turn', 0) or 0),
        }

    def close(self):
        try:
            self.backend.close()
        except Exception:
            pass
        return super().close() if hasattr(super(), 'close') else None
