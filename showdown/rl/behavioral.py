"""BDSpec trackers — docs/rl_env_spec.md v1.1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from showdown.rl.constants import BD_MIN_TURNS, PIVOT_MOVES, STATUS_PROTECT_SETUP
from showdown.rl.snapshot import SlotAction

_PROTECT_IDS = frozenset({
    'protect', 'detect', 'spikyshield', 'banefulbunker',
    'kingsshield', 'silktrap', 'obstruct',
})


def _norm_move(name: str) -> str:
    return str(name or '').strip().lower().replace(' ', '').replace('-', '')


def classify_action(
    action: SlotAction,
    move_name: str = '',
    *,
    force_switch: bool = False,
) -> str:
    """
    Returns 'offensive' | 'status_setup' | 'hard_switch' | 'ignore'.

    Hard switches: explicit SWITCH only (not faint replace / Roar / U-turn).
    Pivot moves → offensive (BD1), not hard_switch (BD2).
    """
    if force_switch and action.kind == 'switch':
        return 'ignore'
    if action.kind == 'switch':
        return 'hard_switch'
    if action.kind == 'pass':
        return 'ignore'
    mid = _norm_move(move_name)
    if mid in PIVOT_MOVES:
        return 'offensive'
    if mid in STATUS_PROTECT_SETUP:
        return 'status_setup'
    if action.kind in ('move', 'mega', 'tera'):
        return 'offensive'
    return 'ignore'


@dataclass
class BehavioralTracker:
    offensive: int = 0
    status_setup: int = 0
    hard_switches: int = 0
    turns: int = 0
    useless_switches: int = 0
    protect_streak: int = 0
    protect_spam_events: int = 0
    _last_hard_from_to: Optional[Tuple[str, str]] = None

    def on_turn_actions(
        self,
        slot_actions: Sequence[SlotAction],
        move_names: Sequence[str],
        *,
        force_switch: Sequence[bool] = (False, False),
    ) -> None:
        self.turns += 1
        saw_protect = False
        for i, act in enumerate(slot_actions):
            mn = move_names[i] if i < len(move_names) else ''
            fs = force_switch[i] if i < len(force_switch) else False
            kind = classify_action(act, mn, force_switch=bool(fs))
            if kind == 'offensive':
                self.offensive += 1
            elif kind == 'status_setup':
                self.status_setup += 1
            elif kind == 'hard_switch':
                self.hard_switches += 1
            if _norm_move(mn) in _PROTECT_IDS:
                saw_protect = True
        if saw_protect:
            self.protect_streak += 1
            if self.protect_streak > 2:
                self.protect_spam_events += 1
        else:
            self.protect_streak = 0

    def note_hard_switch_pingpong(self, from_id: str, to_id: str) -> None:
        if self._last_hard_from_to is not None:
            prev_from, prev_to = self._last_hard_from_to
            if prev_from == to_id and prev_to == from_id:
                self.useless_switches += 1
        self._last_hard_from_to = (from_id, to_id)

    def bd_vector(self) -> Tuple[float, float]:
        den1 = self.offensive + self.status_setup
        bd1 = (self.offensive / den1) if den1 > 0 else 0.5
        bd2 = (self.hard_switches / self.turns) if self.turns > 0 else 0.0
        return (
            max(0.0, min(1.0, float(bd1))),
            max(0.0, min(1.0, float(bd2))),
        )

    def archive_eligible(self) -> bool:
        return self.turns >= BD_MIN_TURNS

    def metrics(self) -> dict:
        bd1, bd2 = self.bd_vector()
        return {
            'bd1_aggressiveness': bd1,
            'bd2_hard_switch_rate': bd2,
            'useless_switches': self.useless_switches,
            'protect_spam': self.protect_spam_events,
            'mean_game_length': float(self.turns),
            'archive_eligible': self.archive_eligible(),
        }
