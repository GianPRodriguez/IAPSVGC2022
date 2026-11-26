"""BDSpec trackers — docs/rl_env_spec.md v1.1.

BD2 desde el 25-11-2026 (I 2.0, decisión del usuario): **foco de ataque** = de los turnos en que los
dos activos usan un ataque de un solo objetivo contra el rival, fracción en que ambos apuntan al mismo
(los movimientos de área no cuentan). La tasa de cambios voluntarios por turno (BD2 anterior) quedaba
en ~0 con PPO, que casi nunca cambia. ``RL_BD2=switch`` restaura la definición anterior.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence, Tuple

from showdown.rl.constants import BD_MIN_TURNS, PIVOT_MOVES, STATUS_PROTECT_SETUP
from showdown.rl.snapshot import SlotAction

_PROTECT_IDS = frozenset({
    'protect', 'detect', 'spikyshield', 'banefulbunker',
    'kingsshield', 'silktrap', 'obstruct',
})


def _norm_move(name: str) -> str:
    return str(name or '').strip().lower().replace(' ', '').replace('-', '')


def bd2_mode() -> str:
    """``focus`` (defecto desde I 2.0) o ``switch`` (BD2 original, ``RL_BD2=switch``)."""
    return 'switch' if os.environ.get('RL_BD2', '').strip().lower() == 'switch' else 'focus'


@lru_cache(maxsize=4096)
def _single_target_attack(move_name: str) -> bool:
    """Ataque (no de estado) que apunta a un solo rival; los de área y desconocidos no cuentan."""
    mid = _norm_move(move_name)
    if not mid:
        return False
    try:
        from showdown.rl import team_preview_obs as P
        m = P._dex()['moves'].get(P.to_id(move_name))
    except Exception:
        m = None
    if not m:
        return False
    return str(m.get('category')) != 'Status' and str(m.get('target')) in ('normal', 'any', 'randomNormal')


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
    paired_attacks: int = 0     # turnos con dos ataques de un solo objetivo contra el rival
    focus_turns: int = 0        # de esos, mismo objetivo
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
        if len(slot_actions) >= 2:
            a, b = slot_actions[0], slot_actions[1]
            na = move_names[0] if len(move_names) > 0 else ''
            nb = move_names[1] if len(move_names) > 1 else ''
            if (a.kind in ('move', 'mega', 'tera') and b.kind in ('move', 'mega', 'tera')
                    and a.target in (0, 1) and b.target in (0, 1)
                    and _single_target_attack(na) and _single_target_attack(nb)):
                self.paired_attacks += 1
                if a.target == b.target:
                    self.focus_turns += 1
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
        if bd2_mode() == 'switch':
            bd2 = (self.hard_switches / self.turns) if self.turns > 0 else 0.0
        else:
            bd2 = (self.focus_turns / self.paired_attacks) if self.paired_attacks > 0 else 0.5
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
            'bd2_hard_switch_rate': (self.hard_switches / self.turns) if self.turns > 0 else 0.0,
            'bd2_focus_rate': (self.focus_turns / self.paired_attacks) if self.paired_attacks > 0 else None,
            'bd2_mode': bd2_mode(),
            'bd2': bd2,
            'useless_switches': self.useless_switches,
            'protect_spam': self.protect_spam_events,
            'mean_game_length': float(self.turns),
            'archive_eligible': self.archive_eligible(),
        }
