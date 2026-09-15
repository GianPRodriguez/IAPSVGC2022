"""Battle / party snapshots for Spec-compliant reward, obs, and BD (no poke-env required)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class MonSnapshot:
    species: str = ''
    ots_species: str = ''  # OTS ground truth (Illusion)
    hp_pct: float = 0.0  # real HP bar [0, 100]; Substitute damage excluded upstream
    max_hp: float = 1.0
    fainted: bool = False
    types: Tuple[str, ...] = ()
    ability: str = ''
    ability_known: str = 'unknown'  # unknown | suspected | confirmed
    item: str = ''
    item_known: str = 'unknown'
    boosts: Dict[str, int] = field(default_factory=dict)
    status: str = ''
    volatiles: Dict[str, float] = field(default_factory=dict)  # name -> countdown or 1.0
    moves: Tuple[str, ...] = ()
    move_pps: Tuple[int, ...] = ()
    spe_tier: float = 0.5  # normalized belief Spe
    atk_belief: float = 0.5
    spa_belief: float = 0.5
    revealed: bool = True
    is_active: bool = False
    just_switched: bool = False
    can_mega: bool = False
    can_tera: bool = False
    tera_type: str = ''

    def display_species(self) -> str:
        return self.ots_species or self.species


@dataclass
class SideSnapshot:
    active: List[Optional[MonSnapshot]] = field(default_factory=lambda: [None, None])
    bench: List[Optional[MonSnapshot]] = field(default_factory=lambda: [None, None])

    def all_mons(self) -> List[MonSnapshot]:
        out: List[MonSnapshot] = []
        for m in list(self.active) + list(self.bench):
            if m is not None:
                out.append(m)
        return out

    def faint_count(self) -> int:
        return sum(1 for m in self.all_mons() if m.fainted or m.hp_pct <= 0.0)

    def team_hp_pct_total(self) -> float:
        """Sum of real HP% over up to 4 mons (Spec: opp_total = 4-mon team)."""
        mons = self.all_mons()
        total = 0.0
        for m in mons[:4]:
            if m.fainted:
                continue
            total += max(0.0, min(100.0, float(m.hp_pct)))
        # Pad missing slots as 0 (already fainted / unknown)
        return total


@dataclass
class FieldSnapshot:
    weather: str = ''
    weather_turns: float = 0.0
    terrain: str = ''
    terrain_turns: float = 0.0
    trick_room: bool = False
    trick_room_turns: float = 0.0
    our_tailwind: bool = False
    our_tailwind_turns: float = 0.0
    opp_tailwind: bool = False
    opp_tailwind_turns: float = 0.0
    our_reflect: bool = False
    our_lightscreen: bool = False
    our_auroraveil: bool = False
    opp_reflect: bool = False
    opp_lightscreen: bool = False
    opp_auroraveil: bool = False
    gravity: bool = False


@dataclass
class BattleSnapshot:
    turn: int = 0
    our: SideSnapshot = field(default_factory=SideSnapshot)
    foe: SideSnapshot = field(default_factory=SideSnapshot)
    field: FieldSnapshot = field(default_factory=FieldSnapshot)
    finished: bool = False
    won: Optional[bool] = None  # True win, False loss, None if unfinished or tie
    tied: bool = False
    force_switch: Tuple[bool, bool] = (False, False)
    player_id: str = 'p1'

    def party_list(self) -> List[Optional[MonSnapshot]]:
        """Canonical Spec order: self a/b, self bench, foe a/b, foe bench."""
        return [
            self.our.active[0] if len(self.our.active) > 0 else None,
            self.our.active[1] if len(self.our.active) > 1 else None,
            self.our.bench[0] if len(self.our.bench) > 0 else None,
            self.our.bench[1] if len(self.our.bench) > 1 else None,
            self.foe.active[0] if len(self.foe.active) > 0 else None,
            self.foe.active[1] if len(self.foe.active) > 1 else None,
            self.foe.bench[0] if len(self.foe.bench) > 0 else None,
            self.foe.bench[1] if len(self.foe.bench) > 1 else None,
        ]


@dataclass
class SlotAction:
    """Decoded per-slot action (auto-regressive atom)."""

    kind: str  # 'move' | 'switch' | 'pass' | 'mega' | 'tera'
    move_index: int = 0
    target: int = 0  # 0 foe_a, 1 foe_b, 2 ally, 3 self/field
    switch_index: int = 0
    raw_id: int = 0


@dataclass
class TeamMaterial:
    """Material counters for reward deltas (real HP only)."""

    faint_count: int = 0
    hp_pct_total: float = 0.0


def material_from_side(side: SideSnapshot) -> TeamMaterial:
    return TeamMaterial(
        faint_count=side.faint_count(),
        hp_pct_total=side.team_hp_pct_total(),
    )
