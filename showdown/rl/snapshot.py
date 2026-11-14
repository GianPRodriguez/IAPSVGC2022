"""Battle / party snapshots for Spec-compliant reward, obs, and BD (no poke-env required)."""
from __future__ import annotations

import dataclasses
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
    # --- Obs v2 (OTS + estado) ----------------------------------------------
    stats: Optional[Dict[str, int]] = None  # stats finales exactos (solo propio)
    nature: str = ''  # naturaleza (rival: hoja abierta)
    seen: bool = True  # rival: ya apareció en combate (OTS muestra 6, trae 4)
    mega_evolved: bool = False
    status_counter: int = 0  # turnos de sueño / toxic
    protect_counter: int = 0  # usos consecutivos de Protect
    # Rival: SP aún posibles por stat según la inferencia {stat: (min, max)}.
    sp_bounds: Optional[Dict[str, Tuple[int, int]]] = None
    # --- Historial (parseado del protocolo del combate) -----------------------
    last_move: str = ''  # id del último movimiento usado
    last_move_ago: int = -1  # turnos desde ese movimiento (-1 = nunca)
    turns_active: int = 0  # turnos completos en campo desde su último cambio
    moves_since_switch: int = 0  # movimientos usados desde que entró
    times_switched_in: int = 0
    hp_lost_last_turn: float = 0.0  # puntos de HP% perdidos en el último turno
    locked_move: str = ''  # Choice / Encore: único movimiento posible
    # --- Rival: la hoja muestra 6, trae 4 -------------------------------------
    p_brought: float = 1.0  # prob. de que esté entre los 4 traídos
    excluded: bool = False  # confirmado NO traído (ya se vieron los 4)

    def display_species(self) -> str:
        return self.ots_species or self.species


BROUGHT_N = 4  # Reg M-C: se traen 4 de 6


@dataclass
class SideSnapshot:
    active: List[Optional[MonSnapshot]] = field(default_factory=lambda: [None, None])
    bench: List[Optional[MonSnapshot]] = field(default_factory=lambda: [None, None])
    # Todos los no activos (rival: los 4 restantes de su hoja de 6, con ``seen``).
    # Vacío en backends que no lo rellenan (mock) → se usa ``bench``.
    reserve: List[MonSnapshot] = field(default_factory=list)
    mega_used: bool = False

    def all_mons(self) -> List[MonSnapshot]:
        out: List[MonSnapshot] = []
        for m in list(self.active) + list(self.bench):
            if m is not None:
                out.append(m)
        return out

    def _material_mons(self) -> Optional[List[MonSnapshot]]:
        """Pokémon que cuentan para el material, o None si no hay ``reserve``.

        Con ``reserve`` (poke-env) solo cuentan los vistos en combate: el rival
        muestra 6 en su hoja pero trae 4, y los no vistos se asumen sanos.
        Antes se tomaban los 2 primeros no activos de la hoja de 6: el conjunto
        cambiaba al entrar/salir Pokémon y producía saltos de HP y de KOs sin
        daño real (recompensa espuria).
        """
        if not self.reserve:
            return None
        mons = [m for m in list(self.active) + list(self.reserve) if m is not None]
        return [m for m in mons if m.seen]

    def faint_count(self) -> int:
        mons = self._material_mons()
        if mons is None:
            mons = self.all_mons()
        return sum(1 for m in mons if m.fainted or m.hp_pct <= 0.0)

    def team_hp_pct_total(self) -> float:
        """Sum of real HP% over the 4 brought mons (Spec: opp_total = 4-mon team)."""
        seen = self._material_mons()
        if seen is not None:
            total = sum(max(0.0, min(100.0, float(m.hp_pct))) for m in seen[:BROUGHT_N]
                        if not m.fainted)
            return total + 100.0 * max(0, BROUGHT_N - len(seen))
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
    # Alineación obs ↔ acciones: ids de movimiento en el orden de
    # ``available_moves[slot]`` (índice de acción 0–3) y especies en el orden de
    # ``available_switches[slot]`` (índice de cambio 0–3). None = no disponible.
    avail_moves: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None
    avail_switches: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None
    # Resumen del último turno: our/foe × protect, switch, faint, fakeout, spread.
    last_turn: Dict[str, float] = dataclasses.field(default_factory=dict)

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
