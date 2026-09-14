"""Utilidad bayesiana V(S) para expectimax 1-ply."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class FieldPositionBonus:
    """Señales de posicionamiento / control de campo (no en FieldContext calc)."""

    our_tailwind: bool = False
    our_trick_room: bool = False
    opp_tailwind: bool = False
    opp_trick_room: bool = False
    tera_available_left: bool = False
    tera_available_right: bool = False
    mega_available_left: bool = False
    mega_available_right: bool = False
    turn: int = 0


def switch_positioning_value(
    *,
    active_matchup: float,
    bench_matchup: float,
    turn: int = 0,
) -> float:
    """
    Valor de reposicionar (mal lead / mal asiento), independiente de P(KO).

    matchup alto = mejor tipado vs actives rivales.
    """
    improvement = float(bench_matchup) - float(active_matchup)
    s = max(0.0, improvement) * 55.0
    # Activo claramente mal tipado: empujar pivot aunque el calc no vea KO.
    if float(active_matchup) < 0.0 and improvement > 0.15:
        s += 45.0
    elif float(active_matchup) < 0.5 and improvement > 0.35:
        s += 28.0
    # Post team-preview: humanos reposicionan pronto; 1-ply solo no lo captura.
    if int(turn or 0) <= 2 and improvement > 0.2:
        s += 35.0
    elif int(turn or 0) <= 4 and improvement > 0.4:
        s += 18.0
    return s


@dataclass
class JointActionUtility:
    """Acumulador de U(a) para un par de acciones nuestras."""

    left_slot: int
    right_slot: int
    left_target: int
    right_target: int
    left_mega: int = 0
    right_mega: int = 0
    left_move: str = ''
    right_move: str = ''
    utility: float = 0.0
    samples: float = 0.0  # masa acumulada (para promedio ponderado)
    components: Dict[str, float] = field(default_factory=dict)

    def add(self, value: float, weight: float = 1.0) -> None:
        w = float(weight)
        self.utility += float(value) * w
        self.samples += w

    @property
    def mean(self) -> float:
        if self.samples <= 1e-12:
            return float('-inf')
        return self.utility / self.samples


def relative_hp_score(
    *,
    our_hp_pct_after: float,
    foe_hp_pct_after: float,
) -> float:
    """Salud relativa residual: aliado mantenido − rival restante."""
    return float(our_hp_pct_after) - float(foe_hp_pct_after)


def field_control_bonus(
    pos: FieldPositionBonus,
    *,
    used_speed_control: bool,
    used_tera: bool,
) -> float:
    s = 0.0
    if pos.our_tailwind or pos.our_trick_room:
        s += 18.0
    if pos.opp_tailwind or pos.opp_trick_room:
        s -= 12.0
    if used_speed_control and not (pos.our_tailwind or pos.our_trick_room):
        s += 22.0
    # Preservar Tera si no se usa este turno
    if (pos.tera_available_left or pos.tera_available_right) and not used_tera:
        s += 8.0
    return s


def switch_preservation_value(
    *,
    stay_p_ko: float,
    bench_incoming_pct: float,
    matchup: float,
    loss_weight: float = 1000.0,
    save_weight: float = 220.0,
    chip_soft_cap: float = 30.0,
    soft_ko_bonus: float = 1000.0,
) -> float:
    """
    Utilidad continua de un switch voluntario (no V→0 duro).

    - Evitar P(KO) al activo: +stay_p_ko * loss_weight (recurso preservado)
    - Bonus suave si el entrante come poco: max(0, cap - E[dmg%])
    - Penaliza chip al bench; tipado suma/resta
    """
    pk = max(0.0, min(1.0, float(stay_p_ko)))
    chip = max(0.0, float(bench_incoming_pct))
    preserve = pk * float(loss_weight) + pk * float(save_weight) * 0.25
    soft_cap_bonus = max(0.0, float(chip_soft_cap) - chip) * 2.5
    chip_pen = (chip / 100.0) * float(soft_ko_bonus) * 0.35
    matchup_term = float(matchup) * 18.0
    return preserve + soft_cap_bonus + matchup_term - chip_pen
