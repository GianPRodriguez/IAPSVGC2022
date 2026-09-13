"""Chance nodes: soft E[damage] y P(KO) desde rangos del calc Smogon.

El bridge solo expone (min, max). Aproximamos las 16 tiradas uniformes
en [0.85, 1.00] interpolando linealmente entre min y max (estándar cuando
no hay histograma completo).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

N_ROLLS = 16
SOFT_KO_BONUS = 1000.0


@dataclass(frozen=True)
class ChanceNode:
    """Nodo de azar sobre un hit concreto."""

    min_damage: int
    max_damage: int
    cur_hp: int
    max_hp: int

    @property
    def rolls(self) -> Tuple[int, ...]:
        return tuple(discrete_rolls(self.min_damage, self.max_damage))

    @property
    def expected_damage(self) -> float:
        r = self.rolls
        return float(sum(r)) / float(len(r)) if r else 0.0

    @property
    def p_ko(self) -> float:
        if self.cur_hp <= 0:
            return 1.0
        r = self.rolls
        if not r:
            return 0.0
        return sum(1 for d in r if d >= self.cur_hp) / float(len(r))

    @property
    def expected_damage_pct(self) -> float:
        if self.max_hp <= 0:
            return 0.0
        return 100.0 * self.expected_damage / float(self.max_hp)

    def soft_ko_value(self, bonus: float = SOFT_KO_BONUS) -> float:
        return self.p_ko * bonus


def discrete_rolls(lo: int, hi: int, n: int = N_ROLLS) -> List[int]:
    """16 daños enteros equiespaciados entre lo y hi (inclusive)."""
    a = int(lo)
    b = int(hi)
    if a > b:
        a, b = b, a
    if n <= 1:
        return [a]
    if a == b:
        return [a] * n
    return [int(round(a + (b - a) * i / float(n - 1))) for i in range(n)]


def chance_from_range(
    rng: Optional[Tuple[int, int]],
    *,
    cur_hp: int,
    max_hp: int,
) -> Optional[ChanceNode]:
    if not rng:
        return None
    return ChanceNode(
        min_damage=int(rng[0]),
        max_damage=int(rng[1]),
        cur_hp=int(cur_hp),
        max_hp=int(max_hp),
    )


def expected_damage(rng: Optional[Tuple[int, int]]) -> float:
    if not rng:
        return 0.0
    rolls = discrete_rolls(int(rng[0]), int(rng[1]))
    return float(sum(rolls)) / float(len(rolls))


def p_ko(rng: Optional[Tuple[int, int]], cur_hp: int) -> float:
    if not rng or cur_hp <= 0:
        return 0.0 if cur_hp > 0 else 1.0
    rolls = discrete_rolls(int(rng[0]), int(rng[1]))
    return sum(1 for d in rolls if d >= int(cur_hp)) / float(len(rolls))


def combined_p_ko(
    ranges: Sequence[Optional[Tuple[int, int]]],
    cur_hp: int,
) -> float:
    """
    P(suma de hits independientes KOea) bajo rolls uniformes.

    Aproxima sumando daños roll-a-roll (mismo índice i en cada hit).
    Suficiente para 1-ply doubles focus fire.
    """
    if cur_hp <= 0:
        return 1.0
    valid = [r for r in ranges if r]
    if not valid:
        return 0.0
    roll_lists = [discrete_rolls(int(r[0]), int(r[1])) for r in valid]
    n = N_ROLLS
    ko = 0
    for i in range(n):
        total = 0
        for rolls in roll_lists:
            total += rolls[min(i, len(rolls) - 1)]
        if total >= int(cur_hp):
            ko += 1
    return ko / float(n)


def combined_expected_damage(
    ranges: Sequence[Optional[Tuple[int, int]]],
) -> float:
    return float(sum(expected_damage(r) for r in ranges))
