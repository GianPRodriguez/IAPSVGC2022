"""Validación Team Preview Random (spec §3)."""
from __future__ import annotations

import math
import random
from collections import Counter

from showdown.battle_bots.random.team_preview import (
    PREVIEW_PICK_COUNT,
    PREVIEW_SLOT_HI,
    PREVIEW_SLOT_LO,
    is_valid_preview_order,
    sample_team_preview_order,
)


def test_constants():
    assert PREVIEW_PICK_COUNT == 4
    assert PREVIEW_SLOT_LO == 1
    assert PREVIEW_SLOT_HI == 6


def test_sample_valid_and_unique():
    rng = random.Random(0)
    for _ in range(50):
        order = sample_team_preview_order(rng)
        assert is_valid_preview_order(order)
        assert len(order) == 4
        assert len(set(order)) == 4


def test_reproducible_with_seed():
    a = sample_team_preview_order(random.Random(42))
    b = sample_team_preview_order(random.Random(42))
    assert a == b
    assert is_valid_preview_order(a)


def test_all_slots_appear_over_many_samples():
    rng = random.Random(1)
    seen = Counter()
    for _ in range(500):
        for i in sample_team_preview_order(rng):
            seen[i] += 1
    assert set(seen) == set(range(1, 7))
    # Ningún slot debe quedar casi nunca elegido
    assert min(seen.values()) > 50


def test_covers_many_distinct_orders():
    rng = random.Random(7)
    orders = {sample_team_preview_order(rng) for _ in range(400)}
    # Espacio = P(6,4) = 360; con 400 samples esperamos decenas de distintas
    assert len(orders) > 80
    assert len(orders) <= math.perm(6, 4)


def test_invalid_orders():
    assert not is_valid_preview_order((1, 2, 3))
    assert not is_valid_preview_order((1, 2, 3, 3))
    assert not is_valid_preview_order((0, 1, 2, 3))
    assert not is_valid_preview_order((1, 2, 3, 7))
