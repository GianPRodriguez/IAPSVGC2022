"""Validación Team Preview Greedy-Solo/Joint (spec §3)."""
from __future__ import annotations

import math

from showdown.battle_bots.greedy_solo.team_preview import (
    PREVIEW_ALPHA,
    PREVIEW_BETA,
    PREVIEW_GAMMA,
    SCORE_EPS,
    best_team_preview_order,
    mon_from_species,
    order_to_team_string,
    score_combination,
)


def _mons(names: list[str]):
    return [mon_from_species(i, n) for i, n in enumerate(names, start=1)]


def test_constants_match_spec():
    assert PREVIEW_ALPHA == 1.0
    assert PREVIEW_BETA == 1.0
    assert PREVIEW_GAMMA == 0.01
    assert SCORE_EPS == 1e-9


def test_exhaustive_search_space_size_implicit():
    our = _mons([
        'Incineroar', 'Flutter Mane', 'Amoonguss', 'Tornadus',
        'Landorus-Therian', 'Rillaboom',
    ])
    foes = _mons(['Calyrex-Shadow', 'Urshifu', 'Chi-Yu', 'Raging Bolt', 'Ogerpon', 'Farigiraf'])
    scored = best_team_preview_order(our, foes)
    assert len(scored.order) == 4
    assert len(set(scored.order)) == 4
    assert all(1 <= i <= 6 for i in scored.order)
    # C(6,4)*C(4,2) = 90 candidatos teóricos
    assert math.comb(6, 4) * math.comb(4, 2) == 90


def test_score_formula_decomposition():
    our = _mons([
        'Incineroar', 'Amoonguss', 'Tornadus', 'Hatterene',
        'Great Tusk', 'Farigiraf',
    ])
    foes = _mons(['Kyogre', 'Groudon', 'Zacian', 'Miraidon', 'Koraidon', 'Terapagos'])
    scored = best_team_preview_order(our, foes)
    by_idx = {m.index: m for m in our}
    ordered = [by_idx[i] for i in scored.order]
    base = score_combination(ordered, foes)
    assert abs(scored.score - base.score) < 1e-9
    assert abs(
        scored.score
        - (PREVIEW_ALPHA * scored.offensive
           + PREVIEW_BETA * scored.defensive
           + PREVIEW_GAMMA * scored.bst)
    ) < 1e-6


def test_same_four_lead_order_does_not_change_score():
    """off/def/BST son invariantes al orden lead dentro del mismo bring."""
    our = _mons([
        'Pikachu', 'Mew', 'Jigglypuff', 'Eevee', 'Snorlax', 'Gengar',
    ])
    foes = _mons(['Charizard', 'Blastoise', 'Venusaur', 'Machamp', 'Alakazam', 'Golem'])
    combo = our[:4]
    s1 = score_combination(combo, foes)
    s2 = score_combination([combo[1], combo[0], combo[2], combo[3]], foes)
    s3 = score_combination([combo[2], combo[3], combo[0], combo[1]], foes)
    assert abs(s1.score - s2.score) < 1e-9
    assert abs(s1.score - s3.score) < 1e-9
    assert abs(s1.offensive - s2.offensive) < 1e-9
    assert abs(s1.bst - s3.bst) < 1e-9


def test_deterministic():
    our = _mons([
        'Incineroar', 'Flutter Mane', 'Amoonguss', 'Tornadus',
        'Landorus-Therian', 'Rillaboom',
    ])
    foes = _mons(['Kyogre', 'Groudon', 'Zacian', 'Miraidon', 'Koraidon', 'Terapagos'])
    a = best_team_preview_order(our, foes)
    b = best_team_preview_order(our, foes)
    assert a.order == b.order
    assert a.score == b.score


def test_pool_lt_4_fallback():
    our = _mons(['Pikachu', 'Mew'])
    foes = _mons(['Kyogre', 'Groudon'])
    scored = best_team_preview_order(our, foes)
    assert len(scored.order) == 4
    assert scored.score == 0.0


def test_order_to_team_string():
    assert order_to_team_string((5, 2, 1, 4)) == '5214'


def test_lead_preference_ignored():
    our = _mons([
        'Incineroar', 'Flutter Mane', 'Amoonguss', 'Tornadus',
        'Landorus-Therian', 'Rillaboom',
    ])
    foes = _mons(['Kyogre', 'Groudon', 'Zacian', 'Miraidon', 'Koraidon', 'Terapagos'])
    a = best_team_preview_order(our, foes)
    b = best_team_preview_order(our, foes, lead_preference=(6, 5, 4, 3))
    assert a.order == b.order


def test_weights_differ_from_expectimax_defaults():
    from showdown.battle_bots.expectimax_bayesian import team_preview as etp
    assert (PREVIEW_ALPHA, PREVIEW_BETA, PREVIEW_GAMMA) != (
        etp.PREVIEW_ALPHA, etp.PREVIEW_BETA, etp.PREVIEW_GAMMA,
    )
