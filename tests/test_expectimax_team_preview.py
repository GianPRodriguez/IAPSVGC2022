"""Validación del contrato Team Preview Expectimax (spec §3)."""
from __future__ import annotations

import math

from showdown.battle_bots.expectimax_bayesian import team_preview as tp
from showdown.battle_bots.greedy_solo.team_preview import (
    PreviewMon,
    best_team_preview_order,
    mon_from_species,
    score_combination,
)


def _mons(names: list[str]) -> list[PreviewMon]:
    return [mon_from_species(i, n) for i, n in enumerate(names, start=1)]


L = tp.LEGACY_PREVIEW_KW  # preview de roles hasta v1.6 (variante expectimax_legacy)


def test_preview_constants_match_spec():
    # v1.7: por defecto = tipado + BST de Greedy, sin roles ni calidad de lead.
    assert (tp.PREVIEW_ALPHA, tp.PREVIEW_BETA, tp.PREVIEW_GAMMA) == (1.0, 1.0, 0.01)
    assert tp.ROLE_WEIGHT == 0.0 and tp.LEAD_QUALITY_WEIGHT == 0.0
    assert (tp.LEGACY_PREVIEW_ALPHA, tp.LEGACY_PREVIEW_BETA, tp.LEGACY_PREVIEW_GAMMA) == (1.05, 1.25, 0.008)
    assert tp.LEGACY_LEAD_QUALITY_WEIGHT == 0.45
    assert tp.ROLE_BONUS_REDIRECT == 28.0
    assert tp.ROLE_BONUS_TAILWIND == 32.0
    assert tp.ROLE_BONUS_TRICK_ROOM == 26.0
    assert tp.ROLE_BONUS_SPEED_CONTROL == 18.0
    assert tp.ROLE_BONUS_FAKE_OUT == 14.0
    assert tp.ROLE_DIVERSITY_BONUS == 12.0


def test_role_sets_are_normalized_ids():
    for s in (
        tp.REDIRECT_SPECIES,
        tp.TAILWIND_SPECIES,
        tp.TRICK_ROOM_SPECIES,
        tp.SPEED_CONTROL_SPECIES,
        tp.FAKE_OUT_SPECIES,
    ):
        assert all(x == x.lower() and ' ' not in x for x in s)
    # Typo histórico eliminado (no matcheaba ninguna especie).
    assert 'riesnther' not in tp.REDIRECT_SPECIES


def test_exhaustive_combo_count_c64_x_c42():
    our = _mons([
        'Incineroar', 'Flutter Mane', 'Amoonguss', 'Tornadus',
        'Landorus-Therian', 'Rillaboom',
    ])
    foes = _mons(['Calyrex-Shadow', 'Urshifu', 'Chi-Yu', 'Raging Bolt', 'Ogerpon', 'Farigiraf'])
    _scored, meta = tp.best_expectimax_preview_with_meta(our, foes)
    assert meta.n_combos_4 == math.comb(6, 4)
    assert meta.n_lead_orders == math.comb(6, 4) * math.comb(4, 2)
    assert meta.n_scored == meta.n_lead_orders
    assert len(_scored.order) == 4
    assert len(set(_scored.order)) == 4


def test_score_formula_decomposition():
    our = _mons([
        'Incineroar', 'Amoonguss', 'Tornadus', 'Hatterene',
        'Great Tusk', 'Farigiraf',
    ])
    foes = _mons(['Kyogre', 'Groudon', 'Incineroar', 'Pikachu', 'Mew', 'Jigglypuff'])

    scored, meta = tp.best_expectimax_preview_with_meta(our, foes, **L)
    by_idx = {m.index: m for m in our}
    ordered = [by_idx[i] for i in scored.order]
    leads = ordered[:2]
    base = score_combination(ordered, foes, alpha=L['alpha'], beta=L['beta'], gamma=L['gamma'])
    role = tp._role_bonus(ordered)
    lead_q = tp._lead_quality(leads, foes)
    expected = base.score + role + L['lead_weight'] * lead_q
    assert abs(scored.score - expected) < 1e-6
    assert meta.chosen is not None
    assert meta.why_lines


def test_roles_prefer_support_over_four_attackers():
    """Con foes genéricos, bring con redirect+TW debe puntuar roles > 0."""
    support = _mons([
        'Amoonguss', 'Tornadus', 'Incineroar', 'Flutter Mane',
        'Landorus-Therian', 'Chi-Yu',
    ])
    attackers = _mons([
        'Chi-Yu', 'Flutter Mane', 'Raging Bolt', 'Urshifu',
        'Ogerpon', 'Calyrex-Shadow',
    ])
    foes = _mons([
        'Kyogre', 'Groudon', 'Zacian', 'Miraidon', 'Koraidon', 'Terapagos',
    ])
    s_sup, _ = tp.best_expectimax_preview_with_meta(support, foes, **L)
    s_atk, _ = tp.best_expectimax_preview_with_meta(attackers, foes, **L)
    role_sup = tp._role_bonus([m for m in support if m.index in s_sup.order])
    role_atk = tp._role_bonus([m for m in attackers if m.index in s_atk.order])
    assert role_sup > role_atk
    assert role_sup >= (
        tp.ROLE_BONUS_REDIRECT + tp.ROLE_BONUS_TAILWIND
        + tp.ROLE_BONUS_SPEED_CONTROL + tp.ROLE_BONUS_FAKE_OUT
        + tp.ROLE_DIVERSITY_BONUS
    )


def test_diversity_bucket_bonus():
    # Solo FO+spdCtrl = un bucket disrupt → sin diversity
    mono = _mons(['Incineroar', 'Rillaboom', 'Chi-Yu', 'Flutter Mane', 'Mew', 'Pikachu'])
    ids = {tp._nid(m.name) for m in mono[:4]}
    flags = tp.role_bucket_flags(ids)
    assert flags['disrupt']
    assert not flags['redirect']
    assert sum(flags.values()) == 1
    # Redirect + TW → diversity
    duo = _mons(['Amoonguss', 'Tornadus', 'Chi-Yu', 'Flutter Mane', 'Mew', 'Pikachu'])
    ids2 = {tp._nid(m.name) for m in duo[:4]}
    assert sum(tp.role_bucket_flags(ids2).values()) >= 2
    assert tp._role_bonus(duo[:4]) >= (
        tp.ROLE_BONUS_REDIRECT + tp.ROLE_BONUS_TAILWIND + tp.ROLE_DIVERSITY_BONUS
    )


def test_differs_from_greedy_on_role_heavy_pool():
    our = _mons([
        'Amoonguss', 'Tornadus', 'Incineroar', 'Hatterene',
        'Flutter Mane', 'Chi-Yu',
    ])
    foes = _mons([
        'Calyrex-Shadow', 'Urshifu-Rapid-Strike', 'Raging Bolt',
        'Ogerpon-Wellspring', 'Farigiraf', 'Landorus-Therian',
    ])
    exp, _ = tp.best_expectimax_preview_with_meta(our, foes, **L)
    gre = best_team_preview_order(our, foes)
    # No se exige orden distinto siempre, pero el score Expectimax debe incluir roles.
    ordered = [m for m in our if m.index in exp.order]
    assert tp._role_bonus(ordered) > 0
    # Pesos distintos ⇒ score base greedy ≠ fórmula expectimax aunque el orden coincida.
    base_g = score_combination(ordered, foes, alpha=1.0, beta=1.0, gamma=0.01)
    base_e = score_combination(ordered, foes, alpha=L['alpha'], beta=L['beta'], gamma=L['gamma'])
    assert abs(base_g.score - base_e.score) > 1e-9 or exp.order != gre.order


def test_pool_lt_4_fallback():
    our = _mons(['Pikachu', 'Mew'])
    foes = _mons(['Kyogre', 'Groudon'])
    scored, meta = tp.best_expectimax_preview_with_meta(our, foes)
    assert len(scored.order) == 4
    assert any('pool < 4' in w for w in meta.why_lines)


def test_indeedee_f_normalizes_to_redirect_and_tr():
    assert 'indeedeef' in tp.REDIRECT_SPECIES
    assert 'indeedeef' in tp.TRICK_ROOM_SPECIES
    m = mon_from_species(1, 'Indeedee-F')
    assert tp._nid(m.name) == 'indeedeef'


def test_default_preview_equals_greedy_typing_bst():
    """v1.7: el preview por defecto elige exactamente lo mismo que Greedy (tipado + BST)."""
    import random
    from showdown.inference_sim.team_factory import active_teams

    rng = random.Random(7)
    teams = active_teams()
    for _ in range(40):
        a, b = rng.sample(teams, 2)
        our = [mon_from_species(i + 1, m.species) for i, m in enumerate(a.mons)]
        foes = [mon_from_species(i + 1, m.species) for i, m in enumerate(b.mons)]
        exp, _ = tp.best_expectimax_preview_with_meta(our, foes)
        assert exp.order == best_team_preview_order(our, foes).order


def test_move_roles_only_count_real_moves():
    """Opción 2: el bono de rol por movimientos ignora la especie."""
    mons = _mons(['Incineroar', 'Tornadus', 'Chi-Yu', 'Flutter Mane'])
    none = {1: {'flareblitz', 'knockoff'}, 2: {'bleakwindstorm'}, 3: set(), 4: set()}
    assert tp._role_bonus_from_moves(mons, none) == 0.0  # especies "de rol" sin el movimiento
    real = {1: {'fakeout'}, 2: {'tailwind'}, 3: set(), 4: set()}
    assert tp._role_bonus_from_moves(mons, real) == (
        tp.ROLE_BONUS_FAKE_OUT + tp.ROLE_BONUS_TAILWIND + tp.ROLE_DIVERSITY_BONUS
    )
