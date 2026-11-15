"""Diversidad de los baselines: muestreo entre casi-empates y resumen de la traza."""
import json
import random
import sys

sys.path.insert(0, '.')

from showdown.battle_bots.diversity_log import sample_near_ties
from showdown.inference_sim import analyze_diversity as ad


def test_margin_zero_is_deterministic_argmax():
    scored = [(10.0, 'a'), (30.0, 'b'), (30.0, 'c'), (5.0, 'd')]
    assert sample_near_ties(scored, 0.0) == ('b', 1)  # primer máximo, como el argmax


def test_samples_only_within_margin():
    scored = [(100.0, 'a'), (95.0, 'b'), (89.0, 'c'), (50.0, 'd')]
    rng = random.Random(0)
    seen = {sample_near_ties(scored, 10.0, rng)[0] for _ in range(200)}
    assert seen == {'a', 'b'}
    assert sample_near_ties(scored, 10.0, rng)[1] == 2


def test_single_candidate_returns_best():
    assert sample_near_ties([(100.0, 'a'), (10.0, 'b')], 5.0) == ('a', 1)
    assert sample_near_ties([], 5.0) == (None, 0)


def test_expectimax_config_default_is_deterministic():
    from showdown.battle_bots.expectimax_bayesian.expectimax import ExpectimaxConfig
    assert ExpectimaxConfig().tie_sample_margin == 0.0


def test_joint_tie_margin_env(monkeypatch):
    from showdown.battle_bots.greedy_joint.pokeenv_player import _joint_tie_margin
    monkeypatch.delenv('PS_JOINT_TIE_MARGIN', raising=False)
    assert _joint_tie_margin() == 0.0
    monkeypatch.setenv('PS_JOINT_TIE_MARGIN', '7.5')
    assert _joint_tie_margin() == 7.5


def test_summary(tmp_path):
    p = tmp_path / 'div.123'
    rows = [
        {'agent': 'greedy_joint', 'same_solo': True, 'tie_cands': 1, 'tie_changed': False},
        {'agent': 'greedy_joint', 'same_solo': False, 'tie_cands': 3, 'tie_changed': True},
        {'agent': 'expectimax_bayesian', 'same_solo': False, 'same_joint': True,
         'tie_cands': 1, 'tie_changed': False},
    ]
    p.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    s = ad.summarize(ad.load(str(tmp_path / 'div')))
    assert s['greedy_joint']['n'] == 2 and s['greedy_joint']['same_solo'] == 0.5
    assert s['greedy_joint']['tie_changed'] == 0.5 and s['greedy_joint']['tie_cands_mean'] == 3
    assert s['expectimax_bayesian']['same_joint'] == 1.0


def _fixture():
    from showdown.battle_bots.greedy_solo.greedy import CalcPoke, MoveOption
    left = CalcPoke(species='Rillaboom', nature='Adamant',
                    stats={'hp': 193, 'atk': 173, 'def': 115, 'spa': 72, 'spd': 100, 'spe': 109},
                    cur_hp=193, types=('Grass', 'Dark'))
    right = CalcPoke(species='Salamence', nature='Naive',
                     stats={'hp': 171, 'atk': 176, 'def': 100, 'spa': 145, 'spd': 90, 'spe': 163},
                     cur_hp=171, types=('Dragon', 'Flying'))
    foes = [
        (1, CalcPoke(species='Pelipper', nature='Modest',
                     stats={'hp': 140, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
                     cur_hp=140, types=('Water', 'Flying'))),
        (2, CalcPoke(species='Toxapex', nature='Calm',
                     stats={'hp': 141, 'atk': 74, 'def': 181, 'spa': 73, 'spd': 187, 'spe': 55},
                     cur_hp=141, types=('Poison', 'Water'))),
    ]
    lm = [MoveOption(1, 'Wood Hammer', 'normal', 'Physical'), MoveOption(2, 'Knock Off', 'normal', 'Physical')]
    rm = [MoveOption(1, 'Double-Edge', 'normal', 'Physical'), MoveOption(2, 'Dragon Claw', 'normal', 'Physical')]
    return left, right, lm, rm, foes


def _fake_batch(reqs):
    return [(30, 36) for _ in reqs]  # todo empata: muchos pares casi iguales


def test_joint_tie_sampling_varies_and_zero_is_fixed():
    import asyncio
    from showdown.battle_bots.expectimax_bayesian.divergence import pair_sig
    from showdown.battle_bots.greedy_joint import joint
    from showdown.damage_calc_bridge import set_damage_range_override
    set_damage_range_override(_fake_batch)
    try:
        args = _fixture()
        fixed = {pair_sig(*asyncio.run(joint.best_joint_double_action(*args))[:2]) for _ in range(5)}
        assert len(fixed) == 1 and joint.LAST_TIE == (1, False)
        varied = {pair_sig(*asyncio.run(joint.best_joint_double_action(*args, tie_margin=1e9))[:2])
                  for _ in range(40)}
        assert len(varied) > 1
        assert joint.LAST_TIE[0] > 1
    finally:
        set_damage_range_override(None)


def test_expectimax_tie_sampling_runs():
    import asyncio
    from showdown.battle_bots.expectimax_bayesian.expectimax import (
        ExpectimaxConfig, best_expectimax_bayesian_action)
    from showdown.damage_calc_bridge import set_damage_range_override
    set_damage_range_override(_fake_batch)
    try:
        left, right, lm, rm, foes = _fixture()
        cfg = ExpectimaxConfig(deadline_ms=5000.0, phase1_budget_ms=2000.0, phase2_end_ms=4800.0,
                               k_profiles=2, max_worlds=2, tie_sample_margin=1e9)
        seen = set()
        for _ in range(12):
            bl, br, _u, meta = asyncio.run(best_expectimax_bayesian_action(
                left, right, lm, rm, foes, config=cfg))
            assert bl is not None and br is not None and not meta.errors, meta.errors
            seen.add((bl.move_slot, bl.target, br.move_slot, br.target))
        assert meta.tie_cands > 1
        assert len(seen) > 1
    finally:
        set_damage_range_override(None)


def test_joint_tie_changed_only_with_several_candidates():
    """'cambió' exige más de un candidato y un par distinto del argmax (antes se comparaba por identidad)."""
    import asyncio
    from showdown.battle_bots.greedy_joint import joint
    from showdown.damage_calc_bridge import set_damage_range_override

    def uneven(reqs):  # daños distintos: con margen chico hay un solo candidato
        return [(10 + 7 * i, 12 + 7 * i) for i, _ in enumerate(reqs)]

    set_damage_range_override(uneven)
    try:
        args = _fixture()
        for _ in range(10):
            asyncio.run(joint.best_joint_double_action(*args, tie_margin=1e-6))
            n, changed = joint.LAST_TIE
            assert not changed or n > 1
            assert n == 1 and not changed
    finally:
        set_damage_range_override(None)
    set_damage_range_override(_fake_batch)
    try:
        flags = []
        for _ in range(40):
            asyncio.run(joint.best_joint_double_action(*_fixture(), tie_margin=1e9))
            n, changed = joint.LAST_TIE
            assert n > 1
            flags.append(changed)
        assert any(flags) and not all(flags)
    finally:
        set_damage_range_override(None)


def test_expectimax_phase1_tie_margin():
    import asyncio
    from showdown.battle_bots.expectimax_bayesian.expectimax import (
        ExpectimaxConfig, best_expectimax_bayesian_action)
    from showdown.damage_calc_bridge import set_damage_range_override
    assert ExpectimaxConfig().phase1_tie_margin == 0.0
    set_damage_range_override(_fake_batch)
    try:
        left, right, lm, rm, foes = _fixture()
        cfg = ExpectimaxConfig(deadline_ms=5000.0, phase1_budget_ms=2000.0, phase2_end_ms=4800.0,
                               k_profiles=2, max_worlds=2, phase1_tie_margin=1e9)
        changed = []
        for _ in range(20):
            bl, br, _u, meta = asyncio.run(best_expectimax_bayesian_action(
                left, right, lm, rm, foes, config=cfg))
            assert bl is not None and br is not None and not meta.errors, meta.errors
            assert meta.p1_tie_cands > 1
            changed.append(meta.p1_tie_changed)
        assert any(changed)
    finally:
        set_damage_range_override(None)
