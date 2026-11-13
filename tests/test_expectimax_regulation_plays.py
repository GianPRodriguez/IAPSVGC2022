"""Jugadas clásicas de Reg M-C en expectimax: Fake Out, sueño, Tailwind / Trick Room."""
from showdown.battle_bots.expectimax_bayesian.expectimax import (
    _fakeout_fails_by_tid,
    _flinch_immune,
    _foe_beats_fakeout,
)
from showdown.battle_bots.expectimax_bayesian.speed_order import acts_before, poke_effective_spe
from showdown.battle_bots.greedy_solo.greedy import CalcPoke
from showdown.battle_bots.move_mechanics.status_effects import (
    expected_effect_score,
    powder_can_affect,
    status_can_affect,
)


def mon(species='mon', spe=100, **kw):
    kw.setdefault('types', ('Normal',))
    return CalcPoke(species=species, stats={'hp': 200, 'atk': 100, 'def': 100, 'spa': 100,
                                            'spd': 100, 'spe': spe}, **kw)


# --- Fake Out ----------------------------------------------------------------
def test_flinch_immunity_by_ability_item_and_substitute():
    atk = mon('attacker')
    assert _flinch_immune(atk, mon(ability='Inner Focus'))
    assert _flinch_immune(atk, mon(ability='Shield Dust'))
    assert _flinch_immune(atk, mon(item='Covert Cloak'))
    assert _flinch_immune(atk, mon(has_substitute=True))
    assert not _flinch_immune(atk, mon(ability='Intimidate'))


def test_mold_breaker_ignores_flinch_immune_abilities_but_not_items():
    atk = mon(ability='Mold Breaker')
    assert not _flinch_immune(atk, mon(ability='Inner Focus'))
    assert _flinch_immune(atk, mon(item='Covert Cloak'))


def test_foe_with_own_fake_out_that_is_faster_beats_ours():
    ours = mon(spe=100, just_switched=True)
    faster = mon(spe=150, just_switched=True)
    assert _foe_beats_fakeout(ours, faster, ['Fake Out'], trick_room=False,
                              our_tailwind=False, opp_tailwind=False)
    slower = mon(spe=60, just_switched=True)
    assert not _foe_beats_fakeout(ours, slower, ['Fake Out'], trick_room=False,
                                  our_tailwind=False, opp_tailwind=False)
    # Trick Room invierte el orden; Tailwind propio lo recupera
    assert _foe_beats_fakeout(ours, slower, ['Fake Out'], trick_room=True,
                              our_tailwind=False, opp_tailwind=False)
    assert not _foe_beats_fakeout(ours, faster, ['Fake Out'], trick_room=False,
                                  our_tailwind=True, opp_tailwind=False)


def test_foe_beats_fakeout_needs_first_turn_and_the_move():
    ours = mon(spe=100, just_switched=True)
    assert not _foe_beats_fakeout(ours, mon(spe=150, just_switched=False), ['Fake Out'],
                                  trick_room=False, our_tailwind=False, opp_tailwind=False)
    assert not _foe_beats_fakeout(ours, mon(spe=150, just_switched=True), ['Close Combat'],
                                  trick_room=False, our_tailwind=False, opp_tailwind=False)


def test_fakeout_fails_by_tid_combines_all_conditions():
    ours = mon('attacker', just_switched=True)
    foes = [(1, mon(ability='Inner Focus')), (2, mon(ability='Intimidate')),
            (3, mon(types=('Ghost',)))]
    out = _fakeout_fails_by_tid(ours, foes, terrain=None)
    assert out == {1: True, 2: False, 3: True}


# --- Sueño (Reg M-C no usa Sleep Clause: Flat Rules) --------------------------
def test_sleep_gates_by_ability_terrain_status_and_substitute():
    atk = mon()
    assert status_can_affect('slp', attacker=atk, defender=mon())
    assert not status_can_affect('slp', attacker=atk, defender=mon(ability='Insomnia'))
    assert not status_can_affect('slp', attacker=atk, defender=mon(ability='Vital Spirit'))
    assert not status_can_affect('slp', attacker=atk, defender=mon(status='par'))
    assert not status_can_affect('slp', attacker=atk, defender=mon(has_substitute=True))
    assert not status_can_affect('slp', attacker=atk, defender=mon(), terrain='electricterrain')
    assert not status_can_affect('slp', attacker=atk, defender=mon(), terrain='mistyterrain')
    assert status_can_affect('slp', attacker=atk, defender=mon(types=('Flying',)),
                             terrain='electricterrain')


def test_powder_sleep_moves_do_not_affect_grass_overcoat_or_goggles():
    assert not powder_can_affect(mon(types=('Grass',)), 'Spore')
    assert not powder_can_affect(mon(ability='Overcoat'), 'Sleep Powder')
    assert not powder_can_affect(mon(item='Safety Goggles'), 'Spore')
    assert powder_can_affect(mon(), 'Spore')


def test_sleep_move_has_value_unless_target_is_immune():
    atk = mon()
    assert expected_effect_score('Spore', attacker=atk, defender=mon()).score > 0.0
    assert expected_effect_score('Spore', attacker=atk, defender=mon(types=('Grass',))).score == 0.0
    assert expected_effect_score('Spore', attacker=atk, defender=mon(status='slp')).score == 0.0


# --- Tailwind / Trick Room ----------------------------------------------------
def test_tailwind_lets_a_slower_mon_move_first_and_trick_room_inverts_it():
    slow, fast = mon(spe=80), mon(spe=120)
    assert not acts_before(poke_effective_spe(slow), 0, poke_effective_spe(fast), 0)
    assert acts_before(poke_effective_spe(slow, tailwind=True), 0, poke_effective_spe(fast), 0)
    assert acts_before(poke_effective_spe(slow), 0, poke_effective_spe(fast), 0, trick_room=True)
    # Tailwind propio bajo Trick Room: vuelve a ir después
    assert not acts_before(poke_effective_spe(slow, tailwind=True), 0,
                           poke_effective_spe(fast), 0, trick_room=True)
