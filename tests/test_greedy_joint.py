"""Smoke tests Greedy-Joint (producto cartesiano + daño combinado)."""
from __future__ import annotations

import asyncio

from showdown.battle_bots.greedy_solo.greedy import CalcPoke, MoveOption
from showdown.battle_bots.greedy_joint.joint import best_joint_double_action
from showdown.damage_calc_bridge import set_damage_range_override


def test_joint_returns_pair_with_positive_score():
    def fake_batch(reqs):
        out = []
        for req in reqs:
            foe = str((req.get('defender') or {}).get('species', '')).lower()
            if 'toxapex' in foe:
                out.append((40, 50))
            else:
                out.append((80, 100))
        return out

    set_damage_range_override(fake_batch)
    try:
        left = CalcPoke(
            species='Rillaboom', nature='Adamant',
            stats={'hp': 193, 'atk': 173, 'def': 115, 'spa': 72, 'spd': 100, 'spe': 109},
            cur_hp=193, types=('Grass', 'Dark'),
        )
        right = CalcPoke(
            species='Salamence', nature='Naive',
            stats={'hp': 171, 'atk': 176, 'def': 100, 'spa': 145, 'spd': 90, 'spe': 163},
            cur_hp=171, types=('Dragon', 'Flying'),
        )
        foes = [
            (1, CalcPoke(
                species='Pelipper', nature='Modest',
                stats={'hp': 140, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
                cur_hp=40, types=('Water', 'Flying'),
            )),
            (2, CalcPoke(
                species='Toxapex', nature='Calm',
                stats={'hp': 141, 'atk': 74, 'def': 181, 'spa': 73, 'spd': 187, 'spe': 55},
                cur_hp=141, types=('Poison', 'Water'),
            )),
        ]
        left_moves = [MoveOption(1, 'Wood Hammer', 'normal', 'Physical')]
        right_moves = [MoveOption(1, 'Double-Edge', 'normal', 'Physical')]

        best_l, best_r, score = asyncio.run(
            best_joint_double_action(
                left, right, left_moves, right_moves, foes,
            )
        )
        assert best_l is not None and best_r is not None
        assert best_l.target in (1, 2)
        assert best_r.target in (1, 2)
        assert score > 0
    finally:
        set_damage_range_override(None)


def test_joint_import_battlebot():
    from showdown.battle_bots.greedy_joint.main import BattleBot
    from showdown.battle_bots.greedy_solo.main import BattleBot as Solo

    assert BattleBot is not Solo
    assert BattleBot.__module__.endswith('greedy_joint.main')


def test_fakeout_no_bonus_vs_ghost_only_field():
    """Sin Scrappy, Fake Out no recibe +25 si ambos foes son Ghost."""
    from showdown.battle_bots.greedy_joint.joint import (
        _FAKE_OUT_BONUS,
        _atoms_from_side,
        _utility_score,
        fake_out_can_affect,
        JointAtom,
    )
    from showdown.battle_bots.greedy_solo.type_chart import fake_out_can_affect as fo_types

    ghost = CalcPoke(species='Gengar', types=('Ghost', 'Poison'), cur_hp=100,
                     stats={'hp': 100, 'atk': 100, 'def': 100, 'spa': 100, 'spd': 100, 'spe': 100})
    normal = CalcPoke(species='Incineroar', types=('Fire', 'Dark'), cur_hp=100,
                      stats={'hp': 100, 'atk': 100, 'def': 100, 'spa': 100, 'spd': 100, 'spe': 100})
    atk = CalcPoke(species='Incineroar', types=('Fire', 'Dark'), ability='Intimidate',
                   just_switched=True, cur_hp=100,
                   stats={'hp': 100, 'atk': 100, 'def': 100, 'spa': 100, 'spd': 100, 'spe': 100})
    scrappy = CalcPoke(species='Kangaskhan', types=('Normal',), ability='Scrappy',
                       just_switched=True, cur_hp=100,
                       stats={'hp': 100, 'atk': 100, 'def': 100, 'spa': 100, 'spd': 100, 'spe': 100})

    assert not fo_types(None, ghost)
    assert fo_types(None, normal)
    assert fo_types('Scrappy', ghost)
    assert not fake_out_can_affect(atk, ghost)
    assert fake_out_can_affect(scrappy, ghost)

    move = MoveOption(1, 'Fake Out', 'normal', 'Physical')
    foes_ghost = [(1, ghost), (2, ghost)]
    meta = [('L', move, 1, ghost, 0), ('L', move, 2, ghost, 0)]
    atoms = _atoms_from_side(
        [move], meta, [(40, 50), (40, 50)], foes_ghost,
        can_mega=False, attacker=atk, weather=None,
    )
    assert all(a.utility != 'fakeout' for a in atoms)
    assert not any(_norm_is_fakeout(a) for a in atoms)

    foes_mixed = [(1, ghost), (2, normal)]
    meta2 = [('L', move, 1, ghost, 0), ('L', move, 2, normal, 0)]
    atoms2 = _atoms_from_side(
        [move], meta2, [(0, 0), (40, 50)], foes_mixed,
        can_mega=False, attacker=atk, weather=None,
    )
    # Con A4 (_MECH_SYNERGY): Fake Out es átomo damage (no utility='fakeout').
    from showdown.battle_bots.greedy_joint.joint import _MECH_SYNERGY
    if _MECH_SYNERGY:
        from showdown.battle_bots.greedy_joint.joint import _norm_move as _nm
        fo_dmg = [a for a in atoms2 if _nm(a.move_name) == 'fakeout']
        assert fo_dmg
        assert all(a.target == 2 for a in fo_dmg)  # solo no-Ghost
        u = _utility_score(
            fo_dmg[0], JointAtom(0, 0, utility='status'),
            foes=foes_mixed, has_damage=False,
        )
        assert u == 0.0  # FO valor en effect_bonus del átomo daño
    else:
        assert any(a.utility == 'fakeout' for a in atoms2)
        fo = next(a for a in atoms2 if a.utility == 'fakeout')
        assert fo.target == 2
        u = _utility_score(fo, JointAtom(0, 0, utility='status'), foes=foes_mixed, has_damage=False)
        from showdown.battle_bots.greedy_joint.joint import _DAMAGE_ISO
        if _DAMAGE_ISO:
            assert u == 0.0
        else:
            assert u >= _FAKE_OUT_BONUS


def _norm_is_fakeout(a) -> bool:
    return str(getattr(a, 'move_name', '')).lower().replace(' ', '') == 'fakeout' and a.utility == 'fakeout'


def test_redirect_plus_ko_beats_double_attack_focus():
    """v_stable: redirect+KO > focus overkill. Propuesta A: daño gana a redirect."""
    from showdown.battle_bots.greedy_joint import joint as J

    foes = [
        (1, CalcPoke(
            species='Pelipper', nature='Modest',
            stats={'hp': 140, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
            cur_hp=30, types=('Water', 'Flying'),
        )),
        (2, CalcPoke(
            species='Toxapex', nature='Calm',
            stats={'hp': 141, 'atk': 74, 'def': 181, 'spa': 73, 'spd': 187, 'spe': 55},
            cur_hp=141, types=('Poison', 'Water'),
        )),
    ]
    redirect = J.JointAtom(
        move_slot=1, target=0, utility='redirect', move_name='Follow Me',
    )
    kill = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 90.0}, max_roll={1: 40}, ko_tids={1},
    )
    chip = J.JointAtom(
        move_slot=2, target=1, utility='damage', move_name='U-turn',
        damage_pct={1: 20.0}, max_roll={1: 10}, ko_tids=set(),
    )
    score_redir = J._score_pair(redirect, kill, foes)
    score_double = J._score_pair(chip, kill, foes)
    assert J._redirect_synergy(redirect, kill, foes) == J._REDIRECT_WITH_KO
    if J._DAMAGE_ISO and J._SYNERGY_LITE:
        # A3: redirect+KO vuelve a valer (escalado); no debe perder vs overkill.
        assert score_redir > score_double
    elif J._DAMAGE_ISO:
        # Chip overkill se anula; kill solo ≈ redirect+kill (sin bonus redirect).
        assert score_double >= score_redir - 1e-6
    else:
        assert score_redir > score_double


def test_redirect_with_high_damage_gets_attack_bonus():
    from showdown.battle_bots.greedy_joint import joint as J

    foes = [
        (1, CalcPoke(
            species='Archaludon', nature='Sassy',
            stats={'hp': 170, 'atk': 125, 'def': 150, 'spa': 145, 'spd': 100, 'spe': 100},
            cur_hp=170, types=('Steel', 'Dragon'),
        )),
    ]
    redirect = J.JointAtom(move_slot=1, target=0, utility='redirect', move_name='Rage Powder')
    heavy = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Double-Edge',
        damage_pct={1: 55.0}, max_roll={1: 80}, ko_tids=set(),
    )
    assert J._redirect_synergy(redirect, heavy, foes) == J._REDIRECT_WITH_ATTACK


def test_speed_aware_faster_ko_wastes_slower_focus():
    """B: el más rápido remata; el hit del lento al mismo foe no suma %."""
    from showdown.battle_bots.greedy_joint import joint as J

    foes = [
        (1, CalcPoke(
            species='Pelipper',
            stats={'hp': 100, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
            cur_hp=40, types=('Water', 'Flying'),
        )),
    ]
    fast = CalcPoke(
        species='Salamence',
        stats={'hp': 170, 'atk': 176, 'def': 100, 'spa': 145, 'spd': 90, 'spe': 200},
        cur_hp=170, types=('Dragon', 'Flying'),
    )
    slow = CalcPoke(
        species='Rillaboom',
        stats={'hp': 190, 'atk': 173, 'def': 115, 'spa': 72, 'spd': 100, 'spe': 50},
        cur_hp=190, types=('Grass', 'Dark'),
    )
    kill = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Double-Edge',
        damage_pct={1: 80.0}, max_roll={1: 50}, ko_tids={1},
    )
    chip = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 60.0}, max_roll={1: 30}, ko_tids=set(),
    )
    # Rápido=kill, lento=chip → chip overkill tras KO.
    s_fast_first = J._score_pair(kill, chip, foes, left_poke=fast, right_poke=slow)
    # Ambos al mismo foe sin Spe (A puro): con SPEED_AWARE off no aplica; con on
    # y pokes, el orden importa. Split a otro foe debería poder sumar más chip.
    foes2 = foes + [
        (2, CalcPoke(
            species='Toxapex',
            stats={'hp': 141, 'atk': 74, 'def': 181, 'spa': 73, 'spd': 187, 'spe': 55},
            cur_hp=141, types=('Poison', 'Water'),
        )),
    ]
    chip_split = J.JointAtom(
        move_slot=1, target=2, utility='damage', move_name='Wood Hammer',
        damage_pct={2: 40.0}, max_roll={2: 30}, ko_tids=set(),
    )
    s_split = J._score_pair(kill, chip_split, foes2, left_poke=fast, right_poke=slow)
    if J._SPEED_AWARE and J._DAMAGE_ISO:
        # Split suma el 40% al segundo foe; focus desperdicia el chip.
        assert s_split > s_fast_first
    if J._FOCUS_SPEED and J._DAMAGE_ISO and not J._SPEED_AWARE:
        # B-lite: mismo efecto en focus (rápido remata → chip lento no suma).
        assert s_split > s_fast_first
    assert J._ordered_atoms(
        kill, chip, left_poke=fast, right_poke=slow, terrain=None,
    )[0] is kill


def test_focus_speed_sequential_ko_wastes_slower():
    """B-lite: si ambos pueden rematar, Spe acredita el hit del más rápido (no anula ambos)."""
    from showdown.battle_bots.greedy_joint import joint as J
    from showdown.battle_bots.greedy_solo.greedy import KO_BONUS

    foes = [
        (1, CalcPoke(
            species='Pelipper',
            stats={'hp': 100, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
            cur_hp=40, types=('Water', 'Flying'),
        )),
    ]
    fast = CalcPoke(
        species='Salamence',
        stats={'hp': 170, 'atk': 176, 'def': 100, 'spa': 145, 'spd': 90, 'spe': 200},
        cur_hp=170, types=('Dragon', 'Flying'),
    )
    slow = CalcPoke(
        species='Rillaboom',
        stats={'hp': 190, 'atk': 173, 'def': 115, 'spa': 72, 'spd': 100, 'spe': 50},
        cur_hp=190, types=('Grass', 'Dark'),
    )
    hit_fast = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Double-Edge',
        damage_pct={1: 80.0}, max_roll={1: 50}, ko_tids={1},
    )
    hit_slow = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 80.0}, max_roll={1: 50}, ko_tids={1},
    )
    s_spe = J._score_pair(hit_fast, hit_slow, foes, left_poke=fast, right_poke=slow)
    s_no = J._score_pair(hit_fast, hit_slow, foes)  # sin pokes → B-lite off
    if J._DAMAGE_ISO and J._FOCUS_SPEED:
        # Sin Spe ambos se anulan mutuamente (solo KO_BONUS); con Spe suma % del rápido.
        assert s_no <= float(KO_BONUS) + 5.0
        # cur_hp/max = 40% → clip del hit rápido a remain, no al damage_pct bruto.
        assert s_spe >= 40.0 + float(KO_BONUS) - 1e-6
        assert s_spe > s_no


def test_focus_clip_prefers_split_without_joint_ko():
    """A2: focus 70+70 sin KO no vale más que remain; split puede ganar."""
    from showdown.battle_bots.greedy_joint import joint as J

    foes = [
        (1, CalcPoke(
            species='Pelipper',
            stats={'hp': 100, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
            cur_hp=100, types=('Water', 'Flying'),
        )),
        (2, CalcPoke(
            species='Toxapex',
            stats={'hp': 100, 'atk': 74, 'def': 181, 'spa': 73, 'spd': 187, 'spe': 55},
            cur_hp=100, types=('Poison', 'Water'),
        )),
    ]
    hit_a = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 70.0}, max_roll={1: 40}, ko_tids=set(),
    )
    hit_b_focus = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Double-Edge',
        damage_pct={1: 70.0}, max_roll={1: 40}, ko_tids=set(),
    )
    hit_b_split = J.JointAtom(
        move_slot=1, target=2, utility='damage', move_name='Double-Edge',
        damage_pct={2: 55.0}, max_roll={2: 40}, ko_tids=set(),
    )
    s_focus = J._score_pair(hit_a, hit_b_focus, foes)
    s_split = J._score_pair(hit_a, hit_b_split, foes)
    if J._DAMAGE_ISO and J._FOCUS_CLIP:
        assert s_focus <= 100.0 + 1e-6
        assert s_split > s_focus


def test_joint_ko_synergy_beats_alone_chip_focus():
    """A2: KO conjunto (nadie solo) gana a focus overkill cuando uno ya remata."""
    from showdown.battle_bots.greedy_joint import joint as J

    foes = [
        (1, CalcPoke(
            species='Pelipper',
            stats={'hp': 100, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
            cur_hp=50, types=('Water', 'Flying'),
        )),
        (2, CalcPoke(
            species='Toxapex',
            stats={'hp': 100, 'atk': 74, 'def': 181, 'spa': 73, 'spd': 187, 'spe': 55},
            cur_hp=80, types=('Poison', 'Water'),
        )),
    ]
    # Uno solo remata foe1; el otro chip al mismo → overkill anulado.
    kill = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 90.0}, max_roll={1: 60}, ko_tids={1},
    )
    waste = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='U-turn',
        damage_pct={1: 30.0}, max_roll={1: 20}, ko_tids=set(),
    )
    # Joint KO en foe2: 40+45 rolls >= 80, nadie solo.
    chip_l = J.JointAtom(
        move_slot=1, target=2, utility='damage', move_name='Wood Hammer',
        damage_pct={2: 45.0}, max_roll={2: 40}, ko_tids=set(),
    )
    chip_r = J.JointAtom(
        move_slot=1, target=2, utility='damage', move_name='Double-Edge',
        damage_pct={2: 50.0}, max_roll={2: 45}, ko_tids=set(),
    )
    s_waste = J._score_pair(kill, waste, foes)
    s_joint = J._score_pair(chip_l, chip_r, foes)
    if J._DAMAGE_ISO:
        assert s_joint > s_waste


def test_protect_react_only_when_low_hp_and_partner_damages():
    """A2: Protect +6 solo si HP%≤40 y partner hace daño."""
    from showdown.battle_bots.greedy_joint import joint as J

    foes = [
        (1, CalcPoke(
            species='Pelipper',
            stats={'hp': 100, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
            cur_hp=100, types=('Water', 'Flying'),
        )),
    ]
    low = CalcPoke(
        species='Rillaboom',
        stats={'hp': 200, 'atk': 173, 'def': 115, 'spa': 72, 'spd': 100, 'spe': 50},
        cur_hp=60, types=('Grass', 'Dark'),  # 30%
    )
    full = CalcPoke(
        species='Salamence',
        stats={'hp': 170, 'atk': 176, 'def': 100, 'spa': 145, 'spd': 90, 'spe': 163},
        cur_hp=170, types=('Dragon', 'Flying'),
    )
    protect = J.JointAtom(move_slot=1, target=0, utility='protect', move_name='Protect')
    hit = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Double-Edge',
        damage_pct={1: 50.0}, max_roll={1: 40}, ko_tids=set(),
    )
    idle = J.JointAtom(move_slot=2, target=0, utility='status', move_name='Growl')
    u_ok = J._utility_score(
        protect, hit, foes=foes, has_damage=True, left_poke=low, right_poke=full,
    )
    u_full = J._utility_score(
        protect, hit, foes=foes, has_damage=True, left_poke=full, right_poke=full,
    )
    u_no_dmg = J._utility_score(
        protect, idle, foes=foes, has_damage=False, left_poke=low, right_poke=full,
    )
    if J._DAMAGE_ISO:
        assert u_ok == J._PROTECT_REACT_BONUS
        assert u_full == 0.0
        assert u_no_dmg == 0.0


def test_synergy_lite_fakeout_hh_need_partner_damage():
    """A3/A4: HH nudge o FO plano según flags; sin partner_dmg = 0."""
    from showdown.battle_bots.greedy_joint import joint as J

    foes = [
        (1, CalcPoke(
            species='Pelipper',
            stats={'hp': 100, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
            cur_hp=100, types=('Water', 'Flying'),
        )),
    ]
    fo = J.JointAtom(move_slot=1, target=1, utility='fakeout', move_name='Fake Out')
    hh = J.JointAtom(move_slot=1, target=0, utility='helpinghand', move_name='Helping Hand')
    hit = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 60.0}, max_roll={1: 40}, ko_tids=set(),
    )
    idle = J.JointAtom(move_slot=2, target=0, utility='status', move_name='Growl')
    if J._DAMAGE_ISO and J._SYNERGY_LITE:
        if J._MECH_SYNERGY:
            # FO damage-atom: deny en effect; A8 suma presión del partner.
            u_fo = J._utility_score(fo, hit, foes=foes, has_damage=True)
            if J._FO_PARTNER_PRESSURE:
                assert u_fo == J._FO_PARTNER_PRESS_BONUS
            else:
                assert u_fo == 0.0
            assert J._utility_score(hh, hit, foes=foes, has_damage=True) == 2.0
        else:
            assert J._utility_score(fo, hit, foes=foes, has_damage=True) == J._FO_PARTNER_BONUS
            assert J._utility_score(hh, hit, foes=foes, has_damage=True) == J._HH_PARTNER_BONUS
        assert J._utility_score(fo, idle, foes=foes, has_damage=False) == 0.0
        assert J._utility_score(hh, idle, foes=foes, has_damage=False) == 0.0


def test_speed_and_screen_partner_bonus_with_damage():
    """A9: Tailwind condicional (+partner daño); pantallas = status."""
    from showdown.battle_bots.greedy_joint import joint as J
    from showdown.battle_bots.greedy_solo.greedy import MoveOption

    foes = [
        (1, CalcPoke(
            species='Pelipper',
            stats={'hp': 100, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
            cur_hp=100, types=('Water', 'Flying'),
        )),
    ]
    tw = J.JointAtom(move_slot=1, target=0, utility='speed', move_name='Tailwind')
    hit = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 50.0}, max_roll={1: 40}, ko_tids=set(),
    )
    idle = J.JointAtom(move_slot=2, target=0, utility='status', move_name='Growl')
    assert J._utility_kind(MoveOption(1, 'Tailwind', 'allies', 'Status')) == 'speed'
    assert J._utility_kind(MoveOption(1, 'Reflect', 'allySide', 'Status')) == 'status'
    if J._DAMAGE_ISO and J._SYNERGY_LITE:
        # Sin pokes: media Spe propia=0 < foe → setup fuerte (o SPEED_PARTNER si A9 off).
        expected = (
            J._FIELD_SETUP_BONUS if J._FIELD_SETUP else J._SPEED_PARTNER_BONUS
        )
        assert J._utility_score(tw, hit, foes=foes, has_damage=True) == expected
        assert J._utility_score(tw, idle, foes=foes, has_damage=False) == 0.0
        # TW ya activo → 0
        if J._FIELD_SETUP:
            assert J._utility_score(
                tw, hit, foes=foes, has_damage=True, our_tailwind=True,
            ) == 0.0


def test_field_setup_trick_room_when_slower():
    from showdown.battle_bots.greedy_joint import joint as J

    slow = CalcPoke(
        species='Rillaboom',
        stats={'hp': 100, 'atk': 1, 'def': 1, 'spa': 1, 'spd': 1, 'spe': 40},
        cur_hp=100, types=('Grass', 'Dark'),
    )
    fast_foe = CalcPoke(
        species='Pelipper',
        stats={'hp': 100, 'atk': 1, 'def': 1, 'spa': 1, 'spd': 1, 'spe': 200},
        cur_hp=100, types=('Water', 'Flying'),
    )
    tr = J.JointAtom(move_slot=1, target=0, utility='speed', move_name='Trick Room')
    hit = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 40.0}, max_roll={1: 30}, ko_tids=set(),
    )
    if not (J._DAMAGE_ISO and J._FIELD_SETUP):
        return
    foes = [(1, fast_foe)]
    assert J._utility_score(
        tr, hit, foes=foes, has_damage=True,
        left_poke=slow, right_poke=slow,
    ) == J._FIELD_SETUP_BONUS
    assert J._utility_score(
        tr, hit, foes=foes, has_damage=True,
        left_poke=slow, right_poke=slow, trick_room=True,
    ) == 0.0


def test_spread_split_bonus_when_focus_cannot_ko():
    from showdown.battle_bots.greedy_joint import joint as J

    foes = [
        (1, CalcPoke(
            species='Pelipper',
            stats={'hp': 100, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
            cur_hp=100, types=('Water', 'Flying'),
        )),
        (2, CalcPoke(
            species='Toxapex',
            stats={'hp': 100, 'atk': 74, 'def': 181, 'spa': 73, 'spd': 187, 'spe': 55},
            cur_hp=100, types=('Poison', 'Water'),
        )),
    ]
    spread = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Rock Slide',
        damage_pct={1: 45.0, 2: 40.0}, max_roll={1: 40, 2: 35}, ko_tids=set(),
    )
    partner = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 40.0}, max_roll={1: 35}, ko_tids=set(),
    )
    focus_a = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 45.0}, max_roll={1: 40}, ko_tids=set(),
    )
    focus_b = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Double-Edge',
        damage_pct={1: 40.0}, max_roll={1: 35}, ko_tids=set(),
    )
    if not (J._DAMAGE_ISO and J._SPREAD_SYNERGY):
        return
    s_spread = J._score_pair(spread, partner, foes)
    s_focus = J._score_pair(focus_a, focus_b, foes)
    assert s_spread > s_focus
    alone = {1: False, 2: False}
    assert J._spread_split_bonus(spread, partner, foes, alone_ko=alone) == J._SPREAD_SPLIT_BONUS
    alone_ko = {1: True, 2: False}
    assert J._spread_split_bonus(spread, partner, foes, alone_ko=alone_ko) == 0.0


def test_prio_win_scale_and_tempo_extreme_speed():
    from showdown.battle_bots.greedy_joint import joint as J

    slow = CalcPoke(
        species='Dragonite',
        stats={'hp': 100, 'atk': 1, 'def': 1, 'spa': 1, 'spd': 1, 'spe': 80},
        cur_hp=100, types=('Dragon', 'Flying'), ability='Multiscale',
    )
    fast_foe = CalcPoke(
        species='Pelipper',
        stats={'hp': 100, 'atk': 1, 'def': 1, 'spa': 1, 'spd': 1, 'spe': 200},
        cur_hp=100, types=('Water', 'Flying'),
    )
    if not J._PRIO_WIN_GATE:
        return
    assert J._prio_win_scale(
        'Extreme Speed', attacker=slow, foe=fast_foe,
    ) == 1.0
    # Fake Out vs Armor Tail / slower prio-loss: use prio0 move name as control
    assert J._prio_win_scale(
        'Tackle', attacker=slow, foe=fast_foe,
    ) == 1.0  # prio≤0 → no gate
    es = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Extreme Speed',
        damage_pct={1: 50.0}, max_roll={1: 40}, ko_tids=set(),
    )
    partner = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 40.0}, max_roll={1: 30}, ko_tids=set(),
    )
    foes = [(1, fast_foe)]
    tempo = J._prio_tempo_bonus(
        es, partner, foes,
        left_poke=slow, right_poke=slow,
        terrain=None, trick_room=False,
        our_tailwind=False, opp_tailwind=False,
    )
    assert tempo == J._PRIO_TEMPO_BONUS


def test_merge_spread_damage_atoms():
    from showdown.battle_bots.greedy_joint import joint as J
    from showdown.battle_bots.greedy_solo.greedy import MoveOption

    moves = [MoveOption(1, 'Rock Slide', 'allAdjacentFoes', 'Physical')]
    a1 = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Rock Slide',
        damage_pct={1: 40.0}, max_roll={1: 30}, ko_tids=set(),
    )
    a2 = J.JointAtom(
        move_slot=1, target=2, utility='damage', move_name='Rock Slide',
        damage_pct={2: 35.0}, max_roll={2: 25}, ko_tids=set(),
    )
    redirect = J.JointAtom(
        move_slot=2, target=0, utility='redirect', move_name='Follow Me',
    )
    merged = J._merge_spread_damage_atoms([a1, a2, redirect], moves)
    assert len(merged) == 2
    spread = next(a for a in merged if a.utility == 'damage')
    assert set(spread.damage_pct) == {1, 2}
    assert spread.max_roll[1] == 30 and spread.max_roll[2] == 25


def test_mech_hh_scales_partner_damage():
    """A4: Helping Hand ×1.5 al daño del partner."""
    from showdown.battle_bots.greedy_joint import joint as J

    foes = [
        (1, CalcPoke(
            species='Pelipper',
            stats={'hp': 100, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
            cur_hp=100, types=('Water', 'Flying'),
        )),
    ]
    hh = J.JointAtom(move_slot=1, target=0, utility='helpinghand', move_name='Helping Hand')
    hit = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 40.0}, max_roll={1: 30}, ko_tids=set(),
    )
    idle = J.JointAtom(move_slot=2, target=0, utility='status', move_name='Growl')
    s_hh = J._score_pair(hh, hit, foes)
    s_idle = J._score_pair(idle, hit, foes)
    if J._DAMAGE_ISO and J._MECH_SYNERGY:
        assert s_hh > s_idle


def test_protect_threat_gate_without_low_hp():
    """A4: Protect por amenaza alta aunque HP% > 40."""
    from showdown.battle_bots.greedy_joint import joint as J

    foes = [
        (1, CalcPoke(
            species='Pelipper',
            stats={'hp': 100, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
            cur_hp=100, types=('Water', 'Flying'),  # Water SE vs Fire
        )),
    ]
    our = CalcPoke(
        species='Charizard',
        stats={'hp': 200, 'atk': 100, 'def': 100, 'spa': 150, 'spd': 100, 'spe': 100},
        cur_hp=200, types=('Fire', 'Flying'),
    )
    partner = CalcPoke(
        species='Rillaboom',
        stats={'hp': 200, 'atk': 173, 'def': 115, 'spa': 72, 'spd': 100, 'spe': 50},
        cur_hp=200, types=('Grass', 'Dark'),
    )
    protect = J.JointAtom(move_slot=1, target=0, utility='protect', move_name='Protect')
    hit = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 50.0}, max_roll={1: 40}, ko_tids=set(),
    )
    threat = J._side_incoming_threat(our, foes)
    u = J._utility_score(
        protect, hit, foes=foes, has_damage=True,
        left_poke=our, right_poke=partner,
        threat_left=threat, threat_right=0.0,
    )
    if J._DAMAGE_ISO and J._OPP_THREAT:
        assert threat >= J._PROTECT_THREAT_PCT
        # Pelipper Spe 101 > Charizard 100 → A7 permite el bono.
        assert u == J._PROTECT_THREAT_BONUS


def test_protect_threat_requires_foe_outrace_when_spe_gate():
    """A7: amenaza alta pero nosotros más rápidos → sin Protect threat bonus."""
    from showdown.battle_bots.greedy_joint import joint as J

    foes = [
        (1, CalcPoke(
            species='Pelipper',
            stats={'hp': 100, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 50},
            cur_hp=100, types=('Water', 'Flying'),
        )),
    ]
    our = CalcPoke(
        species='Charizard',
        stats={'hp': 200, 'atk': 100, 'def': 100, 'spa': 150, 'spd': 100, 'spe': 200},
        cur_hp=200, types=('Fire', 'Flying'),
    )
    partner = CalcPoke(
        species='Rillaboom',
        stats={'hp': 200, 'atk': 173, 'def': 115, 'spa': 72, 'spd': 100, 'spe': 50},
        cur_hp=200, types=('Grass', 'Dark'),
    )
    protect = J.JointAtom(move_slot=1, target=0, utility='protect', move_name='Protect')
    hit = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 50.0}, max_roll={1: 40}, ko_tids=set(),
    )
    threat = J._side_incoming_threat(our, foes)
    u = J._utility_score(
        protect, hit, foes=foes, has_damage=True,
        left_poke=our, right_poke=partner,
        threat_left=threat, threat_right=0.0,
    )
    if J._DAMAGE_ISO and J._OPP_THREAT and J._SPE_RACE_GATE:
        assert threat >= J._PROTECT_THREAT_PCT
        assert not J._foe_acts_before_us(our, foes)
        assert u == 0.0


def test_fo_partner_pressure_prefers_ko_over_idle():
    """A8: Fake Out + partner KO al mismo foe supera FO + idle."""
    from showdown.battle_bots.greedy_joint import joint as J

    foes = [
        (1, CalcPoke(
            species='Pelipper',
            stats={'hp': 100, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 65},
            cur_hp=40, types=('Water', 'Flying'),
        )),
    ]
    fo_user = CalcPoke(
        species='Incineroar',
        stats={'hp': 200, 'atk': 150, 'def': 120, 'spa': 80, 'spd': 100, 'spe': 80},
        cur_hp=200, types=('Fire', 'Dark'),
    )
    partner = CalcPoke(
        species='Rillaboom',
        stats={'hp': 200, 'atk': 173, 'def': 115, 'spa': 72, 'spd': 100, 'spe': 50},
        cur_hp=200, types=('Grass', 'Dark'),
    )
    fo = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Fake Out',
        damage_pct={1: 20.0}, max_roll={1: 12}, ko_tids=set(),
    )
    kill = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 80.0}, max_roll={1: 50}, ko_tids={1},
    )
    idle = J.JointAtom(move_slot=2, target=0, utility='status', move_name='Growl')
    s_ko = J._score_pair(fo, kill, foes, left_poke=fo_user, right_poke=partner)
    s_idle = J._score_pair(fo, idle, foes, left_poke=fo_user, right_poke=partner)
    if J._DAMAGE_ISO and J._FO_PARTNER_PRESSURE:
        bonus = J._fo_partner_pressure_bonus(
            fo, kill, foes, left_poke=fo_user, right_poke=partner,
        )
        assert bonus >= J._FO_PARTNER_KO_BONUS - 1e-6
        assert J._fo_partner_pressure_bonus(
            fo, idle, foes, left_poke=fo_user, right_poke=partner,
        ) == 0.0
        assert s_ko > s_idle


def test_residual_second_prefers_ko_plus_chip_over_waste():
    """A5: KO foe1 + chip foe2 supera KO foe1 + overkill en foe1 (bonus residual)."""
    from showdown.battle_bots.greedy_joint import joint as J

    foes = [
        (1, CalcPoke(
            species='Pelipper',
            stats={'hp': 100, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
            cur_hp=40, types=('Water', 'Flying'),
        )),
        (2, CalcPoke(
            species='Toxapex',
            stats={'hp': 100, 'atk': 74, 'def': 181, 'spa': 73, 'spd': 187, 'spe': 55},
            cur_hp=100, types=('Poison', 'Water'),
        )),
    ]
    kill = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 80.0}, max_roll={1: 50}, ko_tids={1},
    )
    waste = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='U-turn',
        damage_pct={1: 40.0}, max_roll={1: 25}, ko_tids=set(),
    )
    chip_b = J.JointAtom(
        move_slot=1, target=2, utility='damage', move_name='U-turn',
        damage_pct={2: 40.0}, max_roll={2: 25}, ko_tids=set(),
    )
    s_waste = J._score_pair(kill, waste, foes)
    s_split = J._score_pair(kill, chip_b, foes)
    if J._DAMAGE_ISO and J._RESIDUAL_SECOND:
        assert s_split > s_waste
        koed_split = J._pair_koed_tids(kill, chip_b, foes)
        assert 1 in koed_split and 2 not in koed_split
        bonus = J._residual_second_bonus(kill, chip_b, foes, koed=koed_split)
        assert bonus > 0.0
        assert J._residual_second_bonus(
            kill, waste, foes, koed=J._pair_koed_tids(kill, waste, foes),
        ) == 0.0
        # El gap supera el chip crudo: incluye escala residual.
        assert (s_split - s_waste) > float(chip_b.damage_pct[2]) - 1e-6


def test_focus_waste_penalizes_double_hit_on_alone_ko():
    """A6: KO solo + 2º hit mismo foe pierde frente a KO + chip al otro."""
    from showdown.battle_bots.greedy_joint import joint as J

    foes = [
        (1, CalcPoke(
            species='Pelipper',
            stats={'hp': 100, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
            cur_hp=40, types=('Water', 'Flying'),
        )),
        (2, CalcPoke(
            species='Toxapex',
            stats={'hp': 100, 'atk': 74, 'def': 181, 'spa': 73, 'spd': 187, 'spe': 55},
            cur_hp=100, types=('Poison', 'Water'),
        )),
    ]
    kill = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 80.0}, max_roll={1: 50}, ko_tids={1},
    )
    waste = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='U-turn',
        damage_pct={1: 40.0}, max_roll={1: 25}, ko_tids=set(),
    )
    chip_b = J.JointAtom(
        move_slot=1, target=2, utility='damage', move_name='U-turn',
        damage_pct={2: 40.0}, max_roll={2: 25}, ko_tids=set(),
    )
    s_waste = J._score_pair(kill, waste, foes)
    s_split = J._score_pair(kill, chip_b, foes)
    if J._DAMAGE_ISO and J._FOCUS_WASTE:
        assert J._focus_waste_penalty(
            kill, waste, foes,
            alone_ko={1: True, 2: False},
        ) == J._FOCUS_WASTE_PENALTY
        assert J._focus_waste_penalty(
            kill, chip_b, foes,
            alone_ko={1: True, 2: False},
        ) == 0.0
        assert s_split > s_waste


def test_ordered_atoms_respects_trick_room_and_tailwind():
    """Campo activo: TR invierte Spe; Tailwind nuestro dobla Spe vs foe."""
    from showdown.battle_bots.greedy_joint import joint as J

    fast = CalcPoke(
        species='Salamence',
        stats={'hp': 100, 'atk': 1, 'def': 1, 'spa': 1, 'spd': 1, 'spe': 200},
        cur_hp=100, types=('Dragon', 'Flying'),
    )
    slow = CalcPoke(
        species='Rillaboom',
        stats={'hp': 100, 'atk': 1, 'def': 1, 'spa': 1, 'spd': 1, 'spe': 50},
        cur_hp=100, types=('Grass', 'Dark'),
    )
    a_fast = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Double-Edge',
        damage_pct={1: 40.0}, max_roll={1: 30}, ko_tids=set(),
    )
    a_slow = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 40.0}, max_roll={1: 30}, ko_tids=set(),
    )
    order = J._ordered_atoms(
        a_fast, a_slow, left_poke=fast, right_poke=slow, terrain=None,
    )
    assert order[0] is a_fast
    order_tr = J._ordered_atoms(
        a_fast, a_slow, left_poke=fast, right_poke=slow,
        terrain=None, trick_room=True,
    )
    assert order_tr[0] is a_slow

    mid = CalcPoke(
        species='Incineroar',
        stats={'hp': 100, 'atk': 1, 'def': 1, 'spa': 1, 'spd': 1, 'spe': 100},
        cur_hp=100, types=('Fire', 'Dark'),
    )
    foe = CalcPoke(
        species='Pelipper',
        stats={'hp': 100, 'atk': 1, 'def': 1, 'spa': 1, 'spd': 1, 'spe': 150},
        cur_hp=100, types=('Water', 'Flying'),
    )
    foes = [(1, foe)]
    assert J._foe_acts_before_us(mid, foes)
    assert not J._foe_acts_before_us(mid, foes, our_tailwind=True)


def test_utility_kind_classifies_redirect():
    from showdown.battle_bots.greedy_joint.joint import _utility_kind

    assert _utility_kind(MoveOption(1, 'Follow Me', 'self', 'Status')) == 'redirect'
    assert _utility_kind(MoveOption(1, 'Rage Powder', 'self', 'Status')) == 'redirect'
    assert _utility_kind(MoveOption(1, 'Protect', 'self', 'Status')) == 'protect'


def test_joint_atoms_exclude_switches():
    """Decision space is moves only; switches are never JointAtoms."""
    from showdown.battle_bots.greedy_joint.joint import JointAtom, _atoms_from_side

    moves = [
        MoveOption(1, 'Follow Me', 'self', 'Status'),
        MoveOption(2, 'Wood Hammer', 'normal', 'Physical'),
    ]
    atoms = _atoms_from_side(moves, meta=[], ranges=[], foes=[], can_mega=False)
    assert all(isinstance(a, JointAtom) for a in atoms)
    assert {a.utility for a in atoms} == {'redirect'}
    assert all(a.move_name != 'switch' for a in atoms)


def test_robust_player_factories():
    from poke_env.player import RandomPlayer

    from showdown.battle_bots.greedy_joint.pokeenv_player import GreedyJointPlayer
    from showdown.battle_bots.greedy_solo.pokeenv_player import GreedyDamagePlayer
    from showdown.inference_sim.validated_greedy_player import (
        ValidatedGreedyJointPlayer,
        ValidatedGreedyPlayer,
        opponent_player_class,
        validated_player_class,
    )

    assert validated_player_class('greedy_solo') is ValidatedGreedyPlayer
    assert validated_player_class('greedy_joint') is ValidatedGreedyJointPlayer
    assert opponent_player_class('random') is RandomPlayer
    assert opponent_player_class('greedy_solo') is GreedyDamagePlayer
    assert opponent_player_class('greedy_joint') is GreedyJointPlayer
    assert issubclass(ValidatedGreedyJointPlayer, GreedyJointPlayer)
    assert issubclass(ValidatedGreedyPlayer, GreedyDamagePlayer)
