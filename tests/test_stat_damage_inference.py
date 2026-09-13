"""Inferencia Atk/Def/SpA/SpD via dano observado (bridge mockeado)."""
import sys
sys.path.insert(0, '.')

from showdown import stat_inference as si
from showdown.battle import Battle, Pokemon
from showdown.battle_modifier import (
    heal_or_damage,
    note_crit,
    observe_move,
    turn,
)
from showdown.damage_calc_bridge import set_damage_range_override


class TestBattle(Battle):
    async def action_team_preview(self, battle):
        pass

    async def action_battle(self, battle):
        pass


def build_pkmn(name, is_opponent=True, hp=100):
    p = Pokemon(name, hp, is_opponent, '', '', None, None, None, None, None)
    p.HPA_battle_actual = hp
    return p


def _setup_cross_hit(*, user_speed_stats=True):
    si.reset_sp_limits()
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    mine = build_pkmn('whimsicott', is_opponent=False, hp=155)
    mine.nature = 'timid'
    mine.ATK_battle_actual = 78
    mine.DEF_battle_actual = 111
    mine.SPA_battle_actual = 105
    mine.SPD_battle_actual = 95
    mine.VEL_battle_actual = 184
    mine.HPB_battle_base = 155
    mine.objectPkmn = None
    mine.abilityPkmn = 'prankster'

    theirs = build_pkmn('dragonite', is_opponent=True, hp=100)
    theirs.nature = 'adamant'
    theirs.objectPkmn = None
    theirs.abilityPkmn = 'innerfocus'
    # bases de especie ya vienen del constructor / pokedex

    battle.user.active1 = mine
    battle.user.active2 = build_pkmn('torkoal', is_opponent=False, hp=177)
    battle.opponent.active1 = theirs
    battle.opponent.active2 = build_pkmn('blastoise', is_opponent=True, hp=100)
    return battle, mine, theirs


def test_ohko_keeps_high_atk_candidates():
    """
    KO / overkill: el daño reportado es solo el HP restante. Atk altos cuya
    ventana de rolls queda por encima de ese tope deben seguir siendo válidos.
    """
    battle, mine, theirs = _setup_cross_hit()
    belief = battle.stat_beliefs.belief_for(theirs)

    def fake_range(req):
        atk = req['attacker']['stats']['atk']
        if atk >= 185:
            return (200, 240)  # overkill vs Whimsicott 155 HP
        return (40, 50)

    set_damage_range_override(fake_range)
    try:
        observe_move(
            battle,
            ['', 'move', 'p2a: Dragonite', 'Extreme Speed', 'p1a: Whimsicott'],
        )
        mine.HPA_battle_actual = 155
        heal_or_damage(battle, ['', '-damage', 'p1a: Whimsicott', '0 fnt'])

        assert belief.observations >= 1
        assert max(belief.candidates['atk']) == 32
        # Los Atk bajos que no llegan a matar sí se pueden excluir
        assert min(belief.candidates['atk']) > 0 or 32 in belief.candidates['atk']
    finally:
        set_damage_range_override(None)
        si.reset_sp_limits()


def test_opponent_physical_hit_narrows_atk_with_mock_bridge():
    battle, mine, theirs = _setup_cross_hit()
    belief = battle.stat_beliefs.belief_for(theirs)
    before_spe = set(belief.candidates['spe'])

    # Solo SP altos de Atk "pegan" 40 de dano (SP0 adamant Dragonite ya da Atk~169)
    def fake_range(req):
        atk = req['attacker']['stats']['atk']
        if atk >= 185:
            return (38, 42)
        return (8, 12)

    set_damage_range_override(fake_range)
    try:
        observe_move(
            battle,
            ['', 'move', 'p2a: Dragonite', 'Extreme Speed', 'p1a: Whimsicott'],
        )
        # HP nuestro baja 40
        mine.HPA_battle_actual = 155
        heal_or_damage(battle, ['', '-damage', 'p1a: Whimsicott', '115/155'])

        assert belief.observations >= 1
        assert max(belief.candidates['atk']) >= min(belief.candidates['atk'])
        # Debe haberse estrechado respecto al set completo 0..32
        assert len(belief.candidates['atk']) < 33
        # Spe no se toca por esta evidencia (salvo budget extremo)
        assert belief.candidates['spe'] == before_spe or max(belief.candidates['spe']) <= 32
    finally:
        set_damage_range_override(None)
        si.reset_sp_limits()


def test_our_special_hit_on_opponent_percent_narrows_spd():
    battle, mine, theirs = _setup_cross_hit()
    belief = battle.stat_beliefs.belief_for(theirs)

    def fake_range(req):
        spd = req['defender']['stats']['spd']
        hp = req['defender']['stats']['hp']
        # ~22% solo con SpD baja; SpD alta pega mucho menos
        if spd <= 125:
            return (max(1, hp * 20 // 100), max(2, hp * 25 // 100))
        return (max(1, hp * 2 // 100), max(2, hp * 5 // 100))

    set_damage_range_override(fake_range)
    try:
        observe_move(
            battle,
            ['', 'move', 'p1a: Whimsicott', 'Moonblast', 'p2a: Dragonite'],
        )
        theirs.HPA_battle_actual = 100
        theirs.HPB_battle_base = 100
        heal_or_damage(battle, ['', '-damage', 'p2a: Dragonite', '78/100'])

        assert belief.observations >= 1
        assert len(belief.candidates['spd']) < 33
    finally:
        set_damage_range_override(None)
        si.reset_sp_limits()


def test_crit_discards_damage_observation():
    battle, mine, theirs = _setup_cross_hit()
    belief = battle.stat_beliefs.belief_for(theirs)
    before = set(belief.candidates['atk'])

    calls = {'n': 0}

    def fake_range(req):
        calls['n'] += 1
        return (40, 50)

    set_damage_range_override(fake_range)
    try:
        observe_move(
            battle,
            ['', 'move', 'p2a: Dragonite', 'Extreme Speed', 'p1a: Whimsicott'],
        )
        note_crit(battle, ['', '-crit', 'p1a: Whimsicott'])
        mine.HPA_battle_actual = 155
        heal_or_damage(battle, ['', '-damage', 'p1a: Whimsicott', '100/155'])

        assert calls['n'] == 0
        assert belief.candidates['atk'] == before
        assert battle.pending_damage_ctx is None
    finally:
        set_damage_range_override(None)
        si.reset_sp_limits()


def test_residual_damage_does_not_narrow():
    battle, mine, theirs = _setup_cross_hit()
    belief = battle.stat_beliefs.belief_for(theirs)
    before = set(belief.candidates['atk'])

    def fake_range(req):
        return (40, 50)

    set_damage_range_override(fake_range)
    try:
        observe_move(
            battle,
            ['', 'move', 'p2a: Dragonite', 'Extreme Speed', 'p1a: Whimsicott'],
        )
        mine.HPA_battle_actual = 155
        heal_or_damage(
            battle,
            ['', '-damage', 'p1a: Whimsicott', '140/155 brn', '[from] brn'],
        )
        assert belief.candidates['atk'] == before
    finally:
        set_damage_range_override(None)
        si.reset_sp_limits()


def test_status_move_does_not_set_pending_damage():
    battle, _, _ = _setup_cross_hit()
    observe_move(
        battle,
        ['', 'move', 'p2a: Dragonite', 'Dragon Dance', 'p2a: Dragonite'],
    )
    assert battle.pending_damage_ctx is None
    si.reset_sp_limits()


def test_turn_clears_pending_damage():
    battle, _, _ = _setup_cross_hit()
    observe_move(
        battle,
        ['', 'move', 'p2a: Dragonite', 'Extreme Speed', 'p1a: Whimsicott'],
    )
    assert battle.pending_damage_ctx is not None
    turn(battle, ['', 'turn', '2'])
    assert battle.pending_damage_ctx is None
    assert battle.pending_crit_target is None
    si.reset_sp_limits()


def test_observe_damage_batches_unique_stat_values():
    """Varios SP con el mismo stat final => un solo calc (batch/coalesce)."""
    si.reset_sp_limits()
    beliefs = si.TeamStatBeliefs()
    opp = build_pkmn('dragonite', is_opponent=True, hp=100)
    opp.nature = 'adamant'
    beliefs.belief_for(opp)

    calls = {'n': 0}

    def fake_batch(reqs):
        if isinstance(reqs, list):
            calls['n'] += len(reqs)
            return [(10, 12) for _ in reqs]
        calls['n'] += 1
        return (10, 12)

    set_damage_range_override(fake_batch)
    try:
        ctx = si.DamageContext()
        beliefs.observe_damage(
            opponent_pkmn=opp,
            move_name='Extreme Speed',
            category='p',
            opponent_is_attacker=True,
            observed_damage=11,
            damage_is_percent=False,
            known_stats={'hp': 155, 'atk': 78, 'def': 111, 'spa': 105, 'spd': 95, 'spe': 184},
            known_species='whimsicott',
            known_nature='timid',
            ctx=ctx,
        )
        assert calls['n'] < 33
        assert calls['n'] >= 1
    finally:
        set_damage_range_override(None)
        si.reset_sp_limits()


def test_observe_damage_skips_when_intersection_empty():
    si.reset_sp_limits()
    beliefs = si.TeamStatBeliefs()
    opp = build_pkmn('dragonite', is_opponent=True, hp=100)
    opp.nature = 'adamant'
    belief = beliefs.belief_for(opp)
    belief.narrow('atk', {0, 1, 2})

    def fake_range(req):
        return (1, 2)

    set_damage_range_override(fake_range)
    try:
        ctx = si.DamageContext()
        changed = beliefs.observe_damage(
            opponent_pkmn=opp,
            move_name='Extreme Speed',
            category='p',
            opponent_is_attacker=True,
            observed_damage=999,
            damage_is_percent=False,
            known_stats={'hp': 155, 'atk': 78, 'def': 111, 'spa': 105, 'spd': 95, 'spe': 184},
            known_species='whimsicott',
            known_nature='timid',
            ctx=ctx,
        )
        assert changed is False
        assert belief.candidates['atk'] == {0, 1, 2}
        assert belief.skipped >= 1
    finally:
        set_damage_range_override(None)
        si.reset_sp_limits()


def test_bilateral_upper_bound_excludes_high_atk_when_clean():
    """Daño flojo + contexto limpio => se descarta Atk alto (min_roll > observado)."""
    si.reset_sp_limits()
    si.configure_sp_limits(63, 129)
    beliefs = si.TeamStatBeliefs()
    opp = build_pkmn('dragonite', is_opponent=True, hp=100)
    opp.nature = 'adamant'
    belief = beliefs.belief_for(opp)
    belief.candidates['atk'] = set(range(0, 64))

    def fake_batch(reqs):
        out = []
        for req in reqs:
            atk = int(req['attacker']['stats']['atk'])
            # Simula: Atk alto nunca baja de 80; Atk bajo hace ~20-30.
            if atk >= 200:
                out.append((80, 100))
            else:
                out.append((20, 30))
        return out

    set_damage_range_override(fake_batch)
    try:
        ctx = si.DamageContext()  # limpio
        assert ctx.is_clean_for_upper_bound is True
        beliefs.observe_damage(
            opponent_pkmn=opp,
            move_name='Extreme Speed',
            category='p',
            opponent_is_attacker=True,
            observed_damage=25,
            damage_is_percent=False,
            known_stats={'hp': 155, 'atk': 78, 'def': 111, 'spa': 105, 'spd': 95, 'spe': 184},
            known_species='whimsicott',
            known_nature='timid',
            ctx=ctx,
            is_ko=False,
        )
        # Los SP que producen Atk final >= 200 deben salir.
        for sp in list(belief.candidates['atk']):
            final = si.champions_stat(belief.base_stats['atk'], sp, 'adamant', 'atk')
            assert final < 200, f'SP={sp} final={final} debería haberse excluido'
    finally:
        set_damage_range_override(None)
        si.reset_sp_limits()


def test_bilateral_upper_bound_skipped_when_not_clean():
    """Con Helping Hand / sash no se aplica cota superior."""
    si.reset_sp_limits()
    beliefs = si.TeamStatBeliefs()
    opp = build_pkmn('dragonite', is_opponent=True, hp=100)
    opp.nature = 'adamant'
    belief = beliefs.belief_for(opp)
    before = set(belief.candidates['atk'])

    def fake_batch(reqs):
        return [(80, 100) for _ in reqs]

    set_damage_range_override(fake_batch)
    try:
        ctx = si.DamageContext(helping_hand=True)
        assert ctx.is_clean_for_upper_bound is False
        beliefs.observe_damage(
            opponent_pkmn=opp,
            move_name='Extreme Speed',
            category='p',
            opponent_is_attacker=True,
            observed_damage=25,
            damage_is_percent=False,
            known_stats={'hp': 155, 'atk': 78, 'def': 111, 'spa': 105, 'spd': 95, 'spe': 184},
            known_species='whimsicott',
            known_nature='timid',
            ctx=ctx,
        )
        # Sin cota superior: Atk altos siguen vivos (max_roll 100 >= 25).
        assert max(belief.candidates['atk']) == max(before)
    finally:
        set_damage_range_override(None)
        si.reset_sp_limits()


def test_field_payload_aurora_veil_not_emulated_as_screens():
    ctx = si.DamageContext(aurora_veil=True, reflect=True, light_screen=True)
    field = si._field_payload(ctx, 'p')
    assert field['defenderSide'].get('isAuroraVeil') is True
    assert 'isReflect' not in field['defenderSide']
    field_s = si._field_payload(ctx, 's')
    assert field_s['defenderSide'].get('isAuroraVeil') is True
    assert 'isLightScreen' not in field_s['defenderSide']


def test_field_payload_helping_hand_friend_guard_charge():
    ctx = si.DamageContext(
        helping_hand=True,
        friend_guard=True,
        charge=True,
        reflect=True,
    )
    field = si._field_payload(ctx, 'p')
    assert field['attackerSide'].get('isHelpingHand') is True
    assert field['attackerSide'].get('isCharge') is True
    assert field['defenderSide'].get('isFriendGuard') is True
    assert field['defenderSide'].get('isReflect') is True


def test_multiscale_ability_not_uncertain_anymore():
    ctx = si.DamageContext(defender_ability='Multiscale')
    assert ctx.is_uncertain is False
    ctx2 = si.DamageContext(attacker_ability='parentalbond')
    assert ctx2.is_uncertain is True


def test_protect_skips_damage_inference_not_pending():
    """Protect en el defensor: el pending del move se mantiene; el hit no estrecha."""
    battle, mine, theirs = _setup_cross_hit()
    belief = battle.stat_beliefs.belief_for(theirs)
    before = set(belief.candidates['spd'])
    theirs.is_protected = True

    def fake_range(req):
        return (40, 50)

    set_damage_range_override(fake_range)
    try:
        observe_move(
            battle,
            ['', 'move', 'p1a: Whimsicott', 'Moonblast', 'p2a: Dragonite'],
        )
        assert battle.pending_damage_ctx is not None
        theirs.HPA_battle_actual = 100
        heal_or_damage(battle, ['', '-damage', 'p2a: Dragonite', '60/100'])
        assert belief.candidates['spd'] == before
    finally:
        set_damage_range_override(None)
        si.reset_sp_limits()


def test_helping_hand_and_aurora_in_hit_ctx():
    battle, mine, theirs = _setup_cross_hit()
    mine.is_helping_hand = True
    battle.opponent.side_conditions['auroraveil'] = 5
    ally = battle.opponent.active2
    ally.abilityPkmn = 'friendguard'

    seen = {}

    def fake_range(req):
        field = req.get('field') or {}
        seen['hh'] = field.get('attackerSide', {}).get('isHelpingHand')
        seen['av'] = field.get('defenderSide', {}).get('isAuroraVeil')
        seen['fg'] = field.get('defenderSide', {}).get('isFriendGuard')
        seen['reflect'] = field.get('defenderSide', {}).get('isReflect')
        seen['spread'] = field.get('isSpread')
        return (20, 25)

    set_damage_range_override(fake_range)
    try:
        observe_move(
            battle,
            ['', 'move', 'p1a: Whimsicott', 'Moonblast', 'p2a: Dragonite'],
        )
        pending = battle.pending_damage_ctx
        assert pending is not None
        assert pending.get('is_spread') is False
        theirs.HPA_battle_actual = 100
        heal_or_damage(battle, ['', '-damage', 'p2a: Dragonite', '78/100'])
        assert seen.get('hh') is True
        assert seen.get('av') is True
        assert seen.get('fg') is True
        assert seen.get('reflect') is None  # no emular R/LS junto a Aurora
        assert seen.get('spread') is False
    finally:
        set_damage_range_override(None)
        si.reset_sp_limits()


def test_spread_our_special_narrows_each_foe_spd():
    """Sludge Wave nuestro: cada |-damage| % estrecha SpD de ese rival."""
    battle, mine, dragonite = _setup_cross_hit()
    primarina = build_pkmn('primarina', is_opponent=True, hp=100)
    primarina.nature = 'modest'
    primarina.objectPkmn = None
    primarina.abilityPkmn = 'torrent'
    battle.opponent.active2 = primarina

    bel_dn = battle.stat_beliefs.belief_for(dragonite)
    bel_pr = battle.stat_beliefs.belief_for(primarina)
    before_dn = set(bel_dn.candidates['spd'])
    before_pr = set(bel_pr.candidates['spd'])

    def fake_range(req):
        field = req.get('field') or {}
        assert field.get('isSpread') is True
        species = (req['defender'].get('species') or '').lower()
        spd = req['defender']['stats']['spd']
        hp = req['defender']['stats']['hp']
        # Mitad baja del rango Champions de cada especie "encaja" ~25%.
        cut = 128 if 'dragonite' in species else 144
        if spd <= cut:
            return (max(1, hp * 20 // 100), max(2, hp * 28 // 100))
        return (max(1, hp * 2 // 100), max(2, hp * 6 // 100))

    set_damage_range_override(fake_range)
    try:
        observe_move(
            battle,
            ['', 'move', 'p1a: Whimsicott', 'Sludge Wave', 'p2a: Dragonite',
             '[spread] p2a,p2b'],
        )
        assert battle.pending_damage_ctx is not None
        assert battle.pending_damage_ctx['is_spread'] is True

        dragonite.HPA_battle_actual = 100
        heal_or_damage(battle, ['', '-damage', 'p2a: Dragonite', '78/100'])
        assert battle.pending_damage_ctx is not None  # spread sigue vivo
        assert bel_dn.observations >= 1
        assert len(bel_dn.candidates['spd']) < len(before_dn)

        primarina.HPA_battle_actual = 100
        heal_or_damage(battle, ['', '-damage', 'p2b: Primarina', '75/100'])
        assert bel_pr.observations >= 1
        assert len(bel_pr.candidates['spd']) < len(before_pr)
    finally:
        set_damage_range_override(None)
        si.reset_sp_limits()


def test_spread_opp_physical_narrows_atk_from_both_hits():
    """Earthquake rival: dos hits en nosotros aportan Atk del atacante."""
    battle, mine, theirs = _setup_cross_hit()
    ally = battle.user.active2
    ally.nature = 'bold'
    ally.ATK_battle_actual = 85
    ally.DEF_battle_actual = 160
    ally.SPA_battle_actual = 105
    ally.SPD_battle_actual = 110
    ally.VEL_battle_actual = 40
    ally.HPB_battle_base = 177
    ally.objectPkmn = None
    ally.abilityPkmn = 'drought'

    belief = battle.stat_beliefs.belief_for(theirs)
    hits = {'n': 0}

    def fake_range(req):
        hits['n'] += 1
        assert (req.get('field') or {}).get('isSpread') is True
        atk = req['attacker']['stats']['atk']
        if atk >= 185:
            return (38, 42)
        return (8, 12)

    set_damage_range_override(fake_range)
    try:
        observe_move(
            battle,
            ['', 'move', 'p2a: Dragonite', 'Earthquake', 'p1a: Whimsicott',
             '[spread] p1a,p1b'],
        )
        mine.HPA_battle_actual = 155
        heal_or_damage(battle, ['', '-damage', 'p1a: Whimsicott', '115/155'])
        assert belief.observations >= 1
        mid = set(belief.candidates['atk'])

        ally.HPA_battle_actual = 177
        heal_or_damage(battle, ['', '-damage', 'p1b: Torkoal', '137/177'])
        assert hits['n'] >= 2
        assert belief.observations >= 2
        # Segundo hit puede estrechar más o igual
        assert belief.candidates['atk'].issubset(mid) or belief.candidates['atk'] == mid
    finally:
        set_damage_range_override(None)
        si.reset_sp_limits()


def test_spread_friendly_fire_skipped():
    """Hit al aliado del atacante no aporta (solo cruzados)."""
    battle, mine, theirs = _setup_cross_hit()
    ally_opp = battle.opponent.active2
    belief = battle.stat_beliefs.belief_for(theirs)
    before = set(belief.candidates['spa'])
    calls = {'n': 0}

    def fake_range(req):
        calls['n'] += 1
        return (40, 50)

    set_damage_range_override(fake_range)
    try:
        observe_move(
            battle,
            ['', 'move', 'p2a: Dragonite', 'Sludge Wave', 'p1a: Whimsicott',
             '[spread] p1a,p1b,p2b'],
        )
        ally_opp.HPA_battle_actual = 100
        heal_or_damage(battle, ['', '-damage', 'p2b: Blastoise', '70/100'])
        assert calls['n'] == 0
        assert belief.candidates['spa'] == before
        assert battle.pending_damage_ctx is not None

        mine.HPA_battle_actual = 155
        heal_or_damage(battle, ['', '-damage', 'p1a: Whimsicott', '115/155'])
        assert calls['n'] >= 1
        assert belief.observations >= 1
    finally:
        set_damage_range_override(None)
        si.reset_sp_limits()


def test_spread_crit_only_discards_that_target():
    """Crit en un objetivo del spread; el otro hit sí estrecha."""
    battle, mine, theirs = _setup_cross_hit()
    ally = battle.user.active2
    ally.nature = 'bold'
    ally.ATK_battle_actual = 85
    ally.DEF_battle_actual = 160
    ally.SPA_battle_actual = 105
    ally.SPD_battle_actual = 110
    ally.VEL_battle_actual = 40
    ally.HPB_battle_base = 177

    belief = battle.stat_beliefs.belief_for(theirs)
    before = set(belief.candidates['atk'])
    calls = {'n': 0}

    def fake_range(req):
        calls['n'] += 1
        return (38, 42)

    set_damage_range_override(fake_range)
    try:
        observe_move(
            battle,
            ['', 'move', 'p2a: Dragonite', 'Earthquake', 'p1a: Whimsicott',
             '[spread] p1a,p1b'],
        )
        note_crit(battle, ['', '-crit', 'p1a: Whimsicott'])
        # Daño al aliado primero (sin crit): debe inferir
        ally.HPA_battle_actual = 177
        heal_or_damage(battle, ['', '-damage', 'p1b: Torkoal', '137/177'])
        assert calls['n'] >= 1
        assert belief.observations >= 1
        after_ally = set(belief.candidates['atk'])

        # Crit en Whimsicott: no debe llamar al calc de nuevo por ese hit
        before_calls = calls['n']
        mine.HPA_battle_actual = 155
        heal_or_damage(battle, ['', '-damage', 'p1a: Whimsicott', '100/155'])
        assert calls['n'] == before_calls
        assert belief.candidates['atk'] == after_ally
        assert after_ally != before or belief.observations >= 1
    finally:
        set_damage_range_override(None)
        si.reset_sp_limits()


def test_spread_without_tag_via_move_target():
    """Sin [spread] en el log: allAdjacent en moves.json sigue marca is_spread."""
    battle, _, _ = _setup_cross_hit()
    observe_move(
        battle,
        ['', 'move', 'p1a: Whimsicott', 'Sludge Wave', 'p2a: Dragonite'],
    )
    pending = battle.pending_damage_ctx
    assert pending is not None
    assert pending['is_spread'] is True
    si.reset_sp_limits()


def test_equivalent_hp_sp_not_dropped_when_coalescing():
    """Toxapex Calm HP SP 31 y 32 → mismo final 141; no descartar el 32."""
    si.reset_sp_limits()
    assert si.champions_stat(50, 31, 'calm', 'hp') == si.champions_stat(50, 32, 'calm', 'hp')
    beliefs = si.TeamStatBeliefs()
    opp = build_pkmn('toxapex', is_opponent=True, hp=100)
    opp.nature = 'calm'
    belief = beliefs.belief_for(opp)
    assert 31 in belief.candidates['hp'] and 32 in belief.candidates['hp']

    ctx = si.DamageContext(
        terrain='grassyterrain',
        attacker_ability='aerilate',
        attacker_ability_on=True,
        defender_item='leftovers',
        defender_ability='regenerator',
        defender_hp_fraction=1.0,
        attacker_types=['Dragon', 'Flying'],
    )
    beliefs.observe_damage(
        opponent_pkmn=opp,
        move_name='Double-Edge',
        category='p',
        opponent_is_attacker=False,
        observed_damage=60,
        damage_is_percent=True,
        known_stats={'hp': 171, 'atk': 186, 'def': 150, 'spa': 155, 'spd': 99, 'spe': 185},
        known_species='Salamence-Mega',
        known_nature='naive',
        ctx=ctx,
        is_ko=False,
    )
    assert 32 in belief.candidates['hp'], belief.sp_summary_line()
    assert 18 in belief.candidates['def'], belief.sp_summary_line()
    si.reset_sp_limits()


def test_grassy_wood_hammer_percent_keeps_true_toxapex_def():
    """WH ~58% bajo Grassy no debe expulsar Def SP=18 (desync calc/PS)."""
    si.reset_sp_limits()
    beliefs = si.TeamStatBeliefs()
    opp = build_pkmn('toxapex', is_opponent=True, hp=100)
    opp.nature = 'calm'
    belief = beliefs.belief_for(opp)

    ctx = si.DamageContext(
        terrain='grassyterrain',
        attacker_item='sitrusberry',
        attacker_ability='grassysurge',
        defender_item='leftovers',
        defender_ability='regenerator',
        defender_hp_fraction=1.0,
        attacker_types=['Grass', 'Dark'],
    )
    beliefs.observe_damage(
        opponent_pkmn=opp,
        move_name='Wood Hammer',
        category='p',
        opponent_is_attacker=False,
        observed_damage=58,
        damage_is_percent=True,
        known_stats={'hp': 193, 'atk': 187, 'def': 115, 'spa': 72, 'spd': 100, 'spe': 113},
        known_species='Rillaboom',
        known_nature='adamant',
        ctx=ctx,
        is_ko=False,
    )
    assert 18 in belief.candidates['def'], belief.sp_summary_line()
    assert 32 in belief.candidates['hp'], belief.sp_summary_line()
    si.reset_sp_limits()
