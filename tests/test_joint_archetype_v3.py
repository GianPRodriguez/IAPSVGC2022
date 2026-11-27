"""Modo arquetipo nivel 3 de greedy_joint (protocolo §4.17): clima, campo y núcleos clásicos de VGC."""
from __future__ import annotations

from types import SimpleNamespace

from showdown.battle_bots.greedy_joint import archetypes as A


def _team(*mons):
    return [A.mon_info(i + 1, *m) for i, m in enumerate(mons)]


RAIN = _team(('Pelipper', 'Drizzle', 'Damp Rock', ['Hurricane', 'Weather Ball', 'Tailwind', 'Protect']),
             ('Basculegion', 'Swift Swim', 'Choice Scarf', ['Wave Crash', 'Last Respects', 'Aqua Jet', 'Flip Turn']),
             ('Archaludon', 'Stamina', 'Power Herb', ['Electro Shot', 'Draco Meteor', 'Flash Cannon', 'Protect']),
             ('Rillaboom', 'Grassy Surge', 'Miracle Seed', ['Fake Out', 'Grassy Glide', 'Wood Hammer', 'U-turn']),
             ('Incineroar', 'Intimidate', 'Sitrus Berry', ['Fake Out', 'Flare Blitz', 'Parting Shot', 'Knock Off']),
             ('Amoonguss', 'Regenerator', 'Rocky Helmet', ['Spore', 'Rage Powder', 'Pollen Puff', 'Protect']))
DOUBLE = _team(('Charizard', 'Blaze', 'Charizardite Y', ['Heat Wave', 'Solar Beam', 'Overheat', 'Protect']),
               ('Venusaur', 'Chlorophyll', 'Life Orb', ['Sludge Bomb', 'Leaf Storm', 'Sleep Powder', 'Protect']),
               ('Pelipper', 'Drizzle', 'Damp Rock', ['Hurricane', 'Weather Ball', 'Tailwind', 'Protect']),
               ('Archaludon', 'Stamina', 'Power Herb', ['Electro Shot', 'Draco Meteor', 'Flash Cannon', 'Protect']),
               ('Grimmsnarl', 'Prankster', 'Light Clay', ['Reflect', 'Light Screen', 'Parting Shot', 'Spirit Break']),
               ('Garchomp', 'Rough Skin', '', ['Earthquake', 'Dragon Claw', 'Rock Slide', 'Protect']))
TR = _team(('Indeedee-F', 'Psychic Surge', '', ['Follow Me', 'Psychic', 'Helping Hand', 'Protect']),
           ('Hatterene', 'Magic Bounce', '', ['Trick Room', 'Expanding Force', 'Dazzling Gleam', 'Protect']),
           ('Ursaluna', 'Guts', 'Flame Orb', ['Facade', 'Headlong Rush', 'Protect', 'Earthquake']),
           ('Torkoal', 'Drought', 'Charcoal', ['Eruption', 'Heat Wave', 'Protect', 'Solar Beam']),
           ('Dragonite', 'Inner Focus', '', ['Extreme Speed', 'Protect', 'Tailwind', 'Scale Shot']),
           ('Kingambit', 'Defiant', '', ['Kowtow Cleave', 'Sucker Punch', 'Iron Head', 'Protect']))
SAND = _team(('Tyranitar', 'Sand Stream', '', ['Rock Slide', 'Knock Off', 'Protect', 'Low Kick']),
             ('Excadrill', 'Sand Rush', 'Life Orb', ['High Horsepower', 'Iron Head', 'Rock Slide', 'Protect']),
             ('Salamence', 'Intimidate', 'Salamencite', ['Double-Edge', 'Draco Meteor', 'Hurricane', 'Protect']),
             ('Sneasler', 'Unburden', 'Grassy Seed', ['Close Combat', 'Dire Claw', 'Fake Out', 'Protect']),
             ('Rillaboom', 'Grassy Surge', '', ['Fake Out', 'Grassy Glide', 'Wood Hammer', 'U-turn']),
             ('Sinistcha', 'Hospitality', '', ['Rage Powder', 'Matcha Gotcha', 'Trick Room', 'Strength Sap']))
STEEL = _team(('Scizor',), ('Ferrothorn',), ('Metagross',), ('Lucario',), ('Heatran',), ('Kingambit',))
FIRE = _team(('Arcanine',), ('Heatran',), ('Talonflame',), ('Coalossal',), ('Camerupt',), ('Torkoal',))


def test_mega_resolves_weather_ability():
    cz = A.mon_info(1, 'Charizard', 'Blaze', 'Charizardite Y', [])
    assert cz.ability == 'drought'
    ttar = A.mon_info(1, 'Tyranitar', '', 'Tyranitarite', [])
    assert ttar.ability == 'sandstream'


def test_classic_plans_and_leads():
    order, fit = A.preview_order(RAIN, STEEL, (4, 5, 1, 6))
    assert fit.plan.name == 'lluvia' and order[:2] == (1, 2)          # Pelipper + Basculegion abren
    order, fit = A.preview_order(TR, FIRE, (5, 6, 1, 2))
    assert fit.plan.name == 'trick_room' and set(order[:2]) == {1, 2}  # Hatterene (TR) + Indeedee (Follow Me)
    assert {3, 4} <= set(order)                                        # Ursaluna y Torkoal detrás
    names = {f.plan.name for f in A.detect_plans(SAND)}
    assert {'arena', 'campo_hierba'} <= names                          # Excadrill (Sand Rush), Sneasler (semilla)


def test_double_weather_picks_by_matchup():
    assert A.choose_plan(DOUBLE, STEEL).plan.name == 'sol'
    assert A.choose_plan(DOUBLE, FIRE).plan.name == 'lluvia'


def test_no_plan_keeps_base_order():
    plain = _team(('Garchomp', 'Rough Skin', '', ['Earthquake']), ('Kingambit', 'Defiant', '', ['Iron Head']),
                  ('Gholdengo', 'Good as Gold', '', ['Make It Rain']), ('Dragonite', 'Multiscale', '', ['Extreme Speed']))
    order, fit = A.preview_order(plain, STEEL, (2, 1, 4, 3))
    assert fit is None and order == (2, 1, 4, 3)


def _scores(level, partner_move='Hurricane'):
    from showdown.battle_bots.greedy_joint import joint as J
    from showdown.battle_bots.greedy_solo.greedy import CalcPoke
    fit = A.choose_plan(RAIN, STEEL)
    rain = J.JointAtom(move_slot=1, target=0, utility='status', move_name='Rain Dance')
    prot = J.JointAtom(move_slot=2, target=0, utility='protect', move_name='Protect')
    hit = J.JointAtom(move_slot=1, target=1, utility='damage', move_name=partner_move, damage_pct={1: 40.0})
    tok = J._ARCHETYPE_LEVEL_OVERRIDE.set(level)
    ptok = J._ARCHETYPE_PLAN.set((fit, None, None))
    try:
        p = CalcPoke(species='Pelipper')
        kw = dict(foes=[(1, CalcPoke(species='Scizor'))], left_poke=p, right_poke=CalcPoke(species='Basculegion'))
        s_rain = J._utility_score_damage_iso(rain, hit, **kw)
        s_cover = J._utility_score_damage_iso(rain, prot, **kw)
        J._ARCHETYPE_PLAN.set((fit, 'raindance', None))
        s_rain_on = J._utility_score_damage_iso(rain, hit, **kw)
    finally:
        J._ARCHETYPE_PLAN.reset(ptok)
        J._ARCHETYPE_LEVEL_OVERRIDE.reset(tok)
    return s_rain, s_cover, s_rain_on


def test_level3_field_setup_and_cover_bonus():
    from showdown.battle_bots.greedy_joint import joint as J
    s_rain, s_cover, s_rain_on = _scores(3)
    assert s_rain >= J._ARCHETYPE_FIELD_BONUS
    assert s_cover >= J._ARCHETYPE_FIELD_BONUS + J._ARCHETYPE_PROTECT_COVER_BONUS
    assert s_rain_on < J._ARCHETYPE_FIELD_BONUS                      # ya llueve: sin bono
    l2 = _scores(2)
    assert max(l2) < J._ARCHETYPE_FIELD_BONUS                         # el nivel 2 no cambia


def test_forced_switch_prefers_setter_then_abuser():
    from showdown.battle_bots.greedy_joint.pokeenv_player import GreedyJointPlayer
    from showdown.battle_bots.greedy_joint import joint as J
    fit = A.choose_plan(RAIN, STEEL)
    fake = SimpleNamespace(_battle_plan=lambda b: fit)
    pel = SimpleNamespace(species='pelipper', ability='drizzle', item='damprock', moves={'hurricane': 0})
    bas = SimpleNamespace(species='basculegion', ability='swiftswim', item='', moves={'wavecrash': 0})
    inc = SimpleNamespace(species='incineroar', ability='intimidate', item='', moves={'fakeout': 0})
    tok = J._ARCHETYPE_LEVEL_OVERRIDE.set(3)
    try:
        dry = SimpleNamespace(weather={}, fields={})
        assert GreedyJointPlayer._plan_switch_pick(fake, dry, [inc, bas, pel]) is pel
        wet = SimpleNamespace(weather={'RAINDANCE': 1}, fields={})
        assert GreedyJointPlayer._plan_switch_pick(fake, wet, [inc, pel, bas]) is bas
    finally:
        J._ARCHETYPE_LEVEL_OVERRIDE.reset(tok)
    assert GreedyJointPlayer._plan_switch_pick(fake, dry, [inc, bas, pel]) is None   # nivel 0: sin preferencia


def test_wrap_level_covers_preview_and_switch():
    from showdown.battle_bots.greedy_joint import joint as J
    p = SimpleNamespace(choose_move=lambda b: J._archetype_level(), teampreview=lambda b: J._archetype_level(),
                        _forced_switch_order=lambda b: J._archetype_level())
    J.wrap_archetype_level(p, 3)
    assert p.choose_move(None) == 3 and p.teampreview(None) == 3 and p._forced_switch_order(None) == 3


def test_foe_registry_and_events():
    from showdown.rl import pokeenv_backend as B
    from showdown.rl.context_diag import count_events
    assert B.ARCHETYPE_FOE_LEVEL['greedy_joint_arq3'] == 3 and 'greedy_joint_arq3' in B.LIVE_OPPONENTS
    msgs = [['', 'switch', 'p2a: Pelipper', 'Pelipper'],
            ['', '-weather', 'RainDance', '[from] ability: Drizzle', '[of] p2a: Pelipper'],
            ['', '-weather', 'RainDance', '[upkeep]'],
            ['', 'move', 'p1a: Torkoal', 'Sunny Day', 'p1a: Torkoal'],
            ['', '-weather', 'SunnyDay'],
            ['', '-fieldstart', 'move: Psychic Terrain', '[from] ability: Psychic Surge', '[of] p2a: Indeedee']]
    ev = count_events(SimpleNamespace(player_role='p1', _replay_data=msgs))
    assert ev['rival']['lluvia_activado'] == 1 and ev['rival']['clima_activado'] == 1
    assert ev['nuestro']['sol_activado'] == 1
    assert ev['rival']['campo_psiquico_activado'] == 1 and ev['rival']['campo_activado'] == 1
