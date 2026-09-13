"""Champions species lookup: aliases + falsy miss for getPkmnInfo."""
from __future__ import annotations

from API.PokemonData import (
    champions_species_key,
    getPkmnInfo,
    has_pkmn_info,
)
from showdown.battle_bots.greedy_solo.greedy import _resolve_mega_forme


def test_floette_eternal_mega_aliases_to_floette_mega():
    assert champions_species_key('floette-eternal-mega') == 'floette-mega'
    assert has_pkmn_info('floette-eternal-mega')
    info = getPkmnInfo('floette-eternal-mega')
    assert info is not None
    assert info[2] == 74  # Floette-Mega HP base
    assert info[3] == 85  # Atk


def test_basculegion_m_aliases_to_basculegion():
    assert champions_species_key('Basculegion-M') == 'basculegion'
    assert champions_species_key('basculegionm') == 'basculegion'
    assert has_pkmn_info('Basculegion-M')
    from showdown.inference_sim.team_factory import canonicalize_species, to_showdown_export, MonSet, TeamMeta
    assert canonicalize_species('Basculegion-M') == 'Basculegion'
    meta = TeamMeta(
        name='t', source_url='',
        mons=[MonSet('Basculegion-M', 'Choice Scarf', 'Adaptability', 'Jolly', ['Aqua Jet'], {'hp': 0})],
    )
    export = to_showdown_export(meta)
    assert 'Basculegion @' in export
    assert 'Basculegion-M' not in export
    assert meta.mons[0].species == 'Basculegion'


def test_getPkmnInfo_miss_is_none_not_zero_tuple():
    getPkmnInfo.cache_clear()
    info = getPkmnInfo('definitely-not-a-pokemon-xyz')
    assert info is None
    assert not info


def test_resolve_mega_floette_eternal():
    forme = _resolve_mega_forme('floette-eternal', 'Floettite')
    assert forme == 'floette-mega'
    assert has_pkmn_info('floette-eternal-mega')


def test_floette_belief_key_and_fairy_aura_sanitize():
    from showdown.stat_inference import normalize_species, sanitize_ability_for_forme
    from showdown.battle_modifier import _mega_forme_ability

    assert normalize_species('floette-mega') == 'floette-eternal'
    assert normalize_species('Floette-Eternal') == 'floette-eternal'
    assert sanitize_ability_for_forme('floette-eternal', 'Fairy Aura') == 'flowerveil'
    assert sanitize_ability_for_forme('floette-mega', 'Fairy Aura') == 'Fairy Aura'
    assert _mega_forme_ability('floette-eternal-mega') == 'fairyaura'
