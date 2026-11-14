"""Observación de Team Preview v2 (OTS completo) y evaluación pareada del preview."""
from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from showdown.inference_sim.team_factory import MonSet, active_teams
from showdown.rl import team_preview_obs as T


def _salamence(sp=None, nature='Naive'):
    return MonSet(
        'Salamence', 'Salamencite', 'Intimidate', nature,
        ['Hyper Voice', 'Double-Edge', 'Tailwind', 'Protect'],
        sp or {'hp': 1, 'atk': 21, 'def': 0, 'spa': 15, 'spd': 0, 'spe': 29},
    )


def test_dex_snapshot_pinned_to_showdown_commit():
    assert T._dex()['meta']['ps_commit'] == '5b01be810'


def test_vocab_covers_whole_pool():
    for team in active_teams():
        for mon in team.mons:
            feat = T.mon_feat_from_monset(mon, public_only=False)
            prof = T.mon_profile(feat)
            assert prof.present, mon.species
            assert prof.species_idx > 0, mon.species
            assert prof.item_idx > 0, mon.item
            assert prof.ability_idx and min(prof.ability_idx) > 0, mon.ability


def test_own_stats_exact_foe_stats_range_without_sp():
    own = T.mon_feat_from_monset(_salamence(), public_only=False)
    foe = T.mon_feat_from_monset(_salamence(), public_only=True)
    assert own.stats is not None and foe.stats is None
    po, pf = T.mon_profile(own), T.mon_profile(foe)
    for k in T.STAT_KEYS:
        assert po.smin[k] == po.smax[k]
        assert pf.smin[k] < pf.smax[k]
        # El stat real del rival está dentro de su rango público.
        assert pf.smin[k] <= po.smin[k] <= pf.smax[k]
    # La hoja rival no depende de sus SP (no son públicos).
    other_sp = T.mon_feat_from_monset(_salamence(sp={k: 0 for k in T.STAT_KEYS}), public_only=True)
    assert np.array_equal(T._encode_mon(foe), T._encode_mon(other_sp))


def test_mega_form_and_ate_conversion():
    prof = T.mon_profile(T.mon_feat_from_monset(_salamence(), public_only=False))
    assert prof.ability == 'aerilate'
    assert set(prof.types) == {'Dragon', 'Flying'}
    hv = [m for m in prof.damaging if m.mid == 'hypervoice'][0]
    assert hv.type == 'Flying' and hv.spread
    can_mega = prof.sem[2 + 18 + T.N_TYPES]
    assert can_mega == 1.0


def test_type_and_ability_immunities_give_zero_damage():
    indeedee = T.mon_profile(T.PreviewMonFeat('Indeedee-F', moves=('Psychic',), stats=None))
    kingambit = T.mon_profile(T.PreviewMonFeat('Kingambit', ability='Defiant'))
    assert T.expected_damage_frac(indeedee, kingambit) == 0.0
    garchomp = T.mon_profile(T.PreviewMonFeat('Garchomp', moves=('Earthquake',)))
    levitator = T.mon_profile(T.PreviewMonFeat('Rotom-Wash', ability='Levitate'))
    assert T.expected_damage_frac(garchomp, levitator) == 0.0


def test_foe_order_does_not_change_obs():
    a, b = active_teams()[0], active_teams()[1]
    our = [T.mon_feat_from_monset(m, public_only=False) for m in a.mons]
    foe = [T.mon_feat_from_monset(m, public_only=True) for m in b.mons]
    o1 = T.encode_preview_obs(T.PreviewObsBundle(our=our, foe=foe))
    shuffled = foe[:]
    random.Random(7).shuffle(shuffled)
    o2 = T.encode_preview_obs(T.PreviewObsBundle(our=our, foe=shuffled))
    assert np.array_equal(o1, o2)
    # El orden propio SÍ importa (las acciones indexan esas posiciones).
    o3 = T.encode_preview_obs(T.PreviewObsBundle(our=our[::-1], foe=foe))
    assert not np.array_equal(o1, o3)


def test_pokeenv_paths_match_monset_and_hide_foe_sp():
    mon = _salamence()
    pe = SimpleNamespace(
        species='salamence', item='salamencite', ability='intimidate',
        moves={'hypervoice': None, 'doubleedge': None, 'tailwind': None, 'protect': None},
        nature='Naive', evs=[mon.sp[k] for k in T.STAT_KEYS], stats={}, max_hp=None,
    )
    own_pe = T.mon_feat_from_pokeenv(pe, own=True)
    own_ms = T.mon_feat_from_monset(mon, public_only=False)
    assert np.array_equal(T._encode_mon(own_pe), T._encode_mon(own_ms))
    foe_pe = T.mon_feat_from_pokeenv(pe, own=False)
    assert foe_pe.stats is None
    assert np.array_equal(T._encode_mon(foe_pe), T._encode_mon(T.mon_feat_from_monset(mon, public_only=True)))
    pe_unknown = SimpleNamespace(**{**pe.__dict__, 'item': 'unknown_item'})
    assert T.mon_feat_from_pokeenv(pe_unknown, own=False).item is None


def test_metavgc_env_builds_full_ots_obs():
    pytest.importorskip('gymnasium')
    from showdown.rl.team_preview_env import TeamPreviewEnv

    env = TeamPreviewEnv(seed=0, backend='mock', team_source='metavgc')
    obs, info = env.reset()
    assert obs.shape == (T.OBS_DIM,)
    # Bloque propio: flag "stats exactos" = 1; bloque rival: 0.
    assert obs[1] == 1.0
    assert obs[6 * T.MON_DIM + 1] == 0.0
    assert 'our_team' in info and 'foe_team' in info


def test_preview_eval_is_reproducible_and_paired():
    pytest.importorskip('gymnasium')
    from showdown.rl.preview_eval import evaluate_preview_policy
    from showdown.rl.team_preview_env import TeamPreviewEnv

    env = TeamPreviewEnv(seed=0, backend='mock')
    choose = lambda o, m: int(np.flatnonzero(m > 0.5)[-1])  # noqa: E731
    e1 = evaluate_preview_policy(env, choose, n=64, baseline='fixed', eval_seed=11)
    env.reset_batch(5)  # ensuciar el estado del entorno
    e2 = evaluate_preview_policy(env, choose, n=64, baseline='fixed', eval_seed=11)
    assert e1.as_dict() == e2.as_dict()
    assert 0.0 <= e1.mean_wr <= 1.0 and 0.0 <= e1.base_wr <= 1.0
    assert abs(e1.delta_wr - (e1.mean_wr - e1.base_wr)) < 1e-12
    with pytest.raises(RuntimeError):
        evaluate_preview_policy(env, choose, n=8, baseline='pilot')


def test_preview_value_probe_on_mock_contexts():
    pytest.importorskip('gymnasium')
    from showdown.inference_sim.run_preview_value_probe import run_probe, summarize
    from showdown.rl.team_preview_env import TeamPreviewEnv

    env = TeamPreviewEnv(seed=0, backend='mock')
    wins = run_probe(env, n=40, arms=('fixed', 'random'), seed=3)
    s = summarize(wins)
    assert s['fixed']['n'] == 40 and 0.0 <= s['random']['wr'] <= 1.0
    assert abs(s['fixed-random']['delta'] - (s['fixed']['wr'] - s['random']['wr'])) < 1e-12


def test_formation_stats_reports_roles_on_metavgc():
    pytest.importorskip('gymnasium')
    from showdown.rl.preview_metrics import formation_stats
    from showdown.rl.team_preview_env import TeamPreviewEnv

    env = TeamPreviewEnv(seed=1, backend='mock', team_source='metavgc')
    env.reset_batch(48)
    bundles = [b for (b, _c) in env._batch]
    acts = [0] * 48
    st = formation_stats(bundles, acts, env.step_batch(acts))
    assert st['n'] == 48 and st['same_as_fixed'] == 1.0
    assert abs(sum(st['lead_pos']) - 1.0) < 1e-9
    assert st['lead_pos'][0] == st['lead_pos'][1] == 0.5
    assert 'fakeout' in st['roles'] and 'mega' in st['roles']
    for r in st['roles'].values():
        assert 0.0 <= r['lead_rate'] <= 1.0


def test_episode_tracker_outcomes_and_windows(tmp_path):
    from showdown.rl.train_metrics import EpisodeTracker, MetricsLog, read_jsonl

    log = MetricsLog(tmp_path / 'm.jsonl', run_meta={'agent': 'test'})
    tr = EpisodeTracker(log, window=10)
    tr.start(0, 'greedy_joint')
    from showdown.rl.constants import MEGA_BASE, PASS_ID, SWITCH_BASE

    tr.step(0, actions=(0, SWITCH_BASE), reward=0.3, components={'d_faint_opp': 1.0, 'd_hp_opp': 0.6})
    tr.end(0, global_step=1, terminated=True, truncated=False,
           components={'d_win': 1.0}, next_opponent='random')
    tr.step(0, actions=(MEGA_BASE + 1, PASS_ID), reward=-1.0, components={'d_faint_self': 1.0})
    tr.end(0, global_step=2, terminated=False, truncated=True, trunc_reason='turn_cap')
    s = tr.write_window(2, phase='t', learner={'loss': 0.5})
    log.close()
    assert s['by_opp']['greedy_joint']['wr'] == 1.0
    assert s['by_opp']['random']['turn_cap_rate'] == 1.0
    assert s['all']['act']['switch'] == 0.25 and s['all']['act']['mega'] == 0.25
    kinds = [r['kind'] for r in read_jsonl(tmp_path / 'm.jsonl')]
    assert kinds == ['run', 'episode', 'episode', 'window']


def test_real_pokeenv_pokemon_from_teambuilder_matches_monset():
    """Camino de inferencia real: poke-env Pokemon (teambuilder) == MonSet del pool."""
    pytest.importorskip('poke_env')
    from poke_env.battle import Pokemon
    from poke_env.teambuilder import Teambuilder

    from showdown.inference_sim.team_factory import to_showdown_export

    team = active_teams()[0]
    tb = Teambuilder.parse_showdown_team(to_showdown_export(team))
    for tb_mon, ms in zip(tb, team.mons):
        pe = Pokemon(gen=9, teambuilder=tb_mon)
        own_pe = T.mon_feat_from_pokeenv(pe, own=True)
        own_ms = T.mon_feat_from_monset(ms, public_only=False)
        assert own_pe.stats == own_ms.stats, ms.species
        assert np.array_equal(T._encode_mon(own_pe), T._encode_mon(own_ms)), ms.species
        foe_pe = T.mon_feat_from_pokeenv(pe, own=False)
        assert foe_pe.stats is None
        assert np.array_equal(T._encode_mon(foe_pe),
                              T._encode_mon(T.mon_feat_from_monset(ms, public_only=True))), ms.species
