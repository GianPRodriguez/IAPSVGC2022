"""MCTS+PUCT v3, modelo S3 fase 3a (docs/mcts_puct_spec.md v3.0): daño real en tabla,
orden por velocidad/prioridad, área, redirección, precisión. Calculadora falsa (sin Node)."""
from __future__ import annotations

import numpy as np

from showdown.rl.action_space import encode_joint_action, encode_slot_action
from showdown.rl.constants import PASS_ID
from showdown.rl.mcts_puct.openloop import OpenLoopConfig, prepare_root, search_openloop
from showdown.rl.mcts_puct.s3 import S3Model, _key, _point_sp_bounds, move_accuracy, move_target
from showdown.rl.snapshot import BattleSnapshot, MonSnapshot, SideSnapshot

PASS = encode_joint_action(PASS_ID, PASS_ID)


def _mon(sp, types, moves, **kw):
    return MonSnapshot(species=sp, hp_pct=100.0, types=types, moves=moves, **kw)


def _stats(spe):
    return {'hp': 180, 'atk': 150, 'def': 120, 'spa': 120, 'spd': 120, 'spe': spe}


def _snap() -> BattleSnapshot:
    our = SideSnapshot(
        active=[_mon('Garchomp', ('Dragon', 'Ground'), ('earthquake', 'dragonclaw', 'protect', 'rockslide'),
                     stats=_stats(150)),
                _mon('Incineroar', ('Fire', 'Dark'), ('fakeout', 'flareblitz', 'knockoff', 'partingshot'),
                     stats=_stats(80))],
        bench=[_mon('Amoonguss', ('Grass', 'Poison'), ('spore', 'ragepowder', 'pollenpuff', 'protect'),
                    stats=_stats(30)),
               _mon('Gholdengo', ('Steel', 'Ghost'), ('makeitrain', 'shadowball', 'nastyplot', 'protect'),
                    stats=_stats(120))],
    )
    foe = SideSnapshot(
        active=[_mon('Kingambit', ('Dark', 'Steel'), ('kowtowcleave', 'suckerpunch', 'ironhead', 'protect')),
                _mon('Rillaboom', ('Grass',), ('woodhammer', 'grassyglide', 'fakeout', 'uturn'))],
        bench=[_mon('Urshifu', ('Fighting', 'Water'), ('surgingstrikes', 'closecombat', 'aquajet', 'protect')),
               _mon('Flutter Mane', ('Ghost', 'Fairy'), ('moonblast', 'shadowball', 'dazzlinggleam', 'protect'))],
    )
    return BattleSnapshot(turn=1, our=our, foe=foe)


def _fixed(lo, hi):
    """Calculadora falsa: mismo rango absoluto para todo pedido."""
    return lambda reqs: [(lo, hi)] * len(reqs)


def _move(mi, t=0):
    return encode_slot_action(kind='move', move_index=mi, target=t)


def _build(ranges=_fixed(30, 40)):
    root = prepare_root(_snap())
    return root, S3Model.build(root, ranges_fn=ranges)


def test_dex_helpers():
    assert move_target('rockslide') == 'alladjacentfoes'
    assert move_target('earthquake') == 'alladjacent'
    assert move_target('dragonclaw') == 'normal'
    assert move_accuracy('rockslide') == 0.9 and move_accuracy('protect') == 1.0


def test_champions_dex_moves_missing_from_old_dex():
    """El dex antiguo (S2) no tiene Make It Rain ni Kowtow Cleave: con S2 no hacían nada."""
    from showdown.rl.damage_heuristic import move_entry
    from showdown.rl.mcts_puct.s3 import is_damaging, target_class
    assert not move_entry('makeitrain') and not move_entry('kowtowcleave')
    assert is_damaging('makeitrain') and is_damaging('kowtowcleave')
    assert target_class('makeitrain') == 'none' and move_target('makeitrain') == 'alladjacentfoes'
    assert target_class('kowtowcleave') == 'foe' and move_accuracy('kowtowcleave') == 1.0


def test_build_table_covers_both_sides_spread_and_friendly_fire():
    root, m = _build()
    st = m.stats
    assert st.requests > 50 and st.missing == 0 and st.error == ''
    g, inc = root.our.active
    king = root.foe.active[0]
    assert (_key(g), _key(king), 'dragonclaw', False) in m.table
    assert (_key(g), _key(king), 'rockslide', True) in m.table          # área
    assert (_key(g), _key(inc), 'earthquake', True) in m.table          # fuego amigo
    assert (_key(king), _key(g), 'kowtowcleave', False) in m.table      # rival → nuestro
    assert not any(k[2] == 'protect' for k in m.table)                  # estado: sin pedido
    # el banco también está (cambios dentro del árbol)
    ursh = root.foe.bench[0]
    assert (_key(ursh), _key(g), 'closecombat', False) in m.table


def test_faster_ko_prevents_slower_action_and_trick_room_flips():
    root, m = _build(_fixed(1000, 1000))  # todo golpe es KO
    g = root.our.active[0]
    king = root.foe.active[0]
    m.speed[_key(g)], m.speed[_key(king)] = 200, 50
    ours = encode_joint_action(_move(1, 0), PASS_ID)    # Dragon Claw → Kingambit
    theirs = encode_joint_action(_move(2, 0), PASS_ID)  # Iron Head → Garchomp
    s = m.step(root, ours, theirs, np.random.default_rng(0))
    assert s.foe.active[0].fainted and not s.our.active[0].fainted
    m.trick_room = True
    s = m.step(root, ours, theirs, np.random.default_rng(0))
    assert s.our.active[0].fainted and not s.foe.active[0].fainted


def test_priority_beats_speed():
    root, m = _build(_fixed(1000, 1000))
    g, king = root.our.active[0], root.foe.active[0]
    m.speed[_key(g)], m.speed[_key(king)] = 200, 50
    s = m.step(root, encode_joint_action(_move(1, 0), PASS_ID),
               encode_joint_action(_move(1, 0), PASS_ID),  # Sucker Punch (+1)
               np.random.default_rng(0))
    assert s.our.active[0].fainted and not s.foe.active[0].fainted


def test_spread_hits_both_foes_and_earthquake_hits_ally():
    root, m = _build(_fixed(30, 30))
    s = m.step(root, encode_joint_action(_move(0, 3), PASS_ID), PASS, np.random.default_rng(1))  # Earthquake
    assert s.foe.active[0].hp_pct < 100 and s.foe.active[1].hp_pct < 100
    assert s.our.active[1].hp_pct < 100  # Incineroar, aliado
    hits = 0
    for k in range(40):  # Rock Slide 90 %: alguna falla
        s = m.step(root, encode_joint_action(_move(3, 3), PASS_ID), PASS, np.random.default_rng(k))
        hits += sum(x.hp_pct < 100 for x in s.foe.active)
        assert s.our.active[1].hp_pct == 100.0
    assert 50 < hits < 80


def test_single_target_redirects_when_target_fainted():
    root, m = _build(_fixed(30, 30))
    root.foe.active[0].fainted = True
    root.foe.active[0].hp_pct = 0.0
    s = m.step(root, encode_joint_action(_move(1, 0), PASS_ID), PASS, np.random.default_rng(0))
    assert s.foe.active[1].hp_pct < 100


def test_status_move_does_nothing_and_switch_goes_first():
    root, m = _build(_fixed(1000, 1000))
    s = m.step(root, encode_joint_action(_move(2, 3), PASS_ID), PASS, np.random.default_rng(0))  # Protect
    assert all(x.hp_pct == 100 for x in s.foe.active + s.our.active)
    # el rival golpea el hueco 0 y nosotros cambiamos: recibe el golpe el que entra
    s = m.step(root, encode_joint_action(32, PASS_ID), encode_joint_action(_move(2, 0), PASS_ID),
               np.random.default_rng(0))
    assert s.our.active[0].species == 'Amoonguss' and s.our.active[0].fainted
    assert s.our.bench[0].species == 'Garchomp' and not s.our.bench[0].fainted


def test_calc_failure_falls_back_to_s2():
    def boom(_reqs):
        raise RuntimeError('calc down')
    root, m = _build(boom)
    assert m.stats.error and m.stats.missing == m.stats.requests and not m.table
    s = m.step(root, encode_joint_action(_move(1, 0), PASS_ID), PASS, np.random.default_rng(0))
    assert s.foe.active[0].hp_pct < 100


def test_scores_penalize_ally_hit_and_reward_ko():
    root, m = _build(_fixed(30, 40))
    sc = m.slot_scores(root, 0)
    assert sc[3 * 4 + 3] > sc[0 * 4 + 3]  # Rock Slide > Earthquake (que golpea a Incineroar)
    root.foe.active[0].hp_pct = 20.0
    sc2 = m.slot_scores(root, 0)
    assert sc2[1 * 4 + 0] > sc2[1 * 4 + 1]  # rematar a Kingambit vale más (bono de KO)


def test_point_sp_bounds_from_beliefs():
    class B:
        def preferred_sp(self, stat):
            return 32 if stat == 'spe' else 0

    class TB:
        beliefs = {'kingambit': B()}

    out = _point_sp_bounds(TB(), 'Kingambit')
    assert out['spe'] == (32, 32) and out['hp'] == (0, 0)
    assert _point_sp_bounds(TB(), 'Pikachu') is None


def test_search_with_s3_runs_and_reports_calc_telemetry():
    j, root, tel = search_openloop(_snap(), OpenLoopConfig(n_sims=120, seed=0, model='s3'),
                                   ranges_fn=_fixed(30, 40))
    assert tel.model == 's3' and tel.failures == 0 and tel.sims == 120
    assert tel.calc_requests > 50 and tel.calc_missing == 0
    assert tel.root_max_n > 1 and j in root.edges


def test_g_preview_defaults_to_greedy_and_nash_is_opt_in(monkeypatch):
    """Spec v3.1: preview greedy por defecto (team_preview_nash nunca se entrenó)."""
    from showdown.battle_bots.greedy_solo.pokeenv_player import GreedyDamagePlayer
    from showdown.rl.family_preview_mixin import FamilyPreviewMixin
    from showdown.rl.mcts_puct.player import MCTSPuctPlayer

    monkeypatch.setattr(GreedyDamagePlayer, 'teampreview', lambda self, b: '/team greedy')
    monkeypatch.setattr(FamilyPreviewMixin, 'teampreview', lambda self, b: '/team family')
    monkeypatch.delenv('MCTS_PREVIEW', raising=False)
    assert MCTSPuctPlayer.teampreview(object(), None) == '/team greedy'
    monkeypatch.setenv('MCTS_PREVIEW', 'nash')
    assert MCTSPuctPlayer.teampreview(object(), None) == '/team family'


def test_s3_real_calc_path_does_not_persist_to_disk(monkeypatch):
    """S3 llama a la calculadora con persist=False (la caché en disco llenó el servidor)."""
    import showdown.damage_calc_bridge as B
    seen = {}

    def fake(requests, *, chunk_size=None, deadline_monotonic=None, persist=True):
        seen['persist'] = persist
        return [(30, 40)] * len(requests)

    monkeypatch.setenv('MCTS_S3_CALC', '1')
    monkeypatch.setattr(B, 'damage_ranges_batch', fake)
    root = prepare_root(_snap())
    m = S3Model.build(root)
    assert seen.get('persist') is False and m.stats.missing == 0
