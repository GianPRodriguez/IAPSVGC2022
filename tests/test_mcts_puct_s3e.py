"""MCTS+PUCT v7: S3 fase 3e (objetos con efecto dinámico; docs/mcts_puct_spec.md v7.0).

Focus Sash / Sturdy (no contra multigolpe), bayas de resistencia consumidas (×2 después),
Sitrus, Leftovers, recoil de Life Orb (no con Magic Guard), Rocky Helmet, White Herb, semillas de
terreno, puntuación con Sash, y que 3c no cambia. Calculadora falsa (sin Node).
"""
from __future__ import annotations

import numpy as np

from showdown.rl.action_space import encode_joint_action, encode_slot_action
from showdown.rl.constants import PASS_ID, SWITCH_BASE
from showdown.rl.mcts_puct.openloop import OpenLoopConfig, prepare_root, search_openloop
from showdown.rl.mcts_puct.s3 import S3Model, berry_triggers
from showdown.rl.snapshot import BattleSnapshot, MonSnapshot, SideSnapshot


def _mon(sp, types, moves, spe, ability='', item=''):
    st = {'hp': 180, 'atk': 150, 'def': 120, 'spa': 120, 'spd': 120, 'spe': spe}
    return MonSnapshot(species=sp, hp_pct=100.0, types=types, moves=moves, stats=st, ability=ability, item=item)


def _fixed(v):
    return lambda reqs: [(v, v)] * len(reqs)


def _mv(mi, t=0):
    return encode_slot_action(kind='move', move_index=mi, target=t)


def _j(a1=PASS_ID, a2=PASS_ID):
    return encode_joint_action(a1, a2)


def _snap(foe_item='', foe_ability='', our_item='', our_ability='', our_bench=None, foe_bench=None):
    our = SideSnapshot(
        active=[_mon('Urshifu', ('Fighting', 'Water'), ('closecombat', 'surgingstrikes', 'aquajet', 'protect'),
                     spe=120, item=our_item, ability=our_ability),
                _mon('Amoonguss', ('Grass', 'Poison'), ('spore', 'pollenpuff', 'ragepowder', 'protect'), spe=30)],
        bench=[our_bench, None])
    foe = SideSnapshot(
        active=[_mon('Kingambit', ('Dark', 'Steel'), ('kowtowcleave', 'suckerpunch', 'ironhead', 'protect'),
                     spe=50, item=foe_item, ability=foe_ability),
                _mon('Pikachu', ('Electric',), ('thunderbolt', 'fakeout', 'protect', 'volttackle'), spe=90)],
        bench=[foe_bench, None])
    return BattleSnapshot(turn=1, our=our, foe=foe)


def _build(snap, dmg, phase='3e'):
    root = prepare_root(snap)
    return root, S3Model.build(root, ranges_fn=_fixed(dmg), phase=phase)


def _pct(m, att, dfn, mid):
    return m.damage_range(att, dfn, mid, False)[0]


def test_focus_sash_and_sturdy_survive_once_at_full_hp():
    root, m = _build(_snap(foe_item='Focus Sash'), dmg=999)
    s = m.step(root, _j(_mv(0, 0)), _j(), np.random.default_rng(0))   # Close Combat letal
    k = s.foe.active[0]
    assert not k.fainted and abs(k.hp_pct - 1.0) < 1e-9 and k.item == ''
    s2 = m.step(s, _j(_mv(0, 0)), _j(), np.random.default_rng(0))     # sin banda: cae
    assert s2.foe.active[0].fainted
    root, m = _build(_snap(foe_ability='Sturdy'), dmg=999)
    s = m.step(root, _j(_mv(0, 0)), _j(), np.random.default_rng(0))
    assert not s.foe.active[0].fainted
    # 3c no modela la banda
    root, m = _build(_snap(foe_item='Focus Sash'), dmg=999, phase='3c')
    assert m.step(root, _j(_mv(0, 0)), _j(), np.random.default_rng(0)).foe.active[0].fainted


def test_sash_not_vs_multihit_nor_below_full_hp():
    root, m = _build(_snap(foe_item='Focus Sash'), dmg=999)
    s = m.step(root, _j(_mv(1, 0)), _j(), np.random.default_rng(0))   # Surging Strikes (multigolpe)
    assert s.foe.active[0].fainted
    root, m = _build(_snap(foe_item='Focus Sash'), dmg=999)
    root.foe.active[0].hp_pct = 99.0
    assert m.step(root, _j(_mv(0, 0)), _j(), np.random.default_rng(0)).foe.active[0].fainted


def test_resist_berry_consumed_then_double():
    root, m = _build(_snap(foe_item='Chople Berry'), dmg=40)
    att, k = root.our.active[0], root.foe.active[0]
    assert berry_triggers('chopleberry', 'closecombat', k)          # Lucha ×4 contra Dark/Steel
    assert not berry_triggers('chopleberry', 'aquajet', k)
    assert abs(m._mods(att, k, 'closecombat', '') - 1.0) < 1e-9      # con baya: la tabla vale
    s = m.step(root, _j(_mv(0, 0)), _j(), np.random.default_rng(0))
    k2 = s.foe.active[0]
    assert k2.item == ''
    assert abs(m._mods(s.our.active[0], k2, 'closecombat', '') - 2.0) < 1e-9
    assert abs(m._mods(s.our.active[0], k2, 'aquajet', '') - 1.0) < 1e-9  # otro tipo: sin cambio


def test_sitrus_heals_once_below_half():
    root, m = _build(_snap(foe_item='Sitrus Berry'), dmg=40)
    d = _pct(m, root.our.active[0], root.foe.active[0], 'closecombat')
    root.foe.active[0].hp_pct = 40.0 + d                               # queda en 40 → +25
    s = m.step(root, _j(_mv(0, 0)), _j(), np.random.default_rng(0))
    k = s.foe.active[0]
    assert abs(k.hp_pct - 65.0) < 1e-6 and k.item == ''


def test_leftovers_life_orb_and_magic_guard():
    root, m = _build(_snap(foe_item='Leftovers'), dmg=1)
    root.foe.active[0].hp_pct = 50.0
    s = m.step(root, _j(), _j(), np.random.default_rng(0))
    assert abs(s.foe.active[0].hp_pct - 56.25) < 1e-9
    root, m = _build(_snap(our_item='Life Orb'), dmg=1)
    s = m.step(root, _j(_mv(0, 0)), _j(), np.random.default_rng(0))
    assert abs(s.our.active[0].hp_pct - 90.0) < 1e-9
    root, m = _build(_snap(our_item='Life Orb', our_ability='Magic Guard'), dmg=1)
    s = m.step(root, _j(_mv(0, 0)), _j(), np.random.default_rng(0))
    assert s.our.active[0].hp_pct == 100.0


def test_rocky_helmet_on_contact_only():
    root, m = _build(_snap(foe_item='Rocky Helmet'), dmg=1)
    s = m.step(root, _j(_mv(0, 0)), _j(), np.random.default_rng(0))   # Close Combat: contacto
    assert abs(s.our.active[0].hp_pct - (100.0 - 100.0 / 6.0)) < 1e-6
    root, m = _build(_snap(foe_item='Rocky Helmet'), dmg=1)
    root.our.active[1].moves = ('spore', 'pollenpuff', 'ragepowder', 'protect')
    s = m.step(root, _j(PASS_ID, _mv(1, 0)), _j(), np.random.default_rng(0))  # Pollen Puff: especial
    assert s.our.active[1].hp_pct == 100.0


def test_white_herb_cancels_drops_once():
    inc = _mon('Incineroar', ('Fire', 'Dark'), ('flareblitz',), spe=60, ability='Intimidate')
    root, m = _build(_snap(foe_item='White Herb', our_bench=inc), dmg=1)
    s = m.step(root, _j(SWITCH_BASE + 0), _j(), np.random.default_rng(0))  # entra Intimidate
    k, pk = s.foe.active
    assert int((k.boosts or {}).get('atk', 0)) == 0 and k.item == ''
    assert int((pk.boosts or {}).get('atk', 0)) == -1


def test_terrain_seed_on_entry():
    rilla = _mon('Rillaboom', ('Grass',), ('woodhammer',), spe=85, item='Grassy Seed')
    snap = _snap(our_bench=rilla)
    snap.field.terrain = 'grassyterrain'
    root, m = _build(snap, dmg=1)
    s = m.step(root, _j(SWITCH_BASE + 0), _j(), np.random.default_rng(0))
    r = s.our.active[0]
    assert r.species == 'Rillaboom' and int((r.boosts or {}).get('def', 0)) == 1 and r.item == ''


def test_scores_sash_removes_ko_bonus():
    root, m = _build(_snap(foe_item='Focus Sash'), dmg=999)
    root3c, m3c = _build(_snap(foe_item='Focus Sash'), dmg=999, phase='3c')
    assert m.slot_scores(root, 0)[0 * 4 + 0] < m3c.slot_scores(root3c, 0)[0 * 4 + 0]


def test_search_v7_runs():
    j, root, tel = search_openloop(_snap(foe_item='Focus Sash', our_item='Life Orb'),
                                   OpenLoopConfig(n_sims=120, seed=0, model='s3e'), ranges_fn=_fixed(60))
    assert tel.failures == 0 and tel.sims == 120 and j in root.edges
