"""MCTS+PUCT v6: S3 fase 3d, terreno (docs/mcts_puct_spec.md v6.0). Calculadora falsa (sin Node).

Terreno al entrar (Grassy / Psychic Surge) y duración, prioridad de Grassy Glide, bloqueo de
prioridad en Psychic Terrain, Expanding Force, ajuste de daño (×1,3, Earthquake ×0,5),
curación de Grassy, Misty / Electric contra estados.
"""
from __future__ import annotations

import numpy as np

from showdown.rl.action_space import encode_joint_action, encode_slot_action
from showdown.rl.constants import PASS_ID, SWITCH_BASE
from showdown.rl.mcts_puct.openloop import OpenLoopConfig, prepare_root, search_openloop
from showdown.rl.mcts_puct.s3 import S3Model, _tnorm, grounded
from showdown.rl.snapshot import BattleSnapshot, MonSnapshot, SideSnapshot


def _mon(sp, types, moves, spe, ability='', item=''):
    st = {'hp': 180, 'atk': 150, 'def': 120, 'spa': 120, 'spd': 120, 'spe': spe}
    return MonSnapshot(species=sp, hp_pct=100.0, types=types, moves=moves, stats=st, ability=ability, item=item)


def _base() -> BattleSnapshot:
    our = SideSnapshot(
        active=[_mon('Garchomp', ('Dragon', 'Ground'), ('earthquake', 'dragonclaw', 'protect', 'spore'), spe=60),
                _mon('Tornadus', ('Flying',), ('bleakwindstorm', 'tailwind'), spe=50)],
        bench=[_mon('Rillaboom', ('Grass',), ('grassyglide', 'woodhammer', 'fakeout', 'uturn'), spe=40,
                    ability='Grassy Surge'), None],
    )
    foe = SideSnapshot(
        active=[_mon('Kingambit', ('Dark', 'Steel'), ('suckerpunch', 'kowtowcleave', 'protect', 'fakeout'), spe=80),
                _mon('Pikachu', ('Electric',), ('thunderbolt',), spe=100)],
        bench=[_mon('Indeedee-F', ('Psychic', 'Normal'), ('expandingforce', 'followme'), spe=30,
                    ability='Psychic Surge'), None],
    )
    return BattleSnapshot(turn=1, our=our, foe=foe)


def _fixed(v):
    return lambda reqs: [(v, v)] * len(reqs)


def _mv(mi, t=0):
    return encode_slot_action(kind='move', move_index=mi, target=t)


def _j(a1=PASS_ID, a2=PASS_ID):
    return encode_joint_action(a1, a2)


def _build(snap=None, dmg=60, phase='3d'):
    root = prepare_root(snap or _base())
    return root, S3Model.build(root, ranges_fn=_fixed(dmg), phase=phase)


def test_helpers():
    assert _tnorm('GRASSY_TERRAIN') == 'grassy' and _tnorm('psychicterrain') == 'psychic' and _tnorm('') == ''
    assert grounded(_mon('A', ('Ground',), (), 1))
    assert not grounded(_mon('B', ('Flying',), (), 1))
    assert not grounded(_mon('C', ('Psychic',), (), 1, ability='Levitate'))
    assert not grounded(_mon('D', ('Steel',), (), 1, item='Air Balloon'))


def test_grassy_surge_on_entry_and_expiry():
    root, m = _build()
    s = m.step(root, _j(SWITCH_BASE + 0), _j(), np.random.default_rng(0))  # entra Rillaboom
    assert _tnorm(s.field.terrain) == 'grassy' and s.field.terrain_turns == 4.0
    for _ in range(4):
        s = m.step(s, _j(), _j(), np.random.default_rng(0))
    assert s.field.terrain == ''


def test_grassy_glide_priority_depends_on_terrain():
    snap = _base()
    snap.our.active[0] = snap.our.bench[0]
    snap.our.bench[0] = None
    root, m = _build(snap, dmg=400)
    gg = _j(_mv(0, 0))       # Grassy Glide → Kingambit
    ko = _j(_mv(1, 0))       # Kowtow Cleave → Rillaboom
    s = m.step(root, gg, ko, np.random.default_rng(0))  # sin terreno: Kingambit (80) primero
    assert s.our.active[0].fainted and not s.foe.active[0].fainted
    root.field.terrain = 'grassyterrain'
    s = m.step(root, gg, ko, np.random.default_rng(0))  # con Grassy: prioridad +1
    assert s.foe.active[0].fainted and not s.our.active[0].fainted


def test_psychic_terrain_blocks_priority_on_grounded_only():
    root, m = _build()
    root.field.terrain = 'psychicterrain'
    s = m.step(root, _j(), _j(_mv(0, 0)), np.random.default_rng(0))  # Sucker Punch → Garchomp (suelo)
    assert s.our.active[0].hp_pct == 100.0
    s = m.step(root, _j(), _j(_mv(0, 1)), np.random.default_rng(0))  # → Tornadus (volador)
    assert s.our.active[1].hp_pct < 100.0
    root.field.terrain = ''
    s = m.step(root, _j(), _j(_mv(0, 0)), np.random.default_rng(0))
    assert s.our.active[0].hp_pct < 100.0


def test_expanding_force_spread_in_psychic_terrain():
    snap = _base()
    snap.foe.active[1] = snap.foe.bench[0]
    snap.foe.bench[0] = None
    root, m = _build(snap, dmg=36)  # 36/180 = 20 %
    s = m.step(root, _j(), _j(PASS_ID, _mv(0, 0)), np.random.default_rng(0))
    assert s.our.active[0].hp_pct == 80.0 and s.our.active[1].hp_pct == 100.0  # sin terreno: individual
    root.field.terrain = 'psychicterrain'
    s = m.step(root, _j(), _j(PASS_ID, _mv(0, 0)), np.random.default_rng(0))
    exp = 100.0 - 20.0 * 1.5 * 0.75 * 1.3  # poder ×1,5, área ×0,75 y terreno ×1,3 (Psíquico en el suelo)
    assert abs(s.our.active[0].hp_pct - exp) < 1e-6 and abs(s.our.active[1].hp_pct - exp) < 1e-6


def test_terrain_damage_factor_relative_to_root():
    root, m = _build()
    ril = root.our.bench[0]
    king = root.foe.active[0]
    garch = root.our.active[0]
    assert abs(m._mods(ril, king, 'woodhammer', '', 'grassyterrain') - 1.3) < 1e-9
    assert abs(m._mods(garch, king, 'earthquake', '', 'grassyterrain') - 0.5) < 1e-9
    assert abs(m._mods(garch, king, 'earthquake', '', '') - 1.0) < 1e-9


def test_grassy_heal_grounded_only():
    root, m = _build()
    root.field.terrain = 'grassyterrain'
    root.our.active[0].hp_pct = 50.0
    root.our.active[1].hp_pct = 50.0
    s = m.step(root, _j(), _j(), np.random.default_rng(0))
    assert s.our.active[0].hp_pct == 56.25 and s.our.active[1].hp_pct == 50.0


def test_misty_and_electric_block_status():
    root, m = _build()
    root.field.terrain = 'mistyterrain'
    s = m.step(root, _j(_mv(3, 0)), _j(), np.random.default_rng(1))  # Spore a Kingambit (suelo)
    assert s.foe.active[0].status == ''
    root.field.terrain = 'electricterrain'
    s = m.step(root, _j(_mv(3, 0)), _j(), np.random.default_rng(1))
    assert s.foe.active[0].status == ''
    root.field.terrain = ''
    s = m.step(root, _j(_mv(3, 0)), _j(), np.random.default_rng(1))
    assert s.foe.active[0].status == 'slp'


def test_3c_ignores_terrain():
    root, m = _build(phase='3c')
    s = m.step(root, _j(SWITCH_BASE + 0), _j(), np.random.default_rng(0))
    assert s.field.terrain == ''


def test_search_v6_runs():
    j, root, tel = search_openloop(_base(), OpenLoopConfig(n_sims=120, seed=0, model='s3d'), ranges_fn=_fixed(60))
    assert tel.failures == 0 and tel.sims == 120 and j in root.edges
