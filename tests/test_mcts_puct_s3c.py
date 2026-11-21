"""MCTS+PUCT v5: S3 fase 3c (docs/mcts_puct_spec.md v5.0). Calculadora falsa (sin Node).

Tailwind, Trick Room, velocidad dinámica (Icy Wind), estados (sueño, parálisis, quemadura,
veneno) e inmunidades, boosts en el árbol (Swords Dance, selfBoosts), Intimidate (Clear Body,
Defiant), clima por habilidad, pivotes, residuales y duraciones, puntuación.
"""
from __future__ import annotations

import numpy as np

from showdown.rl.action_space import encode_joint_action, encode_slot_action
from showdown.rl.constants import PASS_ID, SWITCH_BASE
from showdown.rl.mcts_puct.openloop import OpenLoopConfig, prepare_root, search_openloop
from showdown.rl.mcts_puct.s3 import S3Model, _key, move_effect3, stage_mult
from showdown.rl.snapshot import BattleSnapshot, MonSnapshot, SideSnapshot


def _mon(sp, types, moves, spe, ability='', item=''):
    st = {'hp': 180, 'atk': 150, 'def': 120, 'spa': 120, 'spd': 120, 'spe': spe}
    return MonSnapshot(species=sp, hp_pct=100.0, types=types, moves=moves, stats=st, ability=ability, item=item)


def _snap(our_a, our_b, foe_a, foe_b, our_bench=None, foe_bench=None) -> BattleSnapshot:
    our = SideSnapshot(active=[our_a, our_b], bench=[our_bench, None])
    foe = SideSnapshot(active=[foe_a, foe_b], bench=[foe_bench, None])
    return BattleSnapshot(turn=1, our=our, foe=foe)


def _fixed(v):
    return lambda reqs: [(v, v)] * len(reqs)


def _mv(mi, t=0):
    return encode_slot_action(kind='move', move_index=mi, target=t)


def _j(a1=PASS_ID, a2=PASS_ID):
    return encode_joint_action(a1, a2)


def _build(snap, dmg=90):
    root = prepare_root(snap)
    return root, S3Model.build(root, ranges_fn=_fixed(dmg), phase='3c')


def _base():
    return _snap(
        _mon('Tornadus', ('Flying',), ('tailwind', 'bleakwindstorm', 'icywind', 'protect'), spe=50),
        _mon('Amoonguss', ('Grass', 'Poison'), ('spore', 'pollenpuff', 'ragepowder', 'protect'), spe=30),
        _mon('Kingambit', ('Dark', 'Steel'), ('kowtowcleave', 'swordsdance', 'trickroom', 'closecombat'), spe=80),
        _mon('Pikachu', ('Electric',), ('thunderbolt', 'willowisp', 'thunderwave', 'uturn'), spe=90),
        our_bench=_mon('Incineroar', ('Fire', 'Dark'), ('flareblitz',), spe=60, ability='Intimidate'),
        foe_bench=_mon('Torkoal', ('Fire',), ('eruption',), spe=20, ability='Drought'),
    )


def test_effect3_classification_and_stage_mult():
    assert move_effect3('tailwind') == 'tailwind' and move_effect3('trickroom') == 'trickroom'
    assert move_effect3('sunnyday') == 'weather' and move_effect3('spore') == 'status'
    assert move_effect3('swordsdance') == 'boost' and move_effect3('partingshot') == 'boost'
    assert move_effect3('closecombat') == ''
    assert stage_mult(2) == 2.0 and stage_mult(-1) == 2.0 / 3.0 and stage_mult(9) == 4.0


def test_tailwind_makes_slower_side_move_first_next_turn():
    root, m = _build(_base(), dmg=400)  # todo golpe es KO
    s1 = m.step(root, _j(_mv(0, 3)), _j(), np.random.default_rng(0))  # Tailwind
    assert s1.field.our_tailwind and s1.field.our_tailwind_turns == 3.0
    # Tornadus (50 ×2 = 100) ataca antes que Kingambit (80): Kingambit cae sin actuar
    s2 = m.step(s1, _j(_mv(1, 0)), _j(_mv(0, 0)), np.random.default_rng(0))
    assert s2.foe.active[0].fainted and not s2.our.active[0].fainted
    # sin Tailwind, Kingambit (80) gana el duelo
    s3 = m.step(root, _j(_mv(1, 0)), _j(_mv(0, 0)), np.random.default_rng(0))
    assert s3.our.active[0].fainted and not s3.foe.active[0].fainted


def test_tailwind_expires():
    root, m = _build(_base())
    s = m.step(root, _j(_mv(0, 3)), _j(), np.random.default_rng(0))
    for _ in range(3):
        s = m.step(s, _j(), _j(), np.random.default_rng(0))
    assert not s.field.our_tailwind


def test_trick_room_reverses_and_cancels():
    root, m = _build(_base(), dmg=400)
    s1 = m.step(root, _j(), _j(_mv(2, 3)), np.random.default_rng(0))  # Trick Room
    assert s1.field.trick_room
    # bajo Trick Room, Tornadus (50) actúa antes que Kingambit (80)
    s2 = m.step(s1, _j(_mv(1, 0)), _j(_mv(0, 0)), np.random.default_rng(0))
    assert s2.foe.active[0].fainted and not s2.our.active[0].fainted
    s3 = m.step(s1, _j(), _j(_mv(2, 3)), np.random.default_rng(0))  # repetirlo lo cancela
    assert not s3.field.trick_room


def test_dynamic_speed_icy_wind_same_turn():
    snap = _base()
    snap.our.active[0].stats['spe'] = 200  # Tornadus usa Icy Wind primero
    snap.our.active[1] = _mon('Garchomp', ('Dragon', 'Ground'), ('dragonclaw',), spe=70)
    root, m = _build(snap, dmg=400)
    # Kingambit 80 → 53 tras Icy Wind: Garchomp (70) ya es más rápido en el mismo turno
    s = m.step(root, _j(_mv(2, 3), _mv(0, 0)), _j(_mv(0, 1)), np.random.default_rng(0))
    assert not s.our.active[1].fainted


def test_sleep_and_immunities():
    root, m = _build(_base(), dmg=60)
    s = m.step(root, _j(PASS_ID, _mv(0, 0)), _j(), np.random.default_rng(1))  # Spore a Kingambit
    k = s.foe.active[0]
    assert k.status == 'slp' and 1 <= k.status_counter <= 3
    s2 = m.step(s, _j(), _j(_mv(0, 0)), np.random.default_rng(0))  # dormido: no ataca
    assert s2.our.active[0].hp_pct == 100.0
    root.foe.active[0].types = ('Grass',)  # Planta: inmune al polvo
    s3 = m.step(root, _j(PASS_ID, _mv(0, 0)), _j(), np.random.default_rng(1))
    assert s3.foe.active[0].status == ''


def test_paralysis_and_burn():
    root, m = _build(_base(), dmg=60)
    s = m.step(root, _j(), _j(PASS_ID, _mv(2, 1)), np.random.default_rng(0))  # Thunder Wave a Amoonguss
    assert s.our.active[1].status == 'par'
    root2, m2 = _build(_base(), dmg=60)
    s = m2.step(root2, _j(), _j(PASS_ID, _mv(2, 0)), np.random.default_rng(0))
    assert s.our.active[0].status == 'par'  # Volador no es inmune a Thunder Wave
    root2.our.active[0].types = ('Ground',)
    s = m2.step(root2, _j(), _j(PASS_ID, _mv(2, 0)), np.random.default_rng(0))
    assert s.our.active[0].status == ''  # Tierra sí lo es
    root2.our.active[0].types = ('Flying',)
    s = m2.step(root2, _j(), _j(PASS_ID, _mv(1, 0)), np.random.default_rng(0))  # Will-O-Wisp a Tornadus
    tor = s.our.active[0]
    assert tor.status == 'brn' and abs(tor.hp_pct - (100.0 - 6.25)) < 1e-6  # residual
    assert abs(m2._mods(tor, s.foe.active[0], 'bleakwindstorm', '') - 1.0) < 1e-9  # especial: no
    burned_k = s.foe.active[0]
    burned_k.status = 'brn'
    assert abs(m2._mods(burned_k, tor, 'kowtowcleave', '') - 0.5) < 1e-9


def test_boosts_in_tree():
    root, m = _build(_base(), dmg=60)
    s = m.step(root, _j(), _j(_mv(1, 3)), np.random.default_rng(0))  # Swords Dance
    k = s.foe.active[0]
    assert k.boosts.get('atk') == 2 and abs(m._mods(k, s.our.active[0], 'kowtowcleave', '') - 2.0) < 1e-9
    s = m.step(root, _j(), _j(_mv(3, 0)), np.random.default_rng(0))  # Close Combat: -1 def/-1 spd
    k = s.foe.active[0]
    assert k.boosts.get('def') == -1 and k.boosts.get('spd') == -1
    assert abs(m._mods(s.our.active[0], k, 'closecombat', '') - 1.5) < 1e-9
    assert root.foe.active[0].boosts in ({}, None)  # copia al escribir: la raíz no cambia


def test_intimidate_clear_body_defiant_and_drought():
    snap = _base()
    snap.foe.active[1].ability = 'Defiant'
    snap.foe.active[0].ability = 'Clear Body'
    root, m = _build(snap)
    s = m.step(root, _j(SWITCH_BASE + 0), _j(), np.random.default_rng(0))  # entra Incineroar
    assert s.our.active[0].species == 'Incineroar'
    assert s.foe.active[0].boosts.get('atk', 0) == 0      # Clear Body
    assert s.foe.active[1].boosts.get('atk', 0) == 1      # Defiant: −1 +2
    s = m.step(root, _j(), _j(SWITCH_BASE + 0), np.random.default_rng(0))  # entra Torkoal: Drought
    assert s.field.weather == 'sunnyday'
    assert abs(m._mods(s.our.active[1], s.foe.active[0], 'flareblitz', s.field.weather) - 1.5) < 1e-9


def test_pivot_and_residual_kill():
    root, m = _build(_base(), dmg=60)
    s = m.step(root, _j(), _j(PASS_ID, _mv(3, 0)), np.random.default_rng(0))  # U-turn
    assert s.foe.active[1].species == 'Torkoal' and s.foe.bench[0].species == 'Pikachu'
    assert s.field.weather == 'sunnyday'  # Drought al entrar por el pivote
    root.our.active[0].hp_pct = 5.0
    root.our.active[0].status = 'brn'
    s = m.step(root, _j(), _j(), np.random.default_rng(0))
    assert s.our.active[0].fainted


def test_scores_3c():
    root, m = _build(_base(), dmg=60)
    sc0 = m.slot_scores(root, 0)
    sc1 = m.slot_scores(root, 1)
    assert sc0[0 * 4 + 3] > 0          # Tailwind: Tornadus es más lento que los dos rivales
    assert sc1[0 * 4 + 0] > 0          # Spore vale el daño negado
    root.field.our_tailwind = True
    assert m.slot_scores(root, 0)[0 * 4 + 3] == 0.0  # ya activo
    sc_icy = sc0[2 * 4 + 3]
    assert sc_icy > 0                   # Icy Wind: daño + control de velocidad


def test_3b_unchanged_by_3c_code():
    root = prepare_root(_base())
    m = S3Model.build(root, ranges_fn=_fixed(60), phase='3b')
    s = m.step(root, _j(_mv(0, 3)), _j(), np.random.default_rng(0))  # Tailwind en 3b: sin efecto
    assert not s.field.our_tailwind


def test_search_v5_runs():
    j, root, tel = search_openloop(_base(), OpenLoopConfig(n_sims=120, seed=0, model='s3c'),
                                   ranges_fn=_fixed(60))
    assert tel.failures == 0 and tel.sims == 120 and j in root.edges
