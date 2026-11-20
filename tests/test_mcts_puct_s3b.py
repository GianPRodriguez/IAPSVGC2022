"""MCTS+PUCT v4: S3 fase 3b (docs/mcts_puct_spec.md v4.0). Calculadora falsa (sin Node).

Protect (y 1/3^n al repetir), Feint, Wide Guard, Fake Out (primer turno, retroceso, Inner
Focus), First Impression, Follow Me / Rage Powder (inmunidad a polvo), Helping Hand.
"""
from __future__ import annotations

import numpy as np

from showdown.rl.action_space import encode_joint_action, encode_slot_action
from showdown.rl.constants import PASS_ID
from showdown.rl.mcts_puct.openloop import OpenLoopConfig, prepare_root, search_openloop
from showdown.rl.mcts_puct.s3 import S3Model, _key, blockable, move_effect
from showdown.rl.snapshot import BattleSnapshot, MonSnapshot, SideSnapshot


def _mon(sp, types, moves, spe=100, **kw):
    st = {'hp': 180, 'atk': 150, 'def': 120, 'spa': 120, 'spd': 120, 'spe': spe}
    return MonSnapshot(species=sp, hp_pct=100.0, types=types, moves=moves, stats=st, **kw)


def _snap(our_moves=None, foe_moves=None) -> BattleSnapshot:
    om = our_moves or [('protect', 'closecombat', 'fakeout', 'wideguard'),
                       ('followme', 'helpinghand', 'ragepowder', 'earthquake')]
    fm = foe_moves or [('closecombat', 'feint', 'rockslide', 'protect'),
                       ('closecombat', 'fakeout', 'protect', 'heatwave')]
    our = SideSnapshot(
        active=[_mon('Urshifu', ('Fighting', 'Water'), om[0], spe=97),
                _mon('Amoonguss', ('Grass', 'Poison'), om[1], spe=30)],
        bench=[_mon('Garchomp', ('Dragon', 'Ground'), ('dragonclaw',), spe=102), None],
    )
    foe = SideSnapshot(
        active=[_mon('Kingambit', ('Dark', 'Steel'), fm[0], spe=50),
                _mon('Incineroar', ('Fire', 'Dark'), fm[1], spe=60)],
        bench=[_mon('Gholdengo', ('Steel', 'Ghost'), ('makeitrain',), spe=84), None],
    )
    return BattleSnapshot(turn=1, our=our, foe=foe)


def _fixed(lo, hi):
    return lambda reqs: [(lo, hi)] * len(reqs)


def _mv(mi, t=0):
    return encode_slot_action(kind='move', move_index=mi, target=t)


def _j(a1=PASS_ID, a2=PASS_ID):
    return encode_joint_action(a1, a2)


def _build(lo=30, hi=30, **kw):
    root = prepare_root(_snap(**kw))
    return root, S3Model.build(root, ranges_fn=_fixed(lo, hi), phase='3b')


def test_effect_classification():
    assert move_effect('protect') == move_effect('spikyshield') == 'protect'
    assert move_effect('endure') == ''  # Endure no protege
    assert move_effect('wideguard') == 'wideguard'
    assert move_effect('followme') == 'redirect' and move_effect('ragepowder') == 'redirect_powder'
    assert move_effect('helpinghand') == 'helpinghand'
    assert blockable('closecombat') and not blockable('feint') and not blockable('protect')


def test_protect_blocks_and_repeat_can_fail():
    root, m = _build()
    s = m.step(root, _j(_mv(0, 3)), _j(_mv(0, 0)), np.random.default_rng(0))  # Protect vs Close Combat
    assert s.our.active[0].hp_pct == 100.0 and s.our.active[0].protect_counter == 1
    fails = 0
    for k in range(60):  # segundo Protect seguido: éxito 1/3
        s2 = m.step(s, _j(_mv(0, 3)), _j(_mv(0, 0)), np.random.default_rng(k))
        fails += s2.our.active[0].hp_pct < 100
    assert 25 < fails < 55
    # otra jugada reinicia el contador
    s3 = m.step(s, _j(_mv(1, 0)), _j(), np.random.default_rng(0))
    assert s3.our.active[0].protect_counter == 0


def test_feint_breaks_protect():
    root, m = _build()
    m.speed[_key(root.foe.active[0])] = 300
    s = m.step(root, _j(_mv(0, 3)), _j(_mv(1, 0)), np.random.default_rng(0))  # Protect vs Feint
    assert s.our.active[0].hp_pct < 100.0


def test_wide_guard_blocks_spread_not_single():
    root, m = _build()
    s = m.step(root, _j(_mv(3, 3)), _j(_mv(2, 3)), np.random.default_rng(2))  # WG vs Rock Slide
    assert s.our.active[0].hp_pct == 100.0 and s.our.active[1].hp_pct == 100.0
    s = m.step(root, _j(_mv(3, 3)), _j(_mv(0, 0)), np.random.default_rng(2))  # WG vs Close Combat
    assert s.our.active[0].hp_pct < 100.0


def test_fake_out_first_turn_flinch_and_inner_focus():
    root, m = _build()
    # Fake Out (+3) a Kingambit: no actúa (su Close Combat no llega)
    s = m.step(root, _j(_mv(2, 0)), _j(_mv(0, 0)), np.random.default_rng(0))
    assert s.foe.active[0].hp_pct < 100.0 and s.our.active[0].hp_pct == 100.0
    # segundo turno en campo: Fake Out falla
    s2 = m.step(s, _j(_mv(2, 0)), _j(_mv(0, 0)), np.random.default_rng(0))
    assert s2.foe.active[0].hp_pct == s.foe.active[0].hp_pct and s2.our.active[0].hp_pct < 100.0
    root.foe.active[0].ability = 'Inner Focus'
    s3 = m.step(root, _j(_mv(2, 0)), _j(_mv(0, 0)), np.random.default_rng(0))
    assert s3.our.active[0].hp_pct < 100.0


def test_follow_me_redirects_and_rage_powder_powder_immunity():
    root, m = _build()
    # Follow Me (Amoonguss): el Close Combat del rival a Urshifu va a Amoonguss
    s = m.step(root, _j(PASS_ID, _mv(0, 3)), _j(_mv(0, 0)), np.random.default_rng(0))
    assert s.our.active[0].hp_pct == 100.0 and s.our.active[1].hp_pct < 100.0
    # Rage Powder no afecta a un atacante tipo Planta
    root.foe.active[0].types = ('Grass',)
    s = m.step(root, _j(PASS_ID, _mv(2, 3)), _j(_mv(0, 0)), np.random.default_rng(0))
    assert s.our.active[0].hp_pct < 100.0 and s.our.active[1].hp_pct == 100.0


def test_helping_hand_boosts_ally():
    root, m = _build(20, 20)
    base = m.step(root, _j(_mv(1, 0)), _j(), np.random.default_rng(0)).foe.active[0].hp_pct
    hh = m.step(root, _j(_mv(1, 0), _mv(1, 2)), _j(), np.random.default_rng(0)).foe.active[0].hp_pct
    assert abs((100 - hh) - 1.5 * (100 - base)) < 1e-6


def test_3a_ignores_support_moves():
    root = prepare_root(_snap())
    m = S3Model.build(root, ranges_fn=_fixed(30, 30), phase='3a')
    s = m.step(root, _j(_mv(0, 3)), _j(_mv(0, 0)), np.random.default_rng(0))
    assert s.our.active[0].hp_pct < 100.0  # sin Protect en 3a


def test_scores_value_support_moves():
    root, m = _build(40, 50)
    sc0 = m.slot_scores(root, 0)
    sc1 = m.slot_scores(root, 1)
    assert sc0[0 * 4 + 3] > 0                 # Protect vale la amenaza entrante
    assert sc0[2 * 4 + 0] > sc0[1 * 4 + 0] * 0.9  # Fake Out: daño + retroceso
    assert sc1[0 * 4 + 3] > 0 and sc1[1 * 4 + 2] > 0  # Follow Me, Helping Hand
    root.our.active[0].protect_counter = 2
    root.our.active[0].turns_active = 1
    sc0b = m.slot_scores(root, 0)
    assert sc0b[0 * 4 + 3] < sc0[0 * 4 + 3] / 5  # Protect repetido vale poco
    assert sc0b[2 * 4 + 0] == 0.0             # Fake Out fuera del primer turno


def test_search_v4_runs():
    j, root, tel = search_openloop(_snap(), OpenLoopConfig(n_sims=120, seed=0, model='s3b'),
                                   ranges_fn=_fixed(30, 40))
    assert tel.failures == 0 and tel.sims == 120 and j in root.edges
