"""MCTS+PUCT v2 (docs/mcts_puct_spec.md v2.0): open-loop real, acciones podadas, hoja corta."""
from __future__ import annotations

import numpy as np

from showdown.rl.action_space import decode_joint_action, decode_slot_action, encode_joint_action
from showdown.rl.constants import MEGA_BASE, PASS_ID, SWITCH_BASE
from showdown.rl.env import MockBattleBackend, VGCEnv
from showdown.rl.mcts_puct import actions as A
from showdown.rl.mcts_puct.agent import MCTSPuctAgent, MCTSPuctConfig
from showdown.rl.mcts_puct.openloop import (
    OpenLoopConfig,
    auto_replace,
    material_value,
    prepare_root,
    search_openloop,
)
from showdown.rl.mcts_puct.simulator import step_sim
from showdown.rl.reward import RewardCandidate
from showdown.rl.snapshot import BattleSnapshot, MonSnapshot, SideSnapshot


def _mon(sp, types, moves, **kw):
    return MonSnapshot(species=sp, hp_pct=100.0, types=types, moves=moves, **kw)


def _snap() -> BattleSnapshot:
    our = SideSnapshot(
        active=[_mon('Garchomp', ('Dragon', 'Ground'), ('earthquake', 'dragonclaw', 'protect', 'rockslide')),
                _mon('Incineroar', ('Fire', 'Dark'), ('fakeout', 'flareblitz', 'knockoff', 'partingshot'))],
        bench=[_mon('Amoonguss', ('Grass', 'Poison'), ('spore', 'ragepowder', 'pollenpuff', 'protect')),
               _mon('Gholdengo', ('Steel', 'Ghost'), ('makeitrain', 'shadowball', 'nastyplot', 'protect'))],
    )
    foe = SideSnapshot(
        active=[_mon('Kingambit', ('Dark', 'Steel'), ('kowtowcleave', 'suckerpunch', 'ironhead', 'protect')),
                _mon('Rillaboom', ('Grass',), ('woodhammer', 'grassyglide', 'fakeout', 'uturn'))],
        bench=[_mon('Urshifu', ('Fighting', 'Water'), ('surgingstrikes', 'closecombat', 'aquajet', 'protect')),
               _mon('Flutter Mane', ('Ghost', 'Fairy'), ('moonblast', 'shadowball', 'dazzlinggleam', 'protect'))],
    )
    return BattleSnapshot(turn=1, our=our, foe=foe)


def _acts(j):
    return tuple(decode_slot_action(a) for a in decode_joint_action(j))


def test_pruned_targets_and_no_pass():
    s = _snap()
    joints = A.pruned_joints(s)
    assert len(joints) < 120  # v1: 663
    for j in joints:
        for slot, act in enumerate(_acts(j)):
            assert act.kind != 'pass'
            if act.kind != 'move':
                continue
            mv = s.our.active[slot].moves[act.move_index]
            cls = A.move_target_class(mv)
            if cls == 'foe':
                assert act.target in (0, 1)
            elif cls == 'none':
                assert act.target == 3  # Protect, Earthquake, Rock Slide, Parting Shot
    for j in joints:
        a1, a2 = decode_joint_action(j)
        assert not (a1 == a2 and SWITCH_BASE <= a1 < SWITCH_BASE + 4)


def test_pass_only_when_nothing_else():
    s = _snap()
    s.our.active[1].fainted = True
    s.our.active[1].hp_pct = 0.0
    s.our.bench = [None, None]
    assert A.slot_candidates(s, 1) == [PASS_ID]


def test_prefer_mega_drops_plain_twin_only():
    plain = encode_joint_action(4, 5)
    mega = encode_joint_action(4 + MEGA_BASE, 5)
    switch = encode_joint_action(SWITCH_BASE, 5)
    out = A.prefer_mega([plain, mega, switch])
    assert mega in out and switch in out and plain not in out


def test_status_move_no_chip_with_flag():
    s = _snap()
    prot = encode_joint_action(2 * 4 + 3, PASS_ID)
    pas = encode_joint_action(PASS_ID, PASS_ID)
    for k in range(30):
        c = step_sim(s, prot, pas, np.random.default_rng(k), status_chip=False)
        assert c.foe.active[0].hp_pct == 100.0 and c.foe.active[1].hp_pct == 100.0
    # default (v1 / MCTS-Nash) conserva el chip de 8-18 %
    hit = sum(step_sim(s, encode_joint_action(2 * 4 + 1, PASS_ID), pas,
                       np.random.default_rng(k)).foe.active[1].hp_pct < 100 for k in range(30))
    assert hit > 0


def test_auto_replace_after_ko():
    s = _snap()
    s.foe.active[0].fainted = True
    s.foe.active[0].hp_pct = 0.0
    s.foe.bench[1].hp_pct = 60.0
    auto_replace(s)
    assert s.foe.active[0].species == 'Urshifu'  # el de más HP
    assert s.foe.bench[0].species == 'Kingambit'


def test_prepare_root_aligns_moves_and_foe_bench():
    s = _snap()
    s.avail_moves = (('rockslide', 'protect'), ())
    sheet = [_mon(n, ('Normal',), ('tackle',), seen=False, p_brought=p)
             for n, p in (('A', 0.2), ('B', 0.9), ('C', 0.6))]
    seen = _mon('Seen', ('Normal',), ('tackle',), seen=True)
    excluded = _mon('X', ('Normal',), ('tackle',), seen=False, excluded=True, p_brought=1.0)
    s.foe.reserve = [seen, excluded] + sheet
    r = prepare_root(s)
    assert r.our.active[0].moves == ('rockslide', 'protect')
    assert [m.species for m in r.foe.bench] == ['Seen', 'B']
    assert s.our.active[0].moves[0] == 'earthquake'  # la raíz original no se toca


def test_material_value_range_and_terminal():
    s = _snap()
    assert material_value(s) == 0.0
    s.foe.active[0].hp_pct = 0.0
    s.foe.active[0].fainted = True
    assert 0 < material_value(s) <= 1
    s.finished, s.won = True, False
    assert material_value(s) == -1.0


def test_search_explores_beyond_id_order_and_is_deterministic():
    s = _snap()
    j1, root, tel = search_openloop(s, OpenLoopConfig(n_sims=200, seed=1))
    j2, _, _ = search_openloop(s, OpenLoopConfig(n_sims=200, seed=1))
    assert j1 == j2
    assert tel.failures == 0 and tel.sims == 200
    assert tel.avg_depth > 1.0   # v1: 1.0 (no pasaba de la raíz)
    assert tel.root_max_n > 1    # v1: todas las aristas con N=1
    assert j1 != min(root.edges)  # v1 elegía el id más bajo por desempate


def test_root_joints_respected():
    s = _snap()
    allowed = A.pruned_joints(s)[-3:]
    j, root, _ = search_openloop(s, OpenLoopConfig(n_sims=40, seed=0), root_joints=allowed)
    assert set(root.edges) <= set(allowed) and j in allowed


def test_single_root_action_is_trivial():
    s = _snap()
    only = A.pruned_joints(s)[0]
    j, _, tel = search_openloop(s, OpenLoopConfig(n_sims=50), root_joints=[only])
    assert j == only and tel.trivial and tel.sims == 0


def test_agent_version_switch_and_mock_game():
    env = VGCEnv(backend=MockBattleBackend(max_turns=40), reward_candidate=RewardCandidate.SPARSE,
                 max_episode_turns=40)
    env.reset(seed=0)
    try:
        for version in (1, 2):
            agent = MCTSPuctAgent(MCTSPuctConfig(n_sims=8, seed=0, version=version))
            agent.act_slots(env._snap)
            assert agent.last_telemetry is not None
        steps = 0
        agent = MCTSPuctAgent(MCTSPuctConfig(n_sims=16, seed=0))
        while steps < 60:
            a1, a2 = agent.act_slots(env._snap)
            _o, _r, term, trunc, _i = env.step((a1, a2))
            steps += 1
            if term or trunc:
                break
        assert steps >= 1
    finally:
        env.close()
