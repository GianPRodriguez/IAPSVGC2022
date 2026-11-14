"""ObsSpec v2 de combate (OTS completo + estado), material del rival y máscaras."""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from showdown.inference_sim.team_factory import active_teams
from showdown.rl import team_preview_obs as P
from showdown.rl.constants import OBS_DIM, PASS_ID
from showdown.rl.observation import build_observation, obs_layout
from showdown.rl.snapshot import BattleSnapshot, FieldSnapshot, MonSnapshot, SideSnapshot


def _mon(ms, *, own: bool, active: bool, seen: bool = True, hp: float = 100.0) -> MonSnapshot:
    stats = P.stats_from_sp(ms.species, ms.sp, ms.nature) if own else None
    return MonSnapshot(
        species=ms.species, ots_species=ms.species, hp_pct=hp, types=(),
        ability=P.to_id(ms.ability), item=P.to_id(ms.item), moves=tuple(P.to_id(m) for m in ms.moves),
        move_pps=(16, 16, 16, 16), is_active=active, stats=stats,
        nature='' if own else ms.nature, seen=seen,
    )


def _snap(field: FieldSnapshot = None, foe_order=None) -> BattleSnapshot:
    a, b = active_teams()[0], active_teams()[1]
    our = [_mon(m, own=True, active=i < 2) for i, m in enumerate(a.mons[:4])]
    foe_all = [_mon(m, own=False, active=i < 2, seen=i < 3) for i, m in enumerate(b.mons)]
    reserve = foe_all[2:]
    if foe_order is not None:
        reserve = [reserve[i] for i in foe_order]
    return BattleSnapshot(
        turn=3,
        our=SideSnapshot(active=our[:2], bench=our[2:4], reserve=our[2:4]),
        foe=SideSnapshot(active=foe_all[:2], bench=reserve[:2], reserve=reserve),
        field=field or FieldSnapshot(),
        avail_moves=(our[0].moves, our[1].moves),
        avail_switches=((our[2].species, our[3].species),) * 2,
    )


def test_dim_finite_and_all_sections_informative():
    obs = build_observation(_snap())
    assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all()
    lay = obs_layout()
    for name in ('our_0', 'our_3', 'foe_0', 'foe_5', 'our_moves', 'foe_moves', 'pairs', 'field'):
        a, b = lay[name]
        assert np.count_nonzero(obs[a:b]) > 0, name


def test_build_is_fast_enough():
    snaps = [_snap() for _ in range(20)]
    t = time.perf_counter()
    for s in snaps:
        build_observation(s)
    assert (time.perf_counter() - t) / len(snaps) < 0.05  # < 50 ms por paso


def test_foe_reserve_order_does_not_matter():
    o1 = build_observation(_snap(foe_order=[0, 1, 2, 3]))
    o2 = build_observation(_snap(foe_order=[3, 1, 0, 2]))
    assert np.array_equal(o1, o2)


def test_move_slots_follow_available_moves_order():
    s = _snap()
    mon = s.our.active[0]
    s.avail_moves = (tuple(reversed(mon.moves)), s.avail_moves[1])
    o_rev = build_observation(s)
    o = build_observation(_snap())
    a, _b = obs_layout()['our_moves']
    from showdown.rl.constants import MOVE_SLOT_DIM
    first = o[a:a + MOVE_SLOT_DIM]
    last_rev = o_rev[a + 3 * MOVE_SLOT_DIM: a + 4 * MOVE_SLOT_DIM]
    assert np.array_equal(first, last_rev)


def test_trick_room_and_tailwind_flip_speed():
    a, b = obs_layout()['pairs']
    n = (b - a) // 3
    base = build_observation(_snap())[a + 2 * n: b]
    tr = build_observation(_snap(FieldSnapshot(trick_room=True)))[a + 2 * n: b]
    assert np.allclose(tr, -base)
    tw = build_observation(_snap(FieldSnapshot(our_tailwind=True)))[a + 2 * n: b]
    assert (tw >= base - 1e-6).all() and (tw > base).any()


def test_foe_material_counts_unseen_as_healthy_and_is_order_stable():
    s = _snap()
    # 3 vistos (2 activos + 1) al 100% + 1 no visto asumido sano = 400.
    assert s.foe.team_hp_pct_total() == pytest.approx(400.0)
    s.foe.active[0].hp_pct = 50.0
    assert s.foe.team_hp_pct_total() == pytest.approx(350.0)
    s2 = _snap(foe_order=[3, 2, 1, 0])
    s2.foe.active[0].hp_pct = 50.0
    assert s2.foe.team_hp_pct_total() == pytest.approx(350.0)
    assert s.foe.faint_count() == 0


def test_rain_boosts_water_damage():
    from showdown.rl.observation import move_damage_frac
    sw = P.mon_profile(P.PreviewMonFeat('Pelipper', moves=('Hurricane', 'Weather Ball', 'Hydro Pump')))
    tgt = P.mon_profile(P.PreviewMonFeat('Arcanine-Hisui'))
    att, dfn = MonSnapshot(hp_pct=100.0), MonSnapshot(hp_pct=100.0)
    hydro = [m for m in sw.damaging if m.mid == 'hydropump'][0]
    dry = move_damage_frac(sw, att, tgt, dfn, hydro, FieldSnapshot(), defender_is_ours=False)
    wet = move_damage_frac(sw, att, tgt, dfn, hydro, FieldSnapshot(weather='raindance'), defender_is_ours=False)
    assert wet == pytest.approx(1.5 * dry, rel=0.05)


def test_mask_filters_by_valid_orders():
    """Máscara de entrenamiento = órdenes válidas de Showdown (pass solo si es legal)."""
    pytest.importorskip('poke_env')
    from poke_env.battle import Move
    from poke_env.player.battle_order import SingleBattleOrder

    from showdown.rl.pe_adapter import pe_slot_mask

    protect = Move('protect', gen=9)
    tackle = Move('tackle', gen=9)
    legal = [SingleBattleOrder(protect), SingleBattleOrder(tackle, move_target=1),
             SingleBattleOrder(tackle, move_target=2)]
    battle = SimpleNamespace(
        force_switch=[False, False], available_switches=[[], []],
        available_moves=[[protect, tackle], []], can_mega_evolve=[False, False],
        can_tera=[False, False], valid_orders=[legal, []],
    )
    m = pe_slot_mask(battle, 0)
    assert m[PASS_ID] == 0.0  # pass no es legal con un activo vivo
    assert m[0 * 4 + 3] == 1.0  # Protect → objetivo self/campo
    assert m[0 * 4 + 0] == 0.0 and m[0 * 4 + 1] == 0.0  # Protect con objetivo rival: inválido
    assert m[1 * 4 + 0] == 1.0 and m[1 * 4 + 1] == 1.0  # Tackle a rival A / B
    assert m[1 * 4 + 2] == 0.0  # Tackle al aliado no está entre las órdenes válidas


def test_stat_inference_narrows_foe_ranges_and_reports_uncertainty():
    from showdown.rl.constants import MON_BLOCK_DIM, OTS_MON_DIM

    s_full = _snap()
    s_inf = _snap()
    foe = s_inf.foe.active[0]
    foe.sp_bounds = {k: (10, 12) for k in P.STAT_KEYS}
    pf_full = P.mon_profile(P.PreviewMonFeat(foe.species, item=foe.item, ability=foe.ability,
                                             nature=foe.nature, moves=foe.moves))
    pf_inf = P.mon_profile(P.PreviewMonFeat(foe.species, item=foe.item, ability=foe.ability,
                                            nature=foe.nature, moves=foe.moves,
                                            sp_bounds=foe.sp_bounds))
    for k in P.STAT_KEYS:
        assert pf_full.smin[k] <= pf_inf.smin[k] <= pf_inf.smax[k] <= pf_full.smax[k]
        assert pf_inf.smax[k] - pf_inf.smin[k] < pf_full.smax[k] - pf_full.smin[k]
    lay = obs_layout()
    a, _ = lay['foe_0']
    dyn = slice(a + OTS_MON_DIM + 40, a + OTS_MON_DIM + 46)
    o_full, o_inf = build_observation(s_full), build_observation(s_inf)
    assert np.allclose(o_full[dyn], 1.0)  # sin inferencia: rango completo
    assert np.allclose(o_inf[dyn], 2.0 / 32.0)
    a0, _ = lay['our_0']
    assert np.allclose(o_full[a0 + OTS_MON_DIM + 40: a0 + OTS_MON_DIM + 46], 0.0)  # propio: exacto


def test_sp_bounds_from_real_stat_belief(monkeypatch):
    from showdown.rl.pe_adapter import player_stat_beliefs, sp_bounds_from_beliefs
    from showdown.stat_inference import StatBelief, normalize_species

    b = StatBelief('Kingambit', {'hp': 100, 'atk': 135, 'def': 120, 'spa': 60, 'spd': 85, 'spe': 50}, 'Adamant')
    b.candidates['spe'] = {0, 1, 2}
    beliefs = SimpleNamespace(beliefs={normalize_species('Kingambit'): b})
    out = sp_bounds_from_beliefs(beliefs, 'Kingambit')
    assert out['spe'] == (0, 2) and out['atk'][0] == 0
    player = SimpleNamespace(iap_battle=SimpleNamespace(stat_beliefs=beliefs))
    monkeypatch.delenv('OBS_STAT_INFERENCE', raising=False)
    assert player_stat_beliefs(player) is None  # apagado por defecto
    monkeypatch.setenv('OBS_STAT_INFERENCE', '1')
    assert player_stat_beliefs(player) is beliefs


def test_replay_grows_lazily_and_keeps_data():
    from showdown.rl.constants import JOINT_ACTION_N
    from showdown.rl.ddqn.replay import ReplayBuffer

    rb = ReplayBuffer(capacity=10_000, obs_dim=8)
    assert len(rb) == 0 and not rb._allocated and rb.origin_fractions() == {}
    m = np.ones(JOINT_ACTION_N, dtype=np.float32)
    for i in range(5000):
        rb.add(s=np.full(8, i % 7, dtype=np.float32), a=i % 576, r=float(i), s_next=np.zeros(8),
               terminated=False, m_next=m)
    assert rb._size == 8192 and len(rb) == 5000
    assert rb._r[4096] == 4096.0 and rb._s[10, 0] == 3.0
    s, a, r, s2, term, m2, opp = rb.sample(32, np.random.default_rng(0))
    assert s.dtype == np.float32 and m2.dtype == np.float32 and m2.min() == 1.0


def test_history_parser_last_move_lock_and_last_turn():
    from showdown.rl.battle_history import parse_history

    msgs = [
        ['', 'turn', '1'],
        ['', 'switch', 'p2a: Kingambit', 'Kingambit, L50', '100/100'],
        ['', 'move', 'p1a: Incineroar', 'Fake Out', 'p2a: Kingambit'],
        ['', '-damage', 'p2a: Kingambit', '90/100'],
        ['', 'move', 'p2a: Kingambit', 'Protect', 'p2a: Kingambit'],
        ['', 'turn', '2'],
        ['', 'move', 'p2a: Kingambit', 'Kowtow Cleave', 'p1a: Incineroar'],
        ['', '-damage', 'p2a: Kingambit', '60/100'],
        ['', 'move', 'p1a: Incineroar', 'Knock Off', 'p2a: Kingambit', '[from]move: Instruct'],
        ['', 'turn', '3'],
    ]
    h = parse_history(msgs)
    assert h.turn == 3
    k = h.mon('p2: Kingambit')
    assert k.last_move == 'kowtowcleave' and k.last_move_turn == 2 and k.moves_since_switch == 2
    assert h.hp_lost_last_turn('p2: Kingambit') == pytest.approx(30.0)
    i = h.mon('p1: Incineroar')
    assert i.last_move == 'fakeout'  # el movimiento [from] Instruct no cuenta como elección
    assert 'protect' not in h.last_turn.get('p2', {})  # en el turno 2 no hubo Protect
    h2 = parse_history(msgs[:5] + [['', 'turn', '2']])
    assert h2.last_turn['p2']['protect'] == 1.0 and h2.last_turn['p1']['fakeout'] == 1.0


def test_excluded_foes_are_dropped_and_brought_probability():
    from showdown.rl.observation import _foe_slots
    from showdown.rl.pe_adapter import _apply_brought_belief

    s = _snap()  # 3 vistos de 6
    _apply_brought_belief(s.foe)
    unseen = [m for m in s.foe.reserve if not m.seen]
    assert len(unseen) == 3 and all(m.p_brought == pytest.approx(1.0 / 3.0) for m in unseen)
    unseen[0].seen = True  # se ve el 4º → los otros 2 quedan descartados
    for m in s.foe.reserve:
        m.p_brought, m.excluded = 1.0, False
    _apply_brought_belief(s.foe)
    assert sum(m.excluded for m in s.foe.reserve) == 2
    slots = _foe_slots(s.foe)
    assert sum(1 for m in slots if m is not None) == 4 and slots[4] is None and slots[5] is None
    a, b = obs_layout()['foe_5']
    assert np.count_nonzero(build_observation(s)[a:b]) == 0


def test_mega_is_a_flag_on_each_move_never_on_switch():
    from showdown.rl.action_space import build_slot_mask, decode_slot_action, encode_slot_action
    from showdown.rl.constants import MEGA_BASE, SWITCH_BASE, SLOT_ACTION_N
    from showdown.rl.snapshot import SlotAction

    for mi in range(4):
        for ti in range(4):
            aid = encode_slot_action(kind='mega', move_index=mi, target=ti)
            act = decode_slot_action(aid)
            assert aid == MEGA_BASE + mi * 4 + ti and act.kind == 'mega'
            assert (act.move_index, act.target) == (mi, ti)
    s = _snap()
    s.our.active[0].can_mega = True
    s.our.active[1].can_mega = True
    m = build_slot_mask(s, 0)
    assert np.array_equal(m[MEGA_BASE:MEGA_BASE + 16], m[0:16])  # mega con cualquier movimiento
    assert decode_slot_action(SWITCH_BASE).kind == 'switch' and SLOT_ACTION_N == 40
    m2 = build_slot_mask(s, 1, other_slot_action=SlotAction('mega'))
    assert m2[MEGA_BASE:MEGA_BASE + 16].sum() == 0  # una sola mega por bando
