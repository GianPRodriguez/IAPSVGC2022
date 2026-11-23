"""Agente H 2.0: SM-MCTS/Nash open-loop con S3 (docs/mcts_puct_nash_spec.md v2.0). Calculadora falsa."""
from __future__ import annotations

import numpy as np

from showdown.rl.action_space import decode_joint_action, encode_joint_action, encode_slot_action
from showdown.rl.constants import PASS_ID
from showdown.rl.mcts_puct_nash.openloop_nash import (
    NashNode,
    NashOLConfig,
    _backup,
    _select,
    search_nash_openloop,
)
from showdown.rl.snapshot import BattleSnapshot, MonSnapshot, SideSnapshot


def _mon(sp, types, moves, spe, hp=100.0):
    st = {'hp': 180, 'atk': 150, 'def': 120, 'spa': 120, 'spd': 120, 'spe': spe}
    return MonSnapshot(species=sp, hp_pct=hp, types=types, moves=moves, stats=st)


def _snap(foe_hp=100.0):
    our = SideSnapshot(active=[_mon('Urshifu', ('Fighting', 'Water'), ('closecombat', 'aquajet', 'protect'), 120),
                               _mon('Amoonguss', ('Grass', 'Poison'), ('spore', 'pollenpuff', 'protect'), 30)],
                       bench=[None, None])
    foe = SideSnapshot(active=[_mon('Kingambit', ('Dark', 'Steel'), ('kowtowcleave', 'suckerpunch', 'protect'), 50, foe_hp),
                               _mon('Pikachu', ('Electric',), ('thunderbolt', 'fakeout', 'protect'), 90, foe_hp)],
                       bench=[None, None])
    return BattleSnapshot(turn=1, our=our, foe=foe)


def _fixed(v):
    return lambda reqs: [(v, v)] * len(reqs)


def test_v2_search_runs_and_returns_root_row():
    cfg = NashOLConfig(n_sims=80, seed=0)
    j, root, tel = search_nash_openloop(_snap(), cfg, ranges_fn=_fixed(30))
    assert tel.failures == 0 and tel.sims == 80
    assert j in root.I and len(root.J) > 1
    assert abs(float(root.sigma.sum()) - 1.0) < 1e-6
    assert tel.root_cells_visited > 1 and -1.0 <= tel.root_value <= 1.0


def test_v2_respects_root_legal_joints():
    mv = lambda mi, t: encode_slot_action(kind='move', move_index=mi, target=t)
    legal = [encode_joint_action(mv(0, 1), mv(1, 1)), encode_joint_action(mv(2, 0), mv(2, 0))]
    j, root, _tel = search_nash_openloop(_snap(), NashOLConfig(n_sims=30, seed=1), root_joints=legal,
                                         ranges_fn=_fixed(30))
    assert set(root.I) <= set(legal) and j in legal


def test_v2_prefers_ko_when_damage_is_lethal():
    """Con daño letal contra rivales a 20 %, la fila ganadora ataca (no protege ni pasa)."""
    j, _root, _tel = search_nash_openloop(_snap(foe_hp=20.0), NashOLConfig(n_sims=120, seed=0),
                                          ranges_fn=_fixed(60))
    a1, a2 = decode_joint_action(j)
    assert a1 != PASS_ID and a2 != PASS_ID


def test_v2_deterministic_given_seed():
    a = search_nash_openloop(_snap(), NashOLConfig(n_sims=40, seed=3), ranges_fn=_fixed(30))[0]
    b = search_nash_openloop(_snap(), NashOLConfig(n_sims=40, seed=3), ranges_fn=_fixed(30))[0]
    assert a == b


def test_selection_is_decoupled_minimax():
    """El rival elige la columna que más nos perjudica, no la que más nos favorece (v1 era optimista)."""
    node = NashNode(I=[10, 11], J=[20, 21], p=np.array([0.5, 0.5]), q=np.array([0.5, 0.5]),
                    n=np.full((2, 2), 50.0), W=np.array([[50.0, -50.0], [40.0, -40.0]]),
                    sigma=np.array([0.5, 0.5]), tau=np.array([0.5, 0.5]), N=200, W_sum=0.0, expanded=True)
    i, j = _select(node, _snap(), c=0.0, check_rows=False)
    assert j == 1           # columna −1 / −0,8: la peor para nosotros
    assert i == 0           # contra τ uniforme la fila 0 vale 0, la 1 vale 0 → empate → menor índice


def test_backup_uses_nash_value_and_fpu():
    node = NashNode(I=[1, 2], J=[3, 4], p=np.array([0.5, 0.5]), q=np.array([0.5, 0.5]),
                    n=np.zeros((2, 2)), W=np.zeros((2, 2)), sigma=np.array([0.5, 0.5]),
                    tau=np.array([0.5, 0.5]), expanded=True)
    parent = NashNode(I=[5], J=[6], p=np.ones(1), q=np.ones(1), n=np.zeros((1, 1)), W=np.zeros((1, 1)),
                      sigma=np.ones(1), tau=np.ones(1), expanded=True)
    _backup([(parent, 0, 0), (node, 0, 0)], z=1.0, t_fp=100)
    # una sola visita: las celdas no visitadas valen la media del nodo (1) → v* = 1, no 0,25
    assert node.N == 1 and abs(parent.W[0, 0] - 1.0) < 1e-9


def test_h_validated_player_defaults_to_meta_and_v2(monkeypatch):
    monkeypatch.delenv('MCTS_NASH_VERSION', raising=False)
    from showdown.rl.mcts_puct.player import MCTSPuctPlayer
    from showdown.rl.mcts_puct_nash.player import MCTSPuctNashPlayer, make_validated_mcts_puct_nash_player_class
    cls = make_validated_mcts_puct_nash_player_class()
    assert cls.default_sp_info_mode == 'meta'
    assert MCTSPuctNashPlayer.teampreview is MCTSPuctPlayer.teampreview
    import inspect
    src = inspect.getsource(MCTSPuctNashPlayer.__init__)
    assert "_env_int('MCTS_NASH_VERSION', 2)" in src


def test_diag_record_writes_row_without_changing_decision(tmp_path, monkeypatch):
    """Paso 0 de H 2.1: el diagnóstico escribe una fila por decisión y no altera la elección."""
    import json
    import os

    from showdown.rl.mcts_puct_nash import diag
    from showdown.rl.mcts_puct_nash.diag import summarize

    cfg = NashOLConfig(n_sims=60, seed=0)
    j, root, tel = search_nash_openloop(_snap(), cfg, ranges_fn=_fixed(30))
    monkeypatch.setenv('MCTS_NASH_DIAG_FILE', str(tmp_path / 'd.jsonl'))
    monkeypatch.setenv('MCTS_NASH_DIAG_G', '0')
    diag.record(_snap(), None, None, j, root, tel, cfg, 0)
    path = tmp_path / f'd.jsonl.{os.getpid()}'
    rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 1 and rows[0]['h'] == j and rows[0]['R'] == len(root.I)
    assert 0.0 <= rows[0]['frac_unvisited'] <= 1.0 and rows[0]['sims'] == 60
    assert 'decisiones no triviales: 1' in summarize(rows)
    j2, _r, _t = search_nash_openloop(_snap(), cfg, ranges_fn=_fixed(30))
    assert j2 == j
