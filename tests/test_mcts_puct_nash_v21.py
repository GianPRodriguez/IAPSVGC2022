"""H 2.1 (protocolo §4.7): variantes de H 2.0 por entorno; los defaults reproducen H 2.0 exactamente."""
from __future__ import annotations

import numpy as np
import pytest

from showdown.rl.mcts_puct_nash.nash import fictitious_play, regret_matching_plus
from showdown.rl.mcts_puct_nash.openloop_nash import NashOLConfig, search_nash_openloop
from test_mcts_puct_nash_v2 import _fixed, _snap

# Joints de H 2.0 (código congelado c70f0f2) en la fixture, n_sims=120, ranges 30+10·seed.
_H20 = {(0, 100.0): 44, (0, 45.0): 204, (1, 100.0): 5, (1, 45.0): 4,
        (2, 100.0): 11, (2, 45.0): 45, (3, 100.0): 204, (3, 45.0): 204}


@pytest.mark.parametrize('seed,hp', sorted(_H20))
def test_defaults_reproduce_h20(seed, hp):
    j, _r, _t = search_nash_openloop(_snap(foe_hp=hp), NashOLConfig(n_sims=120, seed=seed),
                                     ranges_fn=_fixed(30 + 10 * seed))
    assert j == _H20[(seed, hp)]


def test_rm_plus_solves_small_games():
    pennies = np.array([[1.0, -1.0], [-1.0, 1.0]])
    s, t, v, _st = regret_matching_plus(pennies, T=2000)
    assert np.allclose(s, 0.5, atol=0.02) and np.allclose(t, 0.5, atol=0.02) and abs(v) < 0.02
    rps = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    s, t, v, _st = regret_matching_plus(rps, T=2000)
    assert np.allclose(s, 1 / 3, atol=0.03) and abs(v) < 0.02
    dom = np.array([[0.3, 0.6], [0.1, 0.2]])   # fila 0 domina; la columna 0 minimiza
    s, t, v, _st = regret_matching_plus(dom, T=500)
    assert s[0] > 0.95 and t[0] > 0.95 and abs(v - 0.3) < 0.02


def test_rm_plus_warm_start_accumulates():
    rng = np.random.default_rng(0)
    A = rng.uniform(-1, 1, (6, 4))
    _s, _t, v_ref = fictitious_play(A, T=3000)
    st = None
    for _ in range(100):  # 100 backups × 10 iteraciones, como dentro del árbol
        s, t, v, st = regret_matching_plus(A, T=10, state=st)
    assert st.t == 1000 and abs(v - v_ref) < 0.03


def test_small_matrix_and_visits_decision():
    cfg = NashOLConfig(n_sims=120, seed=0, k_self_root=8, k_self=6, k_opp=4, decision='visits')
    j, root, _t = search_nash_openloop(_snap(), cfg, ranges_fn=_fixed(30))
    assert len(root.I) <= 8 and len(root.J) <= 4
    vis = root.n.sum(axis=1)
    assert vis[root.I.index(j)] == vis.max()


def test_rm_plus_search_runs():
    j, root, tel = search_nash_openloop(_snap(), NashOLConfig(n_sims=120, seed=0, solver='rm+'),
                                        ranges_fn=_fixed(30))
    assert tel.failures == 0 and tel.sims == 120 and j in root.I and root.rm is not None


def test_player_env_knobs(monkeypatch):
    from showdown.rl.mcts_puct_nash.player import _h21_knobs
    assert _h21_knobs() == {}
    monkeypatch.setenv('MCTS_NASH_MATRIX', '8,6,4')
    monkeypatch.setenv('MCTS_NASH_DECISION', 'visits')
    monkeypatch.setenv('MCTS_NASH_SOLVER', 'rm+')
    assert _h21_knobs() == {'k_self_root': 8, 'k_self': 6, 'k_opp': 4, 'decision': 'visits', 'solver': 'rm+'}
    monkeypatch.setenv('MCTS_NASH_SOLVER', 'cfr')
    with pytest.raises(ValueError):
        _h21_knobs()
