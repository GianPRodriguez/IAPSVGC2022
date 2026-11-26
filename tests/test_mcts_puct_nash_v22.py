"""H 2.2 (protocolo §4.8, spec de H §4d): backup mixto por entorno; con λ = 0, H3 y B2 sin cambios."""
from __future__ import annotations

import numpy as np
import pytest

from showdown.rl.mcts_puct_nash.openloop_nash import NashNode, NashOLConfig, _backup, search_nash_openloop
from test_mcts_puct_nash_v2 import _fixed, _snap

_H3 = dict(k_self_root=8, k_self=6, k_opp=4, decision='visits')
# Joints con el código congelado de H 2.1 (4db8267 / d8e4bb6), n_sims=120, ranges 30+10·seed.
_REF = {('h3', 0, 100.0): 164, ('h3', 0, 45.0): 164, ('h3', 1, 100.0): 11, ('h3', 1, 45.0): 4,
        ('h3', 2, 100.0): 51, ('h3', 2, 45.0): 45, ('h3', 3, 100.0): 4, ('h3', 3, 45.0): 165,
        ('b2', 0, 100.0): 164, ('b2', 0, 45.0): 164, ('b2', 1, 100.0): 11, ('b2', 1, 45.0): 165,
        ('b2', 2, 100.0): 11, ('b2', 2, 45.0): 165, ('b2', 3, 100.0): 11, ('b2', 3, 45.0): 165}


@pytest.mark.parametrize('arm,seed,hp', sorted(_REF))
def test_h3_and_b2_unchanged(arm, seed, hp):
    kw = dict(_H3, solver='rm+') if arm == 'b2' else dict(_H3)
    j, _r, _t = search_nash_openloop(_snap(foe_hp=hp), NashOLConfig(n_sims=120, seed=seed, **kw),
                                     ranges_fn=_fixed(30 + 10 * seed))
    assert j == _REF[(arm, seed, hp)]


def _two_level(mix: float):
    parent, child = NashNode(), NashNode()
    for nd, a in ((parent, [[0.0, 0.0], [0.0, 0.0]]), (child, [[0.8, -0.2], [0.4, 0.6]])):
        nd.I, nd.J = [1, 2], [3, 4]
        nd.n = np.ones((2, 2))
        nd.W = np.array(a)
        nd.N, nd.W_sum = 4, float(np.sum(a))
    _backup([(parent, 0, 0), (child, 1, 1)], 0.6, 200, 'fp', mix)
    return parent, child


def test_backup_mix_formula():
    p0, c0 = _two_level(0.0)
    p5, c5 = _two_level(0.5)
    v_child = float(c0.sigma @ c0.payoff() @ c0.tau)            # valor de Nash del hijo (FP)
    mean_child = c5.W_sum / c5.N
    assert p0.W[0, 0] == pytest.approx(v_child, abs=0.02)
    assert p5.W[0, 0] == pytest.approx(0.5 * (p0.W[0, 0]) + 0.5 * mean_child, abs=1e-9)


def test_backup_mix_search_runs():
    cfg = NashOLConfig(n_sims=120, seed=0, backup_mix=0.5, **_H3)
    j, root, tel = search_nash_openloop(_snap(), cfg, ranges_fn=_fixed(30))
    assert tel.failures == 0 and tel.sims == 120 and j in root.I


def test_player_backup_mix_knob(monkeypatch):
    from showdown.rl.mcts_puct_nash.player import _h21_knobs
    monkeypatch.setenv('MCTS_NASH_BACKUP_MIX', '0.5')
    assert _h21_knobs() == {'backup_mix': 0.5}
    monkeypatch.setenv('MCTS_NASH_BACKUP_MIX', '1.5')
    with pytest.raises(ValueError):
        _h21_knobs()


def test_player_default_is_h22(monkeypatch):
    """Desde el 25-11-2026 (protocolo §5.9) el player usa H 2.2; las variables lo sobreescriben."""
    from showdown.rl.mcts_puct_nash.player import _nash_kwargs
    assert _nash_kwargs(2) == {'k_self_root': 8, 'k_self': 6, 'k_opp': 4, 'decision': 'visits', 'solver': 'rm+'}
    assert _nash_kwargs(1) == {}
    monkeypatch.setenv('MCTS_NASH_SOLVER', 'fp')
    assert _nash_kwargs(2)['solver'] == 'fp'  # = H 2.1
    monkeypatch.setenv('MCTS_NASH_MATRIX', '12,8,8')
    monkeypatch.setenv('MCTS_NASH_DECISION', 'sigma')
    assert _nash_kwargs(2) == {'k_self_root': 12, 'k_self': 8, 'k_opp': 8, 'decision': 'sigma', 'solver': 'fp'}  # = H 2.0
