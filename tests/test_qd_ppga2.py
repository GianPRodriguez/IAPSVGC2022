"""QD-PPGA 2.0 (agente K 2.0 = PPGA fiel; docs/qd_ppga_spec.md §K 2.0)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from showdown.rl.qd_ppga.ppga import (  # noqa: E402
    SoftArchive, XNES, apply_step, delta_norm, normalized, objective_reward, param_delta,
)
from showdown.rl.vec_env import bd_step_proxies  # noqa: E402


def test_xnes_utilities_and_convergence_on_toy_problem():
    u = XNES.utilities(8)
    assert abs(u.sum()) < 1e-12 and u[0] > u[1] > 0 > u[-1]
    target = np.array([1.5, -0.5, 0.8])
    es = XNES(3, sigma0=1.0, seed=0)
    for _ in range(150):
        x = es.ask(8)
        es.tell([-float(np.sum((xi - target) ** 2)) for xi in x])
    assert np.linalg.norm(es.mu - target) < 0.2
    assert es.sigma < 1.0                      # el paso se reduce al acercarse
    es.reset()
    assert np.allclose(es.mu, 0) and es.sigma == 1.0 and np.allclose(es.B, np.eye(3))


def test_soft_archive_cma_mae_rule():
    sa = SoftArchive(alpha=0.1, t0=-1.0)
    assert sa.update((0, 0), 0.5) == pytest.approx(1.5)        # celda vacía: umbral −1
    assert sa.t[(0, 0)] == pytest.approx(0.9 * -1.0 + 0.1 * 0.5)
    assert sa.update((0, 0), -0.9) == 0.0                      # bajo el umbral: Δ = 0, sin cambio
    t = sa.t[(0, 0)]
    assert sa.update((0, 0), 0.0) == pytest.approx(0.0 - t)    # mejora pequeña sobre el umbral suave


def test_branch_arithmetic_param_deltas():
    base = {'w': torch.zeros(3), 'value.w': torch.zeros(2), 'n': torch.tensor(1)}
    new_f = {'w': torch.tensor([3.0, 4.0, 0.0]), 'value.w': torch.ones(2), 'n': torch.tensor(1)}
    d = param_delta(new_f, base)
    assert set(d) == {'w'}                                     # solo el actor
    g, n = normalized(d)
    assert n == pytest.approx(5.0) and delta_norm(g) == pytest.approx(1.0)
    br = apply_step(base, [g], [2.0], scale=n)                 # θ_μ + s·c·ĝ = θ_μ + 2·Δ
    assert torch.allclose(br['w'], torch.tensor([6.0, 8.0, 0.0]))
    assert torch.equal(br['value.w'], base['value.w'])


def test_markov_proxies_and_objective_rewards():
    assert bd_step_proxies((0, 0, 0, 0), (2, 0, 1, 1)) == (1.0, 1.0)
    assert bd_step_proxies((0, 0, 0, 0), (1, 1, 1, 0)) == (0.5, 0.0)
    assert bd_step_proxies((5, 5, 3, 3), (5, 5, 3, 3)) == (0.5, 0.5)   # turno sin acciones contadas
    r = {'reward': 0.2, 'info': {'bd_step': (0.25, 1.0)}}
    assert objective_reward('f')(r) == pytest.approx(0.2)
    assert objective_reward('m1')(r) == pytest.approx(0.25)
    assert objective_reward('m2')(r) == pytest.approx(1.0)
    assert objective_reward('mix', np.array([1.0, -2.0, 0.5]))(r) == pytest.approx(0.2 - 0.5 + 0.5)
    assert objective_reward('m1')({'reward': 0.0, 'info': {}}) == 0.5   # sin proxy: neutro


def test_train_k2_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv('RL_OBS_CALC', '0')
    from showdown.rl.qd_ppga.train_k2 import run_training
    s = run_training(seed=0, smoke=True, checkpoint_dir=tmp_path / 'k2')
    assert s['iterations'] >= 1 and s['branches'] >= 1 and len(s['xnes_mu']) == 3
    assert (tmp_path / 'k2' / 'best_elite.pt').is_file()
