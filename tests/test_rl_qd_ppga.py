"""Acceptance tests for QD-PPGA (docs/qd_ppga_spec.md)."""
from __future__ import annotations

import numpy as np

from showdown.rl.constants import SPEC_VERSION
from showdown.rl.qd_ppga.arborescence import sample_arborescence_parent
from showdown.rl.qd_ppga.archive import Archive, is_actor_key
from showdown.rl.qd_ppga.train import P_MUT, derived_n_outer


def test_spec_version_qd_ppga_aligned():
    assert SPEC_VERSION == 'v1.8'


def test_derived_n_outer_ppga():
    assert derived_n_outer() == 109
    assert derived_n_outer(t_total=1_000_000, b_boot=100_000, n_emit=4, t_emit=2048) == 109


def test_no_genetic_mutation_contract():
    assert float(P_MUT) == 0.0


def test_arborescence_empty_uses_fallback():
    arch = Archive()
    rng = np.random.default_rng(0)
    fb = {'fc1.weight': np.zeros(2)}
    sd, key, mode = sample_arborescence_parent(arch, rng, alpha=0.5, fallback_state=fb)
    assert mode == 'empty'
    assert key is None
    assert sd is not None


def test_arborescence_frontier_alpha_zero():
    arch = Archive()
    sd_a = {'fc1.weight': np.array([1.0])}
    sd_b = {'fc1.weight': np.array([2.0])}
    arch.try_insert(bd1=0.1, bd2=0.1, fitness=0.5, state_dict=sd_a)
    arch.try_insert(bd1=0.9, bd2=0.9, fitness=0.1, state_dict=sd_b)
    rng = np.random.default_rng(1)
    modes = set()
    for _ in range(40):
        _sd, _k, mode = sample_arborescence_parent(arch, rng, alpha=0.0)
        modes.add(mode)
    assert modes == {'frontier'}


def test_actor_only_keys():
    assert is_actor_key('policy1.weight')
    assert not is_actor_key('value.bias')


def test_smoke_train_mock():
    from showdown.rl.qd_ppga.train import run_training

    arch = run_training(seed=0, smoke=True, checkpoint_dir='logs/qd_ppga_test_smoke')
    assert isinstance(arch, Archive)
    assert arch.coverage() >= 0.0
