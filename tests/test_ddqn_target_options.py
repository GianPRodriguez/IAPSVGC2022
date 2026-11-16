"""Opciones contra la sobreestimación de Q (piloto 15-11-2026): copia dura y objetivo acotado."""
import sys

import numpy as np
import pytest

sys.path.insert(0, '.')

torch = pytest.importorskip('torch')

from showdown.rl.constants import JOINT_ACTION_N, OBS_DIM
from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig


def _agent(**kw):
    ag = DDQNAgent(DDQNConfig(device='cpu', replay_device='cpu', replay_capacity=512, seed=1,
                              batch_size=16, **kw))
    rng = np.random.default_rng(0)
    for k in range(64):
        s = (rng.random(OBS_DIM) < 0.1).astype(np.float32)
        m = np.zeros(JOINT_ACTION_N, dtype=np.float32)
        m[:5] = 1
        ag.replay.add(s=s, a=int(rng.integers(0, 5)), r=float(rng.normal() * 5), s_next=s,
                      terminated=bool(k % 5 == 0), m_next=m)
    return ag


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


def test_defaults_keep_previous_behaviour():
    cfg = DDQNConfig()
    assert cfg.target_update_every == 0 and cfg.target_clip == 0.0


def test_hard_update_copies_only_every_n_steps():
    ag = _agent(target_update_every=3)
    ag.learn()
    assert not _same(ag.online, ag.target)      # paso 1: el objetivo no se toca
    ag.learn()
    assert not _same(ag.online, ag.target)      # paso 2
    ag.learn()
    assert _same(ag.online, ag.target)          # paso 3: copia dura


def test_soft_update_moves_target_every_step():
    ag = _agent()
    before = [p.detach().clone() for p in ag.target.parameters()]
    ag.learn()
    assert any(not torch.equal(b, p) for b, p in zip(before, ag.target.parameters()))


def test_target_clip_bounds_the_target():
    ag = _agent(target_clip=2.0)
    out = ag.learn()
    assert abs(out['mean_target']) <= 2.0 + 1e-6
