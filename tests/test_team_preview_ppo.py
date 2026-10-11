"""Tests for Team Preview Masked PPO (spec v1.0)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from showdown.rl.team_preview_action import (
    N_ACTIONS,
    action_to_team_string,
    decode_preview_action,
    encode_preview_action,
    legal_mask,
)
from showdown.rl.team_preview_obs import OBS_DIM, PreviewObsBundle, encode_preview_obs, mon_feat_from_species


def test_action_roundtrip_all():
    for a in range(N_ACTIONS):
        tup = decode_preview_action(a)
        assert len(set(tup)) == 4
        assert encode_preview_action(tup) == a


def test_team_1234_is_action_zero():
    assert encode_preview_action((0, 1, 2, 3)) == 0
    assert action_to_team_string(0) == '1234'
    assert decode_preview_action(0) == (0, 1, 2, 3)


def test_legal_mask_full_and_small_pool():
    m6 = legal_mask(6)
    assert len(m6) == 360 and all(x == 1.0 for x in m6)
    m4 = legal_mask(4)
    assert sum(m4) == math.perm(4, 4)  # only using 0..3
    assert m4[encode_preview_action((0, 1, 2, 3))] == 1.0
    assert m4[encode_preview_action((0, 1, 2, 5))] == 0.0


def test_obs_dim_512():
    our = [mon_feat_from_species(n) for n in [
        'Incineroar', 'Flutter Mane', 'Amoonguss', 'Tornadus',
        'Landorus-Therian', 'Rillaboom',
    ]]
    foe = [mon_feat_from_species(n) for n in [
        'Kyogre', 'Groudon', 'Chi-Yu', 'Urshifu', 'Whimsicott', 'Indeedee-F',
    ]]
    obs = encode_preview_obs(PreviewObsBundle(our=our, foe=foe))
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    # ctx weather zeros
    assert np.allclose(obs[420 + 36: 420 + 36 + 9], 0.0)


def test_env_smoke_episode():
    gymnasium = pytest.importorskip('gymnasium')
    del gymnasium
    from showdown.rl.team_preview_env import TeamPreviewEnv

    env = TeamPreviewEnv(seed=0, backend='mock')
    obs, info = env.reset()
    assert obs.shape == (512,)
    assert info['action_mask'].sum() == 360
    a = 0
    _o2, r, term, trunc, info2 = env.step(a)
    assert term and not trunc
    assert r in (-1.0, 0.0, 1.0)
    assert info2['team'] == '1234'


def test_ppo_smoke_train():
    torch = pytest.importorskip('torch')
    del torch
    from showdown.rl.team_preview_ppo.train import train

    agent = train(seed=0, smoke=True, updates=1, device='cpu')
    assert agent.env_episodes == 64
    assert agent.update_steps == 1
