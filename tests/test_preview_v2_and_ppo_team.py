"""Preview v2 (90 formaciones canónicas, equipo propio fijo) y equipo fijo / guardia de PPO."""
import sys

import numpy as np
import pytest

sys.path.insert(0, '.')

torch = pytest.importorskip('torch')


def test_canonical_mask_has_90_formations_one_per_class():
    from showdown.rl.team_preview_action import N_ACTIONS, canonical_mask, decode_preview_action
    m = np.asarray(canonical_mask())
    assert m.shape == (N_ACTIONS,) and int(m.sum()) == 90
    classes = {}
    for a in range(N_ACTIONS):
        l1, l2, b1, b2 = decode_preview_action(a)
        classes.setdefault((frozenset((l1, l2)), frozenset((b1, b2))), []).append(a)
    assert len(classes) == 90 and all(len(v) == 4 for v in classes.values())
    assert all(sum(m[a] for a in v) == 1 for v in classes.values())   # un representante por clase


def _agent(canonical):
    from showdown.rl.team_preview_ppo.agent import PreviewPPOAgent, PreviewPPOConfig
    return PreviewPPOAgent(PreviewPPOConfig(device='cpu', canonical=canonical))


def test_canonical_agent_only_picks_canonical_actions():
    from showdown.rl.team_preview_action import canonical_mask, legal_mask
    from showdown.rl.team_preview_obs import OBS_DIM
    canon = np.asarray(canonical_mask())
    ag = _agent(True)
    obs = np.random.default_rng(0).random(OBS_DIM).astype(np.float32)
    acts = {ag.act(obs, np.asarray(legal_mask(), dtype=np.float32))[0] for _ in range(200)}
    assert acts and all(canon[a] == 1 for a in acts)


def test_canonical_flag_survives_save_and_load(tmp_path):
    """Un preview entrenado con 90 formaciones las sigue usando al cargarlo congelado."""
    from showdown.rl.team_preview_action import canonical_mask, legal_mask
    from showdown.rl.team_preview_obs import OBS_DIM
    from showdown.rl.team_preview_ppo.agent import PreviewPPOAgent
    ag = _agent(True)
    ag.save(tmp_path / 'p.pt')
    fresh = PreviewPPOAgent()                     # como frozen_preview: config por defecto
    fresh.load(tmp_path / 'p.pt')
    assert fresh.cfg.canonical is True
    canon = np.asarray(canonical_mask())
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    a, _lp, _v = fresh.act(obs, np.asarray(legal_mask(), dtype=np.float32), greedy=True)
    assert canon[a] == 1


def test_default_preview_agent_keeps_360(tmp_path):
    assert _agent(False).cfg.canonical is False


def test_preview_env_fixed_own_team(monkeypatch):
    from showdown.rl import team_preview_env as TPE
    from showdown.rl.ddqn.train import DEFAULT_AGENT_TEAM
    monkeypatch.setenv('PS_PREVIEW_OWN_TEAM', DEFAULT_AGENT_TEAM)
    own = TPE._fixed_own_team()
    assert own is not None and own.mons[0].species == 'Raichu'
    monkeypatch.delenv('PS_PREVIEW_OWN_TEAM')
    assert TPE._fixed_own_team() is None


def test_ppo_fixed_team_default_and_envspec():
    import inspect
    from showdown.rl.ppo import train as PT
    assert PT.DEFAULT_AGENT_TEAM.endswith('ddqn_team.txt')
    assert inspect.signature(PT.run_training).parameters['agent_team'].default == PT.DEFAULT_AGENT_TEAM


def test_ppo_player_counts_exceptions(monkeypatch):
    import logging
    from poke_env.battle import DoubleBattle
    from poke_env.player.battle_order import DefaultBattleOrder
    from showdown.rl.decision_guard import DecisionStats
    from showdown.rl.ppo import player as mod
    monkeypatch.delenv('PS_RL_STRICT', raising=False)
    monkeypatch.setattr(mod, 'snapshot_from_battle',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('roto')))
    monkeypatch.setattr(mod, 'player_stat_beliefs', lambda _p: None)
    p = object.__new__(mod.PPOPlayer)
    p.decision_stats = DecisionStats(agent='ppo')
    p.greedy = True
    out = mod.PPOPlayer.choose_move(p, DoubleBattle('b-1', 'u', logging.getLogger('t'), gen=9))
    assert isinstance(out, DefaultBattleOrder) and p.decision_stats.exceptions == 1
    monkeypatch.setenv('PS_RL_STRICT', '1')
    with pytest.raises(RuntimeError):
        mod.PPOPlayer.choose_move(p, DoubleBattle('b-2', 'u', logging.getLogger('t'), gen=9))
