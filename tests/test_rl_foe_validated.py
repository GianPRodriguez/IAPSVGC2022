"""Rivales heurísticos del entrenamiento RL con wrapper validated (como run_robust_e2e)."""
import sys

sys.path.insert(0, '.')

import pytest

from showdown.rl import pokeenv_backend as pb


class _FakeValidated:
    def __init__(self, **kw):
        self.kw = kw
        self.resets = []

    def reset_for_next_battle(self, user_team, opp_team):
        self.resets.append((user_team, opp_team))


@pytest.fixture
def fake_validated(monkeypatch):
    from showdown.inference_sim import validated_greedy_player as vgp

    monkeypatch.delenv('PS_RL_FOE_VALIDATED', raising=False)
    monkeypatch.setattr(vgp, 'validated_player_class', lambda kind: _FakeValidated)
    monkeypatch.setattr(vgp, 'ValidatedBattleMixin', _FakeValidated)
    return _FakeValidated


def _metas():
    a, b = pb._default_team_metas(seed=1)
    return a, b


def test_heuristic_foes_are_validated_with_both_teams(fake_validated):
    foe_meta, agent_meta = _metas()
    for kind in ('greedy_solo', 'greedy_joint', 'expectimax_bayesian'):
        foe = pb._make_foe_player(
            kind, account_configuration=None, battle_format='x', team='t',
            server_configuration=None, log_level=40,
            foe_meta=foe_meta, agent_meta=agent_meta,
        )
        assert isinstance(foe, fake_validated)
        # El rival juega con SU equipo y su "opp_team" es el del agente RL.
        assert foe.kw['user_team'] is foe_meta and foe.kw['opp_team'] is agent_meta
        assert foe.kw['soundness_report'] is None


def test_raw_foe_when_disabled_or_without_team_meta(fake_validated, monkeypatch):
    foe_meta, agent_meta = _metas()
    assert pb.foe_validated('greedy_joint') is True
    assert pb.foe_validated('random') is False
    monkeypatch.setenv('PS_RL_FOE_VALIDATED', '0')
    assert pb.foe_validated('greedy_joint') is False


def test_team_meta_detection():
    foe_meta, _ = _metas()
    assert pb._team_meta(foe_meta) is foe_meta
    assert pb._team_meta('Garchomp @ Choice Scarf') is None
    assert pb._team_meta(None) is None


def test_backend_rebuilds_foe_when_validation_mode_changes(fake_validated):
    foe_meta, agent_meta = _metas()
    be = pb.PokeEnvBattleBackend.__new__(pb.PokeEnvBattleBackend)
    be._opponent = 'greedy_joint'
    be._foe = object()  # rival crudo
    assert be._foe_needs_rebuild(foe_meta=foe_meta, agent_meta=agent_meta) is True
    be._foe = fake_validated()
    assert be._foe_needs_rebuild(foe_meta=foe_meta, agent_meta=agent_meta) is False
    # Equipos como export (sin TeamMeta): el validated no puede seguir.
    assert be._foe_needs_rebuild(foe_meta=None, agent_meta=None) is True
    be._update_foe_team('export', foe_meta=foe_meta, agent_meta=agent_meta)
    assert be._foe.resets == [(foe_meta, agent_meta)]


def test_training_foes_get_tie_sampling_by_default(fake_validated, monkeypatch):
    """Rivales de entrenamiento: joint margen 8, expectimax fase 1 margen 3 (diversidad_baselines.md)."""
    monkeypatch.delenv('PS_RL_FOE_TIE', raising=False)
    foe_meta, agent_meta = _metas()
    kw = dict(account_configuration=None, battle_format='x', team='t',
              server_configuration=None, log_level=40, foe_meta=foe_meta, agent_meta=agent_meta)
    joint = pb._make_foe_player('greedy_joint', **kw)
    assert joint.tie_margin == 8.0
    ex = pb._make_foe_player('expectimax_bayesian', **kw)
    assert ex.expectimax_config.phase1_tie_margin == 3.0
    assert ex.expectimax_config.tie_sample_margin == 0.0  # la variante cara nunca se activa
    solo = pb._make_foe_player('greedy_solo', **kw)
    assert not hasattr(solo, 'tie_margin')


def test_tie_sampling_can_be_disabled(fake_validated, monkeypatch):
    foe_meta, agent_meta = _metas()
    kw = dict(account_configuration=None, battle_format='x', team='t',
              server_configuration=None, log_level=40, foe_meta=foe_meta, agent_meta=agent_meta)
    monkeypatch.setenv('PS_RL_FOE_TIE', '0')
    assert pb.foe_tie_enabled() is False
    assert not hasattr(pb._make_foe_player('greedy_joint', **kw), 'tie_margin')
    monkeypatch.delenv('PS_RL_FOE_TIE')
    assert not hasattr(pb._make_foe_player('greedy_joint', tie=False, **kw), 'tie_margin')


def test_joint_player_instance_margin_overrides_env(monkeypatch):
    from showdown.battle_bots.greedy_joint.pokeenv_player import GreedyJointPlayer
    assert GreedyJointPlayer.tie_margin is None  # por defecto determinista (o PS_JOINT_TIE_MARGIN)
