"""Preview entrenado con su propio combate (pilotos RL en preview_live; memorizacion_equipos.md §15)."""
import pytest

from showdown.rl import preview_live as PL


def test_rl_pilots_registered_and_need_checkpoint(monkeypatch, tmp_path):
    assert {'ddqn', 'ppo'} <= set(PL.PILOTS)
    monkeypatch.delenv('PREVIEW_PILOT_CHECKPOINT', raising=False)
    with pytest.raises(FileNotFoundError):
        PL.rl_pilot_checkpoint('ddqn')
    ck = tmp_path / 'c.pt'
    ck.write_bytes(b'x')
    monkeypatch.setenv('PREVIEW_PILOT_CHECKPOINT', str(ck))
    assert PL.rl_pilot_checkpoint('ppo') == str(ck)
    assert PL.rl_pilot_checkpoint('greedy_joint') == str(ck)  # heurístico: no se exige


def test_foe_pilot_defaults_to_greedy_joint(monkeypatch):
    monkeypatch.delenv('PREVIEW_FOE_PILOT', raising=False)
    assert PL.foe_pilot_for('ddqn') == 'greedy_joint'
    assert PL.foe_pilot_for('expectimax_bayesian') == 'expectimax_bayesian'
    monkeypatch.setenv('PREVIEW_FOE_PILOT', 'expectimax_bayesian')
    assert PL.foe_pilot_for('ppo') == 'expectimax_bayesian'
    monkeypatch.setenv('PREVIEW_FOE_PILOT', 'ddqn')
    with pytest.raises(ValueError):
        PL.foe_pilot_for('ppo')


@pytest.mark.parametrize('pilot', ['ddqn', 'ppo'])
def test_rl_pilot_class_keeps_forced_order_and_turn_cap(pilot):
    cls = PL._pilot_class(pilot)
    assert hasattr(cls, 'forced_order') and cls.max_turns == 40
    assert cls.__mro__[1].__name__ in ('DDQNPlayer', 'PPOPlayer')
