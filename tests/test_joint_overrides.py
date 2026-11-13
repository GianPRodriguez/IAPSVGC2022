import importlib

import pytest


def test_joint_overrides_apply_and_reject_unknown(monkeypatch):
    from showdown.battle_bots.greedy_joint import joint
    monkeypatch.setenv('JOINT_OVERRIDES', '_OPP_THREAT=0;_RESIDUAL_SCALE=0.2')
    try:
        importlib.reload(joint)
        assert joint._OPP_THREAT is False and joint._RESIDUAL_SCALE == 0.2
        monkeypatch.setenv('JOINT_OVERRIDES', '_NO_EXISTE=1')
        with pytest.raises(ValueError):
            joint._apply_env_overrides()
    finally:
        monkeypatch.delenv('JOINT_OVERRIDES', raising=False)
        importlib.reload(joint)
    assert joint._OPP_THREAT is True
