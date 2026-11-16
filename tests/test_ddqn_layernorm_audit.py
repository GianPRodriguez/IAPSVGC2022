"""LayerNorm opcional (punto 7) y auditoría de la observación."""
import sys

import numpy as np
import pytest

sys.path.insert(0, '.')

torch = pytest.importorskip('torch')

from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig


def test_layernorm_off_by_default_and_on_when_asked():
    a = DDQNAgent(DDQNConfig(device='cpu', replay_device='cpu', replay_capacity=64))
    b = DDQNAgent(DDQNConfig(device='cpu', replay_device='cpu', replay_capacity=64, layer_norm=True))
    assert a.online.layer_norm is False and isinstance(a.online.norm, torch.nn.Identity)
    assert b.online.layer_norm is True and isinstance(b.online.norm, torch.nn.LayerNorm)


def test_checkpoint_with_layernorm_loads_into_default_agent(tmp_path):
    b = DDQNAgent(DDQNConfig(device='cpu', replay_device='cpu', replay_capacity=64, layer_norm=True))
    b.save(tmp_path / 'ln.pt')
    a = DDQNAgent(DDQNConfig(device='cpu', replay_device='cpu', replay_capacity=64))
    a.load(tmp_path / 'ln.pt')                     # reconstruye la red con LayerNorm
    assert a.online.layer_norm is True
    for p, q in zip(a.online.parameters(), b.online.parameters()):
        assert torch.equal(p, q)


def test_obs_audit_summary_flags_constant_and_large_dims():
    from showdown.rl.constants import OBS_DIM
    from showdown.rl.obs_audit import summarize
    x = np.zeros((10, OBS_DIM), dtype=np.float32)
    x[:, 5] = np.arange(10)            # varía y llega a 9 (> 5)
    x[:, 6] = 0.3                      # constante distinta de cero
    rep = summarize(x)
    assert rep['dims'] == OBS_DIM and rep['muestras'] == 10
    assert rep['constantes_total'] == OBS_DIM - 1
    assert rep['fuera_5_total'] == 1 and rep['top_max_abs'][0]['idx'] == 5
