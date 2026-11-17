"""Valores Q por hueco en DDQN (memorizacion_equipos.md §10; ddqn_spec 1.21)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from showdown.rl.action_space import encode_joint_action  # noqa: E402
from showdown.rl.constants import JOINT_ACTION_N, OBS_DIM, SLOT_ACTION_N  # noqa: E402
from showdown.rl.ddqn.network import QNetwork  # noqa: E402


def test_joint_head_unchanged():
    q = QNetwork()
    keys = list(q.state_dict())
    assert 'net.4.weight' in keys and not any(k.startswith(('v.', 'a1.', 'a2.')) for k in keys)
    assert q(torch.zeros(2, OBS_DIM)).shape == (2, JOINT_ACTION_N)


@pytest.mark.parametrize('rank', [0, 4])
def test_factored_matches_formula_and_joint_index(rank):
    torch.manual_seed(0)
    q = QNetwork(q_head='factored', q_rank=rank).eval()
    x = torch.randn(3, OBS_DIM)
    out = q(x)
    assert out.shape == (3, JOINT_ACTION_N)
    with torch.no_grad():
        z = q._trunk(q.norm(q.encoder(x)))
        v, a1, a2 = q.v(z)[:, 0], q.a1(z), q.a2(z)
        for b, i, j in ((0, 0, 0), (1, 5, 33), (2, 39, 7)):
            ref = v[b] + a1[b, i] + a2[b, j]
            if rank:
                u1 = q.u1(z).view(3, SLOT_ACTION_N, rank)
                u2 = q.u2(z).view(3, SLOT_ACTION_N, rank)
                ref = ref + (u1[b, i] * u2[b, j]).sum()
            assert torch.allclose(out[b, encode_joint_action(i, j)], ref, atol=1e-5)


def test_factored_has_far_fewer_output_params():
    def head(m):
        return sum(p.numel() for n, p in m.named_parameters() if not n.startswith(('encoder', 'norm', 'net.0', 'net.2')))
    assert head(QNetwork(q_head='factored', q_rank=8)) < 0.4 * head(QNetwork())


def test_bad_head_raises():
    with pytest.raises(ValueError):
        QNetwork(q_head='nope')


def _fill(agent, n=40, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros(JOINT_ACTION_N, np.float32)
    mask[:200] = 1.0
    out = None
    for _ in range(n):
        s = rng.random(OBS_DIM).astype(np.float32)
        out = agent.observe(s=s, a=int(rng.integers(0, 200)), r=0.1, s_next=s,
                            terminated=False, m_next=mask) or out
    return out


def test_learn_save_load_and_legal_greedy(tmp_path, monkeypatch):
    monkeypatch.setenv('RL_REPLAY_DEVICE', 'cpu')
    monkeypatch.setenv('RL_SYM_AUG', '1')  # la simetría permuta índices conjuntos: debe seguir funcionando
    from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
    a = DDQNAgent(DDQNConfig(batch_size=8, warmup_steps=0, train_freq=1, q_head='factored', q_rank=4))
    out = _fill(a)
    assert np.isfinite(out['loss'])
    mask = np.zeros(JOINT_ACTION_N, np.float32)
    mask[[3, 77, 1201]] = 1.0
    assert a.greedy(np.zeros(OBS_DIM, np.float32), mask) in (3, 77, 1201)
    path = a.save(tmp_path / 'f.pt')
    b = DDQNAgent(DDQNConfig(device='cpu'))  # por defecto 'joint': se reconstruye al cargar
    b.load(path)
    assert b.online.q_head == 'factored' and b.online.q_rank == 4 and b.target.q_head == 'factored'
    x = torch.randn(2, OBS_DIM)
    a.online.eval(); b.online.eval()
    assert torch.allclose(a.online(x), b.online(x))


def test_old_checkpoint_loads_as_joint(tmp_path):
    from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
    a = DDQNAgent(DDQNConfig(device='cpu'))
    path = a.save(tmp_path / 'old.pt')
    payload = torch.load(path, weights_only=False)
    for k in ('q_head', 'q_rank'):
        payload['config'].pop(k)
    torch.save(payload, path)
    b = DDQNAgent(DDQNConfig(device='cpu', q_head='factored'))
    b.load(path)
    assert b.online.q_head == 'joint'


def test_cli_flags():
    import inspect
    from showdown.rl.ddqn import train as T
    sig = inspect.signature(T.run_training)
    assert sig.parameters['q_head'].default is None and sig.parameters['q_rank'].default is None
    src = inspect.getsource(T._main)
    assert '--q-head' in src and '--q-rank' in src
