"""Remedios contra la memorización en DDQN (memorizacion_equipos.md §8; ddqn_spec 1.19)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from showdown.rl.constants import JOINT_ACTION_N, MON_BLOCK_DIM, OBS_DIM  # noqa: E402
from showdown.rl.ddqn import regularize as REG  # noqa: E402
from showdown.rl.observation import obs_layout  # noqa: E402


def test_bench_columns_cover_only_inactive_foes():
    lay = obs_layout()
    cols = REG.bench_columns()
    assert len(cols) == 4
    allc = np.concatenate(cols)
    assert len(set(allc.tolist())) == allc.size  # disjuntas
    for k, c in enumerate(cols):
        a, b = lay[f'foe_{k + 2}']
        assert set(range(a, b)) <= set(c.tolist())
        assert c.size == MON_BLOCK_DIM + 12  # bloque + 3 pares × 4 Pokémon propios
    protected = set()
    for name in ('foe_0', 'foe_1', 'our_0', 'our_1', 'our_2', 'our_3', 'our_moves', 'foe_moves', 'field'):
        protected |= set(range(*lay[name]))
    assert not protected & set(allc.tolist())
    p0, p1 = lay['pairs']
    pairs = [x for x in allc.tolist() if p0 <= x < p1]
    assert len(pairs) == 4 * 12 and all(((x - p0) % 6) >= 2 for x in pairs)  # columnas j = 2..5


def test_numpy_and_torch_versions_match():
    s = np.random.default_rng(0).random((64, OBS_DIM)).astype(np.float32)
    a, fa = REG.foe_bench_dropout(s, 0.5, np.random.default_rng(7))
    b, fb = REG.foe_bench_dropout_t(torch.as_tensor(s), 0.5, np.random.default_rng(7))
    assert np.allclose(a, b.numpy()) and fa == fb and 0.3 < fa < 0.7
    assert not np.allclose(a, s)
    untouched = np.setdiff1d(np.arange(OBS_DIM), np.concatenate(REG.bench_columns()))
    assert np.array_equal(a[:, untouched], s[:, untouched])


def test_dropout_keeps_state_dict_and_is_off_in_eval():
    from showdown.rl.ddqn.network import QNetwork
    q0, q1 = QNetwork(), QNetwork(dropout=0.5)
    assert list(q0.state_dict()) == list(q1.state_dict())  # checkpoints intercambiables
    q1.load_state_dict(q0.state_dict())
    x = torch.randn(8, OBS_DIM)
    q0.eval(); q1.eval()
    assert torch.allclose(q0(x), q1(x))
    q1.train()
    assert not torch.allclose(q1(x), q1(x))


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


def test_defaults_unchanged_and_regularized_learn_runs(tmp_path, monkeypatch):
    monkeypatch.setenv('RL_REPLAY_DEVICE', 'cpu')
    from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
    base = dict(batch_size=8, warmup_steps=0, train_freq=1, sym_aug=False)
    torch.manual_seed(0)  # los pesos iniciales dependen del generador global de torch
    a = DDQNAgent(DDQNConfig(**base))
    torch.manual_seed(0)
    b = DDQNAgent(DDQNConfig(**base, weight_decay=0.0, dropout=0.0, foe_bench_dropout=0.0))
    oa, ob = _fill(a), _fill(b)
    assert oa['loss'] == ob['loss'] and 'foe_bench_dropped' not in oa  # 0 = sin cambios
    assert type(a.optim).__name__ == 'Adam'

    r = DDQNAgent(DDQNConfig(**base, weight_decay=1e-4, dropout=0.2, foe_bench_dropout=0.5))
    assert type(r.optim).__name__ == 'AdamW'
    out = _fill(r)
    assert np.isfinite(out['loss']) and 0.0 < out['foe_bench_dropped'] <= 1.0
    path = r.save(tmp_path / 'r.pt')
    c = DDQNAgent(DDQNConfig(device='cpu'))
    c.load(path)
    assert c.cfg.dropout == 0.2 and c.online.dropout == 0.2 and c.cfg.foe_bench_dropout == 0.5


def test_cli_flags_reach_config():
    import inspect
    from showdown.rl.ddqn import train as T
    sig = inspect.signature(T.run_training)
    for k in ('weight_decay', 'dropout', 'foe_bench_dropout'):
        assert k in sig.parameters and sig.parameters[k].default == 0.0
    src = inspect.getsource(T._main)
    assert '--foe-bench-dropout' in src and '--weight-decay' in src and '--dropout' in src
