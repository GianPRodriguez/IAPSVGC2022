"""Validación periódica, mejor checkpoint y reanudación de DDQN (ddqn_spec 1.16)."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, '.')

torch = pytest.importorskip('torch')

from showdown.rl.ddqn import train as T
from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig


def _agent():
    return DDQNAgent(DDQNConfig(device='cpu', replay_device='cpu', replay_capacity=256, seed=0))


def _rows(wrs):
    return [SimpleNamespace(foe=f, n_games=10, wins=int(w * 10), losses=10 - int(w * 10), ties=0,
                            mean_return=0.0, win_rate=w) for f, w in zip(T.VAL_FOES, wrs)]


def test_validator_runs_on_schedule_and_keeps_best(tmp_path, monkeypatch):
    from showdown.rl import train_report
    from showdown.rl.train_metrics import MetricsLog
    scores = iter([[0.5, 0.1, 0.0, 0.0], [0.6, 0.3, 0.1, 0.1], [0.4, 0.1, 0.0, 0.0]])
    calls = []

    def fake_suite(agent, env, *, foes, n_games, seed):
        calls.append((tuple(foes), n_games, seed))
        return _rows(next(scores)), ''
    monkeypatch.setattr(train_report, 'run_suite_ddqn_act', fake_suite)
    monkeypatch.setattr(T, '_make_eval_env', lambda **kw: SimpleNamespace(close=lambda: None))
    mlog = MetricsLog(tmp_path / 'train_metrics.jsonl')
    v = T._Validator(_agent(), tmp_path, mlog, every=100, games=10, start_ps=False, port=None)
    for g in (50, 99, 100, 150, 230, 299, 300):
        v.maybe(g)
    assert [h['step'] for h in v.history] == [100, 230, 300]
    assert v.best['step'] == 230                                   # el mejor, no el último
    assert json.loads((tmp_path / 'best.json').read_text())['step'] == 230
    assert (tmp_path / 'best.pt').is_file() and (tmp_path / 'step_300.pt').is_file()
    assert all(c == (T.VAL_FOES, 10, T.VAL_SEED) for c in calls)  # mismas semillas siempre
    evals = [json.loads(l) for l in open(tmp_path / 'train_metrics.jsonl') if '"eval"' in l]
    assert len(evals) == 3


def test_validation_uses_deterministic_foes_and_four_heuristics():
    assert T.VAL_FOES == ('random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian')
    from showdown.rl.pokeenv_backend import PokeEnvBattleBackend
    assert PokeEnvBattleBackend(opponent='greedy_joint', foe_tie=False)._foe_tie is False
    assert PokeEnvBattleBackend(opponent='greedy_joint')._foe_tie is None   # entrenamiento: sorteo


def test_checkpoint_keeps_optimizer_state(tmp_path):
    ag = _agent()
    ag.optim.state[next(iter(ag.online.parameters()))] = {'step': torch.tensor(7.0)}
    ag.env_steps, ag.grad_steps = 1234, 56
    ag.save(tmp_path / 'c.pt')
    other = _agent()
    other.load(tmp_path / 'c.pt')
    assert other.env_steps == 1234 and other.grad_steps == 56
    assert other.optim.state_dict()['state'], 'el estado de Adam debe restaurarse'


def test_resume_hold_blocks_learning_until_replay_refills():
    ag = _agent()
    ag.cfg.warmup_steps, ag.cfg.batch_size, ag.cfg.train_freq = 0, 8, 1
    ag.env_steps = 10_000
    ag.min_replay_to_learn = 50
    s = np.zeros(ag.replay.obs_dim, dtype=np.float32)
    m = np.zeros(1600, dtype=np.float32)
    m[0] = 1
    outs = [ag.observe(s=s, a=0, r=0.0, s_next=s, terminated=False, m_next=m) for _ in range(60)]
    assert all(o is None for o in outs[:49]) and any(o for o in outs[49:])


def test_resume_continues_the_schedule(tmp_path):
    """Corre 120 pasos (mock), reanuda con un currículo de 300: hace solo los 180 que faltan."""
    sched = [T.TrainPhase(opponent='mock', steps=300)]
    T.run_training(seed=0, schedule=[T.TrainPhase(opponent='mock', steps=120)], device='cpu',
                   checkpoint_dir=tmp_path / 'a', suite_games=0, eval_every=0, agent_team='')
    ag = T.run_training(seed=0, schedule=sched, device='cpu', checkpoint_dir=tmp_path / 'a',
                        suite_games=0, eval_every=0, agent_team='',
                        resume=str(tmp_path / 'a' / 'final.pt'))
    assert ag.env_steps == 300
