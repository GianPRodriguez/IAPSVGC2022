"""PPO: validación periódica, best.pt, métricas en train_metrics.jsonl y reanudación."""
import json
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, '.')

torch = pytest.importorskip('torch')

from showdown.rl.ppo import train as PT


def _rows(wrs):
    return [SimpleNamespace(foe=f, n_games=10, wins=0, losses=0, ties=0, mean_return=0.0, win_rate=w)
            for f, w in zip(PT.VAL_FOES, wrs)]


def test_ppo_validator_logs_updates_and_keeps_best(tmp_path, monkeypatch):
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    from showdown.rl.train_metrics import MetricsLog
    scores = iter([[0.9, 0.5, 0.4, 0.3], [0.8, 0.2, 0.1, 0.1]])
    monkeypatch.setattr(PT, 'run_suite_ppo_act', lambda *a, **k: (_rows(next(scores)), ''))
    monkeypatch.setattr(PT, '_make_ppo_eval_env', lambda **kw: SimpleNamespace(close=lambda: None))
    mlog = MetricsLog(tmp_path / 'train_metrics.jsonl')
    v = PT._PPOValidator(PPOAgent(PPOConfig(device='cpu')), tmp_path, mlog, every=100, games=10,
                         start_ps=False, port=None, log_every=40)
    for g in (30, 45, 100, 130, 210):
        v.hook(g, {'loss': 0.1, 'entropy': 2.0, 'approx_kl': 0.01})
    assert [h['step'] for h in v.history] == [100, 210] and v.best['step'] == 100
    assert (tmp_path / 'best.pt').is_file()
    recs = [json.loads(l) for l in open(tmp_path / 'train_metrics.jsonl')]
    assert [r['step'] for r in recs if r['kind'] == 'update'] == [45, 100, 130, 210]
    assert sum(r['kind'] == 'eval' for r in recs) == 2


def test_ppo_checkpoint_keeps_optimizer(tmp_path):
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    a = PPOAgent(PPOConfig(device='cpu'))
    a.optim.state[next(iter(a.net.parameters()))] = {'step': torch.tensor(3.0)}
    a.env_steps = 777
    a.save(tmp_path / 'p.pt')
    b = PPOAgent(PPOConfig(device='cpu'))
    b.load(tmp_path / 'p.pt')
    assert b.env_steps == 777 and b.optim.state_dict()['state']


def test_ppo_resume_continues_schedule(tmp_path):
    PT.run_training(seed=0, schedule=[PT.TrainPhase(opponent='mock', steps=120)], device='cpu',
                    checkpoint_dir=tmp_path / 'a', suite_games=0, eval_every=0, agent_team='')
    ag = PT.run_training(seed=0, schedule=[PT.TrainPhase(opponent='mock', steps=300)], device='cpu',
                         checkpoint_dir=tmp_path / 'a', suite_games=0, eval_every=0, agent_team='',
                         resume=str(tmp_path / 'a' / 'final.pt'))
    rep = json.loads((tmp_path / 'a' / 'reporte.json').read_text(encoding='utf-8'))
    assert rep['global_steps'] == 300
    assert ag.env_steps >= 300
