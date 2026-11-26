"""QD-PGA 2.0 (agente J 2.0): PGA-ME fiel con acciones discretas — docs/qd_pga_map_elites_spec.md §J 2.0.

Por generación (QDax ``PGAMEEmitter``): padres uniformes; mitad Iso+LineDD, mitad rama de gradiente
(``n_pg`` pasos de gradiente de política esperado contra Q₁ con Adam nuevo) más el actor codicioso;
todos los hijos se evalúan (todas sus transiciones van al replay); inserción por comparación pareada
(regla común de I/J/K); luego ``n_critic`` pasos del crítico gemelo y ``n_critic/2`` del actor codicioso.
Arranque, calibración de descriptores, mosaico, reevaluación en ``val`` y despliegue: como I 2.0.
Presupuesto: pasos de entorno totales (toda la interacción es evaluación de hijos).

    python -m showdown.rl.qd_pga.train_j2 --seed 0 --smoke
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from showdown.rl.constants import OBS_DIM
from showdown.rl.ppo.agent import PPOAgent, PPOConfig
from showdown.rl.qd_pga.pgame import JointReplay, TwinCritic, iso_line, pg_steps, to_torch
from showdown.rl.qd_ppo.archive import Archive
from showdown.rl.qd_ppo.batch_insert import evaluate_insert_batch
from showdown.rl.qd_ppo.matchups import make_matchups, play_matchups
from showdown.rl.qd_ppo.mosaic import make_sampler
from showdown.rl.qd_ppo.train2 import (
    DEFAULT_TEAM, _agent_from, _make_env, _reeval, _state, calibrate_bounds,
)
from showdown.rl.train_log import TrainHeartbeat

BUDGET = 1_000_000
N_GA = 8
N_PG = 7
N_PG_STEPS = 100
PG_LR = 1e-4
GREEDY_LR = 1e-4
N_CRITIC = 300
CRITIC_LR = 3e-4
POLICY_DELAY = 2
WARMUP_CRITIC = 2000
REPLAY = 200_000
BATCH = 256
PAIRED_GAMES = 32
G_MIN = 16
CALIBRATE_GAMES = 512


def run_training(
    *, seed: int = 0, device: str = 'cpu', checkpoint_dir: str | Path = 'logs/qd_pga2',
    init_checkpoint: Optional[str] = None, budget: int = BUDGET, n_ga: int = N_GA, n_pg: int = N_PG,
    n_pg_steps: int = N_PG_STEPS, pg_lr: float = PG_LR, greedy_lr: float = GREEDY_LR,
    n_critic: int = N_CRITIC, critic_lr: float = CRITIC_LR, warmup_critic: int = WARMUP_CRITIC,
    replay_size: int = REPLAY, batch: int = BATCH, paired_games: int = PAIRED_GAMES, g_min: int = G_MIN,
    calibrate_games: int = CALIBRATE_GAMES, reeval_top: int = 10, reeval_games: int = 300,
    mosaic: str = 'ppo_v2', smoke: bool = False, start_ps: bool = False, agent_team: str = DEFAULT_TEAM,
    num_envs: int = 1, envs_per_server: int = 4, base_port: Optional[int] = None,
) -> Dict[str, Any]:
    import torch

    from showdown.rl.parallel import describe_resources, raise_nofile_limit, resolve_device

    raise_nofile_limit()
    device = resolve_device(device)
    if smoke:
        budget, n_ga, n_pg, n_pg_steps, n_critic = 600, 1, 1, 2, 4
        warmup_critic, batch, paired_games, g_min, calibrate_games = 4, 16, 3, 1, 4
        reeval_top, reeval_games, replay_size = 2, 3, 5000
    mock = bool(smoke or not start_ps)
    reeval_split = None if mock else 'val'

    agent = PPOAgent(PPOConfig(seed=seed, device=device))
    agent._ensure_torch()
    if init_checkpoint:
        agent.load(init_checkpoint)
    agent.cfg.seed, agent.cfg.device = int(seed), str(device)
    os.environ['RL_OBS_ABILITIES'] = '1' if agent.cfg.obs_abilities else '0'
    cfg = agent.cfg
    init_state = _state(agent)
    dev = torch.device(device)
    print(f'[qd_pga2] {describe_resources()} device={device} init={init_checkpoint or "(nuevo)"} '
          f'P5={cfg.obs_abilities} GA={n_ga} PG={n_pg}+codicioso', flush=True)

    rng = np.random.default_rng(int(seed) + 4049)
    sampler, heur = make_sampler(mosaic), make_sampler('heur')
    ckpt = Path(checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    env = _make_env(mock=mock, seed=seed, start_ps=start_ps, num_envs=num_envs,
                    envs_per_server=envs_per_server, base_port=base_port, agent_team=agent_team,
                    mask_kind='joint')
    replay = JointReplay(replay_size, OBS_DIM)
    critic = TwinCritic(OBS_DIM, dev, lr=critic_lr, gamma=float(cfg.gamma))
    greedy = _agent_from(cfg, init_state)          # actor codicioso (de PPO v2), optimizador persistente
    greedy_targ = copy.deepcopy(greedy.net)
    for p in greedy_targ.parameters():
        p.requires_grad_(False)
    greedy_opt = torch.optim.Adam([p for n, p in greedy.net.named_parameters()
                                   if not n.startswith('value.')], lr=greedy_lr)
    hb = TrainHeartbeat(tag='qd_pga2', t_total=max(1, budget), log_every=10_000, log_every_sec=120.0)
    steps = {'n': 0}

    def rec(t):
        steps['n'] += 1
        replay.add(t)

    def play(a, matches):
        return play_matchups(a, env, matches, reward='dense', on_transition=rec)

    stats = {'children': 0, 'ga': 0, 'pg': 0, 'greedy': 0, 'accepts': 0, 'new_cells': 0, 'replaced': 0,
             'rejected_gmin': 0, 'critic_steps': 0, 'greedy_steps': 0}
    try:
        calib = play(agent, make_matchups(rng, calibrate_games, sampler))
        archive = Archive(bounds=calibrate_bounds(calib.eligible_bds()))
        bd0 = calib.bd() or (0.5, 0.5)
        archive.try_insert(bd1=bd0[0], bd2=bd0[1], fitness=calib.fitness, state_dict=init_state,
                           meta={'stage': 'init'}, returns=calib.valid_returns())
        print(f'[qd_pga2] calibración f={calib.fitness:+.3f} bd={bd0} límites={archive.bounds} '
              f'replay={len(replay)}', flush=True)
        for _ in range(int(warmup_critic)):
            critic.update(to_torch(replay.sample(batch, rng), dev), greedy_targ)
        stats['critic_steps'] += int(warmup_critic)

        gen = 0
        while steps['n'] < budget:
            gen += 1
            offspring = []
            for _ in range(int(n_ga)):
                p1, k1 = archive.sample_parent(rng, fallback_state=init_state)
                p2, _k2 = archive.sample_parent(rng, fallback_state=init_state)
                offspring.append(('ga', iso_line(p1, p2, rng), k1))
            for _ in range(int(n_pg)):
                p, k = archive.sample_parent(rng, fallback_state=init_state)
                child = _agent_from(cfg, p)
                pg_steps(child.net, critic, replay, rng, n_steps=n_pg_steps, lr=pg_lr, batch=batch)
                offspring.append(('pg', _state(child), k))
            offspring.append(('greedy', _state(greedy), None))
            # todos los hijos a la vez; ocupantes en los mismos enfrentamientos (regla común de I/J/K)
            for kind, _sd, _k in offspring:
                stats['children'] += 1
                stats[kind] += 1
            evaluate_insert_batch(
                archive, env, lambda sd: _agent_from(cfg, sd),
                [(sd, {'gen': gen, 'op': kind, 'parent': list(pk) if pk else None})
                 for kind, sd, pk in offspring],
                rng, sampler, paired_games=paired_games, g_min=g_min, stats=stats, reward='dense',
                on_transition=rec)
            # crítico (TD3 con SARSA esperado) y actor codicioso con retardo 2
            for j in range(int(n_critic)):
                critic.update(to_torch(replay.sample(batch, rng), dev), greedy_targ)
                stats['critic_steps'] += 1
                if (j + 1) % POLICY_DELAY == 0:
                    pg_steps(greedy.net, critic, replay, rng, n_steps=1, lr=greedy_lr, batch=batch,
                             optim=greedy_opt)
                    stats['greedy_steps'] += 1
                    with torch.no_grad():
                        for p, pt in zip(greedy.net.parameters(), greedy_targ.parameters()):
                            pt.mul_(1.0 - critic.tau).add_(critic.tau * p)
            best = archive.best()
            hb.log(steps['n'], phase=f'gen={gen}', extras={
                'cov': round(archive.coverage(), 3), 'qd': round(archive.qd_score(), 2),
                'best_f': round(best.fitness, 3) if best else None,
                'acc': f"{stats['accepts']}/{stats['children']}", 'replay': len(replay),
            }, force=True)
            archive.save(ckpt / 'archive.pt')

        ranked = sorted(archive.cells.items(), key=lambda kv: kv[1].fitness, reverse=True)[:int(reeval_top)]
        rows = _reeval(cfg, env, rng, heur, [(f'{k[0]},{k[1]}', e.state_dict) for k, e in ranked],
                       reeval_games, reeval_split)
        for row, (k, e) in zip(rows, ranked):
            row.update({'cell': list(k), 'bd': list(e.bd), 'archive_fitness': e.fitness,
                        'n_eval': e.n_eval, 'op': e.meta.get('op')})
        best_row = max(rows, key=lambda r: r['fitness']) if rows else None
        if best_row is not None:
            _agent_from(cfg, ranked[rows.index(best_row)][1].state_dict).save(ckpt / 'best_elite.pt')
        archive.save(ckpt / 'archive.pt')
        (ckpt / 'reeval.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        summary = {'seed': seed, 'init_checkpoint': init_checkpoint, 'global_steps': steps['n'],
                   'generations': gen, 'bounds': [list(b) for b in archive.bounds],
                   'coverage': archive.coverage(), 'qd_score': archive.qd_score(),
                   'cells': len(archive.cells), 'best_reeval': best_row, 'replay': len(replay), **stats}
    finally:
        env.close()
    (ckpt / 'summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    hb.log(steps['n'], phase='DONE', extras={k: summary.get(k) for k in
                                             ('coverage', 'qd_score', 'cells', 'accepts', 'children')},
           force=True)
    return summary


def main(argv: Optional[list] = None) -> int:
    from showdown.rl.foe_pool import trainer_defaults
    from showdown.rl.frozen_preview import add_preview_cli_args, configure_preview

    p = argparse.ArgumentParser(description='QD-PGA 2.0 (PGA-ME fiel, acciones discretas)')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--checkpoint-dir', default='logs/qd_pga2')
    p.add_argument('--init-checkpoint', default=None)
    p.add_argument('--budget', type=int, default=BUDGET)
    p.add_argument('--n-ga', type=int, default=N_GA)
    p.add_argument('--n-pg', type=int, default=N_PG)
    p.add_argument('--n-pg-steps', type=int, default=N_PG_STEPS)
    p.add_argument('--n-critic', type=int, default=N_CRITIC)
    p.add_argument('--paired-games', type=int, default=PAIRED_GAMES)
    p.add_argument('--g-min', type=int, default=G_MIN)
    p.add_argument('--calibrate-games', type=int, default=CALIBRATE_GAMES)
    p.add_argument('--reeval-top', type=int, default=10)
    p.add_argument('--reeval-games', type=int, default=300)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--start-ps', action='store_true')
    p.add_argument('--agent-team', default=DEFAULT_TEAM)
    p.add_argument('--num-envs', type=int, default=1)
    p.add_argument('--envs-per-server', type=int, default=4)
    p.add_argument('--base-port', type=int, default=None)
    add_preview_cli_args(p)
    a = p.parse_args(argv)
    with trainer_defaults():
        print('[qd_pga2] ' + configure_preview('qd_pga', checkpoint=a.preview_checkpoint,
                                              required=a.require_preview), flush=True)
        run_training(seed=a.seed, device=a.device, checkpoint_dir=a.checkpoint_dir,
                     init_checkpoint=a.init_checkpoint, budget=a.budget, n_ga=a.n_ga, n_pg=a.n_pg,
                     n_pg_steps=a.n_pg_steps, n_critic=a.n_critic, paired_games=a.paired_games,
                     g_min=a.g_min, calibrate_games=a.calibrate_games, reeval_top=a.reeval_top,
                     reeval_games=a.reeval_games, smoke=a.smoke, start_ps=a.start_ps,
                     agent_team='' if str(a.agent_team).lower() in ('pool', 'none', '') else a.agent_team,
                     num_envs=a.num_envs, envs_per_server=a.envs_per_server, base_port=a.base_port)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
