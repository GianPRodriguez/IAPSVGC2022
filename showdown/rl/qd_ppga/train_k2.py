"""QD-PPGA 2.0 (agente K 2.0): PPGA fiel — docs/qd_ppga_spec.md §K 2.0 (Batra et al., ICLR 2024).

Por iteración (Algoritmo 1): VPPO (k+1 copias del punto de búsqueda θ_μ, cada una ``n1`` iteraciones
de PPO con su recompensa: r_f, δ1, δ2; gradientes = diferencias de parámetros, normalizados); evaluar
θ_μ; ``lam`` ramas θ_μ + s·Σ c_j ĝ_j con c ~ xNES, evaluadas e insertadas (Δ del archivo suave CMA-MAE
para el ranking; archivo de resultados con la comparación pareada de I/J/K); xNES.tell; paseo: ``n2``
iteraciones de PPO de θ_μ con la recompensa combinada por la media de xNES; reinicio si el archivo no
cambió. Arranque, calibración, mosaico, reevaluación en ``val`` y despliegue: como I 2.0.
Presupuesto: pasos de entorno totales (VPPO, paseo y evaluación).

    python -m showdown.rl.qd_ppga.train_k2 --seed 0 --smoke
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from showdown.rl.ppo.agent import PPOAgent, PPOConfig
from showdown.rl.qd_ppga.ppga import (
    SoftArchive, XNES, apply_step, normalized, objective_reward, param_delta, ppo_iterations,
)
from showdown.rl.qd_ppo.archive import Archive
from showdown.rl.qd_ppo.matchups import make_matchups, paired_diff, play_matchups
from showdown.rl.qd_ppo.mosaic import make_sampler
from showdown.rl.qd_ppo.train2 import (
    DEFAULT_TEAM, _agent_from, _make_env, _reeval, _state, calibrate_bounds,
)
from showdown.rl.train_log import TrainHeartbeat

BUDGET = 1_000_000
N1 = 3
N2 = 3
ROLLOUT = 1024
LAM = 8
SIGMA_G = 1.0
ALPHA = 0.1
PAIRED_GAMES = 32
G_MIN = 16
CALIBRATE_GAMES = 512
OBJECTIVES = ('f', 'm1', 'm2')


def run_training(
    *, seed: int = 0, device: str = 'cpu', checkpoint_dir: str | Path = 'logs/qd_ppga2',
    init_checkpoint: Optional[str] = None, budget: int = BUDGET, n1: int = N1, n2: int = N2,
    rollout: int = ROLLOUT, lam: int = LAM, sigma_g: float = SIGMA_G, alpha: float = ALPHA,
    paired_games: int = PAIRED_GAMES, g_min: int = G_MIN, calibrate_games: int = CALIBRATE_GAMES,
    reeval_top: int = 10, reeval_games: int = 300, mosaic: str = 'ppo_v2', smoke: bool = False,
    start_ps: bool = False, agent_team: str = DEFAULT_TEAM, num_envs: int = 2, envs_per_server: int = 4,
    base_port: Optional[int] = None,
) -> Dict[str, Any]:
    from showdown.rl.parallel import describe_resources, raise_nofile_limit, resolve_device
    from showdown.rl.vec_rollout import make_qd_env

    raise_nofile_limit()
    device = resolve_device(device)
    if smoke:
        budget, n1, n2, rollout, lam = 900, 1, 1, 64, 2
        paired_games, g_min, calibrate_games, reeval_top, reeval_games = 2, 1, 4, 2, 2
    mock = bool(smoke or not start_ps)
    reeval_split = None if mock else 'val'

    base = PPOAgent(PPOConfig(seed=seed, device=device, rollout_steps=rollout))
    base._ensure_torch()
    if init_checkpoint:
        base.load(init_checkpoint)
    base.cfg.seed, base.cfg.device, base.cfg.rollout_steps = int(seed), str(device), int(rollout)
    os.environ['RL_OBS_ABILITIES'] = '1' if base.cfg.obs_abilities else '0'
    cfg = base.cfg
    init_state = _state(base)
    print(f'[qd_ppga2] {describe_resources()} device={device} init={init_checkpoint or "(nuevo)"} '
          f'P5={cfg.obs_abilities} N1={n1} N2={n2} lam={lam} sigma_g={sigma_g} alpha={alpha}', flush=True)

    rng = np.random.default_rng(int(seed) + 6151)
    sampler, heur = make_sampler(mosaic), make_sampler('heur')
    if mock:  # el entorno vectorizado simulado no tiene rivales reales
        sampler = heur = (lambda _rng: 'mock')
    ckpt = Path(checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    if mock:  # PPGA necesita el entorno vectorizado (proxies por paso); en mock, síncrono
        env = make_qd_env(make_single=lambda: None, mock=True, num_envs=max(2, num_envs), seed=seed,
                          start_ps=False, account_prefix='QdPpga2', combat_family='qd_ppga',
                          subprocess=False)
    else:
        env = _make_env(mock=False, seed=seed, start_ps=start_ps, num_envs=num_envs,
                        envs_per_server=envs_per_server, base_port=base_port, agent_team=agent_team)
    hb = TrainHeartbeat(tag='qd_ppga2', t_total=max(1, budget), log_every=10_000, log_every_sec=120.0)
    steps = {'n': 0}

    def count(_t):
        steps['n'] += 1

    def play(state, matches):
        return play_matchups(_agent_from(cfg, state), env, matches, on_transition=count)

    xnes = XNES(1 + 2, sigma0=sigma_g, seed=int(seed))
    soft = SoftArchive(alpha=alpha)
    stats = {'iterations': 0, 'branches': 0, 'accepts': 0, 'new_cells': 0, 'replaced': 0, 'restarts': 0,
             'rejected_gmin': 0}

    def evaluate_insert(state, meta, archive):
        """Evalúa ``state``; archivo de resultados (pareada) y Δ del archivo suave. Devuelve (Δ, cambió)."""
        matches = make_matchups(rng, paired_games, sampler)
        res = play(state, matches)
        bd = res.bd()
        if bd is None or len(res.eligible_bds()) < int(g_min):
            stats['rejected_gmin'] += 1
            return 0.0, False
        key = archive.key_of(*bd)
        delta = soft.update(key, res.fitness)
        inc = archive.cells.get(key)
        if inc is None:
            archive.try_insert(bd1=bd[0], bd2=bd[1], fitness=res.fitness, state_dict=state, meta=meta,
                               returns=res.valid_returns(), force=True)
            stats['accepts'] += 1
            stats['new_cells'] += 1
            return delta, True
        res_i = play(inc.state_dict, matches)
        inc.add_games(res_i.valid_returns())
        d = paired_diff(res, res_i)
        if d > 0:
            meta['paired_diff'] = d
            archive.try_insert(bd1=bd[0], bd2=bd[1], fitness=res.fitness, state_dict=state, meta=meta,
                               returns=res.valid_returns(), force=True)
            stats['accepts'] += 1
            stats['replaced'] += 1
            return delta, True
        return delta, delta > 0

    try:
        calib = play_matchups(base, env, make_matchups(rng, calibrate_games, sampler), on_transition=count)
        archive = Archive(bounds=calibrate_bounds(calib.eligible_bds()))
        bd0 = calib.bd() or (0.5, 0.5)
        archive.try_insert(bd1=bd0[0], bd2=bd0[1], fitness=calib.fitness, state_dict=init_state,
                           meta={'stage': 'init'}, returns=calib.valid_returns())
        soft.update(archive.key_of(*bd0), calib.fitness)
        print(f'[qd_ppga2] calibración f={calib.fitness:+.3f} bd={bd0} límites={archive.bounds}', flush=True)

        mu_state = dict(init_state)
        it = 0
        while steps['n'] < budget:
            it += 1
            stats['iterations'] += 1
            # 1. VPPO: gradientes = diferencias de parámetros tras n1 iteraciones de PPO por objetivo
            grads, norms = [], []
            for obj in OBJECTIVES:
                copy_agent = _agent_from(cfg, mu_state)
                steps['n'] += ppo_iterations(copy_agent, env, n_iters=n1, rollout=rollout, rng=rng,
                                             sampler=sampler, reward_fn=objective_reward(obj),
                                             heartbeat=hb, hb_offset=steps['n'], phase=f'it={it}:{obj}')
                g, n = normalized(param_delta(_state(copy_agent), mu_state))
                grads.append(g)
                norms.append(n)
            scale = float(np.mean([n for n in norms if n > 0])) if any(n > 0 for n in norms) else 0.0
            # 2. evaluar el punto de búsqueda
            _d, changed = evaluate_insert(mu_state, {'iter': it, 'op': 'mu'}, archive)
            # 3. ramas
            coefs = xnes.ask(lam)
            deltas = []
            for c in coefs:
                if steps['n'] >= budget:
                    deltas.append(0.0)
                    continue
                stats['branches'] += 1
                br = apply_step(mu_state, grads, c, scale)
                d, ch = evaluate_insert(br, {'iter': it, 'op': 'branch', 'coef': [float(x) for x in c]},
                                        archive)
                deltas.append(d)
                changed = changed or ch
            # 4. xNES con el ranking de Δ
            xnes.tell(deltas)
            # 5. paseo: n2 iteraciones de PPO con la recompensa combinada por la media de xNES
            walker = _agent_from(cfg, mu_state)
            steps['n'] += ppo_iterations(walker, env, n_iters=n2, rollout=rollout, rng=rng, sampler=sampler,
                                         reward_fn=objective_reward('mix', xnes.mu), heartbeat=hb,
                                         hb_offset=steps['n'], phase=f'it={it}:walk')
            mu_state = _state(walker)
            # 6. reinicio si el archivo no cambió
            if not changed:
                stats['restarts'] += 1
                xnes.reset()
                mu_state, _k = archive.sample_parent(rng, fallback_state=init_state)
            best = archive.best()
            hb.log(steps['n'], phase=f'it={it}', extras={
                'cov': round(archive.coverage(), 3), 'qd': round(archive.qd_score(), 2),
                'best_f': round(best.fitness, 3) if best else None, 'mu_c': np.round(xnes.mu, 2).tolist(),
                'sigma': round(xnes.sigma, 3), 'acc': f"{stats['accepts']}/{stats['branches']}",
                'restarts': stats['restarts'],
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
                   'bounds': [list(b) for b in archive.bounds], 'coverage': archive.coverage(),
                   'qd_score': archive.qd_score(), 'cells': len(archive.cells), 'best_reeval': best_row,
                   'xnes_mu': xnes.mu.tolist(), 'xnes_sigma': xnes.sigma, **stats}
    finally:
        env.close()
    (ckpt / 'summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    hb.log(steps['n'], phase='DONE', extras={k: summary.get(k) for k in
                                             ('coverage', 'qd_score', 'cells', 'accepts', 'restarts')},
           force=True)
    return summary


def main(argv: Optional[list] = None) -> int:
    from showdown.rl.foe_pool import trainer_defaults
    from showdown.rl.frozen_preview import add_preview_cli_args, configure_preview

    p = argparse.ArgumentParser(description='QD-PPGA 2.0 (PPGA fiel)')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--checkpoint-dir', default='logs/qd_ppga2')
    p.add_argument('--init-checkpoint', default=None)
    p.add_argument('--budget', type=int, default=BUDGET)
    p.add_argument('--n1', type=int, default=N1)
    p.add_argument('--n2', type=int, default=N2)
    p.add_argument('--rollout', type=int, default=ROLLOUT)
    p.add_argument('--lam', type=int, default=LAM)
    p.add_argument('--sigma-g', type=float, default=SIGMA_G)
    p.add_argument('--alpha', type=float, default=ALPHA)
    p.add_argument('--paired-games', type=int, default=PAIRED_GAMES)
    p.add_argument('--g-min', type=int, default=G_MIN)
    p.add_argument('--calibrate-games', type=int, default=CALIBRATE_GAMES)
    p.add_argument('--reeval-top', type=int, default=10)
    p.add_argument('--reeval-games', type=int, default=300)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--start-ps', action='store_true')
    p.add_argument('--agent-team', default=DEFAULT_TEAM)
    p.add_argument('--num-envs', type=int, default=32)
    p.add_argument('--envs-per-server', type=int, default=4)
    p.add_argument('--base-port', type=int, default=None)
    add_preview_cli_args(p)
    a = p.parse_args(argv)
    with trainer_defaults():
        print('[qd_ppga2] ' + configure_preview('qd_ppga', checkpoint=a.preview_checkpoint,
                                               required=a.require_preview), flush=True)
        run_training(seed=a.seed, device=a.device, checkpoint_dir=a.checkpoint_dir,
                     init_checkpoint=a.init_checkpoint, budget=a.budget, n1=a.n1, n2=a.n2,
                     rollout=a.rollout, lam=a.lam, sigma_g=a.sigma_g, alpha=a.alpha,
                     paired_games=a.paired_games, g_min=a.g_min, calibrate_games=a.calibrate_games,
                     reeval_top=a.reeval_top, reeval_games=a.reeval_games, smoke=a.smoke,
                     start_ps=a.start_ps,
                     agent_team='' if str(a.agent_team).lower() in ('pool', 'none', '') else a.agent_team,
                     num_envs=a.num_envs, envs_per_server=a.envs_per_server, base_port=a.base_port)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
