"""QD-PPO 2.0 (agente I 2.0) — docs/qd_ppo_spec.md §I 2.0 (decisiones del usuario, 25-11-2026).

MAP-Elites con PPO como operador de variación, **desde PPO v2**:

1. Arranque: actor-crítico de un checkpoint de PPO v2 (observación P5, equipo fijo y preview del
   checkpoint/CLI). Validación reservada: los rivales salen de ``fit``.
2. Descriptores calibrados: el arranque juega ``calibrate_games`` enfrentamientos; los límites de la
   rejilla son los percentiles 5–95 de los descriptores por partida. Esas partidas también dan la
   estimación inicial del arranque, que es la primera élite.
3. Generación: padre uniforme entre celdas ocupadas; hijo = copia completa (actor + crítico, optimizador
   nuevo) mejorada con ``child_updates`` actualizaciones PPO (``rollout`` pasos cada una, Cand. 2).
4. Comparación pareada: el hijo juega ``paired_games`` enfrentamientos nuevos; si su celda está ocupada,
   el ocupante juega los mismos y suma esas partidas a su estimación. El hijo reemplaza solo si gana
   más en esas partidas (diferencia pareada > 0). Celda vacía: entra directo.
5. Despliegue: las ``reeval_top`` élites con mejor estimación se reevalúan en ``reeval_games``
   enfrentamientos comunes con equipos ``val`` contra los tres heurísticos; π* = la mejor.

``--control``: misma receta sin archivo (PPO v2 que sigue entrenando con el mismo mosaico, preview,
equipo y pasos). Es el control de la hipótesis central "I > PPO v2 + el mismo presupuesto".

    python -m showdown.rl.qd_ppo.train2 --seed 0 --smoke
    python -m showdown.rl.qd_ppo.train2 --seed 0 --init-checkpoint <ppo_v2>/final.pt --start-ps \\
        --num-envs 32 --device cuda --preview-checkpoint <preview.pt> --checkpoint-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from showdown.rl.ppo.agent import PPOAgent, PPOConfig
from showdown.rl.qd_ppo.archive import Archive, UNIT_BOUNDS
from showdown.rl.qd_ppo.improve import ppo_improve
from showdown.rl.qd_ppo.matchups import MatchResults, make_matchups, paired_diff, play_matchups
from showdown.rl.qd_ppo.mosaic import make_sampler
from showdown.rl.train_log import TrainHeartbeat

DEFAULT_TEAM = str(Path(__file__).resolve().parents[1] / 'data' / 'ddqn_team.txt')
T_EXTRA = 1_000_000
N_EMIT = 2
CHILD_UPDATES = 2
ROLLOUT = 2048
PAIRED_GAMES = 32
G_MIN = 16
CALIBRATE_GAMES = 512
REEVAL_TOP = 10
REEVAL_GAMES = 300


def _state(agent: PPOAgent) -> Dict[str, Any]:
    return {k: v.detach().cpu().clone() for k, v in agent.net.state_dict().items()}


def _agent_from(cfg: PPOConfig, state: Dict[str, Any]) -> PPOAgent:
    """Agente nuevo (optimizador nuevo) con los pesos completos de ``state``."""
    a = PPOAgent(PPOConfig(**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__}))
    a._ensure_torch()
    a.net.load_state_dict(state)
    return a


def calibrate_bounds(bds: List[Tuple[float, float]], lo_q: float = 5.0, hi_q: float = 95.0,
                     min_width: float = 0.05) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Límites de la rejilla: percentiles de los descriptores por partida (ancho mínimo ``min_width``)."""
    if len(bds) < 2:
        return UNIT_BOUNDS
    arr = np.asarray(bds, dtype=np.float64)
    out = []
    for d in range(2):
        lo, hi = float(np.percentile(arr[:, d], lo_q)), float(np.percentile(arr[:, d], hi_q))
        if hi - lo < min_width:
            c = 0.5 * (lo + hi)
            lo, hi = c - min_width / 2, c + min_width / 2
        out.append((max(0.0, lo), min(1.0, hi)))
    return out[0], out[1]


def _make_env(*, mock: bool, seed: int, start_ps: bool, num_envs: int, envs_per_server: int,
              base_port: Optional[int], agent_team: str):
    from showdown.rl.env import MockBattleBackend, VGCEnv
    from showdown.rl.reward import RewardCandidate
    from showdown.rl.vec_rollout import make_qd_env

    def single():
        if mock:
            backend = MockBattleBackend(max_turns=40)
        else:
            from showdown.inference_sim.team_factory import load_team_file
            from showdown.rl.pokeenv_backend import PokeEnvBattleBackend
            backend = PokeEnvBattleBackend(
                opponent='random', start_ps=start_ps, account_prefix='QdPpo2',
                combat_family='qd_ppo', our_team=load_team_file(agent_team) if agent_team else None,
            )
        return VGCEnv(reward_candidate=RewardCandidate.DENSE, backend=backend, max_episode_turns=40)

    return make_qd_env(
        make_single=single, mock=mock, num_envs=num_envs, seed=seed, start_ps=start_ps,
        account_prefix='QdPpo2', combat_family='qd_ppo', envs_per_server=envs_per_server,
        base_port=base_port, agent_team='' if mock else agent_team,
    )


def run_training(
    *,
    seed: int = 0,
    device: str = 'cpu',
    checkpoint_dir: str | Path = 'logs/qd_ppo2',
    init_checkpoint: Optional[str] = None,
    t_extra: int = T_EXTRA,
    n_emit: int = N_EMIT,
    child_updates: int = CHILD_UPDATES,
    rollout: int = ROLLOUT,
    paired_games: int = PAIRED_GAMES,
    g_min: int = G_MIN,
    calibrate_games: int = CALIBRATE_GAMES,
    bd_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    reeval_top: int = REEVAL_TOP,
    reeval_games: int = REEVAL_GAMES,
    reeval_split: Optional[str] = 'val',
    mosaic: str = 'ppo_v2',
    control: bool = False,
    smoke: bool = False,
    start_ps: bool = False,
    agent_team: str = DEFAULT_TEAM,
    num_envs: int = 1,
    envs_per_server: int = 4,
    base_port: Optional[int] = None,
    log_every: int = 10_000,
    log_every_sec: float = 120.0,
) -> Dict[str, Any]:
    from showdown.rl.parallel import describe_resources, raise_nofile_limit, resolve_device

    raise_nofile_limit()
    device = resolve_device(device)
    if smoke:
        t_extra, rollout = min(t_extra, 1024), min(rollout, 128)
        paired_games, g_min, calibrate_games = min(paired_games, 4), min(g_min, 1), min(calibrate_games, 8)
        reeval_top, reeval_games = min(reeval_top, 2), min(reeval_games, 4)
        log_every, log_every_sec = 256, 15.0
    mock = bool(smoke or not start_ps)
    if mock:
        reeval_split = None

    # ---------------------------------------------------------------- arranque
    agent = PPOAgent(PPOConfig(seed=seed, device=device, rollout_steps=rollout))
    agent._ensure_torch()
    if init_checkpoint:
        agent.load(init_checkpoint)
    agent.cfg.seed, agent.cfg.device, agent.cfg.rollout_steps = int(seed), str(device), int(rollout)
    # La observación del checkpoint (P5) también en los workers, que heredan el entorno.
    os.environ['RL_OBS_ABILITIES'] = '1' if agent.cfg.obs_abilities else '0'
    cfg = agent.cfg
    init_state = _state(agent)
    print(f'[qd_ppo2] recursos: {describe_resources()} device={device} num_envs={num_envs} '
          f'init={init_checkpoint or "(nuevo)"} P5={cfg.obs_abilities} control={control} '
          f'mosaico={mosaic} equipo={agent_team or "pool"}', flush=True)

    rng = np.random.default_rng(int(seed) + 2027)
    sampler = make_sampler(mosaic)
    heur = make_sampler('heur')
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    env = _make_env(mock=mock, seed=seed, start_ps=start_ps, num_envs=num_envs,
                    envs_per_server=envs_per_server, base_port=base_port, agent_team=agent_team)
    hb = TrainHeartbeat(tag='qd_ppo2', t_total=max(1, int(t_extra)), log_every=int(log_every),
                        log_every_sec=float(log_every_sec))
    summary: Dict[str, Any] = {'seed': seed, 'init_checkpoint': init_checkpoint, 'control': control,
                               'mosaic': mosaic, 't_extra': t_extra, 'smoke': smoke}
    try:
        if control:
            summary.update(_run_control(agent, env, rng, sampler, t_extra, hb, ckpt_dir))
            final_state = _state(agent)
            reeval = _reeval(cfg, env, rng, heur, [('control_final', final_state)],
                             reeval_games, reeval_split)
            summary['reeval'] = reeval
            agent.save(ckpt_dir / 'final.pt')
        else:
            summary.update(_run_qd(cfg, agent, init_state, env, rng, sampler, hb, ckpt_dir,
                                   t_extra=t_extra, n_emit=n_emit, child_updates=child_updates,
                                   rollout=rollout, paired_games=paired_games, g_min=g_min,
                                   calibrate_games=calibrate_games, bd_bounds=bd_bounds,
                                   reeval_top=reeval_top, reeval_games=reeval_games,
                                   reeval_split=reeval_split, heur=heur))
    finally:
        env.close()
    (ckpt_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    hb.log(int(summary.get('global_steps', 0)), phase='DONE', extras={
        k: summary.get(k) for k in ('coverage', 'qd_score', 'best_reeval', 'accepts', 'children')
    }, force=True)
    return summary


def _run_control(agent, env, rng, sampler, t_extra, hb, ckpt_dir) -> Dict[str, Any]:
    env, taken, metrics = ppo_improve(agent, env, steps=t_extra, rng=rng, global_steps=0,
                                      t_total=t_extra, use_mosaic=True, heartbeat=hb,
                                      phase='control', foe_sampler=sampler)
    return {'global_steps': int(taken), 'last_metrics': metrics}


def _reeval(cfg, env, rng, heur, candidates: List[Tuple[str, Dict[str, Any]]], games: int,
            split: Optional[str]) -> List[Dict[str, Any]]:
    """Mismos ``games`` enfrentamientos (equipos ``split``, tres heurísticos) para todos."""
    matches = make_matchups(rng, games, heur, split=split)
    rows = []
    for name, state in candidates:
        res = play_matchups(_agent_from(cfg, state), env, matches)
        v = res.valid_returns()
        wins = sum(1 for r in v if r > 0)
        rows.append({'name': name, 'fitness': res.fitness, 'win_rate': wins / max(1, len(v)),
                     'games': len(v)})
        print(f'[qd_ppo2] reeval {name}: f={res.fitness:+.3f} wr={wins / max(1, len(v)):.3f} '
              f'({len(v)} partidas, split={split})', flush=True)
    return rows


def _run_qd(cfg, agent, init_state, env, rng, sampler, hb, ckpt_dir, *, t_extra, n_emit,
            child_updates, rollout, paired_games, g_min, calibrate_games, bd_bounds, reeval_top,
            reeval_games, reeval_split, heur) -> Dict[str, Any]:
    # ------------------------------------------------ calibración + primera élite
    calib = play_matchups(agent, env, make_matchups(rng, calibrate_games, sampler))
    bounds = bd_bounds or calibrate_bounds(calib.eligible_bds())
    archive = Archive(bounds=bounds)
    bd0 = calib.bd() or (0.5, 0.5)
    archive.try_insert(bd1=bd0[0], bd2=bd0[1], fitness=calib.fitness, state_dict=init_state,
                       meta={'stage': 'init'}, returns=calib.valid_returns())
    print(f'[qd_ppo2] calibración: {len(calib.valid_returns())} partidas f={calib.fitness:+.3f} '
          f'bd={bd0} límites={bounds}', flush=True)
    stats = {'children': 0, 'accepts': 0, 'new_cells': 0, 'replaced': 0, 'rejected_gmin': 0,
             'paired_games_incumbent': 0}
    steps = 0
    it = 0
    while steps < t_extra:
        it += 1
        for _em in range(int(n_emit)):
            if steps >= t_extra:
                break
            parent_state, pkey = archive.sample_parent(rng, fallback_state=init_state)
            child = _agent_from(cfg, parent_state)
            env, taken, _m = ppo_improve(child, env, steps=int(rollout) * int(child_updates), rng=rng,
                                         global_steps=steps, t_total=t_extra, use_mosaic=True,
                                         heartbeat=hb, phase=f'gen={it}', foe_sampler=sampler)
            steps += taken
            stats['children'] += 1
            matches = make_matchups(rng, paired_games, sampler)
            res_c = play_matchups(child, env, matches)
            bd = res_c.bd()
            if bd is None or len(res_c.eligible_bds()) < int(g_min):
                stats['rejected_gmin'] += 1
                continue
            key = archive.key_of(*bd)
            inc = archive.cells.get(key)
            meta = {'gen': it, 'parent': list(pkey) if pkey else None, 'steps': steps}
            if inc is None:
                archive.try_insert(bd1=bd[0], bd2=bd[1], fitness=res_c.fitness,
                                   state_dict=_state(child), meta=meta,
                                   returns=res_c.valid_returns(), force=True)
                stats['accepts'] += 1
                stats['new_cells'] += 1
                continue
            res_i = play_matchups(_agent_from(cfg, inc.state_dict), env, matches)
            inc.add_games(res_i.valid_returns())
            stats['paired_games_incumbent'] += len(res_i.valid_returns())
            d = paired_diff(res_c, res_i)
            if d > 0:
                meta['paired_diff'] = d
                archive.try_insert(bd1=bd[0], bd2=bd[1], fitness=res_c.fitness,
                                   state_dict=_state(child), meta=meta,
                                   returns=res_c.valid_returns(), force=True)
                stats['accepts'] += 1
                stats['replaced'] += 1
        best = archive.best()
        hb.log(steps, phase=f'gen={it}', extras={
            'cov': round(archive.coverage(), 3), 'qd': round(archive.qd_score(), 2),
            'best_f': round(best.fitness, 3) if best else None,
            'acc': f"{stats['accepts']}/{stats['children']}",
        }, force=True)
        archive.save(ckpt_dir / 'archive.pt')

    # ------------------------------------------------ despliegue: reevaluación de las mejores
    ranked = sorted(archive.cells.items(), key=lambda kv: kv[1].fitness, reverse=True)[:int(reeval_top)]
    cands = [(f'{k[0]},{k[1]}', e.state_dict) for k, e in ranked]
    rows = _reeval(cfg, env, rng, heur, cands, reeval_games, reeval_split)
    for row, (k, e) in zip(rows, ranked):
        row.update({'cell': list(k), 'bd': list(e.bd), 'archive_fitness': e.fitness, 'n_eval': e.n_eval})
    best_row = max(rows, key=lambda r: r['fitness']) if rows else None
    if best_row is not None:
        best_state = dict(ranked[rows.index(best_row)][1].state_dict)
        out = _agent_from(cfg, best_state)
        out.save(ckpt_dir / 'best_elite.pt')
    archive.save(ckpt_dir / 'archive.pt')
    (ckpt_dir / 'reeval.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return {
        'global_steps': steps, 'generations': it, 'bounds': [list(b) for b in archive.bounds],
        'coverage': archive.coverage(), 'qd_score': archive.qd_score(), 'cells': len(archive.cells),
        'best_reeval': best_row, **stats,
    }


def main(argv: Optional[list] = None) -> int:
    from showdown.rl.foe_pool import trainer_defaults

    p = argparse.ArgumentParser(description='QD-PPO 2.0 (MAP-Elites + PPO desde PPO v2)')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--checkpoint-dir', default='logs/qd_ppo2')
    p.add_argument('--init-checkpoint', default=None, help='final.pt de PPO v2 (arranque)')
    p.add_argument('--t-extra', type=int, default=T_EXTRA)
    p.add_argument('--n-emit', type=int, default=N_EMIT)
    p.add_argument('--child-updates', type=int, default=CHILD_UPDATES)
    p.add_argument('--rollout', type=int, default=ROLLOUT)
    p.add_argument('--paired-games', type=int, default=PAIRED_GAMES)
    p.add_argument('--g-min', type=int, default=G_MIN)
    p.add_argument('--calibrate-games', type=int, default=CALIBRATE_GAMES)
    p.add_argument('--bd-bounds', default='', help='lo1,hi1,lo2,hi2 (sin calibración)')
    p.add_argument('--reeval-top', type=int, default=REEVAL_TOP)
    p.add_argument('--reeval-games', type=int, default=REEVAL_GAMES)
    p.add_argument('--mosaic', default='ppo_v2', choices=('ppo_v2', 'v1', 'heur'))
    p.add_argument('--control', action='store_true', help='PPO v2 sigue entrenando sin archivo')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--start-ps', action='store_true')
    p.add_argument('--agent-team', default=DEFAULT_TEAM, help='"pool" = sorteo por reinicio')
    p.add_argument('--num-envs', type=int, default=1)
    p.add_argument('--envs-per-server', type=int, default=4)
    p.add_argument('--base-port', type=int, default=None)
    from showdown.rl.frozen_preview import add_preview_cli_args, configure_preview
    add_preview_cli_args(p)
    args = p.parse_args(argv)
    bounds = None
    if args.bd_bounds:
        v = [float(x) for x in args.bd_bounds.split(',')]
        bounds = ((v[0], v[1]), (v[2], v[3]))
    with trainer_defaults():
        print('[qd_ppo2] ' + configure_preview('qd_ppo', checkpoint=args.preview_checkpoint,
                                              required=args.require_preview), flush=True)
        if args.num_envs == 0:
            from showdown.rl.parallel import default_num_envs
            args.num_envs = default_num_envs()
        run_training(
            seed=args.seed, device=args.device, checkpoint_dir=args.checkpoint_dir,
            init_checkpoint=args.init_checkpoint, t_extra=args.t_extra, n_emit=args.n_emit,
            child_updates=args.child_updates, rollout=args.rollout, paired_games=args.paired_games,
            g_min=args.g_min, calibrate_games=args.calibrate_games, bd_bounds=bounds,
            reeval_top=args.reeval_top, reeval_games=args.reeval_games, mosaic=args.mosaic,
            control=args.control, smoke=args.smoke, start_ps=args.start_ps,
            agent_team='' if str(args.agent_team).lower() in ('pool', 'none', '') else args.agent_team,
            num_envs=args.num_envs, envs_per_server=args.envs_per_server, base_port=args.base_port,
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
