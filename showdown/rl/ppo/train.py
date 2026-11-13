"""Masked PPO training — docs/ppo_spec.md.

Examples::

  python -m showdown.rl.ppo.train --seed 0 --opponent mock --steps 20000
  python -m showdown.rl.ppo.train --seed 0 --opponent curriculum --start-ps --expand-omega
  python -m showdown.rl.ppo.train --seed 0 --opponent self --reward anneal --bc-checkpoint logs/ppo_bc.pt
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from showdown.rl.action_space import build_autoregressive_masks, decode_slot_action, build_slot_mask
from showdown.rl.ddqn.curriculum import (
    ExpandPhase,
    expand_phases_from_exclusive,
    merge_weights,
    omega_sampling_probs,
    parse_weight_overrides,
    sample_from_omega,
)
from showdown.rl.ddqn.train import (
    DEFAULT_CURRICULUM_PHASE_STEPS,
    TrainPhase,
    parse_phases,
    resolve_schedule,
)
from showdown.rl.env import MockBattleBackend, VGCEnv
from showdown.rl.observation import build_observation_ppo
from showdown.rl.ppo.agent import PPOAgent, PPOConfig
from showdown.rl.ppo.selfplay import SelfPlayPool
from showdown.rl.reward import RewardCandidate
from showdown.rl.train_log import TrainHeartbeat, schedule_total_steps
from showdown.rl.train_report import (
    SUITE_FOES,
    TrainReport,
    resolve_suite_games,
    run_suite_ppo_act,
    write_train_report,
)

LIVE_OPPONENTS = (
    'random',
    'greedy_solo',
    'greedy_joint',
    'expectimax_bayesian',
)
ALL_OPPONENTS = ('mock',) + LIVE_OPPONENTS + ('curriculum', 'self')


def _slot_masks(env: VGCEnv, *, slot1_action_id: Optional[int] = None):
    live = env.backend.action_masks(slot1_action_id=slot1_action_id)
    if live is not None:
        return (
            np.asarray(live[0], dtype=np.float32),
            np.asarray(live[1], dtype=np.float32),
        )
    assert env._snap is not None
    return build_autoregressive_masks(env._snap, slot1_action_id=slot1_action_id)


def _m2_fn(env: VGCEnv):
    def _fn(a1: int) -> np.ndarray:
        live = env.backend.action_masks(slot1_action_id=a1)
        if live is not None:
            return np.asarray(live[1], dtype=np.float32)
        assert env._snap is not None
        return build_slot_mask(env._snap, 1, other_slot_action=decode_slot_action(a1))

    return _fn


def _reward_candidate(mode: str, *, self_play: bool, anneal_frac: float) -> RewardCandidate:
    key = str(mode or 'candidate2').strip().lower()
    if key in ('candidate1', 'c1', 'sparse'):
        return RewardCandidate.SPARSE
    if key == 'anneal' and self_play:
        return RewardCandidate.SPARSE if anneal_frac >= 0.5 else RewardCandidate.DENSE
    return RewardCandidate.DENSE


def _make_env(
    opponent: str,
    *,
    seed: int,
    start_ps: bool,
    reward: RewardCandidate,
) -> VGCEnv:
    if opponent == 'mock':
        backend = MockBattleBackend(max_turns=40)
    else:
        from showdown.rl.pokeenv_backend import PokeEnvBattleBackend
        backend = PokeEnvBattleBackend(
            opponent=opponent,
            start_ps=start_ps,
            account_prefix=f'Ppo{opponent[:4].title()}',
            combat_family='ppo',
        )
    return VGCEnv(
        reward_candidate=reward,
        backend=backend,
        max_episode_turns=40,
    )


def _ensure_opponent(
    env: VGCEnv,
    opponent: str,
    *,
    seed: int,
    start_ps: bool,
    reward: RewardCandidate,
) -> VGCEnv:
    if opponent == 'mock':
        if isinstance(env.backend, MockBattleBackend):
            env.reward_candidate = reward
            return env
        env.close()
        return _make_env('mock', seed=seed, start_ps=False, reward=reward)

    backend = env.backend
    if hasattr(backend, 'set_opponent') and hasattr(backend, 'opponent'):
        if getattr(backend, 'opponent', None) != opponent:
            backend.set_opponent(opponent)
        env.reward_candidate = reward
        return env
    env.close()
    return _make_env(opponent, seed=seed, start_ps=start_ps, reward=reward)


def _bootstrap_value(agent: PPOAgent, obs: np.ndarray) -> float:
    import torch

    agent.net.eval()
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).view(1, -1)
        _z1, _z2, v = agent.net(x)
        return float(v.item())


def _collect_and_update(
    agent: PPOAgent,
    env: VGCEnv,
    *,
    steps: int,
    omega: Sequence[str],
    expand_omega: bool,
    weights: Dict[str, float],
    rng: np.random.Generator,
    global_step_offset: int,
    ckpt_dir: Path,
    heartbeat: TrainHeartbeat,
    phase_label: str,
    seed: int,
    start_ps: bool,
    reward_mode: str,
    self_play: bool,
    pool: Optional[SelfPlayPool],
) -> Tuple[VGCEnv, int, dict]:
    current = getattr(getattr(env, 'backend', None), 'opponent', None) or (
        omega[-1] if omega else 'mock'
    )
    anneal_frac = 0.0
    reward = _reward_candidate(reward_mode, self_play=self_play, anneal_frac=anneal_frac)
    if expand_omega and current not in omega and omega:
        current = omega[0]
        env = _ensure_opponent(
            env, current, seed=seed, start_ps=start_ps, reward=reward,
        )

    obs, info = env.reset(seed=seed + global_step_offset)
    assert env._snap is not None
    obs = build_observation_ppo(env._snap)
    m1, _ = _slot_masks(env)
    metrics_last: dict = {}
    ep_counts: Dict[str, int] = {o: 0 for o in omega}
    ep_counts[current] = ep_counts.get(current, 0) + 1
    local_done = 0

    while local_done < int(steps):
        a1, a2, logp, value = agent.act(obs, m1, _m2_fn(env), greedy=False)
        m2 = _m2_fn(env)(a1)
        next_obs, reward_t, terminated, truncated, info = env.step((a1, a2))
        assert env._snap is not None
        next_obs = build_observation_ppo(env._snap)
        skip = bool(info.get('skip_buffer'))
        trunc_reason = info.get('trunc_reason')
        eng = info.get('engine') or {}
        if isinstance(eng, dict) and eng.get('used_fallback'):
            skip = True

        boot_v = 0.0
        if trunc_reason == 'turn_cap' and not skip:
            boot_v = _bootstrap_value(agent, next_obs)

        agent.observe(
            obs=obs, a1=a1, a2=a2, logp=logp, reward=float(reward_t), value=value,
            terminated=bool(terminated), m1=m1, m2=m2,
            skip_buffer=skip, trunc_reason=trunc_reason,
            bootstrap_value=boot_v,
        )
        local_done += 1
        obs, m1 = next_obs, _slot_masks(env)[0]

        if terminated or truncated:
            if self_play and pool is not None and len(pool) > 0:
                # Opponent is frozen checkpoint weights — live foes stay in omega mode.
                pass
            if expand_omega and len(omega) > 1 and not self_play:
                nxt = sample_from_omega(omega, rng, weights=weights)
                if nxt != current:
                    env = _ensure_opponent(
                        env, nxt, seed=seed, start_ps=start_ps, reward=reward,
                    )
                    current = nxt
                ep_counts[current] = ep_counts.get(current, 0) + 1
            obs, info = env.reset()
            assert env._snap is not None
            obs = build_observation_ppo(env._snap)
            m1, _ = _slot_masks(env)

        if agent.is_rollout_full() or local_done >= int(steps):
            last_v = 0.0
            if not (terminated or truncated) and not skip:
                last_v = _bootstrap_value(agent, obs)
            elif trunc_reason == 'turn_cap':
                last_v = boot_v
            metrics_last = agent.update(last_value=last_v)

        gstep = global_step_offset + local_done
        heartbeat.log(
            gstep,
            phase=phase_label,
            extras={
                'opp': current,
                'loss': metrics_last.get('loss'),
                'kl': metrics_last.get('approx_kl'),
                'clip': metrics_last.get('clip_fraction'),
                'ent': metrics_last.get('entropy'),
                'pool': len(pool) if pool else 0,
            },
        )
        if heartbeat.should_checkpoint(gstep):
            agent.save(ckpt_dir / 'last.pt')
            heartbeat.mark_checkpoint(gstep)

    return env, int(local_done), {'metrics': metrics_last, 'ep_counts': ep_counts}


def _collect_and_update_vec(
    agent: PPOAgent,
    venv,
    *,
    steps: int,
    omega: Sequence[str],
    expand_omega: bool,
    weights: Dict[str, float],
    rng: np.random.Generator,
    global_step_offset: int,
    ckpt_dir: Path,
    heartbeat: TrainHeartbeat,
    phase_label: str,
    unlock: str,
) -> Tuple[int, dict]:
    """Igual que :func:`_collect_and_update` pero con N entornos en paralelo.

    Pasos asíncronos: la red decide en lote para los entornos que ya
    respondieron y no espera al más lento (p. ej. uno reiniciando batalla).
    Antes de cada actualización PPO se esperan los entornos en vuelo, así el
    lote es on-policy y el GAE por entorno queda exacto.

    ``steps`` cuenta pasos de entorno totales (sumando los N entornos), así el
    presupuesto de una fase no cambia al variar ``--num-envs``.
    """
    n = int(venv.num_envs)
    obs, m1 = venv.set_opponent(unlock)
    current = [unlock] * n

    def _sample_next() -> Optional[str]:
        if expand_omega and len(omega) > 1:
            return sample_from_omega(omega, rng, weights=weights)
        return None

    pending = [_sample_next() for _ in range(n)]
    ep_counts: Dict[str, int] = {o: 0 for o in omega}
    ep_counts[unlock] = ep_counts.get(unlock, 0) + n
    metrics_last: dict = {}
    local_done = 0
    issued = 0
    last_skip = [False] * n
    restarts = 0
    # Datos del paso en vuelo por entorno: (a1, a2, logp, value, m1, m2, obs)
    flight: Dict[int, tuple] = {}
    ready = list(range(n))

    def _issue(idx: List[int]) -> None:
        nonlocal issued
        if not idx:
            return
        a1, a2, logp, value, m2 = agent.act_batch(
            obs[idx], m1[idx],
            lambda a1s, _idx=idx: venv.mask2_subset(_idx, a1s),
            greedy=False,
        )
        venv.step_async(
            idx,
            list(zip(a1.tolist(), a2.tolist())),
            [pending[i] for i in idx],
        )
        for k, i in enumerate(idx):
            flight[i] = (int(a1[k]), int(a2[k]), float(logp[k]), float(value[k]),
                         m1[i].copy(), m2[k], obs[i].copy())
        issued += len(idx)

    def _absorb(results) -> List[int]:
        nonlocal local_done, restarts
        boot_idx = [
            k for k, (_i, r) in enumerate(results)
            if r['info']['trunc_reason'] == 'turn_cap'
            and not (r['info']['skip_buffer'] or r['info']['used_fallback'])
        ]
        boot_v = np.zeros(len(results), dtype=np.float32)
        if boot_idx:
            boot_v[boot_idx] = agent.values_batch(
                np.stack([results[k][1]['final_obs'] for k in boot_idx])
            )
        done_idx = []
        for k, (i, r) in enumerate(results):
            a1_i, a2_i, logp_i, v_i, m1_i, m2_i, obs_i = flight.pop(i)
            info = r['info']
            skip = bool(info['skip_buffer'] or info['used_fallback'])
            last_skip[i] = skip
            agent.observe(
                obs=obs_i, a1=a1_i, a2=a2_i, logp=logp_i,
                reward=float(r['reward']), value=v_i,
                terminated=bool(r['terminated']), m1=m1_i, m2=m2_i,
                skip_buffer=skip, trunc_reason=info['trunc_reason'],
                bootstrap_value=float(boot_v[k]), env_index=i,
            )
            if r.get('worker_restarted'):
                restarts += 1
            if r['terminated'] or r['truncated']:
                current[i] = r.get('next_opponent') or current[i]
                ep_counts[current[i]] = ep_counts.get(current[i], 0) + 1
                pending[i] = _sample_next()
            obs[i] = r['obs']
            m1[i] = r['mask']
            local_done += 1
            done_idx.append(i)
        return done_idx

    while local_done < int(steps):
        budget = int(steps) - issued
        _issue(ready[:max(0, budget)])
        if not flight:
            break
        ready = _absorb(venv.step_wait_any())

        if agent.is_rollout_full() or local_done >= int(steps):
            if flight:
                ready = ready + _absorb(venv.drain())
            last_v = agent.values_batch(obs)
            for i in range(n):
                if last_skip[i]:
                    last_v[i] = 0.0
            metrics_last = agent.update(last_value=last_v)

        gstep = global_step_offset + local_done
        heartbeat.log(
            gstep,
            phase=phase_label,
            extras={
                'envs': n,
                'opp': unlock,
                'loss': metrics_last.get('loss'),
                'kl': metrics_last.get('approx_kl'),
                'clip': metrics_last.get('clip_fraction'),
                'ent': metrics_last.get('entropy'),
                'restarts': restarts,
            },
        )
        if heartbeat.should_checkpoint(gstep):
            agent.save(ckpt_dir / 'last.pt')
            heartbeat.mark_checkpoint(gstep)

    if flight:
        _absorb(venv.drain())
    return int(local_done), {
        'metrics': metrics_last, 'ep_counts': ep_counts, 'worker_restarts': restarts,
    }


def run_training(
    *,
    seed: int = 0,
    schedule: Sequence[TrainPhase],
    device: str = 'cpu',
    checkpoint_dir: str | Path = 'logs/ppo_checkpoints',
    log_every: int = 5_000,
    log_every_sec: float = 60.0,
    start_ps: bool = False,
    expand_omega: bool = True,
    include_expectimax: bool = True,
    weight_overrides: Optional[Dict[str, float]] = None,
    reward_mode: str = 'candidate2',
    bc_checkpoint: Optional[str] = None,
    self_play: bool = False,
    rollout_steps: Optional[int] = None,
    suite_games: Optional[int] = None,
    smoke: bool = False,
    lr: Optional[float] = None,
    clip_eps: Optional[float] = None,
    num_envs: int = 1,
    envs_per_server: int = 4,
    base_port: Optional[int] = None,
) -> PPOAgent:
    if not schedule and not self_play:
        raise ValueError('schedule vacío')
    from showdown.rl.parallel import describe_resources, raise_nofile_limit, resolve_device

    raise_nofile_limit()
    device = resolve_device(device)
    num_envs = max(1, int(num_envs))
    if num_envs > 1 and self_play:
        raise ValueError('--num-envs > 1 aún no soporta --opponent self')
    print(f'[ppo] recursos: {describe_resources()} device={device} num_envs={num_envs}', flush=True)

    weights = merge_weights(weight_overrides)
    cfg = PPOConfig(seed=seed, device=device)
    if rollout_steps is not None:
        cfg.rollout_steps = int(rollout_steps)
    if lr is not None:
        cfg.lr = float(lr)
    if clip_eps is not None:
        cfg.clip_eps = float(clip_eps)
    agent = PPOAgent(cfg)
    if bc_checkpoint:
        agent.load_bc_weights(bc_checkpoint)
        print(f'[ppo] loaded BC checkpoint {bc_checkpoint} lr={agent.cfg.lr_post_bc}')

    rng = np.random.default_rng(int(seed) + 17)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pool = SelfPlayPool(root=ckpt_dir / 'pool') if self_play else None
    global_steps = 0
    phase_log: List[dict] = []

    if self_play:
        # Self-play stage: train vs uniform pool; seed pool with initial snapshot
        assert pool is not None
        pool.push(agent, tag='init')
        sched = [TrainPhase('mock' if not start_ps else 'random', 50_000)]
        if schedule:
            sched = list(schedule)
        expand_sched = [
            ExpandPhase(unlock=p.opponent, steps=p.steps, omega=(p.opponent,))
            for p in sched
        ]
    elif expand_omega:
        sched = list(schedule)
        if not include_expectimax:
            sched = [p for p in sched if p.opponent != 'expectimax_bayesian']
            if not sched:
                raise ValueError('schedule vacío tras excluir expectimax')
        expand_sched = expand_phases_from_exclusive(sched)
    else:
        expand_sched = [
            ExpandPhase(unlock=p.opponent, steps=p.steps, omega=(p.opponent,))
            for p in schedule
        ]

    first_opp = expand_sched[0].omega[0]
    reward = _reward_candidate(reward_mode, self_play=self_play, anneal_frac=0.0)
    venv = None
    env: Optional[VGCEnv] = None
    if num_envs > 1:
        from showdown.rl.vec_env import EnvSpec, make_vec_env

        agent.use_vec_buffer(num_envs)
        any_live = any(o != 'mock' for ph in expand_sched for o in ph.omega)
        venv = make_vec_env(
            EnvSpec(
                opponent=first_opp,
                account_prefix='Ppo',
                combat_family='ppo',
                reward=reward.value,
                obs_kind='ppo',
                mask_kind='slot',
            ),
            num_envs=num_envs,
            seed=seed,
            envs_per_server=envs_per_server,
            base_port=base_port,
            start_ps=start_ps,
        )
        if any_live and first_opp == 'mock':
            # Los servidores de fases live posteriores se aseguran al cambiar de rival.
            pass
    else:
        env = _make_env(first_opp, seed=seed, start_ps=start_ps, reward=reward)
    t_total = schedule_total_steps(expand_sched)
    hb = TrainHeartbeat(
        tag='ppo',
        t_total=max(1, t_total),
        log_every=int(log_every),
        log_every_sec=float(log_every_sec),
    )

    try:
        assert env is not None or venv is not None
        for pi, phase in enumerate(expand_sched):
            probs = omega_sampling_probs(phase.omega, weights=weights)
            mode = 'self-play' if self_play else (
                'expand-Omega' if expand_omega else 'exclusive'
            )
            print(
                f'[ppo] === phase {pi + 1}/{len(expand_sched)} ({mode}): '
                f'unlock={phase.unlock} steps={phase.steps} '
                f'Omega={list(phase.omega)} '
                f'P={ {k: round(v, 3) for k, v in probs.items()} } ===',
                flush=True,
            )
            if venv is not None:
                if phase.unlock != 'mock':
                    _ensure_vec_servers(venv, start_ps=start_ps)
                done, info = _collect_and_update_vec(
                    agent, venv,
                    steps=phase.steps,
                    omega=phase.omega,
                    expand_omega=expand_omega and not self_play,
                    weights=weights,
                    rng=rng,
                    global_step_offset=global_steps,
                    ckpt_dir=ckpt_dir,
                    heartbeat=hb,
                    phase_label=f'phase={pi}:{phase.unlock}',
                    unlock=phase.unlock,
                )
                global_steps += done
                phase_log.append({
                    'index': pi,
                    'unlock': phase.unlock,
                    'omega': list(phase.omega),
                    'steps_done': done,
                    'sampling_probs': probs,
                    'ep_counts': info.get('ep_counts', {}),
                    'worker_restarts': info.get('worker_restarts', 0),
                    'num_envs': num_envs,
                })
                agent.save(ckpt_dir / f'phase_{pi}_{phase.unlock}.pt')
                hb.log(
                    global_steps,
                    phase=f'phase={pi}:{phase.unlock}:end',
                    extras={'envs': num_envs},
                    force=True,
                )
                continue
            env = _ensure_opponent(
                env, phase.unlock, seed=seed, start_ps=start_ps, reward=reward,
            )
            env, done, info = _collect_and_update(
                agent, env,
                steps=phase.steps,
                omega=phase.omega,
                expand_omega=expand_omega and not self_play,
                weights=weights,
                rng=rng,
                global_step_offset=global_steps,
                ckpt_dir=ckpt_dir,
                heartbeat=hb,
                phase_label=f'phase={pi}:{phase.unlock}',
                seed=seed,
                start_ps=start_ps,
                reward_mode=reward_mode,
                self_play=self_play,
                pool=pool,
            )
            global_steps += done
            if pool is not None:
                pool.push(agent, tag=f'gate_{pi}')
            phase_log.append({
                'index': pi,
                'unlock': phase.unlock,
                'omega': list(phase.omega),
                'steps_done': done,
                'sampling_probs': probs,
                'ep_counts': info.get('ep_counts', {}),
                'pool_size': len(pool) if pool else 0,
            })
            agent.save(ckpt_dir / f'phase_{pi}_{phase.unlock}.pt')
            hb.log(
                global_steps,
                phase=f'phase={pi}:{phase.unlock}:end',
                extras={'pool': len(pool) if pool else 0},
                force=True,
            )

        agent.save(ckpt_dir / 'final.pt')
        n_suite = resolve_suite_games(suite_games=suite_games, smoke=smoke)
        suite_rows, suite_note = [], ''
        if n_suite > 0 and env is None and venv is not None:
            last_opp = expand_sched[-1].unlock
            venv.close()
            venv = None
            env = _make_env(last_opp, seed=seed, start_ps=start_ps, reward=reward)
        if n_suite > 0 and env is not None:
            print(
                f'[ppo] offline suite: {n_suite} BO1 × {len(SUITE_FOES)} foes',
                flush=True,
            )
            suite_rows, suite_note = run_suite_ppo_act(
                agent, env, n_games=n_suite, seed=seed,
            )
        report = TrainReport(
            agent='ppo',
            seed=seed,
            device=device,
            smoke=smoke,
            global_steps=global_steps,
            t_total=max(1, t_total),
            elapsed_sec=hb.elapsed(),
            steps_per_sec=hb.rate(max(1, global_steps)),
            config={
                'expand_omega': bool(expand_omega),
                'self_play': bool(self_play),
                'reward_mode': reward_mode,
                'weights': weights,
                'lr': float(cfg.lr),
                'clip_eps': float(cfg.clip_eps),
                'num_envs': int(num_envs),
                'envs_per_server': int(envs_per_server),
            },
            metrics={
                'env_steps': agent.env_steps,
                'update_steps': agent.update_steps,
                'pool_size': len(pool) if pool else 0,
            },
            phases=phase_log,
            artifacts={
                'final_pt': str(ckpt_dir / 'final.pt'),
                'train_meta_json': str(ckpt_dir / 'train_meta.json'),
                'checkpoint_dir': str(ckpt_dir),
            },
            suite_games_per_foe=n_suite,
            suite=suite_rows,
            suite_note=suite_note,
            notes=[
                'Train uses Cand. 2 (or anneal); suite uses Cand. 1 sparse.',
                'Self-play runs may need live PS for meaningful suite foes.',
                'Live console heartbeats complement this static report.',
            ],
        )
        write_train_report(ckpt_dir, report)
        meta = {
            'seed': seed,
            'device': device,
            'expand_omega': bool(expand_omega),
            'self_play': bool(self_play),
            'reward_mode': reward_mode,
            'weights': weights,
            'phase_log': phase_log,
            'env_steps': agent.env_steps,
            'update_steps': agent.update_steps,
            'total_steps': global_steps,
            'suite': [asdict(s) for s in suite_rows],
        }
        (ckpt_dir / 'train_meta.json').write_text(
            json.dumps(meta, indent=2) + '\n', encoding='utf-8',
        )
        mean_wr = (
            float(sum(s.win_rate for s in suite_rows) / len(suite_rows))
            if suite_rows else None
        )
        hb.log(
            global_steps,
            phase='DONE',
            extras={
                'expand_omega': int(bool(expand_omega)),
                'self_play': int(bool(self_play)),
                'MEAN_WR': mean_wr,
                'reporte': str(ckpt_dir / 'reporte.txt'),
            },
            force=True,
        )
    finally:
        if venv is not None:
            venv.close()
        if env is not None:
            env.close()

    return agent


def _ensure_vec_servers(venv, *, start_ps: bool) -> None:
    """Asegura que los servidores Showdown de los workers estén arriba."""
    ports = sorted({s.port for s in venv.specs if s.port is not None})
    if not ports:
        return
    from showdown.inference_sim.ps_server import ensure_ps_ports
    ensure_ps_ports(ports, start=start_ps)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description='Train Masked PPO (auto-regressive Slot1→Slot2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cpu', help='cpu | cuda | cuda:N | auto')
    p.add_argument('--steps', type=int, default=20_000)
    p.add_argument(
        '--opponent', default='mock',
        help='mock|random|greedy_solo|greedy_joint|expectimax_bayesian|curriculum|self',
    )
    p.add_argument('--phases', default=None)
    p.add_argument(
        '--curriculum-phase-steps', type=int, default=DEFAULT_CURRICULUM_PHASE_STEPS,
    )
    p.add_argument('--expand-omega', dest='expand_omega', action='store_true', default=True)
    p.add_argument('--no-expand-omega', dest='expand_omega', action='store_false')
    p.add_argument('--include-expectimax', action='store_true', default=True)
    p.add_argument('--no-expectimax', dest='include_expectimax', action='store_false')
    p.add_argument('--opponent-weights', default=None)
    p.add_argument('--reward', default='candidate2',
                   help='candidate2|candidate1|anneal')
    p.add_argument('--bc-checkpoint', default=None)
    p.add_argument('--rollout-steps', type=int, default=None,
                   help='Override rollout length (default 2048; use 256 for smoke)')
    p.add_argument('--checkpoint-dir', default='logs/ppo_checkpoints')
    p.add_argument('--log-every', type=int, default=5_000,
                   help='Heartbeat every N env steps (also --log-every-sec)')
    p.add_argument('--log-every-sec', type=float, default=60.0,
                   help='Heartbeat at least every N seconds')
    p.add_argument('--start-ps', action='store_true')
    p.add_argument(
        '--suite-games', type=int, default=None,
        help='BO1 per foe for offline suite (default 20; 0=skip; smoke default 0)',
    )
    p.add_argument('--smoke', action='store_true', help='Treat as smoke (suite default 0)')
    p.add_argument('--lr', type=float, default=None, help='Override PPOConfig.lr')
    p.add_argument('--clip-eps', type=float, default=None, help='Override PPOConfig.clip_eps')
    p.add_argument('--num-envs', type=int, default=1,
                   help='Entornos paralelos (procesos). Servidor: ~24; 0 = automático')
    p.add_argument('--envs-per-server', type=int, default=4,
                   help='Entornos por servidor Showdown (puertos base-port..)')
    p.add_argument('--base-port', type=int, default=None,
                   help='Primer puerto Showdown (default PS_PORT=8000)')
    args = p.parse_args(argv)
    if args.num_envs == 0:
        from showdown.rl.parallel import default_num_envs
        args.num_envs = default_num_envs()

    self_play = str(args.opponent).lower().replace('-', '_') == 'self'
    if self_play:
        schedule = resolve_schedule(
            opponent='mock',
            steps=args.steps,
            phases=args.phases,
            curriculum_phase_steps=args.curriculum_phase_steps,
        ) if args.phases else [TrainPhase('mock', int(args.steps))]
    else:
        schedule = resolve_schedule(
            opponent=args.opponent,
            steps=args.steps,
            phases=args.phases,
            curriculum_phase_steps=args.curriculum_phase_steps,
        )

    expand = bool(args.expand_omega)
    if args.opponent == 'curriculum' and args.phases is None:
        expand = True

    overrides = parse_weight_overrides(args.opponent_weights)
    print(
        '[ppo] schedule:',
        ', '.join(f'{ph.opponent}:{ph.steps}' for ph in schedule),
        f'| expand_omega={expand} self_play={self_play} reward={args.reward}',
    )
    run_training(
        seed=args.seed,
        schedule=schedule,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        log_every_sec=args.log_every_sec,
        start_ps=bool(args.start_ps),
        expand_omega=expand,
        include_expectimax=bool(args.include_expectimax),
        weight_overrides=overrides or None,
        reward_mode=args.reward,
        bc_checkpoint=args.bc_checkpoint,
        self_play=self_play,
        rollout_steps=args.rollout_steps,
        suite_games=args.suite_games,
        smoke=bool(args.smoke),
        lr=args.lr,
        clip_eps=args.clip_eps,
        num_envs=args.num_envs,
        envs_per_server=args.envs_per_server,
        base_port=args.base_port,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
