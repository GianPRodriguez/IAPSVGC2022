"""DDQN training with exclusive or monotonic-Ω opponent schedules.

Examples::

  # Offline smoke
  python -m showdown.rl.ddqn.train --seed 0 --steps 20000 --opponent mock

  # Exclusive phases (one foe at a time; shared replay)
  python -m showdown.rl.ddqn.train --seed 0 --start-ps --no-expand-omega ^
    --phases random:100k,greedy_solo:200k,greedy_joint:300k

  # Monotonic Ω (default for curriculum / recommended)
  python -m showdown.rl.ddqn.train --seed 0 --start-ps --expand-omega ^
    --phases random:100k,greedy_solo:200k,greedy_joint:300k,expectimax_bayesian:50k

  # Spec expand curriculum (random→solo→joint→expectimax)
  python -m showdown.rl.ddqn.train --seed 0 --opponent curriculum --start-ps
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from showdown.rl.action_space import (
    build_slot_mask,
    decode_joint_action,
    decode_slot_action,
    legal_joint_mask,
    legal_joint_mask_from_slot_masks,
)
from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
from showdown.rl.train_log import TrainHeartbeat, schedule_total_steps
from showdown.rl.train_report import (
    SUITE_FOES,
    TrainReport,
    resolve_suite_games,
    run_suite_ddqn_act,
    write_train_report,
)
from showdown.rl.ddqn.curriculum import (
    ExpandPhase,
    expand_phases_from_exclusive,
    merge_weights,
    omega_sampling_probs,
    opponent_code,
    parse_weight_overrides,
    sample_from_omega,
)
from showdown.rl.env import MockBattleBackend, VGCEnv
from showdown.rl.observation import build_observation_ddqn
from showdown.rl.reward import RewardCandidate
from showdown.rl.train_metrics import EpisodeTracker, MetricsLog

LIVE_OPPONENTS = (
    'random',
    'greedy_solo',
    'greedy_joint',
    'expectimax_bayesian',
)
OFFLINE_OPPONENTS = ('mock',)
ALL_OPPONENTS = OFFLINE_OPPONENTS + LIVE_OPPONENTS + ('curriculum',)

DEFAULT_CURRICULUM_PHASE_STEPS = 500_000


@dataclass(frozen=True)
class TrainPhase:
    opponent: str
    steps: int


def _parse_steps_token(raw: str) -> int:
    s = str(raw).strip().lower().replace('_', '')
    m = re.fullmatch(r'(\d+(?:\.\d+)?)([kmb]?)', s)
    if not m:
        raise ValueError(f'pasos inválidos: {raw!r} (ej. 100000, 100k, 1.5m)')
    val = float(m.group(1))
    suf = m.group(2)
    mult = {'': 1, 'k': 1_000, 'm': 1_000_000, 'b': 1_000_000_000}[suf]
    n = int(val * mult)
    if n <= 0:
        raise ValueError(f'pasos deben ser > 0: {raw!r}')
    return n


def parse_phases(spec: str) -> List[TrainPhase]:
    text = str(spec or '').strip()
    if not text:
        raise ValueError('--phases vacío')
    out: List[TrainPhase] = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if ':' not in part:
            raise ValueError(
                f'fase inválida {part!r}; usa oponente:pasos '
                f'(ej. random:100000,greedy_joint:50k)'
            )
        opp, steps_s = part.split(':', 1)
        opp_key = opp.strip().lower().replace('-', '_')
        if opp_key not in OFFLINE_OPPONENTS and opp_key not in LIVE_OPPONENTS:
            raise ValueError(
                f'oponente desconocido {opp!r}. '
                f'Opciones: {", ".join(OFFLINE_OPPONENTS + LIVE_OPPONENTS)}'
            )
        out.append(TrainPhase(opponent=opp_key, steps=_parse_steps_token(steps_s)))
    if not out:
        raise ValueError('--phases no contiene fases válidas')
    return out


def default_curriculum_phases(
    phase_steps: int = DEFAULT_CURRICULUM_PHASE_STEPS,
) -> List[TrainPhase]:
    """Exclusive unlock markers (used to build ExpandPhase when expand-omega)."""
    n = int(phase_steps)
    return [
        TrainPhase('random', n),
        TrainPhase('greedy_solo', n),
        TrainPhase('greedy_joint', n),
        TrainPhase('expectimax_bayesian', n),
    ]


def resolve_schedule(
    *,
    opponent: str,
    steps: int,
    phases: Optional[str],
    curriculum_phase_steps: int = DEFAULT_CURRICULUM_PHASE_STEPS,
) -> List[TrainPhase]:
    if phases:
        return parse_phases(phases)
    key = str(opponent or 'mock').strip().lower().replace('-', '_')
    if key == 'curriculum':
        return default_curriculum_phases(curriculum_phase_steps)
    if key not in OFFLINE_OPPONENTS and key not in LIVE_OPPONENTS:
        raise ValueError(
            f'--opponent inválido: {opponent!r}. '
            f'Opciones: {", ".join(ALL_OPPONENTS)} o usa --phases'
        )
    return [TrainPhase(opponent=key, steps=int(steps))]


def _joint_masks(env: VGCEnv):
    assert env._snap is not None
    live = env.backend.action_masks(slot1_action_id=None)
    if live is not None:
        m1, _ = live

        def _m2(a1: int):
            pair = env.backend.action_masks(slot1_action_id=a1)
            if pair is not None:
                return pair[1]
            return build_slot_mask(env._snap, 1, other_slot_action=decode_slot_action(a1))

        return legal_joint_mask_from_slot_masks(m1, _m2)
    return legal_joint_mask(env._snap)


# Equipo fijo de DDQN (ddqn_spec 1.14): el agente juega SIEMPRE con este equipo; los rivales
# sortean los suyos del grupo `train`. '' = el equipo del agente también se sortea del pool.
DEFAULT_AGENT_TEAM = str(Path(__file__).resolve().parents[1] / 'data' / 'ddqn_team.txt')
_AGENT_TEAM = ''  # lo fija run_training (los sitios que crean un entorno suelto lo leen)


def _make_env(opponent: str, *, seed: int, start_ps: bool) -> VGCEnv:
    if opponent == 'mock':
        backend = MockBattleBackend(max_turns=40)
    else:
        from showdown.rl.pokeenv_backend import PokeEnvBattleBackend
        our_team = None
        if _AGENT_TEAM:
            from showdown.inference_sim.team_factory import load_team_file
            our_team = load_team_file(_AGENT_TEAM)
        backend = PokeEnvBattleBackend(
            opponent=opponent,
            start_ps=start_ps,
            account_prefix=f'Ddqn{opponent[:4].title()}',
            combat_family='ddqn',
            our_team=our_team,
        )
    return VGCEnv(
        reward_candidate=RewardCandidate.DENSE,
        backend=backend,
        max_episode_turns=40,
    )


def _ensure_opponent(env: VGCEnv, opponent: str, *, seed: int, start_ps: bool) -> VGCEnv:
    """Return env whose backend plays ``opponent`` (may replace env)."""
    if opponent == 'mock':
        if isinstance(env.backend, MockBattleBackend):
            return env
        env.close()
        return _make_env('mock', seed=seed, start_ps=False)

    backend = env.backend
    if hasattr(backend, 'set_opponent') and hasattr(backend, 'opponent'):
        if getattr(backend, 'opponent', None) != opponent:
            backend.set_opponent(opponent)
        return env
    env.close()
    return _make_env(opponent, seed=seed, start_ps=start_ps)


def _learner_snapshot(agent: DDQNAgent, metrics_last: dict) -> dict:
    """Estado del learner para la ventana de métricas (tesis)."""
    snap = {k: v for k, v in (metrics_last or {}).items()
            if k in ('loss', 'mean_q', 'mean_target', 'td_error') or k.startswith('sym_')}
    snap.update({
        'eps': float(agent.epsilon),
        'env_steps': int(agent.env_steps),
        'grad_steps': int(agent.grad_steps),
        'replay_len': len(agent.replay),
        'replay_frac': agent.replay.origin_fractions(),
    })
    return snap


def _run_steps(
    agent: DDQNAgent,
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
    tracker: Optional[EpisodeTracker] = None,
    eval_hook=None,
) -> Tuple[VGCEnv, int, dict]:
    """
    Run ``steps`` interactions.

    If expand_omega: on each episode boundary sample foe from Ω.
    Else: keep the single foe already set on env (exclusive).
    """
    current = getattr(getattr(env, 'backend', None), 'opponent', None) or (
        omega[-1] if omega else 'mock'
    )
    if expand_omega and current not in omega and omega:
        current = omega[0]
        env = _ensure_opponent(env, current, seed=seed, start_ps=start_ps)

    obs, info = env.reset(seed=seed + global_step_offset)
    assert env._snap is not None
    obs = build_observation_ddqn(env._snap)
    mask = _joint_masks(env)
    metrics_last: dict = {}
    opp_code = opponent_code(current)
    ep_counts: Dict[str, int] = {o: 0 for o in omega}
    ep_counts[current] = ep_counts.get(current, 0) + 1
    if tracker is not None:
        tracker.start(0, current)

    for local in range(1, int(steps) + 1):
        a_joint = agent.act(obs, mask, epsilon=None)
        a1, a2 = decode_joint_action(a_joint)
        next_obs, reward, terminated, truncated, info = env.step((a1, a2))
        assert env._snap is not None
        next_obs = build_observation_ddqn(env._snap)
        skip = bool(info.get('skip_buffer'))
        eng = info.get('engine') or {}
        used_fb = bool(isinstance(eng, dict) and eng.get('used_fallback'))
        if used_fb:
            skip = True
        if tracker is not None:
            tracker.step(0, actions=(a1, a2), reward=float(reward),
                         components=info.get('reward_components'),
                         used_fallback=used_fb, skipped=skip)
        m_next = _joint_masks(env) if not (terminated or truncated) else mask
        train_m = agent.observe(
            s=obs, a=a_joint, r=float(reward), s_next=next_obs,
            terminated=bool(terminated), m_next=m_next,
            skip_buffer=skip, truncated=bool(truncated),
            opponent_code=opp_code,
        )
        if train_m:
            metrics_last = train_m
        obs, mask = next_obs, m_next

        if terminated or truncated:
            ended_info = info
            if expand_omega and len(omega) > 1:
                nxt = sample_from_omega(omega, rng, weights=weights)
                if nxt != current:
                    env = _ensure_opponent(env, nxt, seed=seed, start_ps=start_ps)
                    current = nxt
                opp_code = opponent_code(current)
                ep_counts[current] = ep_counts.get(current, 0) + 1
            if tracker is not None:
                tracker.end(0, global_step=global_step_offset + local,
                            terminated=bool(terminated), truncated=bool(truncated),
                            components=ended_info.get('reward_components'),
                            trunc_reason=ended_info.get('trunc_reason'),
                            next_opponent=current,
                            foe_state=(ended_info.get('engine') or {}).get('foe_state')
                            if isinstance(ended_info.get('engine'), dict) else None)
            obs, info = env.reset()
            assert env._snap is not None
            obs = build_observation_ddqn(env._snap)
            mask = _joint_masks(env)

        gstep = global_step_offset + local
        loss = metrics_last.get('loss', float('nan'))
        extras = {
            'opp': current,
            'eps': float(agent.epsilon),
            'replay': len(agent.replay),
            'loss': float(loss) if loss == loss else None,
        }
        if tracker is not None and heartbeat.should_log(gstep):
            tracker.write_window(gstep, phase=phase_label,
                                 learner=_learner_snapshot(agent, metrics_last))
            extras.update(tracker.console_extras())
        heartbeat.log(gstep, phase=phase_label, extras=extras)
        if heartbeat.should_checkpoint(gstep):
            agent.save(ckpt_dir / 'last.pt')
            heartbeat.mark_checkpoint(gstep)
        if eval_hook is not None:
            eval_hook(gstep)

    return env, int(steps), {'metrics': metrics_last, 'ep_counts': ep_counts}


def _run_steps_vec(
    agent: DDQNAgent,
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
    tracker: Optional[EpisodeTracker] = None,
    eval_hook=None,
) -> Tuple[int, dict]:
    """Como :func:`_run_steps` pero con N entornos en paralelo (asíncrono).

    ``steps`` cuenta pasos de entorno totales. Cada transición pasa por
    ``agent.observe`` (misma razón actualización/dato que con 1 entorno).
    """
    n = int(venv.num_envs)
    obs, masks = venv.set_opponent(unlock)
    current = [unlock] * n

    def _sample_next() -> Optional[str]:
        if expand_omega and len(omega) > 1:
            return sample_from_omega(omega, rng, weights=weights)
        return None

    pending = [_sample_next() for _ in range(n)]
    if tracker is not None:
        for i in range(n):
            tracker.start(i, unlock)
    ep_counts: Dict[str, int] = {o: 0 for o in omega}
    ep_counts[unlock] = ep_counts.get(unlock, 0) + n
    metrics_last: dict = {}
    local = 0
    issued = 0
    restarts = 0
    flight: Dict[int, tuple] = {}  # i -> (a_joint, obs, mask)
    ready = list(range(n))

    while local < int(steps):
        idx = ready[:max(0, int(steps) - issued)]
        if idx:
            a_joint = agent.act_batch(obs[idx], masks[idx], epsilon=None)
            venv.step_async(
                idx,
                [decode_joint_action(int(a)) for a in a_joint],
                [pending[i] for i in idx],
            )
            for k, i in enumerate(idx):
                flight[i] = (int(a_joint[k]), obs[i].copy(), masks[i].copy())
            issued += len(idx)
        if not flight:
            break
        ready = []
        for i, r in venv.step_wait_any():
            a_i, obs_i, mask_i = flight.pop(i)
            info = r['info']
            done = bool(r['terminated'] or r['truncated'])
            skip = bool(info['skip_buffer'] or info['used_fallback'])
            if tracker is not None:
                tracker.step(i, actions=decode_joint_action(a_i), reward=float(r['reward']),
                             components=info.get('components'),
                             used_fallback=bool(info['used_fallback']), skipped=skip)
            s_next = r['final_obs'] if done else r['obs']
            m_next = mask_i if done else r['mask']
            train_m = agent.observe(
                s=obs_i, a=a_i, r=float(r['reward']), s_next=s_next,
                terminated=bool(r['terminated']), m_next=m_next,
                skip_buffer=skip, truncated=bool(r['truncated']),
                opponent_code=opponent_code(current[i]),
            )
            if train_m:
                metrics_last = train_m
            if r.get('worker_restarted'):
                restarts += 1
            if done:
                current[i] = r.get('next_opponent') or current[i]
                ep_counts[current[i]] = ep_counts.get(current[i], 0) + 1
                pending[i] = _sample_next()
                if tracker is not None:
                    tracker.end(i, global_step=global_step_offset + local + 1,
                                terminated=bool(r['terminated']), truncated=bool(r['truncated']),
                                components=info.get('components'),
                                trunc_reason=info.get('trunc_reason'),
                                next_opponent=current[i],
                                foe_state=info.get('foe_state'))
            obs[i] = r['obs']
            masks[i] = r['mask']
            local += 1
            ready.append(i)

        gstep = global_step_offset + local
        loss = metrics_last.get('loss', float('nan'))
        extras = {
            'envs': n,
            'opp': unlock,
            'eps': float(agent.epsilon),
            'replay': len(agent.replay),
            'loss': float(loss) if loss == loss else None,
            'restarts': restarts,
        }
        if tracker is not None and heartbeat.should_log(gstep):
            tracker.write_window(gstep, phase=phase_label,
                                 learner=_learner_snapshot(agent, metrics_last),
                                 worker_restarts=restarts)
            extras.update(tracker.console_extras())
        heartbeat.log(gstep, phase=phase_label, extras=extras)
        if heartbeat.should_checkpoint(gstep):
            agent.save(ckpt_dir / 'last.pt')
            heartbeat.mark_checkpoint(gstep)
        if eval_hook is not None:
            eval_hook(gstep)

    for i, r in venv.drain():
        flight.pop(i, None)
    return int(local), {
        'metrics': metrics_last, 'ep_counts': ep_counts, 'worker_restarts': restarts,
    }


# --------------------------------------------------------------------------- #
# Validación periódica y mejor checkpoint (ddqn_spec 1.16; protocolo §3c)
# --------------------------------------------------------------------------- #
# Rivales DETERMINISTAS (sin sorteo de casi-empates), los 4 heurísticos y semillas fijas: las
# semillas fijan los equipos rivales (grupo `train`), así que todas las validaciones de una
# corrida juegan exactamente los mismos emparejamientos. Nunca usa los equipos de `eval`.
VAL_FOES: Tuple[str, ...] = ('random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian')
VAL_SEED = 900_000
VAL_PORT_OFFSET = 100  # servidor Showdown propio para la validación (base_port + 100)


def _make_eval_env(*, start_ps: bool, port: Optional[int]) -> VGCEnv:
    from showdown.rl.foe_pool import validation_split
    from showdown.rl.pokeenv_backend import PokeEnvBattleBackend
    our_team = None
    if _AGENT_TEAM:
        from showdown.inference_sim.team_factory import load_team_file
        our_team = load_team_file(_AGENT_TEAM)
    backend = PokeEnvBattleBackend(
        opponent=VAL_FOES[0], start_ps=start_ps, port=port, account_prefix='DdqnVal',
        combat_family='ddqn', our_team=our_team, foe_tie=False,
        team_split=validation_split(),  # 'val' con RL_HOLDOUT_VAL=1 (equipos no entrenados)
    )
    return VGCEnv(reward_candidate=RewardCandidate.DENSE, backend=backend, max_episode_turns=40)


def _foe_pool_meta() -> dict:
    """Opciones del pool de rivales (showdown/rl/foe_pool.py) para el registro de la corrida."""
    from showdown.rl.foe_pool import holdout_val, synth_prob, train_subset, training_split
    import os as _os
    return {'split': training_split(), 'holdout_val': holdout_val(), 'subset': train_subset(),
            'synth_prob': synth_prob(), 'synth_file': _os.environ.get('RL_FOE_SYNTH_FILE', '')}


def _restore_best(validator, ckpt_dir: Path, resume: str) -> Optional[dict]:
    """Al reanudar, recupera el mejor checkpoint previo (``best.json``) para no reemplazarlo por uno peor.

    Busca primero en el directorio de la corrida y después en el del checkpoint de origen; si el
    mejor está en otro directorio, copia su ``best.pt`` y ``best.json`` aquí.
    """
    import shutil
    for d in (Path(ckpt_dir), Path(resume).parent if resume else None):
        if d is None or not (d / 'best.json').is_file():
            continue
        try:
            rec = json.loads((d / 'best.json').read_text(encoding='utf-8'))
            float(rec['mean_wr'])
        except Exception:
            continue
        if d.resolve() != Path(ckpt_dir).resolve():
            if not (d / 'best.pt').is_file():
                continue
            shutil.copy2(d / 'best.pt', Path(ckpt_dir) / 'best.pt')
            shutil.copy2(d / 'best.json', Path(ckpt_dir) / 'best.json')
        validator.best = rec
        print(f"[resume] mejor checkpoint previo: paso {rec.get('step')} media={float(rec['mean_wr']):.3f}", flush=True)
        return rec
    return None


class _Validator:
    """Evalúa la política sin exploración cada ``every`` pasos y guarda ``best.pt``.

    Criterio del mejor checkpoint: tasa de victorias media contra los 4 rivales de validación.
    """

    def __init__(self, agent, ckpt_dir: Path, mlog, *, every: int, games: int,
                 start_ps: bool, port: Optional[int]):
        self.agent, self.ckpt_dir, self.mlog = agent, Path(ckpt_dir), mlog
        self.every, self.games = int(every), int(games)
        self.start_ps, self.port = bool(start_ps), port
        self.next = self.every
        self.best: Optional[dict] = None
        self.history: List[dict] = []
        self.env: Optional[VGCEnv] = None

    @property
    def enabled(self) -> bool:
        return self.every > 0 and self.games > 0

    def maybe(self, gstep: int) -> None:
        if not self.enabled or gstep < self.next:
            return
        while self.next <= gstep:
            self.next += self.every
        self.run(gstep)

    def run(self, gstep: int) -> Optional[dict]:
        if not self.enabled:
            return None
        import time as _time
        from showdown.rl.train_report import run_suite_ddqn_act
        if self.env is None:
            self.env = _make_eval_env(start_ps=self.start_ps, port=self.port)
        t0 = _time.time()
        rows, _note = run_suite_ddqn_act(self.agent, self.env, foes=VAL_FOES,
                                         n_games=self.games, seed=VAL_SEED)
        mean = float(np.mean([r.win_rate for r in rows])) if rows else float('nan')
        rec = {'step': int(gstep), 'mean_wr': mean, 'games_per_foe': self.games,
               'rows': [dict(vars(r)) for r in rows], 'sec': round(_time.time() - t0, 1)}
        self.history.append(rec)
        self.mlog.write('eval', **rec)
        self.agent.save(self.ckpt_dir / f'step_{int(gstep)}.pt')
        improved = self.best is None or mean > float(self.best['mean_wr'])
        if improved:
            self.best = rec
            self.agent.save(self.ckpt_dir / 'best.pt')
            (self.ckpt_dir / 'best.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
        by = ' '.join(f'{r.foe[:6]}={r.win_rate:.2f}' for r in rows)
        print(f'[ddqn] validación paso {gstep}: media={mean:.3f} | {by} | {rec["sec"]}s'
              f'{" | NUEVO MEJOR -> best.pt" if improved else ""}', flush=True)
        return rec

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None


def _default_ps_port() -> int:
    from showdown.inference_sim.paths import PS_PORT
    return int(PS_PORT)


def run_training(
    *,
    seed: int = 0,
    schedule: Sequence[TrainPhase],
    device: str = 'cpu',
    checkpoint_dir: str | Path = 'logs/ddqn_checkpoints',
    log_every: int = 5_000,
    log_every_sec: float = 60.0,
    start_ps: bool = False,
    expand_omega: bool = True,
    include_expectimax: bool = True,
    weight_overrides: Optional[Dict[str, float]] = None,
    suite_games: Optional[int] = None,
    smoke: bool = False,
    lr: Optional[float] = None,
    gamma: Optional[float] = None,
    target_update_every: Optional[int] = None,
    target_clip: Optional[float] = None,
    eval_every: int = 250_000,
    eval_games: int = 50,
    resume: str = '',
    layer_norm: bool = False,
    weight_decay: Optional[float] = None,
    dropout: Optional[float] = None,
    foe_bench_dropout: Optional[float] = None,
    train_freq: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_envs: int = 1,
    envs_per_server: int = 4,
    base_port: Optional[int] = None,
    agent_team: str = DEFAULT_AGENT_TEAM,
) -> DDQNAgent:
    if not schedule:
        raise ValueError('schedule vacío')
    global _AGENT_TEAM
    _AGENT_TEAM = str(agent_team or '')
    if _AGENT_TEAM:
        print(f'[ddqn] equipo fijo del agente: {_AGENT_TEAM}', flush=True)
    from showdown.rl.parallel import describe_resources, raise_nofile_limit, resolve_device

    raise_nofile_limit()
    device = resolve_device(device)
    num_envs = max(1, int(num_envs))
    print(f'[ddqn] recursos: {describe_resources()} device={device} num_envs={num_envs}', flush=True)

    weights = merge_weights(weight_overrides)
    cfg = DDQNConfig(seed=seed, device=device)
    if lr is not None:
        cfg.lr = float(lr)
    if gamma is not None:
        cfg.gamma = float(gamma)
    cfg.layer_norm = bool(layer_norm)
    if weight_decay is not None:  # None = valor por defecto de DDQNConfig (REG, ddqn_spec 1.20)
        cfg.weight_decay = float(weight_decay)
    if dropout is not None:
        cfg.dropout = float(dropout)
    if foe_bench_dropout is not None:
        cfg.foe_bench_dropout = float(foe_bench_dropout)
    if train_freq is not None:
        cfg.train_freq = int(train_freq)
    if batch_size is not None:
        cfg.batch_size = int(batch_size)
    if target_update_every is not None:
        cfg.target_update_every = int(target_update_every)
    if target_clip is not None:
        cfg.target_clip = float(target_clip)
    if cfg.target_update_every or cfg.target_clip:
        print(f'[ddqn] objetivo: copia dura cada {cfg.target_update_every} pasos de gradiente '
              f'| clip={cfg.target_clip}', flush=True)
    agent = DDQNAgent(cfg)
    resume_skip = 0
    if resume:
        agent.load(resume, map_location=str(device))
        agent.min_replay_to_learn = int(cfg.warmup_steps)
        resume_skip = int(agent.env_steps)
        print(f'[ddqn] reanudando desde {resume}: {resume_skip} pasos hechos, grad_steps='
              f'{agent.grad_steps}; el replay se rellena antes de volver a aprender', flush=True)
    rng = np.random.default_rng(int(seed) + 17)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_steps = 0
    phase_log: List[dict] = []

    if expand_omega:
        # Drop expectimax from exclusive markers if disabled
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
    venv = None
    env: Optional[VGCEnv] = None
    if num_envs > 1:
        from showdown.rl.vec_env import EnvSpec, make_vec_env

        venv = make_vec_env(
            EnvSpec(
                opponent=first_opp,
                account_prefix='Ddqn',
                combat_family='ddqn',
                reward=RewardCandidate.DENSE.value,
                obs_kind='ddqn',
                mask_kind='joint',
                agent_team=_AGENT_TEAM,
            ),
            num_envs=num_envs,
            seed=seed,
            envs_per_server=envs_per_server,
            base_port=base_port,
            start_ps=start_ps,
        )
    else:
        env = _make_env(first_opp, seed=seed, start_ps=start_ps)
    t_total = schedule_total_steps(expand_sched)
    hb = TrainHeartbeat(
        tag='ddqn',
        t_total=max(1, t_total),
        log_every=int(log_every),
        log_every_sec=float(log_every_sec),
    )

    mlog = MetricsLog(ckpt_dir / 'train_metrics.jsonl', run_meta={
        'agent': 'ddqn', 'seed': seed, 'device': device, 'num_envs': num_envs,
        'foe_pool': _foe_pool_meta(),
        'lr': float(cfg.lr), 'gamma': float(cfg.gamma), 'expand_omega': bool(expand_omega),
        'target_clip': float(cfg.target_clip), 'target_update_every': int(cfg.target_update_every),
        'layer_norm': bool(cfg.layer_norm), 'train_freq': int(cfg.train_freq),
        'regularization': {'weight_decay': float(cfg.weight_decay), 'dropout': float(cfg.dropout),
                           'foe_bench_dropout': float(cfg.foe_bench_dropout)},
        'batch_size': int(cfg.batch_size), 'agent_team': _AGENT_TEAM,
        'schedule': [{'unlock': p.unlock, 'steps': p.steps, 'omega': list(p.omega)}
                     for p in expand_sched],
    })
    tracker = EpisodeTracker(mlog, window=200)
    val_port = (base_port or _default_ps_port()) + VAL_PORT_OFFSET
    offline_only = all(o in OFFLINE_OPPONENTS for p in expand_sched for o in p.omega)
    if offline_only and eval_every > 0:
        # La validación juega contra rivales heurísticos en Showdown: no aplica a corridas solo con mock.
        print('[ddqn] validación desactivada: el calendario solo usa rivales offline (mock)', flush=True)
        eval_every = 0
    validator = _Validator(agent, ckpt_dir, mlog, every=eval_every, games=eval_games,
                           start_ps=start_ps, port=val_port)
    if resume_skip:
        validator.next = (resume_skip // max(1, validator.every) + 1) * max(1, validator.every)
        _restore_best(validator, ckpt_dir, resume)

    try:
        assert env is not None or venv is not None
        for pi, phase in enumerate(expand_sched):
            phase_steps = int(phase.steps)
            if resume_skip >= phase_steps:
                resume_skip -= phase_steps
                global_steps += phase_steps
                print(f'[ddqn] fase {pi + 1} ya completada antes de reanudar ({phase.unlock})', flush=True)
                continue
            global_steps += resume_skip  # parte de esta fase ya hecha antes de reanudar
            phase_steps -= resume_skip
            resume_skip = 0
            probs = omega_sampling_probs(phase.omega, weights=weights)
            mode = 'expand-Omega' if expand_omega else 'exclusive'
            print(
                f'[ddqn] === phase {pi + 1}/{len(expand_sched)} ({mode}): '
                f'unlock={phase.unlock} steps={phase.steps} '
                f'Omega={list(phase.omega)} '
                f'P={ {k: round(v, 3) for k, v in probs.items()} } ===',
                flush=True,
            )
            if venv is not None:
                if phase.unlock != 'mock':
                    from showdown.rl.ppo.train import _ensure_vec_servers
                    _ensure_vec_servers(venv, start_ps=start_ps)
                done, info = _run_steps_vec(
                    agent, venv,
                    steps=phase_steps,
                    omega=phase.omega,
                    expand_omega=expand_omega,
                    weights=weights,
                    rng=rng,
                    global_step_offset=global_steps,
                    ckpt_dir=ckpt_dir,
                    heartbeat=hb,
                    phase_label=f'phase={pi}:{phase.unlock}',
                    unlock=phase.unlock,
                    tracker=tracker,
                    eval_hook=validator.maybe,
                )
                global_steps += done
                phase_log.append({
                    'index': pi,
                    'unlock': phase.unlock,
                    'omega': list(phase.omega),
                    'steps_planned': phase.steps,
                    'steps_done': done,
                    'sampling_probs': probs,
                    'ep_counts': info.get('ep_counts', {}),
                    'buf_frac_end': agent.replay.origin_fractions(),
                    'worker_restarts': info.get('worker_restarts', 0),
                    'num_envs': num_envs,
                })
                agent.save(ckpt_dir / f'phase_{pi}_{phase.unlock}.pt')
                tracker.write_window(global_steps, phase=f'phase={pi}:{phase.unlock}:end',
                                     learner=_learner_snapshot(agent, info.get('metrics', {})))
                hb.log(
                    global_steps,
                    phase=f'phase={pi}:{phase.unlock}:end',
                    extras={'replay': len(agent.replay), 'envs': num_envs},
                    force=True,
                )
                continue
            # Ensure unlock foe exists at phase start
            env = _ensure_opponent(env, phase.unlock, seed=seed, start_ps=start_ps)
            env, done, info = _run_steps(
                agent, env,
                steps=phase_steps,
                omega=phase.omega,
                expand_omega=expand_omega,
                weights=weights,
                rng=rng,
                global_step_offset=global_steps,
                ckpt_dir=ckpt_dir,
                heartbeat=hb,
                phase_label=f'phase={pi}:{phase.unlock}',
                seed=seed,
                start_ps=start_ps,
                tracker=tracker,
                eval_hook=validator.maybe,
            )
            global_steps += done
            phase_log.append({
                'index': pi,
                'unlock': phase.unlock,
                'omega': list(phase.omega),
                'steps_planned': phase.steps,
                'steps_done': done,
                'sampling_probs': probs,
                'ep_counts': info.get('ep_counts', {}),
                'buf_frac_end': agent.replay.origin_fractions(),
            })
            agent.save(ckpt_dir / f'phase_{pi}_{phase.unlock}.pt')
            tracker.write_window(global_steps, phase=f'phase={pi}:{phase.unlock}:end',
                                 learner=_learner_snapshot(agent, info.get('metrics', {})))
            hb.log(
                global_steps,
                phase=f'phase={pi}:{phase.unlock}:end',
                extras={'replay': len(agent.replay)},
                force=True,
            )

        agent.save(ckpt_dir / 'final.pt')
        if validator.enabled and (not validator.history or validator.history[-1]['step'] != global_steps):
            validator.run(global_steps)  # el mejor checkpoint también considera la política final
        validator.close()
        from showdown.rl.train_metrics import FOE_STATE
        _fs = FOE_STATE.flush()  # último tramo + total de la corrida (contraste de estado del rival)
        mlog.write('foe_state', step=int(global_steps), **_fs)
        print(f"[ddqn] contraste de estado del rival: {_fs['total']}", flush=True)
        n_suite = resolve_suite_games(suite_games=suite_games, smoke=smoke)
        suite_rows, suite_note = [], ''
        if n_suite > 0 and env is None and venv is not None:
            last_opp = expand_sched[-1].unlock
            venv.close()
            venv = None
            env = _make_env(last_opp, seed=seed, start_ps=start_ps)
        if n_suite > 0 and env is not None:
            print(
                f'[ddqn] offline suite: {n_suite} BO1 × {len(SUITE_FOES)} foes',
                flush=True,
            )
            suite_rows, suite_note = run_suite_ddqn_act(
                agent, env, n_games=n_suite, seed=seed,
            )
        report = TrainReport(
            agent='ddqn',
            seed=seed,
            device=device,
            smoke=smoke,
            global_steps=global_steps,
            t_total=max(1, t_total),
            elapsed_sec=hb.elapsed(),
            steps_per_sec=hb.rate(max(1, global_steps)),
            config={
                'expand_omega': bool(expand_omega),
                'weights': weights,
                'schedule': [asdict(p) for p in schedule],
                'lr': float(cfg.lr),
                'gamma': float(cfg.gamma),
                'num_envs': int(num_envs),
                'envs_per_server': int(envs_per_server),
            },
            metrics={
                'best_validation': validator.best,
                'validations': [{'step': h['step'], 'mean_wr': h['mean_wr']} for h in validator.history],
                'env_steps': agent.env_steps,
                'grad_steps': agent.grad_steps,
                'epsilon_final': float(agent.epsilon),
                'replay_len': len(agent.replay),
                'buf_frac_final': agent.replay.origin_fractions(),
            },
            phases=phase_log,
            artifacts={
                'train_metrics_jsonl': str(ckpt_dir / 'train_metrics.jsonl'),
                'final_pt': str(ckpt_dir / 'final.pt'),
                'train_meta_json': str(ckpt_dir / 'train_meta.json'),
                'checkpoint_dir': str(ckpt_dir),
            },
            suite_games_per_foe=n_suite,
            suite=suite_rows,
            suite_note=suite_note,
            notes=[
                'Primary learning signal: Cand. 2 during train; suite uses Cand. 1 sparse return.',
                'Compare MEAN_WR / per-foe WR across seeds for thesis tables.',
                'Live console heartbeats complement this static report.',
            ],
        )
        write_train_report(ckpt_dir, report)
        mlog.write('suite', step=global_steps, rows=[asdict(s) for s in suite_rows], note=suite_note)
        meta = {
            'seed': seed,
            'device': device,
            'expand_omega': bool(expand_omega),
            'weights': weights,
            'exclusive_schedule': [asdict(p) for p in schedule],
            'expand_schedule': [
                {'unlock': p.unlock, 'steps': p.steps, 'omega': list(p.omega)}
                for p in expand_sched
            ],
            'phase_log': phase_log,
            'env_steps': agent.env_steps,
            'grad_steps': agent.grad_steps,
            'total_steps': global_steps,
            'buf_frac_final': agent.replay.origin_fractions(),
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
                'MEAN_WR': mean_wr,
                'reporte': str(ckpt_dir / 'reporte.txt'),
            },
            force=True,
        )
    finally:
        mlog.close()
        if venv is not None:
            venv.close()
        if env is not None:
            env.close()

    return agent


def _main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description='Train joint DDQN with exclusive or monotonic-Ω curricula',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Ejemplos:\n'
            '  --opponent mock --steps 20000\n'
            '  --opponent curriculum --start-ps          # expand-Ω (default)\n'
            '  --phases random:50k,greedy_solo:100k,greedy_joint:100k --start-ps\n'
            '  --no-expand-omega --phases random:50k,greedy_joint:50k --start-ps\n'
            '  --opponent-weights expectimax_bayesian:0.2,greedy_joint:1.5\n'
        ),
    )
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cpu', help='cpu | cuda | cuda:N | auto')
    p.add_argument('--steps', type=int, default=20_000)
    p.add_argument(
        '--opponent', default='mock',
        help='mock|random|greedy_solo|greedy_joint|expectimax_bayesian|curriculum',
    )
    p.add_argument(
        '--phases', default=None,
        help='oponente:pasos,... (define orden de unlock si --expand-omega)',
    )
    p.add_argument(
        '--curriculum-phase-steps', type=int, default=DEFAULT_CURRICULUM_PHASE_STEPS,
        help='Pasos por unlock cuando --opponent curriculum',
    )
    p.add_argument(
        '--expand-omega', dest='expand_omega', action='store_true', default=True,
        help='Expansión monotónica de Ω; muestrear oponente por episodio (default)',
    )
    p.add_argument(
        '--no-expand-omega', dest='expand_omega', action='store_false',
        help='Fases excluyentes (un solo oponente por tramo; replay sigue compartido)',
    )
    p.add_argument(
        '--include-expectimax', action='store_true', default=True,
        help='Incluir expectimax_bayesian en curriculum expand (default)',
    )
    p.add_argument(
        '--no-expectimax', dest='include_expectimax', action='store_false',
        help='Omitir Expectimax del curriculum (más rápido)',
    )
    p.add_argument(
        '--opponent-weights', default=None,
        help='Overrides: nombre:peso,... (ej. expectimax_bayesian:0.35,greedy_joint:1.5)',
    )
    p.add_argument('--checkpoint-dir', default='logs/ddqn_checkpoints')
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
    p.add_argument('--lr', type=float, default=None, help='Override DDQNConfig.lr')
    p.add_argument('--gamma', type=float, default=None, help='Override DDQNConfig.gamma')
    p.add_argument('--target-update-every', type=int, default=None,
                   help='Copia dura de la red objetivo cada N pasos de gradiente (0 = suave con tau; defecto 0)')
    p.add_argument('--eval-every', type=int, default=250_000,
                   help='Validación sin exploración cada N pasos con rivales deterministas; guarda best.pt (0 = off)')
    p.add_argument('--eval-games', type=int, default=50,
                   help='Partidas BO1 por rival en cada validación (4 rivales)')
    p.add_argument('--layer-norm', action='store_true', help='LayerNorm tras el codificador (punto 7)')
    p.add_argument('--weight-decay', type=float, default=None,
                   help='AdamW con ese decaimiento (por defecto 0,01; 0 = Adam). memorizacion_equipos.md §8')
    p.add_argument('--dropout', type=float, default=None,
                   help='Dropout tras cada capa oculta de la red Q (por defecto 0,1; 0 = sin dropout)')
    p.add_argument('--foe-bench-dropout', type=float, default=None,
                   help='Prob. de borrar cada Pokémon rival no activo de la obs al aprender (§8, b)')
    p.add_argument('--train-freq', type=int, default=None, help='Pasos de entorno por paso de gradiente (defecto 4)')
    p.add_argument('--batch-size', type=int, default=None, help='Lote del aprendiz (defecto 256)')
    p.add_argument('--resume', default='',
                   help='Reanudar desde un checkpoint (p. ej. <dir>/last.pt); el replay se rellena')
    p.add_argument('--target-clip', type=float, default=None,
                   help='Acota el objetivo y a [-c, c] (0 = sin acotar; defecto 2.0, ddqn_spec 1.15)')
    p.add_argument('--num-envs', type=int, default=1,
                   help='Entornos paralelos (procesos). Servidor: ~24; 0 = automático')
    p.add_argument('--envs-per-server', type=int, default=4,
                   help='Entornos por servidor Showdown (puertos base-port..)')
    p.add_argument('--base-port', type=int, default=None,
                   help='Primer puerto Showdown (default PS_PORT=8000)')
    p.add_argument('--agent-team', default=DEFAULT_AGENT_TEAM,
                   help='Equipo fijo del agente (archivo en formato export; default: '
                        'showdown/rl/data/ddqn_team.txt). "pool" = sortearlo del grupo train')
    from showdown.rl.frozen_preview import add_preview_cli_args, configure_preview
    add_preview_cli_args(p)
    args = p.parse_args(argv)
    print(f'[ddqn] ' + configure_preview('ddqn', checkpoint=args.preview_checkpoint,
                                          required=args.require_preview), flush=True)
    if args.num_envs == 0:
        from showdown.rl.parallel import default_num_envs
        args.num_envs = default_num_envs()

    schedule = resolve_schedule(
        opponent=args.opponent,
        steps=args.steps,
        phases=args.phases,
        curriculum_phase_steps=args.curriculum_phase_steps,
    )
    # curriculum implies expand unless user forced exclusive after
    expand = bool(args.expand_omega)
    if args.opponent == 'curriculum' and args.phases is None:
        expand = True

    overrides = parse_weight_overrides(args.opponent_weights)
    print(
        '[ddqn] schedule:',
        ', '.join(f'{ph.opponent}:{ph.steps}' for ph in schedule),
        f'| expand_omega={expand}',
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
        suite_games=args.suite_games,
        smoke=bool(args.smoke),
        lr=args.lr,
        gamma=args.gamma,
        target_update_every=args.target_update_every,
        target_clip=args.target_clip,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        resume=args.resume,
        layer_norm=bool(args.layer_norm),
        weight_decay=args.weight_decay, dropout=args.dropout,
        foe_bench_dropout=args.foe_bench_dropout,
        train_freq=args.train_freq,
        batch_size=args.batch_size,
        num_envs=args.num_envs,
        envs_per_server=args.envs_per_server,
        base_port=args.base_port,
        agent_team='' if str(args.agent_team).lower() in ('pool', 'none', '') else args.agent_team,
    )
    return 0


def main(argv: Optional[list] = None) -> int:
    """Validación reservada por defecto (``RL_HOLDOUT_VAL=0`` la apaga; foe_pool.trainer_defaults)."""
    from showdown.rl.foe_pool import trainer_defaults
    with trainer_defaults():
        return _main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
