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
import time
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


def _make_env(opponent: str, *, seed: int, start_ps: bool) -> VGCEnv:
    if opponent == 'mock':
        backend = MockBattleBackend(max_turns=40)
    else:
        from showdown.rl.pokeenv_backend import PokeEnvBattleBackend
        backend = PokeEnvBattleBackend(
            opponent=opponent,
            start_ps=start_ps,
            account_prefix=f'Ddqn{opponent[:4].title()}',
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
    log_every: int,
    t0: float,
    phase_label: str,
    seed: int,
    start_ps: bool,
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

    for local in range(1, int(steps) + 1):
        a_joint = agent.act(obs, mask, epsilon=None)
        a1, a2 = decode_joint_action(a_joint)
        next_obs, reward, terminated, truncated, info = env.step((a1, a2))
        assert env._snap is not None
        next_obs = build_observation_ddqn(env._snap)
        skip = bool(info.get('skip_buffer'))
        eng = info.get('engine') or {}
        if isinstance(eng, dict) and eng.get('used_fallback'):
            skip = True
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
            if expand_omega and len(omega) > 1:
                nxt = sample_from_omega(omega, rng, weights=weights)
                if nxt != current:
                    env = _ensure_opponent(env, nxt, seed=seed, start_ps=start_ps)
                    current = nxt
                opp_code = opponent_code(current)
                ep_counts[current] = ep_counts.get(current, 0) + 1
            obs, info = env.reset()
            assert env._snap is not None
            obs = build_observation_ddqn(env._snap)
            mask = _joint_masks(env)

        gstep = global_step_offset + local
        if gstep % int(log_every) == 0:
            elapsed = time.perf_counter() - t0
            frac_ep = {
                k: (v / max(1, sum(ep_counts.values())))
                for k, v in ep_counts.items()
            }
            print(
                f'[ddqn] {phase_label} local={local}/{steps} global={gstep} '
                f'opp={current} Omega={list(omega)} eps={agent.epsilon:.3f} '
                f'replay={len(agent.replay)} loss={metrics_last.get("loss", float("nan")):.4f} '
                f'buf_frac={ {k: round(v, 3) for k, v in agent.replay.origin_fractions().items()} } '
                f'ep_frac={ {k: round(v, 3) for k, v in frac_ep.items()} } '
                f'elapsed={elapsed:.1f}s'
            )
            agent.save(ckpt_dir / 'last.pt')

    return env, int(steps), {'metrics': metrics_last, 'ep_counts': ep_counts}


def run_training(
    *,
    seed: int = 0,
    schedule: Sequence[TrainPhase],
    device: str = 'cpu',
    checkpoint_dir: str | Path = 'logs/ddqn_checkpoints',
    log_every: int = 1000,
    start_ps: bool = False,
    expand_omega: bool = True,
    include_expectimax: bool = True,
    weight_overrides: Optional[Dict[str, float]] = None,
) -> DDQNAgent:
    if not schedule:
        raise ValueError('schedule vacío')

    weights = merge_weights(weight_overrides)
    cfg = DDQNConfig(seed=seed, device=device)
    agent = DDQNAgent(cfg)
    rng = np.random.default_rng(int(seed) + 17)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
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
    env: Optional[VGCEnv] = _make_env(first_opp, seed=seed, start_ps=start_ps)

    try:
        assert env is not None
        for pi, phase in enumerate(expand_sched):
            probs = omega_sampling_probs(phase.omega, weights=weights)
            mode = 'expand-Omega' if expand_omega else 'exclusive'
            print(
                f'[ddqn] === phase {pi + 1}/{len(expand_sched)} ({mode}): '
                f'unlock={phase.unlock} steps={phase.steps} '
                f'Omega={list(phase.omega)} '
                f'P={ {k: round(v, 3) for k, v in probs.items()} } ==='
            )
            # Ensure unlock foe exists at phase start
            env = _ensure_opponent(env, phase.unlock, seed=seed, start_ps=start_ps)
            env, done, info = _run_steps(
                agent, env,
                steps=phase.steps,
                omega=phase.omega,
                expand_omega=expand_omega,
                weights=weights,
                rng=rng,
                global_step_offset=global_steps,
                ckpt_dir=ckpt_dir,
                log_every=log_every,
                t0=t0,
                phase_label=f'phase={pi}:{phase.unlock}',
                seed=seed,
                start_ps=start_ps,
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
    finally:
        if env is not None:
            env.close()

    agent.save(ckpt_dir / 'final.pt')
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
    }
    (ckpt_dir / 'train_meta.json').write_text(
        json.dumps(meta, indent=2) + '\n', encoding='utf-8',
    )
    print(
        f'[ddqn] DONE expand_omega={expand_omega} steps={global_steps} '
        f'saved={ckpt_dir / "final.pt"}'
    )
    return agent


def main(argv: Optional[list] = None) -> int:
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
    p.add_argument('--device', default='cpu')
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
    p.add_argument('--log-every', type=int, default=1000)
    p.add_argument('--start-ps', action='store_true')
    args = p.parse_args(argv)

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
        start_ps=bool(args.start_ps),
        expand_omega=expand,
        include_expectimax=bool(args.include_expectimax),
        weight_overrides=overrides or None,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
