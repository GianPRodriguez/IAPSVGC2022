"""Final train reports — human + AI readable (reporte.txt / reporte.json)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from showdown.rl.train_log import format_count, format_duration

# Spec suite foes (docs/*_spec.md offline suite)
SUITE_FOES: Tuple[str, ...] = ('random', 'greedy_joint', 'expectimax_bayesian')
SUITE_GAMES_DEFAULT = 20  # full thesis run often uses 200; CLI override
SUITE_GAMES_FULL = 200


@dataclass
class SuiteFoeResult:
    foe: str
    n_games: int
    wins: int
    losses: int
    ties: int
    mean_return: float
    win_rate: float


@dataclass
class TrainReport:
    """Structured payload for reporte.txt / reporte.json."""

    agent: str
    status: str = 'DONE'
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    seed: Optional[int] = None
    device: str = 'cpu'
    smoke: bool = False
    # Budget
    global_steps: int = 0
    t_total: int = 0
    elapsed_sec: float = 0.0
    steps_per_sec: float = 0.0
    # Outcome (agent-specific flat dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    phases: List[Dict[str, Any]] = field(default_factory=list)
    # Artifacts
    artifacts: Dict[str, str] = field(default_factory=dict)
    # Offline suite
    suite_games_per_foe: int = 0
    suite: List[SuiteFoeResult] = field(default_factory=list)
    suite_note: str = ''
    # Free-form notes
    notes: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _fmt_val(v: Any) -> str:
    if v is None:
        return 'null'
    if isinstance(v, float):
        if v != v:
            return 'nan'
        if abs(v) >= 1000:
            return f'{v:.4g}'
        return f'{v:.6g}'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def render_reporte_txt(report: TrainReport) -> str:
    """Plain-text report optimized for humans and LLM ingestion."""
    lines: List[str] = []
    lines.append(f'# Train Report — {report.agent}')
    lines.append(f'status: {report.status}')
    lines.append(f'generated_at_utc: {report.generated_at}')
    lines.append(f'seed: {_fmt_val(report.seed)}')
    lines.append(f'device: {report.device}')
    lines.append(f'smoke: {_fmt_val(report.smoke)}')
    lines.append('')
    lines.append('## Config')
    if report.config:
        for k, v in report.config.items():
            lines.append(f'{k}: {_fmt_val(v)}')
    else:
        lines.append('(none)')
    lines.append('')
    lines.append('## Budget')
    pct = (
        100.0 * float(report.global_steps) / float(max(1, report.t_total))
        if report.t_total
        else 0.0
    )
    lines.append(
        f'steps: {report.global_steps} / {report.t_total} '
        f'({pct:.1f}%) [{format_count(report.global_steps)}/'
        f'{format_count(report.t_total)}]'
    )
    lines.append(f'elapsed: {format_duration(report.elapsed_sec)} ({report.elapsed_sec:.1f}s)')
    lines.append(f'steps_per_sec: {report.steps_per_sec:.2f}')
    lines.append('')
    lines.append('## Outcome metrics')
    if report.metrics:
        for k, v in report.metrics.items():
            lines.append(f'{k}: {_fmt_val(v)}')
    else:
        lines.append('(none)')
    lines.append('')
    if report.phases:
        lines.append('## Phases')
        for i, ph in enumerate(report.phases):
            lines.append(f'### phase[{i}]')
            for k, v in ph.items():
                lines.append(f'{k}: {_fmt_val(v)}')
        lines.append('')
    lines.append('## Artifacts')
    if report.artifacts:
        for k, v in report.artifacts.items():
            lines.append(f'{k}: {v}')
    else:
        lines.append('(none)')
    lines.append('')
    lines.append('## Offline suite (Cand. 1 greedy BO1)')
    lines.append(f'suite_games_per_foe: {report.suite_games_per_foe}')
    if report.suite_note:
        lines.append(f'note: {report.suite_note}')
    if report.suite:
        for s in report.suite:
            lines.append(
                f'{s.foe}: WR={s.win_rate:.3f} '
                f'(W={s.wins} L={s.losses} T={s.ties} n={s.n_games}) '
                f'mean_return={s.mean_return:.3f}'
            )
        mean_wr = float(np.mean([s.win_rate for s in report.suite]))
        lines.append(f'MEAN_WR: {mean_wr:.3f}')
    else:
        lines.append('(suite skipped or empty)')
    lines.append('')
    lines.append('## Analysis notes')
    if report.notes:
        for n in report.notes:
            lines.append(f'- {n}')
    else:
        lines.append('- See Outcome metrics and Offline suite for thesis tables.')
    lines.append('')
    lines.append('## Machine twin')
    lines.append('Companion JSON: reporte.json (same directory).')
    lines.append('')
    return '\n'.join(lines)


def write_train_report(
    out_dir: Path | str,
    report: TrainReport,
    *,
    basename: str = 'reporte',
) -> Tuple[Path, Path]:
    """
    Write ``reporte.txt`` + ``reporte.json`` under ``out_dir``.

    Returns (txt_path, json_path).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    txt_path = out / f'{basename}.txt'
    json_path = out / f'{basename}.json'
    txt_path.write_text(render_reporte_txt(report), encoding='utf-8')
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(
        f'[{report.agent}] wrote {txt_path} and {json_path}',
        flush=True,
    )
    return txt_path, json_path


def _classify_return(ret: float) -> str:
    if ret > 0.05:
        return 'win'
    if ret < -0.05:
        return 'loss'
    return 'tie'


def run_suite_ppo_act(
    agent,
    env,
    *,
    foes: Sequence[str] = SUITE_FOES,
    n_games: int,
    seed: int = 0,
    mock_note: bool = False,
) -> Tuple[List[SuiteFoeResult], str]:
    """
    Cand. 1 greedy suite for agents with ``act(obs, m1, m2_fn, greedy=True)``.

    On MockBattleBackend, foe switching is a no-op — note is set accordingly.
    """
    from showdown.rl.observation import build_observation_ppo
    from showdown.rl.qd_ppo.evaluate import _ensure_foe, _m2_fn, _slot_masks
    from showdown.rl.reward import RewardCandidate

    if n_games <= 0:
        return [], 'suite_games=0 (skipped)'

    prev = env.reward_candidate
    env.reward_candidate = RewardCandidate.SPARSE
    results: List[SuiteFoeResult] = []
    is_mock = type(env.backend).__name__ == 'MockBattleBackend'
    note = ''
    if is_mock or mock_note:
        note = (
            'MockBattleBackend: foe labels are nominal (backend ignores set_opponent); '
            'WR reflects mock dynamics only. Use --start-ps for live suite.'
        )

    try:
        for fi, foe in enumerate(foes):
            env = _ensure_foe(env, foe, seed=seed)
            wins = losses = ties = 0
            returns: List[float] = []
            for gi in range(int(n_games)):
                obs, info = env.reset(seed=int(seed) + 100_000 + fi * 10_000 + gi)
                assert env._snap is not None
                obs = build_observation_ppo(env._snap)
                m1, _ = _slot_masks(env)
                steps = 0
                while steps < 80:
                    a1, a2, _lp, _v = agent.act(obs, m1, _m2_fn(env), greedy=True)
                    obs, _r, term, trunc, info = env.step((a1, a2))
                    assert env._snap is not None
                    obs = build_observation_ppo(env._snap)
                    m1, _ = _slot_masks(env)
                    steps += 1
                    if term or trunc:
                        break
                ret = float(info.get('episode_return', 0.0))
                returns.append(ret)
                kind = _classify_return(ret)
                if kind == 'win':
                    wins += 1
                elif kind == 'loss':
                    losses += 1
                else:
                    ties += 1
            n = len(returns)
            wr = float(wins) / float(max(1, n))
            results.append(
                SuiteFoeResult(
                    foe=str(foe),
                    n_games=n,
                    wins=wins,
                    losses=losses,
                    ties=ties,
                    mean_return=float(np.mean(returns)) if returns else 0.0,
                    win_rate=wr,
                )
            )
    finally:
        env.reward_candidate = prev
    return results, note


def run_suite_ddqn_act(
    agent,
    env,
    *,
    foes: Sequence[str] = SUITE_FOES,
    n_games: int,
    seed: int = 0,
) -> Tuple[List[SuiteFoeResult], str]:
    """Cand. 1 greedy suite for DDQN joint ``act(obs, mask, epsilon=0)``."""
    from showdown.rl.action_space import decode_joint_action
    from showdown.rl.ddqn.train import _joint_masks
    from showdown.rl.observation import build_observation_ddqn
    from showdown.rl.reward import RewardCandidate

    if n_games <= 0:
        return [], 'suite_games=0 (skipped)'

    prev = env.reward_candidate
    env.reward_candidate = RewardCandidate.SPARSE
    results: List[SuiteFoeResult] = []
    is_mock = type(env.backend).__name__ == 'MockBattleBackend'
    note = ''
    if is_mock:
        note = (
            'MockBattleBackend: foe labels nominal; use --start-ps for live suite.'
        )

    try:
        for fi, foe in enumerate(foes):
            backend = env.backend
            if (
                hasattr(backend, 'set_opponent')
                and getattr(backend, 'opponent', None) != foe
                and type(backend).__name__ != 'MockBattleBackend'
            ):
                backend.set_opponent(foe)
            wins = losses = ties = 0
            returns: List[float] = []
            for gi in range(int(n_games)):
                obs, info = env.reset(seed=int(seed) + 200_000 + fi * 10_000 + gi)
                assert env._snap is not None
                obs = build_observation_ddqn(env._snap)
                mask = _joint_masks(env)
                steps = 0
                while steps < 80:
                    a = agent.act(obs, mask, epsilon=0.0)
                    a1, a2 = decode_joint_action(int(a))
                    obs, _r, term, trunc, info = env.step((a1, a2))
                    assert env._snap is not None
                    obs = build_observation_ddqn(env._snap)
                    mask = _joint_masks(env)
                    steps += 1
                    if term or trunc:
                        break
                ret = float(info.get('episode_return', 0.0))
                returns.append(ret)
                kind = _classify_return(ret)
                if kind == 'win':
                    wins += 1
                elif kind == 'loss':
                    losses += 1
                else:
                    ties += 1
            n = len(returns)
            results.append(
                SuiteFoeResult(
                    foe=str(foe),
                    n_games=n,
                    wins=wins,
                    losses=losses,
                    ties=ties,
                    mean_return=float(np.mean(returns)) if returns else 0.0,
                    win_rate=float(wins) / float(max(1, n)),
                )
            )
    finally:
        env.reward_candidate = prev
    return results, note


def resolve_suite_games(*, suite_games: Optional[int], smoke: bool) -> int:
    """Default: 0 on smoke, SUITE_GAMES_DEFAULT otherwise; explicit overrides."""
    if suite_games is not None:
        return max(0, int(suite_games))
    return 0 if smoke else SUITE_GAMES_DEFAULT

