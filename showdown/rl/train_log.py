"""Unified train heartbeat: progress %, rate, ETA (human-readable console)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


def format_duration(seconds: float) -> str:
    """Human duration: 45s / 12m30s / 1h05m."""
    s = max(0.0, float(seconds))
    if s < 60:
        return f'{s:.0f}s'
    if s < 3600:
        m = int(s // 60)
        rem = int(s % 60)
        return f'{m}m{rem:02d}s'
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    return f'{h}h{m:02d}m'


def format_count(n: float) -> str:
    """Compact counts: 1.2M / 420k / 512."""
    x = float(n)
    ax = abs(x)
    if ax >= 1_000_000:
        return f'{x / 1_000_000:.2f}M'.replace('.00M', 'M')
    if ax >= 10_000:
        return f'{x / 1000:.0f}k'
    if ax >= 1000:
        return f'{x / 1000:.1f}k'
    return f'{int(x)}'


def format_extras(extras: Optional[Mapping[str, Any]]) -> str:
    if not extras:
        return ''
    parts = []
    for k, v in extras.items():
        if v is None:
            continue
        if isinstance(v, float):
            if v != v:  # NaN
                parts.append(f'{k}=nan')
            elif abs(v) >= 100 or (abs(v) > 0 and abs(v) < 0.01):
                parts.append(f'{k}={v:.3g}')
            else:
                parts.append(f'{k}={v:.3f}')
        else:
            parts.append(f'{k}={v}')
    return (' | ' + ' '.join(parts)) if parts else ''


@dataclass
class TrainHeartbeat:
    """
    Shared console heartbeat for long RL / QD runs.

    Logs when ``log_every`` steps elapse **or** ``log_every_sec`` wall time
    (whichever comes first after the previous log), always on forced flush.
    """

    tag: str
    t_total: int
    log_every: int = 5_000
    log_every_sec: float = 60.0
    ckpt_every: int = 25_000
    t0: float = field(default_factory=time.perf_counter)
    _last_log_steps: int = 0
    _last_log_t: float = field(default_factory=time.perf_counter)
    _last_ckpt_steps: int = 0

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def rate(self, steps: int) -> float:
        e = self.elapsed()
        return float(steps) / e if e > 1e-9 else 0.0

    def eta_seconds(self, steps: int) -> Optional[float]:
        total = int(self.t_total)
        if total <= 0 or steps <= 0:
            return None
        r = self.rate(steps)
        if r <= 1e-9:
            return None
        rem = max(0, total - int(steps))
        return rem / r

    def progress(self, steps: int) -> float:
        total = max(1, int(self.t_total))
        return min(1.0, max(0.0, float(steps) / float(total)))

    def should_log(self, steps: int, *, force: bool = False) -> bool:
        if force:
            return True
        if int(steps) <= 0:
            return False
        if int(self.log_every) > 0 and (int(steps) - self._last_log_steps) >= int(self.log_every):
            return True
        if float(self.log_every_sec) > 0:
            if (time.perf_counter() - self._last_log_t) >= float(self.log_every_sec):
                return True
        return False

    def should_checkpoint(self, steps: int) -> bool:
        if int(self.ckpt_every) <= 0:
            return False
        if int(steps) - self._last_ckpt_steps >= int(self.ckpt_every):
            return True
        return False

    def mark_checkpoint(self, steps: int) -> None:
        self._last_ckpt_steps = int(steps)

    def line(
        self,
        steps: int,
        *,
        extras: Optional[Mapping[str, Any]] = None,
        phase: str = '',
    ) -> str:
        pct = 100.0 * self.progress(steps)
        rate = self.rate(steps)
        eta = self.eta_seconds(steps)
        eta_s = format_duration(eta) if eta is not None else '?'
        phase_bit = f' {phase}' if phase else ''
        return (
            f'[{self.tag}]{phase_bit} {pct:5.1f}% | '
            f'steps {format_count(steps)}/{format_count(self.t_total)} | '
            f'{rate:.0f} step/s | ETA {eta_s} | '
            f'elapsed {format_duration(self.elapsed())}'
            f'{format_extras(extras)}'
        )

    def log(
        self,
        steps: int,
        *,
        extras: Optional[Mapping[str, Any]] = None,
        phase: str = '',
        force: bool = False,
    ) -> bool:
        """Print heartbeat if due. Returns True if printed."""
        if not self.should_log(steps, force=force):
            return False
        print(self.line(steps, extras=extras, phase=phase), flush=True)
        self._last_log_steps = int(steps)
        self._last_log_t = time.perf_counter()
        return True


def schedule_total_steps(schedule) -> int:
    """Sum ``.steps`` over phase-like objects."""
    return int(sum(int(getattr(p, 'steps', 0)) for p in schedule))
