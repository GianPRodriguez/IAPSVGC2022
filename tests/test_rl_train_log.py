"""Unit tests for unified train heartbeat."""
from __future__ import annotations

import time

from showdown.rl.train_log import (
    TrainHeartbeat,
    format_count,
    format_duration,
    schedule_total_steps,
)


def test_format_duration_and_count():
    assert format_duration(45) == '45s'
    assert format_duration(90).startswith('1m')
    assert format_count(512) == '512'
    assert format_count(12_000) == '12k'
    assert 'M' in format_count(1_500_000)


def test_heartbeat_eta_and_progress():
    hb = TrainHeartbeat(tag='test', t_total=1000, log_every=100, log_every_sec=0)
    hb.t0 = time.perf_counter() - 10.0  # 10s elapsed
    assert abs(hb.progress(250) - 0.25) < 1e-9
    rate = hb.rate(250)
    assert rate > 0
    eta = hb.eta_seconds(250)
    assert eta is not None and eta > 0
    line = hb.line(250, extras={'loss': 0.12}, phase='phase=0')
    assert '[test]' in line
    assert '25.0%' in line
    assert 'ETA' in line
    assert 'loss=0.120' in line


def test_should_log_steps_and_force():
    hb = TrainHeartbeat(tag='t', t_total=10_000, log_every=500, log_every_sec=9999)
    assert not hb.should_log(100)
    assert hb.should_log(500)
    assert hb.log(500, extras={'a': 1})
    assert not hb.should_log(600)
    assert hb.log(600, force=True)


def test_schedule_total_steps():
    class P:
        def __init__(self, steps):
            self.steps = steps

    assert schedule_total_steps([P(100), P(200)]) == 300
