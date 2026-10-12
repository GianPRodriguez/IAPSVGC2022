"""Tune HP QD-PPO combate (independiente de Preview QD-PGA).

Ejemplos::

  python -m showdown.inference_sim.run_qd_ppo_hp_tune --dry-run
  python -m showdown.inference_sim.run_qd_ppo_hp_tune --seed 0 --max-trials 6 --suite-games 4
  python -m showdown.inference_sim.run_qd_ppo_hp_tune --seed 0 --max-trials 6 --start-ps --no-smoke --t-total 50000

Knobs: lr, clip_eps. Tracking: logs/qd_ppo_hp_tune_*.
"""
from __future__ import annotations

from typing import Optional, Sequence

from showdown.inference_sim.qd_hp_tune import run_qd_agent_hp_tune


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_qd_agent_hp_tune(
        agent='qd_ppo',
        train_module='showdown.rl.qd_ppo.train',
        argv=argv,
    )


if __name__ == '__main__':
    raise SystemExit(main())
