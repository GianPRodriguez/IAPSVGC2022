"""Tune HP QD-PGA combate (independiente de Preview QD-PGA).

Ejemplos::

  python -m showdown.inference_sim.run_qd_pga_hp_tune --dry-run
  python -m showdown.inference_sim.run_qd_pga_hp_tune --seed 0 --max-trials 6 --suite-games 4

Knobs: lr, clip_eps. Tracking: logs/qd_pga_hp_tune_*.
"""
from __future__ import annotations

from typing import Optional, Sequence

from showdown.inference_sim.qd_hp_tune import run_qd_agent_hp_tune


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_qd_agent_hp_tune(
        agent='qd_pga',
        train_module='showdown.rl.qd_pga.train',
        argv=argv,
    )


if __name__ == '__main__':
    raise SystemExit(main())
