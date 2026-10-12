"""Constantes del protocolo de Conclusión (docs/conclusion_eval_protocol.md)."""
from __future__ import annotations

from pathlib import Path

from showdown.inference_sim.paths import REPO_ROOT

CROSS_EPS = 0.025
DEFAULT_SNAPSHOT_N = 250
DEFAULT_N_PAIRS = 100
# 100 AB + 100 BA
BATTLES_PER_CROSS = 2 * DEFAULT_N_PAIRS
DQ_FAIL_RATE = 0.02
WILSON_Z = 1.959963984540054  # ~95%

HEURISTIC_IDS = (
    'random',
    'greedy_solo',
    'greedy_joint',
    'expectimax_bayesian',
)

OUT_DIR = REPO_ROOT / 'logs' / 'conclusion_eval'
SNAPSHOT_PATH = OUT_DIR / 'snapshot.json'
PIPELINES_PATH = OUT_DIR / 'pipelines.json'
RESULTS_PATH = OUT_DIR / 'results.json'
PROGRESS_PATH = OUT_DIR / 'progress.json'
REPORT_PATH = OUT_DIR / 'report.txt'


def battles_per_cross(n_pairs: int = DEFAULT_N_PAIRS) -> int:
    return 2 * int(n_pairs)
