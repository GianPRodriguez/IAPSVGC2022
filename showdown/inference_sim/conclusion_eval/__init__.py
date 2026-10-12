"""Evaluación final (Conclusión) — docs/conclusion_eval_protocol.md."""

from showdown.inference_sim.conclusion_eval.constants import (
    CROSS_EPS,
    DEFAULT_N_PAIRS,
    DEFAULT_SNAPSHOT_N,
    DQ_FAIL_RATE,
    HEURISTIC_IDS,
)
from showdown.inference_sim.conclusion_eval.pipelines import (
    DEFAULT_PIPELINES,
    PipelineSpec,
    load_pipelines,
)
from showdown.inference_sim.conclusion_eval.stats import (
    cross_points,
    rank_pipelines,
    wilson_ci,
)

__all__ = [
    'CROSS_EPS',
    'DEFAULT_N_PAIRS',
    'DEFAULT_SNAPSHOT_N',
    'DQ_FAIL_RATE',
    'HEURISTIC_IDS',
    'DEFAULT_PIPELINES',
    'PipelineSpec',
    'load_pipelines',
    'cross_points',
    'rank_pipelines',
    'wilson_ci',
]
