"""Registro canónico de pipelines (congelar justo antes del run)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from showdown.inference_sim.conclusion_eval.constants import PIPELINES_PATH


@dataclass(frozen=True)
class PipelineSpec:
    """Un bot completo: combate + preview de familia (documentado)."""

    id: str
    combat: str
    preview: str
    group: str  # 'family' | 'heuristic'
    checkpoint: str = ''  # path opcional; vacío = default del player


# Orden canónico = índices i < j para asientos p1/p2.
DEFAULT_PIPELINES: List[PipelineSpec] = [
    PipelineSpec('ddqn', 'ddqn', 'team_preview_ppo', 'family'),
    PipelineSpec('ppo', 'ppo', 'team_preview_ppo', 'family'),
    PipelineSpec('mcts_puct', 'mcts_puct', 'team_preview_nash', 'family'),
    PipelineSpec(
        'mcts_puct_nash', 'mcts_puct_nash', 'team_preview_nash', 'family',
    ),
    PipelineSpec('qd_ppo', 'qd_ppo', 'team_preview_qd_pga', 'family'),
    PipelineSpec('qd_pga', 'qd_pga', 'team_preview_qd_pga', 'family'),
    PipelineSpec('qd_ppga', 'qd_ppga', 'team_preview_qd_pga', 'family'),
    PipelineSpec('random', 'random', 'random', 'heuristic'),
    PipelineSpec('greedy_solo', 'greedy_solo', 'greedy_solo', 'heuristic'),
    PipelineSpec('greedy_joint', 'greedy_joint', 'greedy_solo', 'heuristic'),
    PipelineSpec(
        'expectimax_bayesian',
        'expectimax_bayesian',
        'expectimax_bayesian',
        'heuristic',
    ),
]


def load_pipelines(path: Optional[Path] = None) -> List[PipelineSpec]:
    """Carga pipelines.json si existe; si no, DEFAULT_PIPELINES."""
    p = Path(path) if path is not None else PIPELINES_PATH
    if not p.is_file():
        return list(DEFAULT_PIPELINES)
    raw = json.loads(p.read_text(encoding='utf-8'))
    rows = raw.get('pipelines') if isinstance(raw, dict) else raw
    out: List[PipelineSpec] = []
    for row in rows or []:
        out.append(
            PipelineSpec(
                id=str(row['id']),
                combat=str(row.get('combat') or row['id']),
                preview=str(row.get('preview') or ''),
                group=str(row.get('group') or 'family'),
                checkpoint=str(row.get('checkpoint') or ''),
            )
        )
    if len(out) < 2:
        raise ValueError(f'pipelines insuficientes en {p}')
    return out


def save_pipelines(
    pipelines: Sequence[PipelineSpec],
    path: Optional[Path] = None,
    *,
    note: str = '',
) -> Path:
    p = Path(path) if path is not None else PIPELINES_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'note': note or (
            'Congelar justo antes del run de Conclusión '
            '(docs/conclusion_eval_protocol.md §1).'
        ),
        'pipelines': [asdict(x) for x in pipelines],
    }
    p.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return p


def pipeline_index_map(pipelines: Sequence[PipelineSpec]) -> dict:
    return {pl.id: i for i, pl in enumerate(pipelines)}
