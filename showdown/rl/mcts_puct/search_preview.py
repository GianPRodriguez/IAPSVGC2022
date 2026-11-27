"""Preview por búsqueda para los agentes de árbol (G y, después, H). Decisión del usuario 2026-11-21.

El preview greedy (tipado + BST) elige bien los 4 Pokémon pero **no elige los leads**: su
puntuación ignora el orden, así que los leads salen en el orden de la hoja. Este preview mira
hacia adelante con el mismo modelo de turno que el combate (S3):

1. Candidatas propias: las ``our_combos`` mejores combinaciones de 4 según el greedy, cada una con
   sus 6 parejas de leads.
2. Candidatas del rival (hoja abierta: especies, movimientos, objetos, habilidades; SP del prior):
   sus ``foe_combos`` mejores combinaciones según la misma regla, con sus 6 parejas de leads.
3. Una tabla de daño S3 para los 6 × 6 Pokémon (un lote a la calculadora).
4. Para cada par (nuestra formación, la suya): ``n_rollouts`` simulaciones de ``turns`` turnos
   (políticas por softmax de la puntuación S3 en ambos lados, efectos al entrar incluidos) y
   valor = material al final, en [−1, 1].
5. Decisión sobre la matriz: ``mean`` (mejor promedio contra las formaciones del rival) o
   ``maximin`` (mejor peor caso). Desempate: el orden greedy.

Si se agota ``deadline_sec`` o algo falla, se devuelve el orden greedy (``info['fallback']``).
"""
from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from showdown.rl.mcts_puct import actions as A
from showdown.rl.mcts_puct.openloop import material_value
from showdown.rl.mcts_puct.search import make_sim_rng
from showdown.rl.snapshot import BattleSnapshot, FieldSnapshot, MonSnapshot, SideSnapshot


@dataclass
class SearchPreviewConfig:
    our_combos: int = 3
    foe_combos: int = 2
    n_rollouts: int = 8
    turns: int = 3
    temperature: float = 10.0   # softmax de la puntuación S3 en las simulaciones (ambos lados)
    eps: float = 0.10
    decision: str = 'mean'      # 'mean' | 'maximin'
    phase: str = '3c'           # fase de S3 usada en la simulación
    deadline_sec: float = 25.0
    seed: int = 0


@dataclass
class PreviewInfo:
    order: Tuple[int, ...] = ()
    greedy_order: Tuple[int, ...] = ()
    fallback: str = ''
    n_ours: int = 0
    n_foe: int = 0
    rollouts: int = 0
    elapsed_sec: float = 0.0
    calc_requests: int = 0
    calc_missing: int = 0
    values: Dict[Tuple[int, ...], float] = field(default_factory=dict)


def _preview_mons(mons: Sequence[MonSnapshot]):
    from showdown.battle_bots.greedy_solo.team_preview import mon_from_species
    return [mon_from_species(i + 1, m.display_species() or m.species) for i, m in enumerate(mons)]


def top_combos(mons: Sequence[MonSnapshot], foes: Sequence[MonSnapshot], k: int) -> List[Tuple[int, ...]]:
    """Las ``k`` mejores combinaciones de 4 (índices 1-based) según la regla greedy, y su orden."""
    from showdown.battle_bots.greedy_solo.team_preview import score_combination
    ours, theirs = _preview_mons(mons), _preview_mons(foes)
    scored = []
    for combo in itertools.combinations(ours, 4):
        sc = score_combination(list(combo), theirs)
        scored.append((-sc.score, -sc.bst, tuple(m.index for m in combo)))
    scored.sort()
    return [c for _s, _b, c in scored[:max(1, int(k))]]


def formations(combo: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """Las 6 parejas de leads de una combinación: (lead1, lead2, banca1, banca2), 1-based."""
    out = []
    for lead in itertools.combinations(combo, 2):
        bench = [i for i in combo if i not in lead]
        out.append(tuple(lead) + tuple(bench))
    return out


def _side(mons: Sequence[MonSnapshot], order: Tuple[int, ...]) -> SideSnapshot:
    pick = [copy.copy(mons[i - 1]) for i in order]
    for j, m in enumerate(pick):
        m.is_active = j < 2
        m.turns_active = 0
        m.protect_counter = 0
        m.hp_pct = 100.0
        m.fainted = False
    return SideSnapshot(active=pick[:2], bench=pick[2:4])


def _start(model, ours: Sequence[MonSnapshot], foes: Sequence[MonSnapshot],
           o_ord: Tuple[int, ...], f_ord: Tuple[int, ...]) -> BattleSnapshot:
    s = BattleSnapshot(turn=1, our=_side(ours, o_ord), foe=_side(foes, f_ord), field=FieldSnapshot())
    for ours_side, side in ((True, s.our), (False, s.foe)):  # efectos al entrar de los leads
        for m in side.active:
            model.on_entry(s, m, ours_side)
    return s


def choose_formation(
    ours: Sequence[MonSnapshot],
    foes: Sequence[MonSnapshot],
    cfg: Optional[SearchPreviewConfig] = None,
    *,
    stat_beliefs=None,
    ranges_fn=None,
    foe_brought: Optional[Sequence[str]] = None,
) -> Tuple[Tuple[int, ...], PreviewInfo]:
    """Formación (lead1, lead2, banca1, banca2), 1-based, para ``/team``.

    Opcional (protocolo §4.19): ``MCTS_ARCH=1`` suma la formación del plan de cada lado; ``foe_brought`` (serie BO3,
    ``MCTS_SERIES=1``) limita al rival a los 4 que trajo antes, con sus 6 parejas de leads.
    """
    from showdown.rl.mcts_puct.s3 import S3Model

    cfg = cfg or SearchPreviewConfig()
    t0 = time.perf_counter()
    info = PreviewInfo()
    greedy = formations(top_combos(ours, foes, 1)[0])[0]
    info.greedy_order = greedy
    try:
        if len(ours) < 4 or len(foes) < 4:
            raise ValueError('equipos incompletos')
        our_forms = [f for c in top_combos(ours, foes, cfg.our_combos) for f in formations(c)]
        foe_forms = [f for c in top_combos(foes, ours, cfg.foe_combos) for f in formations(c)]
        from showdown.rl.mcts_puct import arch_ext as _X
        if _X.arch_on():
            for forms, team, opp in ((our_forms, ours, foes), (foe_forms, foes, ours)):
                pf = _X.plan_formation(team, opp)
                if pf is not None and pf not in forms:
                    forms.append(pf)
        if foe_brought:
            seen = _X.seen_formations(foes, foe_brought)
            if seen:
                foe_forms = seen
        info.n_ours, info.n_foe = len(our_forms), len(foe_forms)

        # Tabla S3 para los 6 × 6: raíz "ancha" con todos en activos + banca.
        base_our = [copy.copy(m) for m in ours]
        base_foe = [copy.copy(m) for m in foes]
        wide = BattleSnapshot(turn=1, our=SideSnapshot(active=base_our[:2], bench=base_our[2:]),
                              foe=SideSnapshot(active=base_foe[:2], bench=base_foe[2:]), field=FieldSnapshot())
        model = S3Model.build(wide, stat_beliefs=stat_beliefs, ranges_fn=ranges_fn, phase=cfg.phase)
        info.calc_requests, info.calc_missing = model.stats.requests, model.stats.missing
        from showdown.rl.mcts_puct.s3 import target_class
        scorer, tfn = model.joint_scores, target_class

        vals = np.zeros((len(our_forms), len(foe_forms)), dtype=np.float64)
        sim = 0
        for i, o in enumerate(our_forms):
            for j, f in enumerate(foe_forms):
                tot = 0.0
                for r in range(int(cfg.n_rollouts)):
                    if time.perf_counter() - t0 > cfg.deadline_sec:
                        raise TimeoutError('deadline')
                    rng = make_sim_rng(cfg.seed, i * 1000 + j, r)
                    s = _start(model, base_our, base_foe, o, f)
                    for _t in range(int(cfg.turns)):
                        if s.finished:
                            break
                        a_self = A.sample_opponent_joint(A.flip_view(s), rng, temperature=cfg.temperature,
                                                         eps=cfg.eps, scorer=scorer, target_fn=tfn)
                        a_opp = A.sample_opponent_joint(s, rng, temperature=cfg.temperature,
                                                        eps=cfg.eps, scorer=scorer, target_fn=tfn)
                        s = model.step(s, a_self, a_opp, rng)
                        if not s.finished:
                            model.replace_fainted(s)
                    tot += material_value(s)
                    sim += 1
                vals[i, j] = tot / max(1, int(cfg.n_rollouts))
        info.rollouts = sim
        if cfg.decision == 'maximin':
            score = vals.min(axis=1)
        else:
            score = vals.mean(axis=1)
        info.values = {our_forms[i]: float(score[i]) for i in range(len(our_forms))}
        best = max(range(len(our_forms)), key=lambda i: (round(float(score[i]), 9), our_forms[i] == greedy))
        info.order = our_forms[best]
    except Exception as exc:  # deadline, calculadora, datos incompletos → greedy
        info.fallback = repr(exc)[:160]
        info.order = greedy
    info.elapsed_sec = time.perf_counter() - t0
    return info.order, info


__all__ = ['SearchPreviewConfig', 'PreviewInfo', 'choose_formation', 'top_combos', 'formations']
