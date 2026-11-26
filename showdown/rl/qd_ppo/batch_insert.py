"""Inserción por lotes con comparación pareada (regla común de I/J/K 2.0, docs/qd_ppo_spec.md §I 2.0).

Fase A: todos los candidatos de la generación juegan sus enfrentamientos a la vez. Fase B: para cada
candidato cuya celda está ocupada, el ocupante juega **los mismos** enfrentamientos (también a la vez) y
suma esas partidas a su estimación. Decisión por celda: entra el candidato con mayor diferencia pareada
positiva frente al ocupante; en una celda vacía, el candidato de mayor fitness.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from showdown.rl.qd_ppo.matchups import make_matchups, paired_diff, play_matchups_multi


def evaluate_insert_batch(archive, env, agent_of: Callable[[Dict[str, Any]], Any],
                          candidates: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]], rng, sampler, *,
                          paired_games: int, g_min: int, stats: Dict[str, int], reward: str = 'sparse',
                          on_transition=None) -> List[Dict[str, Any]]:
    """``candidates`` = [(state, meta)]. Devuelve por candidato {res, key, inserted, eligible}."""
    if not candidates:
        return []
    matches = [make_matchups(rng, paired_games, sampler) for _ in candidates]
    res_c = play_matchups_multi([agent_of(sd) for sd, _m in candidates], env, matches, reward=reward,
                                on_transition=on_transition)
    out: List[Dict[str, Any]] = []
    need: List[int] = []
    for i, ((sd, meta), res) in enumerate(zip(candidates, res_c)):
        bd = res.bd()
        ok = bd is not None and len(res.eligible_bds()) >= int(g_min)
        if not ok:
            stats['rejected_gmin'] = stats.get('rejected_gmin', 0) + 1
        key = archive.key_of(*bd) if ok else None
        out.append({'res': res, 'key': key, 'bd': bd, 'eligible': ok, 'inserted': False})
        if ok and key in archive.cells:
            need.append(i)
    # Fase B: ocupantes en los mismos enfrentamientos de cada candidato
    if need:
        res_i = play_matchups_multi([agent_of(archive.cells[out[i]['key']].state_dict) for i in need], env,
                                    [matches[i] for i in need], reward=reward, on_transition=on_transition)
        for i, r in zip(need, res_i):
            archive.cells[out[i]['key']].add_games(r.valid_returns())
            out[i]['paired_diff'] = paired_diff(out[i]['res'], r)
    by_key: Dict[Tuple[int, int], List[int]] = {}
    for i, o in enumerate(out):
        if o['eligible']:
            by_key.setdefault(o['key'], []).append(i)
    for key, idx in by_key.items():
        if key in archive.cells:
            pos = [i for i in idx if out[i].get('paired_diff', 0.0) > 0]
            if not pos:
                continue
            win = max(pos, key=lambda i: out[i]['paired_diff'])
            stats['replaced'] = stats.get('replaced', 0) + 1
        else:
            win = max(idx, key=lambda i: out[i]['res'].fitness)
            stats['new_cells'] = stats.get('new_cells', 0) + 1
        sd, meta = candidates[win]
        meta = dict(meta)
        if 'paired_diff' in out[win]:
            meta['paired_diff'] = out[win]['paired_diff']
        bd = out[win]['bd']
        archive.try_insert(bd1=bd[0], bd2=bd[1], fitness=out[win]['res'].fitness, state_dict=sd, meta=meta,
                           returns=out[win]['res'].valid_returns(), force=True)
        out[win]['inserted'] = True
        stats['accepts'] = stats.get('accepts', 0) + 1
    return out
