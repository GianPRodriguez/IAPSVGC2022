"""Calibración de la inferencia por daño contra la verdad terreno.

En el harness de pruebas se conocen los SP verdaderos del rival. Este módulo
registra, en cada observación de daño, cuánto se aleja el rango del calc
(damage-calc con las stats verdaderas) de lo que Showdown realmente dibujó:

    res_hi = max(0, min_roll - observado)   # el calc predice MÁS daño del visto
    res_lo = max(0, observado - max_roll)   # el calc predice MENOS daño del visto

en las unidades del protocolo (puntos de HP si el daño es absoluto, puntos
porcentuales si es porcentaje del HP rival). Una tolerancia es segura si es
mayor que el residuo máximo observado en su categoría.

Uso: ``install(path)`` una vez por proceso; el jugador validado llama a
``set_truth(opp_team)`` al empezar cada serie. Activar con
``PS_CALIB_LOG=<archivo.jsonl>`` en ``run_robust_e2e`` (usar ``--workers 1``:
la verdad es por proceso).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

_STATE: Dict[str, Any] = {'meta': None, 'path': None, 'n': 0}


def set_truth(opp_meta) -> None:
    _STATE['meta'] = opp_meta


def install(path: Optional[str] = None) -> bool:
    """Instala el registro si hay ruta (argumento o ``PS_CALIB_LOG``). Devuelve si quedó activo."""
    path = path or os.environ.get('PS_CALIB_LOG', '').strip()
    if not path:
        return False
    from showdown.stat_inference import set_calibration_sink

    _STATE['path'] = str(path)
    set_calibration_sink(_sink)
    return True


def uninstall() -> None:
    from showdown.stat_inference import set_calibration_sink

    set_calibration_sink(None)
    _STATE['path'] = None


def _sink(ev: Dict[str, Any]) -> None:
    meta = _STATE.get('meta')
    path = _STATE.get('path')
    if meta is None or not path:
        return
    from showdown.inference_sim.soundness import _lookup_truth
    from showdown.stat_inference import champions_stat
    from teams.team_converter import normalize_name

    true_sp, _final = _lookup_truth(meta, normalize_name(ev['species']))
    if not true_sp:
        return
    stat = ev['stat']
    nature = ev['nature']
    tsp = int(true_sp.get(stat, 0))
    base = int(ev['base'])
    if base <= 0:
        return
    true_final = champions_stat(base, tsp, nature, stat)
    hp_final = None
    if ev['hp_base']:
        hp_final = champions_stat(int(ev['hp_base']), int(true_sp.get('hp', 0)), nature, 'hp')

    groups = ev['groups']
    cand = [g for g in groups if g[0] == true_final and (hp_final is None or g[1] == hp_final)]
    if not cand:
        cand = [g for g in groups if g[0] == true_final]  # HP representante cercano
    rec: Dict[str, Any] = {
        'species': ev['species'], 'stat': stat, 'observed': ev['observed'],
        'percent': ev['percent'], 'opp_attacks': ev['opponent_is_attacker'],
        'ko': ev['is_ko'], 'clean': ev['clean'], 'spread': ev['spread'],
        'move': ev['move'], 'cat': ev['category'], 'true_final': true_final,
        'n_groups': len(groups),
    }
    if not cand:
        rec['true_in_groups'] = False
    else:
        _fu, _fh, hp_val, rmin, rmax = cand[0]
        scale = (100.0 / hp_val) if (ev['percent'] and hp_val) else 1.0
        obs = ev['observed']
        rec.update({
            'true_in_groups': True,
            'hp_val': int(hp_val),
            'min': round(rmin * scale, 2), 'max': round(rmax * scale, 2),
            'res_hi': round(max(0.0, rmin * scale - obs), 2),
            'res_lo': round(max(0.0, obs - rmax * scale), 2),
        })
    try:
        with open(path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec) + '\n')
        _STATE['n'] += 1
    except OSError:
        pass
