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

_STATE: Dict[str, Any] = {'meta': None, 'path': None, 'n': 0, 'series': 0, 'seq': 0}

# Rejilla de tolerancias para el análisis de acumulación de evidencia
# (activar con PS_CALIB_PROFILE=1; añade el perfil de exclusión a cada línea).
PROFILE_TOL_ABS = (2, 4, 8, 14, 28)
PROFILE_TOL_PCT = (3, 6, 12, 20, 32)


def set_truth(opp_meta) -> None:
    """Nueva serie (BO3): cambia la verdad terreno y reinicia el orden de observaciones."""
    if opp_meta is not _STATE.get('meta'):
        _STATE['series'] += 1
        _STATE['seq'] = 0
    _STATE['meta'] = opp_meta


def _profile(ev: Dict[str, Any]) -> Dict[str, Any]:
    """Qué valores finales quedan excluidos con cada tolerancia (imitando a producción).

    Un valor final se considera compatible si ALGÚN grupo (HP) con ese final encaja,
    igual que ``observe_damage``. ``lo`` = solo cota inferior; ``bi`` = con cota
    superior (solo golpes no KO).
    """
    from showdown.stat_inference import _rolls_match_absolute, _rolls_match_percent

    percent = bool(ev['percent'])
    obs = int(ev['observed'])
    grid = PROFILE_TOL_PCT if percent else PROFILE_TOL_ABS
    finals = sorted({g[0] for g in ev['groups']})
    out: Dict[str, Any] = {'finals': finals}
    for mode in ('lo', 'bi'):
        if mode == 'bi' and ev['is_ko']:
            continue
        for t in grid:
            excluded = []
            for fu in finals:
                ok = False
                for g in ev['groups']:
                    if g[0] != fu:
                        continue
                    if percent:
                        ok = _rolls_match_percent(
                            g[3], g[4], g[2], obs, is_ko=bool(ev['is_ko']),
                            bilateral=(mode == 'bi'), tol_lo=t, tol_hi=t,
                        )
                    else:
                        ok = _rolls_match_absolute(
                            g[3], g[4], obs, is_ko=bool(ev['is_ko']),
                            bilateral=(mode == 'bi'), tol_lo=t, tol_hi=t,
                        )
                    if ok:
                        break
                if not ok:
                    excluded.append(fu)
            out[f'{mode}{t}'] = excluded
    return out


def _speed_sink(ev: Dict[str, Any]) -> None:
    """Registra cada observación de orden de velocidad con el SP de Spe verdadero."""
    meta = _STATE.get('meta')
    path = _STATE.get('path')
    if meta is None or not path:
        return
    from showdown.inference_sim.soundness import _lookup_truth
    from teams.team_converter import normalize_name

    true_sp, _final = _lookup_truth(meta, normalize_name(ev['species']))
    if not true_sp:
        return
    tsp = int(true_sp.get('spe', 0))
    rec = dict(ev)
    rec.update({'kind': 'speed', 'true_spe_sp': tsp,
                'true_in_before': tsp in ev['before'] if ev['before'] is not None else None,
                'true_in_after': tsp in ev['after'],
                'series': _STATE['series'], 'seq': _STATE['seq'], 'pid': os.getpid()})
    _STATE['seq'] += 1
    try:
        with open(path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec) + chr(10))
    except OSError:
        pass



def install(path: Optional[str] = None) -> bool:
    """Instala el registro si hay ruta (argumento o ``PS_CALIB_LOG``). Devuelve si quedó activo."""
    path = path or os.environ.get('PS_CALIB_LOG', '').strip()
    if not path:
        return False
    from showdown.stat_inference import set_calibration_sink

    _STATE['path'] = str(path)
    set_calibration_sink(_sink)
    from showdown.stat_inference import set_speed_sink
    set_speed_sink(_speed_sink)
    return True


def uninstall() -> None:
    from showdown.stat_inference import set_calibration_sink

    set_calibration_sink(None)
    from showdown.stat_inference import set_speed_sink
    set_speed_sink(None)
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
        'series': _STATE['series'], 'seq': _STATE['seq'], 'pid': os.getpid(),
    }
    _STATE['seq'] += 1
    if os.environ.get('PS_CALIB_PROFILE', '').strip() in ('1', 'true', 'yes'):
        rec['profile'] = _profile(ev)
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
