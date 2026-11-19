"""¿Alcanza la calculadora real para la observación de DDQN/PPO? (P5 completo, 19-11-2026)

La observación pediría ~80 cálculos de daño por paso (2 nuestros × 4 movimientos × 2 rivales × 3 modos,
más los movimientos rivales y el aliado). El entrenamiento corre a ~300 pasos/s con 32 entornos, es
decir ~24 000 cálculos/s. Este script simula N procesos de entorno que piden lotes de ``--batch``
cálculos al servidor TCP compartido (como el entrenamiento) con casos del pool, y mide:
cálculos/s agregados, latencia por lote (p50 / p95) y pasos/s equivalentes.

Modo ``--unique``: cada pedido distinto (HP al azar) → sin aciertos de caché (peor caso).

    python scripts/analysis/bench_calc_obs.py --procs 32 --batch 80 --seconds 60 [--unique]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

WEATHERS = (None, 'Sun', 'Rain', 'Sand', 'Snow')
TERRAINS = (None, 'Grassy', 'Psychic', 'Electric', 'Misty')


def _pool():
    from showdown.battle_bots.greedy_solo.greedy import CalcPoke
    from showdown.inference_sim.team_factory import active_teams
    mons = []
    for split in ('train', 'eval'):
        for t in active_teams(split):
            for m in t.mons:
                mons.append((CalcPoke(species=m.species, nature=m.nature or 'Serious', item=m.item or None,
                                      ability=m.ability or None), list(m.moves)))
    return mons


def _worker(args):
    seconds, batch, unique, seed = args
    import dataclasses
    from showdown.battle_bots.greedy_solo.greedy import FieldContext, build_calc_request
    from showdown.damage_calc_bridge import damage_ranges_batch
    rng = random.Random(seed)
    mons = _pool()
    lat, n, err = [], 0, 0
    t_end = time.time() + seconds
    while time.time() < t_end:
        reqs = []
        for _ in range(batch):
            (a, moves), (d, _m) = rng.choice(mons), rng.choice(mons)
            if unique:
                d = dataclasses.replace(d, cur_hp=rng.randint(1, 100))
            ctx = FieldContext(weather=rng.choice(WEATHERS), terrain=rng.choice(TERRAINS))
            reqs.append(build_calc_request(a, d, rng.choice(moves), field_ctx=ctx, is_spread=rng.random() < 0.3))
        t0 = time.time()
        out = damage_ranges_batch(reqs)
        lat.append(time.time() - t0)
        n += len(reqs)
        err += sum(1 for r in out if r is None)
    return n, err, lat


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=32)
    ap.add_argument('--batch', type=int, default=80)
    ap.add_argument('--seconds', type=float, default=60.0)
    ap.add_argument('--unique', action='store_true')
    ap.add_argument('--out', default='')
    a = ap.parse_args(argv)
    from showdown.battle_bots.greedy_solo.calc_async import warm_damage_calc
    warm_damage_calc()
    t0 = time.time()
    with mp.get_context('spawn').Pool(a.procs) as pool:
        res = pool.map(_worker, [(a.seconds, a.batch, a.unique, i) for i in range(a.procs)])
    wall = time.time() - t0
    n = sum(r[0] for r in res)
    err = sum(r[1] for r in res)
    lat = sorted(x for r in res for x in r[2])
    rep = {'procs': a.procs, 'batch': a.batch, 'unique': a.unique, 'seconds': a.seconds,
           'calculos': n, 'errores': err, 'calculos_por_s': round(n / a.seconds, 1),
           'pasos_por_s_equivalentes': round(n / a.seconds / a.batch, 1),
           'latencia_lote_p50_ms': round(1000 * statistics.median(lat), 1) if lat else None,
           'latencia_lote_p95_ms': round(1000 * lat[int(0.95 * (len(lat) - 1))], 1) if lat else None,
           'wall_s': round(wall, 1)}
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
