"""¿Cuánto importa el Team Preview? Medición previa a entrenar los previews.

Sobre los MISMOS contextos (pares de equipos MetaVGC), nuestro lado juega con
π_c congelado y solo cambia la formación de preview:

- ``pilot``: preview propio de π_c (heurística de greedy_joint / expectimax);
- ``fixed``: ``/team 1234`` (orden de exportación; fallback del contrato);
- ``random``: formación uniforme entre las 360 legales.

El rival siempre usa el preview propio de π_c. La diferencia de WR entre brazos
acota cuánto puede aportar un preview aprendido con este piloto. Si
``pilot − random`` es de 1–2 puntos, un bandit de preview con ruido de BO1
necesitará muchísimos episodios para distinguir mejoras: conviene saberlo antes
de gastar CPU entrenando.

Ejemplo (servidor)::

  python -m showdown.inference_sim.run_preview_value_probe --n 3000 \\
      --num-workers 28 --start-ps --out ~/iapsvgc_runs/preview_value_probe.json

Costo: ``n × brazos`` BO1 (≈5–8 BO1/s con 16–28 workers).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ARMS = ('pilot', 'fixed', 'random')


def _ci95(x: np.ndarray) -> float:
    return float(1.96 * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float('nan')


def summarize(wins: Dict[str, np.ndarray]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for arm, w in wins.items():
        out[arm] = {'n': int(len(w)), 'wr': float(w.mean()), 'ci95': _ci95(w)}
    arms = list(wins)
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            d = wins[a] - wins[b]
            out[f'{a}-{b}'] = {'delta': float(d.mean()), 'ci95': _ci95(d)}
    return out


def run_probe(env, *, n: int, arms: Sequence[str], seed: int) -> Dict[str, np.ndarray]:
    from showdown.rl.preview_eval import wr_from_rewards

    env.reseed(int(seed))
    _obs, infos = env.reset_batch(int(n))
    masks = [np.asarray(info['action_mask']) for info in infos]
    rng = np.random.default_rng(int(seed) + 1)
    wins: Dict[str, np.ndarray] = {}
    for arm in arms:
        t0 = time.time()
        if arm == 'pilot':
            r = env.step_batch_pilot()
        elif arm == 'fixed':
            r = env.step_batch([0] * int(n))
        elif arm == 'random':
            r = env.step_batch([int(rng.choice(np.flatnonzero(m > 0.5))) for m in masks])
        else:
            raise ValueError(f'brazo desconocido {arm!r}; opciones: {", ".join(ARMS)}')
        wins[arm] = wr_from_rewards(r)
        print(f'[probe] {arm}: WR={wins[arm].mean():.4f} n={n} ({time.time() - t0:.0f}s)', flush=True)
    return wins


def main(argv: Optional[Sequence[str]] = None) -> int:
    from showdown.rl.preview_live import PILOTS, make_preview_env, provenance_meta

    ap = argparse.ArgumentParser(description='Valor del Team Preview bajo π_c congelado (BO1 live)')
    ap.add_argument('--n', type=int, default=3000, help='Contextos (pares de equipos)')
    ap.add_argument('--arms', default=','.join(ARMS), help='Brazos: pilot,fixed,random')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pilot', choices=PILOTS, default='greedy_joint')
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--envs-per-server', type=int, default=4)
    ap.add_argument('--base-port', type=int, default=None)
    ap.add_argument('--start-ps', action='store_true')
    ap.add_argument('--max-turns', type=int, default=40)
    ap.add_argument('--out', default='logs/preview_value_probe.json')
    args = ap.parse_args(list(argv) if argv is not None else None)

    arms: List[str] = [a.strip() for a in args.arms.split(',') if a.strip()]
    live = {
        'backend': 'live', 'num_workers': args.num_workers,
        'envs_per_server': args.envs_per_server, 'base_port': args.base_port,
        'start_ps': args.start_ps, 'pilot': args.pilot, 'max_turns': args.max_turns,
    }
    env, pool = make_preview_env(args.seed, live)
    try:
        wins = run_probe(env, n=args.n, arms=arms, seed=args.seed)
        prov = provenance_meta(env, pool)
    finally:
        if pool is not None:
            pool.close()
    summary = summarize(wins)
    for k, v in summary.items():
        if 'wr' in v:
            print(f'{k:>14}: WR {v["wr"]:.4f} +- {v["ci95"]:.4f}')
        else:
            print(f'{k:>14}: d {v["delta"]:+.4f} +- {v["ci95"]:.4f}')
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'n': args.n, 'seed': args.seed, 'pilot': args.pilot, 'arms': arms,
        'summary': summary, **prov,
    }, indent=2) + '\n', encoding='utf-8')
    print(f'OK {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
