r"""Diagnóstico del paso 0 de H 2.1 (spec de H §4b, 23-11-2026). Apagado por defecto.

Con ``MCTS_NASH_DIAG_FILE=<ruta>``, cada decisión no trivial de H 2.0 añade una fila JSON a
``<ruta>.<pid>`` con:

* dispersión de la matriz de la raíz: filas, columnas, celdas visitadas, visitas por celda;
* concentración de la mezcla de Nash: \(\sigma^*\) (máximo, entropía normalizada, soporte) y \(\tau^*\);
* qué fila elegirían otras reglas sobre la misma \(\hat A\): más visitada, mejor respuesta al prior
  del rival \(q\) y la mezcla \(\tfrac12(\hat A\tau^* + \hat A q)\);
* con ``MCTS_NASH_DIAG_G=1`` (por defecto), qué joint elige G 5.2 en el mismo estado (mismas raíz,
  creencias y semilla) y su tiempo.

No cambia la decisión de H. Se resume con ``python -m showdown.rl.mcts_puct_nash.diag <ruta>*``.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Iterable, List, Optional, Sequence

import numpy as np


def enabled() -> bool:
    return bool(os.environ.get('MCTS_NASH_DIAG_FILE'))


def _entropy_norm_over(p: np.ndarray, n: int) -> float:
    """Entropía normalizada por log(n) (n = filas o columnas disponibles)."""
    p = np.asarray(p, dtype=np.float64)
    if n <= 1:
        return 0.0
    q = p[p > 1e-12]
    q = q / q.sum()
    return float(-(q * np.log(q)).sum() / math.log(n))


def record(snap, root_joints: Optional[Sequence[int]], stat_beliefs, joint: int, root, tel,
           cfg, seed: int) -> None:
    """Escribe una fila por decisión no trivial. Nunca lanza."""
    try:
        if tel is None or getattr(tel, 'trivial', False) or len(root.I) <= 1:
            return
        from showdown.rl.mcts_puct_nash.nash import fictitious_play
        A_hat = root.payoff()
        n = np.asarray(root.n, dtype=np.float64)
        R, C = A_hat.shape
        sigma, tau, v = fictitious_play(A_hat, T=int(cfg.t_fp_root))
        q = np.asarray(root.q, dtype=np.float64)
        q = q / q.sum() if q.sum() > 0 else np.full(C, 1.0 / C)
        I = [int(i) for i in root.I]
        row_vis = n.sum(axis=1)
        v_tau = A_hat @ tau
        v_q = A_hat @ q
        v_mix = 0.5 * v_tau + 0.5 * v_q
        visited = n[n > 0]
        h_idx = I.index(int(joint)) if int(joint) in I else -1
        row = {
            'turn': int(getattr(snap, 'turn', 0) or 0),
            'R': int(R), 'C': int(C), 'sims': int(n.sum()),
            'cells_visited': int((n > 0).sum()),
            'frac_unvisited': float((n == 0).mean()),
            'cell_visits_med': float(np.median(visited)) if len(visited) else 0.0,
            'cell_visits_max': float(n.max()),
            'top_row_share': float(row_vis.max() / max(1.0, row_vis.sum())),
            'rows_lt5': int((row_vis < 5).sum()),
            'sigma_max': float(sigma.max()), 'sigma_H': _entropy_norm_over(sigma, R),
            'sigma_support': int((sigma > 0.05).sum()),
            'tau_max': float(tau.max()), 'tau_H': _entropy_norm_over(tau, C),
            'tau_q_tv': float(0.5 * np.abs(tau - q).sum()),
            'v_nash': float(v),
            'h': int(joint),
            'h_visit': I[int(np.argmax(row_vis))],
            'h_q': I[int(np.argmax(v_q))],
            'h_mix': I[int(np.argmax(v_mix))],
            'gap_q': float(v_q.max() - (v_q[h_idx] if h_idx >= 0 else v_q.max())),
            'h_sec': float(getattr(tel, 'elapsed_sec', 0.0) or 0.0),
        }
        if os.environ.get('MCTS_NASH_DIAG_G', '1') != '0':
            from showdown.rl.mcts_puct.agent import MCTSPuctAgent, MCTSPuctConfig
            g = MCTSPuctAgent(MCTSPuctConfig(n_sims=200, seed=int(seed), version=5))
            t0 = time.perf_counter()
            gj = int(g.act(snap, root_joints=root_joints, stat_beliefs=stat_beliefs))
            row['g'] = gj
            row['g_sec'] = float(time.perf_counter() - t0)
            row['g_in_rows'] = gj in I
            if gj in I:
                k = I.index(gj)
                row['g_sigma'] = float(sigma[k])
                row['g_vis_share'] = float(row_vis[k] / max(1.0, row_vis.sum()))
                row['g_gap_tau'] = float(v_tau.max() - v_tau[k])
        path = f"{os.environ['MCTS_NASH_DIAG_FILE']}.{os.getpid()}"
        with open(path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(row) + '\n')
    except Exception as exc:  # pragma: no cover - diagnóstico, nunca rompe la partida
        try:
            path = f"{os.environ.get('MCTS_NASH_DIAG_FILE', 'diag')}.{os.getpid()}.err"
            with open(path, 'a', encoding='utf-8') as fh:
                fh.write(repr(exc)[:300] + '\n')
        except Exception:
            pass


def _load(paths: Iterable[str]) -> List[dict]:
    rows = []
    for p in paths:
        if p.endswith('.err'):
            continue
        with open(p, encoding='utf-8') as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def summarize(rows: List[dict]) -> str:
    if not rows:
        return 'sin filas'
    a = lambda k: np.array([r[k] for r in rows if k in r], dtype=np.float64)  # noqa: E731
    pct = lambda x: f'{100 * np.mean(x):.1f} %'  # noqa: E731
    q3 = lambda x: f'{np.percentile(x, 25):.2f} / {np.median(x):.2f} / {np.percentile(x, 75):.2f}'  # noqa: E731
    out = [f'decisiones no triviales: {len(rows)}',
           f'matriz raíz R×C (mediana): {np.median(a("R")):.0f} × {np.median(a("C")):.0f}',
           f'fracción de celdas sin visitar (p25/med/p75): {q3(a("frac_unvisited"))}',
           f'visitas por celda visitada, mediana (p25/med/p75): {q3(a("cell_visits_med"))}',
           f'filas con < 5 visitas (p25/med/p75): {q3(a("rows_lt5"))}',
           f'cuota de la fila más visitada (p25/med/p75): {q3(a("top_row_share"))}',
           f'σ* máx (p25/med/p75): {q3(a("sigma_max"))}   entropía norm.: {q3(a("sigma_H"))}   soporte>5 %: {q3(a("sigma_support"))}',
           f'τ* máx (p25/med/p75): {q3(a("tau_max"))}   entropía norm.: {q3(a("tau_H"))}   TV(τ*, q): {q3(a("tau_q_tv"))}',
           '',
           'acuerdo de la elección de H (argmax σ*) con otras reglas sobre la misma Â:',
           f'  más visitada: {pct(a("h") == a("h_visit"))}',
           f'  mejor respuesta al prior q: {pct(a("h") == a("h_q"))}',
           f'  mezcla ½(Âτ*+Âq): {pct(a("h") == a("h_mix"))}',
           f'  pérdida de H frente a la mejor respuesta a q, en valor [−1,1] (p25/med/p75): {q3(a("gap_q"))}']
    g = [r for r in rows if 'g' in r]
    if g:
        ag = lambda k: np.mean([r['h' if k == 'h' else k] == r['g'] for r in g])  # noqa: E731
        inrows = [r for r in g if r.get('g_in_rows')]
        out += ['',
                f'comparación con G 5.2 en los mismos estados ({len(g)} decisiones):',
                f'  H = G: {100 * ag("h"):.1f} %   más visitada = G: {100 * ag("h_visit"):.1f} %   '
                f'mejor resp. a q = G: {100 * ag("h_q"):.1f} %   mezcla = G: {100 * ag("h_mix"):.1f} %',
                f'  joint de G dentro de las filas de H: {100 * len(inrows) / len(g):.1f} %']
        if inrows:
            gs = np.array([r['g_sigma'] for r in inrows])
            gv = np.array([r['g_vis_share'] for r in inrows])
            out += [f'  σ* que H le da al joint de G (p25/med/p75): {q3(gs)}   σ* = 0: {pct(gs < 1e-6)}',
                    f'  cuota de visitas de H en el joint de G (p25/med/p75): {q3(gv)}']
        out += [f'  tiempo por decisión: H {np.mean(a("h_sec")):.2f} s, G {np.mean([r["g_sec"] for r in g]):.2f} s']
    early = [r for r in rows if r['turn'] <= 2]
    if early and g:
        out += [f'  H = G en turnos 1–2: {100 * np.mean([r["h"] == r["g"] for r in early if "g" in r]):.1f} % '
                f'({len(early)} decisiones)']
    return '\n'.join(out)


if __name__ == '__main__':
    print(summarize(_load(sys.argv[1:])))
