"""Resumen de la vigesimosexta cola: P5 completo (calculadora real) + preview v2.1, PPO y DDQN.

Mismas tablas que ``kickoff_summary25`` (mismos nombres de archivos), más: el preview v2.1 y el uso de la
calculadora en cada entrenamiento (fracción de observaciones con respaldo y pedidos sin respuesta), y la
comparación contra PPO v1 en espejo con la calibración (v1 contra v1 = 59,8 %).
Decisión del usuario: medir antes de las corridas oficiales; si algún agente pierde más de 2 puntos en
`val`, se revisa antes de seguir.

    python -m showdown.rl.kickoff_summary26 ~/iapsvgc_runs/p5full ~/iapsvgc_runs/ppo_v2 ~/iapsvgc_runs/ppo_v1 \
        ~/iapsvgc_runs/kickoff15
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from showdown.rl import kickoff_summary25 as K25

CALIBRACION_V1 = 0.598  # v1 contra v1 en espejo (vigesimocuarta cola, memorizacion_equipos §17.1)


def _calc_usage(txt: Path) -> str:
    try:
        lines = [ln for ln in txt.read_text(encoding='utf-8', errors='replace').splitlines()
                 if 'calculadora real en la observación' in ln]
    except Exception:
        return 'sin registro'
    if not lines:
        return 'sin registro'
    m_fb = re.search(r"'fallback_rate': ([0-9.e-]+)", lines[-1])
    m_ms = re.search(r"'missing_rate': ([0-9.e-]+)", lines[-1])
    m_ob = re.search(r"'obs': (\d+)", lines[-1])
    if not (m_fb and m_ms and m_ob):
        return lines[-1][-120:]
    return (f"{int(m_ob.group(1))} obs; respaldo {100 * float(m_fb.group(1)):.2f} %; "
            f"pedidos sin respuesta {100 * float(m_ms.group(1)):.2f} %")


def _speed(txt: Path) -> str:
    try:
        m = re.findall(r'\| ([0-9.]+) step/s \| ETA 0s \| elapsed ([0-9hms]+)', txt.read_text(encoding='utf-8', errors='replace'))
    except Exception:
        return '—'
    return f'{m[-1][0]} pasos/s, {m[-1][1]}' if m else '—'


def build(out: Path, v2: Path, v1: Path, q15: Path) -> str:
    md = K25.build(out, v2, v1, q15)
    md = md.replace('# P5: corrección de habilidades en la observación (vigesimoquinta cola)',
                    '# P5 completo (calculadora real) + preview v2.1 (vigesimosexta cola)', 1)
    md = md.replace("(50 % = igual que v1).", f"(referencia: v1 contra v1 = {100 * CALIBRACION_V1:.1f} %, calibración).")
    L = ['', '## Preview v2.1', '']
    rep = out / 'preview_v21' / 'reporte.txt'
    try:
        mw = re.search(r'MEAN_WR:\s*([\d.]+)', rep.read_text(encoding='utf-8'))
        L.append(f'Entrenado con la receta del v2 y la observación nueva. MEAN_WR final: {mw.group(1) if mw else "—"}.')
    except Exception:
        L.append('Sin reporte del preview v2.1.')
    L += ['', '## Calculadora real en el entrenamiento', '',
          '| Corrida | Uso de la calculadora | Velocidad final |', '|---|---|---|']
    for name in ('ppo_p5_s0', 'ppo_p5_s1', 'ddqn_p5_s0', 'ddqn_p5_s1'):
        txt = out / f'{name}.txt'
        L.append(f'| {name} | {_calc_usage(txt)} | {_speed(txt)} |')
    return md + '\n'.join(L) + '\n'


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    out = Path(a[0]).expanduser()
    v2 = Path(a[1]).expanduser() if len(a) > 1 else out.parent / 'ppo_v2'
    v1 = Path(a[2]).expanduser() if len(a) > 2 else out.parent / 'ppo_v1'
    q15 = Path(a[3]).expanduser() if len(a) > 3 else out.parent / 'kickoff15'
    (out / 'kickoff_summary26.md').write_text(build(out, v2, v1, q15), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
