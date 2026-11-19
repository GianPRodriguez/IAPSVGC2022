"""¿Cuánto se aparta el sorteo (P1) del modo determinista? (memorizacion_equipos.md §17.2)

Lee los ``policy_diag --sample`` con ``divergencia`` y resume, por temperatura: % de decisiones
(con más de una opción legal) en que la jugada completa o el hueco 1 difieren de lo que habría
elegido el modo determinista, lo mismo solo en el turno 1, % de partidas con al menos una
decisión distinta, y la tasa de victorias contra los 3 heurísticos.

    python scripts/analysis/divergencia_p1.py ~/iapsvgc_runs/p1_div
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

HEUR = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')


def summarize(paths):
    c, w, n = Counter(), 0, 0
    for p in paths:
        d = json.loads(Path(p).read_text(encoding='utf-8'))
        for foe, cnt in (d.get('divergencia') or {}).items():
            if foe in HEUR:
                c.update(cnt)
        for foe in HEUR:
            r = d.get('resultados', {}).get(foe, {})
            w += r.get('win', 0)
            n += sum(r.values())
    pct = lambda a, b: f'{100 * c[a] / c[b]:.1f} %' if c[b] else '—'  # noqa: E731
    return {
        'jugada': pct('todas_jugada_distinta', 'todas_decisiones'),
        'hueco1': pct('todas_hueco1_distinto', 'todas_decisiones'),
        't1_jugada': pct('turno_1_jugada_distinta', 'turno_1_decisiones'),
        'partidas': pct('partidas_con_alguna_distinta', 'partidas'),
        'decisiones': c['todas_decisiones'],
        'victorias': f'{100 * w / n:.1f} %' if n else '—',
    }


def main(argv=None) -> int:
    d = Path((sys.argv[1:] if argv is None else argv)[0]).expanduser()
    lines = ['| Temperatura | Jugada distinta | Hueco 1 distinto | Turno 1: jugada distinta | '
             'Partidas con alguna distinta | Decisiones | Victorias (`val`) |', '|---|---|---|---|---|---|---|']
    for t in ('0.5', '1.0'):
        r = summarize(sorted(glob.glob(str(d / f'T{t}_s*_val.json'))))
        lines.append(f"| {t} | {r['jugada']} | {r['hueco1']} | {r['t1_jugada']} | {r['partidas']} | "
                     f"{r['decisiones']} | {r['victorias']} |")
    out = '\n'.join(lines) + '\n'
    (d / 'divergencia_p1.md').write_text(out, encoding='utf-8')
    print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
