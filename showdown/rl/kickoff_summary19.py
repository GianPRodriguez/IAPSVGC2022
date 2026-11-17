"""Resumen de la decimonovena cola: decisiones según el plan del rival y uso real de los arquetipos.

Lee ``<dir>/diag/{nombre}.json`` (policy_diag ``--context`` con equipo del agente al azar en cada
partida) y escribe ``<dir>/kickoff_summary19.md`` (memorizacion_equipos.md §12).

    python -m showdown.rl.kickoff_summary19 ~/iapsvgc_runs/kickoff19
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from showdown.rl.context_diag import summarize

HEUR = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')
FOES = ('random',) + HEUR
AGENTS = (('DDQN (REG 0,03 + dropout 0,1)', ('ddqn_s0', 'ddqn_s1')), ('PPO (simetría)', ('ppo_s0', 'ppo_s1')))
EVENTS = (('trick_room', 'usa Trick Room'), ('trick_room_activado', 'activa Trick Room'),
          ('tailwind_activado', 'activa Tailwind'), ('fake_out', 'usa Fake Out'), ('redirect', 'usa redirección'))


def _load(d: Path, name: str):
    try:
        return json.loads((d / 'diag' / f'{name}.json').read_text(encoding='utf-8'))
    except Exception:
        return None


def build(d: Path) -> str:
    L = ['# Resumen de la decimonovena cola: ¿los agentes juegan contra el plan del rival?', '',
         'Equipo del agente **al azar en cada partida** (grupo `eval`, como el rival; solo para este '
         'análisis: los agentes se entrenaron con el equipo fijo). Rivales con equipos `eval`, sin '
         'exploración. Decisiones libres contra los 3 heurísticos (sin pasos obligatorios ni cambios '
         'forzados). Un contexto puede coincidir con otros.', '']
    for title, names in AGENTS:
        reps = [r for r in (_load(d, n) for n in names) if r]
        L += [f'## {title}', '']
        if not reps:
            L += ['Sin datos.', '']
            continue
        res = defaultdict(Counter)
        ctx = defaultdict(Counter)
        ev_games = defaultdict(lambda: defaultdict(Counter))  # rival → lado → evento → partidas con ≥1
        n_games = Counter()
        for r in reps:
            for f, c in r.get('resultados', {}).items():
                res[f].update(c)
            for c, v in r.get('contexto', {}).items():
                ctx[c].update(v)
            for g in r.get('partidas', []):
                n_games[g['rival']] += 1
                for side, evs in (g.get('eventos') or {}).items():
                    for k, v in evs.items():
                        if v:
                            ev_games[g['rival']][side][k] += 1
        L += ['**Victorias con equipo propio al azar:** ' + ', '.join(
            f"{f} {100 * res[f].get('win', 0) / max(1, sum(res[f].values())):.0f} %" for f in FOES if res[f]), '']
        L += ['### Decisiones según el contexto (% de las decisiones libres en ese contexto)', ''] + summarize(ctx) + ['']
        L += ['### Uso real de movimientos clave (% de partidas en que ocurre al menos una vez)', '',
              '| Rival | Lado | ' + ' | '.join(t for _, t in EVENTS) + ' |', '|---|---|' + '---|' * len(EVENTS)]
        for f in FOES:
            if not n_games[f]:
                continue
            for side in ('rival', 'nuestro'):
                cells = [f"{100 * ev_games[f][side][k] / n_games[f]:.0f} %" for k, _ in EVENTS]
                L.append(f'| {f} | {side} | ' + ' | '.join(cells) + ' |')
        L.append('')
    return '\n'.join(L) + '\n'


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    d = Path(args[0]).expanduser()
    (d / 'kickoff_summary19.md').write_text(build(d), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
