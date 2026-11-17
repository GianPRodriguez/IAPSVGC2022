"""¿DDQN y PPO rinden distinto según el arquetipo del equipo rival? (memorizacion_equipos.md §11)

Etiqueta cada equipo rival por su estrategia, a partir de su hoja (movimientos y habilidades):
Trick Room, Tailwind, Fake Out (2 o más usuarios), redirección y clima. Un equipo puede tener
varias etiquetas. Compara la tasa de victorias contra los 3 heurísticos con equipos `eval` y con
los equipos propios de entrenamiento, para DDQN (REG) y PPO (con simetría).

Uso: python scripts/analysis/analisis_arquetipos.py <dir con kickoff12/ kickoff14/ kickoff15/ (diag/*.json)>
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from showdown.inference_sim.team_factory import active_teams  # noqa: E402

D = Path(sys.argv[1])
HEUR = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')
TEAMS = {t.source_url: t for s in ('train', 'eval') for t in active_teams(s)}


def _id(x):
    return ''.join(c for c in str(x or '').lower() if c.isalnum())


WEATHER = {'drought', 'drizzle', 'sandstream', 'snowwarning', 'orichalcumpulse'}


def tags(team):
    moves = [{_id(m) for m in (getattr(mon, 'moves', None) or [])} for mon in team.mons]
    abil = {_id(getattr(mon, 'ability', '')) for mon in team.mons}
    out = set()
    if any('trickroom' in m for m in moves):
        out.add('Trick Room')
    if any('tailwind' in m for m in moves):
        out.add('Tailwind')
    if sum('fakeout' in m for m in moves) >= 2:
        out.add('Fake Out ×2+')
    if any(m & {'followme', 'ragepowder'} for m in moves):
        out.add('Redirección')
    if abil & WEATHER:
        out.add('Clima')
    if not out & {'Trick Room', 'Tailwind'}:
        out.add('Sin control de velocidad (TR/TW)')
    return out


AGENTS = {
    'DDQN (REG)': {'eval': ['kickoff14/diag/REG_s0_final_eval', 'kickoff14/diag/REG_s1_final_eval',
                            'kickoff14/diag/REG_s0_best_eval', 'kickoff14/diag/REG_s1_best_eval',
                            'kickoff15/diag/R_wd03_s0_final_eval', 'kickoff15/diag/R_wd03_s1_final_eval'],
                   'own': ['kickoff14/diag/REG_s0_best_own', 'kickoff14/diag/REG_s1_best_own']},
    'PPO (simetría)': {'eval': ['kickoff12/diag/PPO_sym_s0_final_eval', 'kickoff12/diag/PPO_sym_s1_final_eval',
                                'kickoff12/diag/PPO_sym_s0_best_eval', 'kickoff12/diag/PPO_sym_s1_best_eval'],
                       'own': ['kickoff12/diag/PPO_sym_s0_best_own', 'kickoff12/diag/PPO_sym_s1_best_own']},
}


def games(paths, foes=HEUR):
    out = []
    for p in paths:
        f = D / f'{p}.json'
        if f.is_file():
            out += [g for g in json.loads(f.read_text())['partidas'] if g['rival'] in foes]
    return out


def rate(gs):
    n = len(gs)
    return (sum(g['res'] == 'win' for g in gs) / n, n) if n else (float('nan'), 0)


def fmt(r):
    p, n = r
    if not n:
        return '—'
    return f'{100 * p:.0f} % (n={n}, ± {100 * 1.96 * math.sqrt(p * (1 - p) / n):.0f})'


ev_teams = [t for t in active_teams('eval')]
tag_count = defaultdict(int)
for t in ev_teams:
    for k in tags(t):
        tag_count[k] += 1
fit_count = defaultdict(int)
for t in active_teams('fit'):
    for k in tags(t):
        fit_count[k] += 1
ALL_TAGS = ('Trick Room', 'Tailwind', 'Sin control de velocidad (TR/TW)', 'Fake Out ×2+', 'Redirección', 'Clima')

print('# Rendimiento por arquetipo del equipo rival (3 heurísticos)\n')
print('| Arquetipo | Equipos `eval` | Equipos `fit` | ' + ' | '.join(f'{a} `eval`' for a in AGENTS) + ' | ' +
      ' | '.join(f'{a} propios' for a in AGENTS) + ' |')
print('|---|---|---|' + '---|' * (2 * len(AGENTS)))
for k in ('(todos)',) + ALL_TAGS:
    cells = []
    for split in ('eval', 'own'):
        for a, src in AGENTS.items():
            gs = games(src[split])
            if k != '(todos)':
                gs = [g for g in gs if g['equipo'] in TEAMS and k in tags(TEAMS[g['equipo']])]
            cells.append(fmt(rate(gs)))
    ne = len(ev_teams) if k == '(todos)' else tag_count[k]
    nf = len(active_teams('fit')) if k == '(todos)' else fit_count[k]
    print(f'| {k} | {ne} | {nf} | ' + ' | '.join(cells) + ' |')

print('\n# Solo contra expectimax (el rival que mejor ejecuta la estrategia del equipo), equipos `eval`\n')
print('| Arquetipo | ' + ' | '.join(AGENTS) + ' |')
print('|---|' + '---|' * len(AGENTS))
for k in ('(todos)',) + ALL_TAGS:
    cells = []
    for a, src in AGENTS.items():
        gs = games(src['eval'], foes=('expectimax_bayesian',))
        if k != '(todos)':
            gs = [g for g in gs if g['equipo'] in TEAMS and k in tags(TEAMS[g['equipo']])]
        cells.append(fmt(rate(gs)))
    print(f'| {k} | ' + ' | '.join(cells) + ' |')
