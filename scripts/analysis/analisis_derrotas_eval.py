"""Por qué DDQN pierde con equipos no vistos (memorizacion_equipos.md §7).

Uso: python scripts/analysis/analisis_derrotas_eval.py <dir con kickoff8/ kickoff9/ kickoff11/ (diag/*.json)>
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from showdown.inference_sim.team_factory import active_teams  # noqa: E402

D = Path(sys.argv[1])
HEUR = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')

teams = {t.source_url: t for s in ('train', 'eval') for t in active_teams(s)}
fit = active_teams('fit')
fit_species = defaultdict(int)
fit_sets = set()


def set_key(m):
    return (m.species, getattr(m, 'item', ''), getattr(m, 'ability', ''),
            tuple(sorted(getattr(m, 'moves', []) or [])))


for t in fit:
    for m in t.mons:
        fit_species[m.species] += 1
        fit_sets.add(set_key(m))

GROUPS = {
    'DDQN actual (B, C: máscara, sin sim.)': ['kickoff11/diag/B_s0_best', 'kickoff11/diag/B_s0_final',
                                               'kickoff11/diag/B_s1_best', 'kickoff11/diag/B_s1_final',
                                               'kickoff11/diag/C2M_s0_best', 'kickoff11/diag/C2M_s0_final'],
    'DDQN anterior (cola 8)': ['kickoff8/diag/F170_s0', 'kickoff8/diag/F170_s1', 'kickoff8/diag/SY50_s0',
                               'kickoff8/diag/SY50_s1', 'kickoff8/diag/S50_s0', 'kickoff8/diag/S100_s0'],
    'PPO pilotos (400k)': ['kickoff9/diag/ppo_sym', 'kickoff9/diag/ppo_nosym'],
}


def load(prefix, split):
    cands = [D / f'{prefix}_{split}.json']
    if prefix.endswith('_best') or prefix.endswith('_final'):
        base, kind = prefix.rsplit('_', 1)
        cands = [D / f'{base}_{kind}_{"own" if split in ("own", "train") else "eval"}.json']
    if split == 'own':
        cands.append(D / f'{prefix}_train.json')
    for p in cands:
        if p.is_file():
            return json.loads(p.read_text())
    return None


def games(group, split):
    out = []
    for pre in GROUPS[group]:
        r = load(pre, split)
        if r:
            out += [g for g in r['partidas'] if g['rival'] in HEUR]
    return out


def rate(gs):
    n = len(gs)
    w = sum(g['res'] == 'win' for g in gs)
    return w / n if n else float('nan'), n


P = print
P('# A. Perfil de acciones: propios frente a eval (por 100 decisiones; partidas = decisiones por partida)\n')
for group in GROUPS:
    for split in ('own', 'eval'):
        agg = defaultdict(float)
        ngames = 0
        for pre in GROUPS[group]:
            r = load(pre, split)
            if not r:
                continue
            for foe in HEUR:
                for k, v in r['por_rival'].get(foe, {}).items():
                    agg[k] += v
            ngames += sum(1 for g in r['partidas'] if g['rival'] in HEUR)
        dec = agg['decisiones'] or 1
        keys = ('ataque_rival', 'area_rivales', 'protect', 'mega', 'cambio_voluntario', 'cambio_forzado',
                'otro_estado', 'pass_obligatorio', 'ataque_aliado')
        cells = ' '.join(f'{k}={100 * agg.get(k, 0) / dec:.1f}' for k in keys)
        P(f'{group:40s} {split:4s} dec/partida={dec / max(1, ngames):.1f}  {cells}')
P()

P('# B. Dificultad por equipo eval (heurísticos)\n')
per_team = {}
for group in GROUPS:
    tw = defaultdict(lambda: [0, 0])
    for g in games(group, 'eval'):
        tw[g['equipo']][0] += g['res'] == 'win'
        tw[g['equipo']][1] += 1
    per_team[group] = tw
    p_all, n_all = rate(games(group, 'eval'))
    ns = np.array([v[1] for v in tw.values()])
    ws = np.array([v[0] for v in tw.values()])
    # sobredispersión: varianza observada de las tasas frente a la binomial esperada
    obs = np.sum((ws - ns * p_all) ** 2 / (ns * p_all * (1 - p_all)))
    dof = len(ns) - 1
    P(f'{group}: {len(tw)} equipos, {n_all} partidas, media {100 * p_all:.1f} %, '
      f'chi2 de dispersión {obs:.0f} con {dof} gl (≈{dof} si no hay equipos más difíciles que otros; z={(obs - dof) / math.sqrt(2 * dof):.1f})')
    rates = sorted(((v[0] / v[1], v[1], k) for k, v in tw.items() if v[1] >= 6))
    P('  equipos con 0–20 % (n ≥ 6): ' + str(sum(1 for r in rates if r[0] <= 0.2)) + ' de ' + str(len(rates)))
    P('  equipos con ≥ 80 % (n ≥ 6): ' + str(sum(1 for r in rates if r[0] >= 0.8)) + ' de ' + str(len(rates)))

# correlación entre grupos
gnames = list(GROUPS)
for i in range(len(gnames)):
    for j in range(i + 1, len(gnames)):
        a, b = per_team[gnames[i]], per_team[gnames[j]]
        common = [k for k in a if k in b and a[k][1] >= 3 and b[k][1] >= 2]
        x = np.array([a[k][0] / a[k][1] for k in common])
        y = np.array([b[k][0] / b[k][1] for k in common])
        if len(common) > 5:
            P(f'correlación por equipo {gnames[i]} vs {gnames[j]}: r={np.corrcoef(x, y)[0, 1]:.2f} (n={len(common)} equipos)')
# fiabilidad dentro del grupo DDQN actual: mitad A vs mitad B de checkpoints
half = {}
for h, pres in (('a', GROUPS['DDQN actual (B, C: máscara, sin sim.)'][0::2]),
                ('b', GROUPS['DDQN actual (B, C: máscara, sin sim.)'][1::2])):
    tw = defaultdict(lambda: [0, 0])
    for pre in pres:
        r = load(pre, 'eval')
        for g in (r or {}).get('partidas', []):
            if g['rival'] in HEUR:
                tw[g['equipo']][0] += g['res'] == 'win'
                tw[g['equipo']][1] += 1
    half[h] = tw
common = [k for k in half['a'] if k in half['b'] and half['a'][k][1] >= 3 and half['b'][k][1] >= 3]
x = np.array([half['a'][k][0] / half['a'][k][1] for k in common])
y = np.array([half['b'][k][0] / half['b'][k][1] for k in common])
P(f'fiabilidad (DDQN actual, mitad de checkpoints vs la otra mitad): r={np.corrcoef(x, y)[0, 1]:.2f} (n={len(common)})')
P()

P('# C. ¿Pierde por especies o sets que no conoce? (DDQN actual, equipos eval, heurísticos)\n')
gs = games('DDQN actual (B, C: máscara, sin sim.)', 'eval')
by_unseen_sp = defaultdict(list)
by_known_sets = defaultdict(list)
for g in gs:
    t = teams.get(g['equipo'])
    if not t:
        continue
    uns = sum(1 for m in t.mons if fit_species.get(m.species, 0) == 0)
    ks = sum(1 for m in t.mons if set_key(m) in fit_sets)
    by_unseen_sp[uns].append(g)
    by_known_sets[ks].append(g)
for k in sorted(by_unseen_sp):
    p, n = rate(by_unseen_sp[k])
    P(f'  especies nunca vistas en fit = {k}: {100 * p:.1f} % (n={n})')
for k in sorted(by_known_sets):
    p, n = rate(by_known_sets[k])
    P(f'  sets idénticos a uno de fit = {k}/6: {100 * p:.1f} % (n={n})')
P()

P('# D. Especies del rival: tasa con equipos eval frente a propios (DDQN actual, n ≥ 25 en eval)\n')
own = games('DDQN actual (B, C: máscara, sin sim.)', 'own')


def by_species(gs):
    d = defaultdict(list)
    for g in gs:
        t = teams.get(g['equipo'])
        if t:
            for m in t.mons:
                d[m.species].append(g)
    return d


se, so = by_species(gs), by_species(own)
rows = []
for sp, g_e in se.items():
    if len(g_e) < 25:
        continue
    pe, ne = rate(g_e)
    po, no = rate(so.get(sp, []))
    rows.append((pe, sp, ne, po, no))
P(f'  media eval {100 * rate(gs)[0]:.1f} %, propios {100 * rate(own)[0]:.1f} %')
for pe, sp, ne, po, no in sorted(rows):
    P(f'  {sp:18s} eval {100 * pe:5.1f} % (n={ne:3d})   propios {100 * po:5.1f} % (n={no:3d})   en fit: {fit_species.get(sp, 0)} equipos')
P()

P('# E. ¿Pierde contra los mismos equipos que PPO? Equipos eval más difíciles para DDQN actual\n')
a, b = per_team['DDQN actual (B, C: máscara, sin sim.)'], per_team['PPO pilotos (400k)']
hard = sorted(((v[0] / v[1], v[1], k) for k, v in a.items() if v[1] >= 8))[:12]
for r_, n_, k in hard:
    t = teams.get(k)
    pb = b.get(k, [0, 0])
    sp = ', '.join(m.species for m in t.mons) if t else '?'
    P(f'  DDQN {100 * r_:4.0f} % (n={n_:2d})  PPO {100 * pb[0] / max(1, pb[1]):4.0f} % (n={pb[1]})  {sp}')
