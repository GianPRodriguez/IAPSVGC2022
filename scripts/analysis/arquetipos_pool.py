"""Arquetipos del pool de equipos (memorizacion_equipos.md §16).

Etiqueta los 300 equipos (train / eval) por sus hojas: clima (fuente + abusador), Trick Room,
Tailwind, campos, redirección, Fake Out, setup, etc., y les asigna un arquetipo primario
(precedencia: Trick Room > clima > Tailwind > Campo Psíquico > equilibrado). También el equipo fijo.

    python scripts/analysis/arquetipos_pool.py
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from showdown.inference_sim.team_factory import active_teams, load_team_file


def i(x):
    return ''.join(c for c in str(x or '').lower() if c.isalnum())


def feats(team):
    mv = [{i(m) for m in mon.moves} for mon in team.mons]
    ab = [i(mon.ability) for mon in team.mons]
    it = [i(mon.item) for mon in team.mons]
    allm = set().union(*mv)
    has = lambda s: sum(1 for m in mv if m & s)
    f = {}
    # clima: fuente (habilidad, mega o movimiento) + abusador
    sun_src = sum(a in ('drought', 'orichalcumpulse') for a in ab) + sum(x == 'charizarditey' for x in it) + has({'sunnyday'})
    rain_src = sum(a in ('drizzle', 'primordialsea') for a in ab) + has({'raindance'})
    sand_src = sum(a in ('sandstream',) for a in ab) + has({'sandstorm'}) + sum(x in ('tyranitarite',) for x in it)
    snow_src = sum(a in ('snowwarning',) for a in ab) + has({'snowscape', 'hail'}) + sum(x == 'abomasite' for x in it)
    sun_ab = sum(a in ('chlorophyll', 'solarpower', 'protosynthesis', 'flowergift', 'harvest') for a in ab)
    rain_ab = sum(a in ('swiftswim', 'raindish', 'drysking') for a in ab)
    sand_ab = sum(a in ('sandrush', 'sandforce', 'sandveil') for a in ab)
    snow_ab = sum(a in ('slushrush', 'icebody') for a in ab) + has({'auroraveil', 'blizzard'})
    f['Sol'] = sun_src > 0 and (sun_ab > 0 or has({'weatherball', 'solarbeam', 'heatwave'}) >= 1)
    f['Lluvia'] = rain_src > 0
    f['Arena'] = sand_src > 0
    f['Nieve'] = snow_src > 0
    f['_abusa_clima'] = (sun_src and sun_ab) or (rain_src and rain_ab) or (sand_src and sand_ab) or (snow_src and snow_ab)
    f['Trick Room'] = has({'trickroom'}) > 0
    f['Tailwind'] = has({'tailwind'}) > 0
    f['Control de velocidad (Icy Wind / Electroweb)'] = has({'icywind', 'electroweb', 'thunderwave', 'scaryface'}) > 0
    terr = {'psychicsurge': 'Campo Psíquico', 'electricsurge': 'Campo Eléctrico', 'grassysurge': 'Campo de Hierba',
            'mistysurge': 'Campo de Niebla', 'hadronengine': 'Campo Eléctrico'}
    for a in ab:
        if a in terr:
            f[terr[a]] = True
    for mvname, lab in (('psychicterrain', 'Campo Psíquico'), ('electricterrain', 'Campo Eléctrico'),
                        ('grassyterrain', 'Campo de Hierba'), ('mistyterrain', 'Campo de Niebla')):
        if mvname in allm:
            f[lab] = True
    f['Redirección (Follow Me / Rage Powder)'] = has({'followme', 'ragepowder'}) > 0
    f['Fake Out ×2+'] = has({'fakeout'}) >= 2
    f['Intimidate ×2+'] = sum(a == 'intimidate' for a in ab) >= 2
    f['Perish Song'] = has({'perishsong'}) > 0
    f['Setup (Swords Dance / Nasty Plot / Dragon Dance…)'] = has({'swordsdance', 'nastyplot', 'dragondance', 'calmmind', 'quiverdance', 'bellydrum', 'bulkup', 'shellsmash', 'coaching'}) >= 1
    f['Pantallas (Reflect / Light Screen / Aurora Veil)'] = has({'reflect', 'lightscreen', 'auroraveil'}) > 0
    f['Wide Guard'] = has({'wideguard'}) > 0
    f['Sueño (Spore / Sleep Powder…)'] = has({'spore', 'sleeppowder', 'yawn', 'hypnosis', 'darkvoid'}) > 0
    f['Trick / Encore / Taunt (disrupción)'] = has({'encore', 'taunt', 'trick', 'imprison', 'disable'}) > 0
    return f


def primary(f):
    if f['Trick Room']:
        return 'Trick Room'
    for w in ('Sol', 'Lluvia', 'Arena', 'Nieve'):
        if f[w]:
            return w
    if f['Tailwind']:
        return 'Tailwind'
    if f.get('Campo Psíquico'):
        return 'Campo Psíquico'
    return 'Equilibrado / ofensivo (sin plan de campo)'


rows = defaultdict(Counter)
prim = defaultdict(Counter)
combos = Counter()
for split in ('train', 'eval'):
    for t in active_teams(split):
        f = feats(t)
        for k, v in f.items():
            if v and not k.startswith('_'):
                rows[k][split] += 1
        prim[primary(f)][split] += 1
        speed = [k for k in ('Trick Room', 'Tailwind', 'Sol', 'Lluvia', 'Arena', 'Nieve') if f[k]]
        combos[' + '.join(speed) or '—'] += 1
ours = feats(load_team_file(REPO / 'showdown/rl/data/ddqn_team.txt'))
print('PRIMARIO')
for k, c in sorted(prim.items(), key=lambda x: -sum(x[1].values())):
    print(f'{k}|{c["train"]}|{c["eval"]}|{sum(c.values())}')
print('ETIQUETAS')
for k, c in sorted(rows.items(), key=lambda x: -sum(x[1].values())):
    print(f'{k}|{c["train"]}|{c["eval"]}|{sum(c.values())}|{"sí" if ours.get(k) else ""}')
print('COMBOS', combos.most_common(10))
print('NUESTRO', {k: v for k, v in ours.items() if v})
