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


from showdown.rl.team_archetypes import feats, primary  # noqa: E402


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
