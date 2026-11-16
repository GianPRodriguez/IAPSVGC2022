"""Equipos rivales sintéticos por recombinación (16-11-2026, respuesta a la memorización).

**Por qué.** DDQN gana 84 % con equipos rivales de entrenamiento y 50 % con equipos no vistos
(cuarta cola). Los equipos ``eval`` solo cambian la combinación de especies (se separaron por
grupos de clones), así que el agente aprende combinaciones concretas, no a jugar contra
especies y sets.

**Diseño.** El metajuego real de torneo tiene poca variación: los 300 equipos lo reflejan bien.
Por eso los sintéticos no son al azar, sino **variantes de equipos reales**:

1. Se toma un equipo real del grupo de entrenamiento (``fit``).
2. Se reemplazan k de sus 6 Pokémon (k = 1, 2 o 3; por defecto con pesos 0,5 / 0,35 / 0,15)
   por Pokémon de otros equipos de entrenamiento, **cada uno con su set real completo**
   (objeto, habilidad, naturaleza, movimientos y SP).
3. Los donantes se sortean entre todas las apariciones de Pokémon del grupo, así que se
   respeta la frecuencia de uso del metajuego.
4. Reglas: sin especies repetidas (por número de Pokédex: Arcanine y Arcanine-Hisui cuentan
   como la misma) ni objetos repetidos. Después, el validador de Showdown de Reg M-C revisa
   cada equipo y se descartan los inválidos.
5. Se descartan combinaciones de especies que ya existen en ``fit``, ``val`` o ``eval``: un
   sintético nunca reproduce un equipo de evaluación ni de validación.

El núcleo del equipo (al menos 3 de 6) queda intacto, así que el equipo sigue siendo
coherente. **Límite:** no crea especies ni sets que no estén en el pool, así que no cubre
equipos realmente fuera del metajuego; solo combinaciones nuevas de piezas conocidas.

    python -m showdown.rl.synth_teams --n 2000 --out showdown/rl/data/synth_teams.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

DEFAULT_OUT = Path(__file__).resolve().parent / 'data' / 'synth_teams.json'
DEFAULT_SEED = 20261116
DEFAULT_WEIGHTS = (0.5, 0.35, 0.15)  # k = 1, 2, 3 reemplazos


def _sid(species: str) -> str:
    return ''.join(ch for ch in str(species or '').lower() if ch.isalnum())


_DEX_NUM: Dict[str, int] = {}


def dex_num(species: str) -> int:
    """Número de Pokédex (la cláusula de especie de VGC compara por número)."""
    sid = _sid(species)
    if sid not in _DEX_NUM:
        try:
            from poke_env.data import GenData
            entry = GenData.from_gen(9).pokedex.get(sid) or {}
            _DEX_NUM[sid] = int(entry.get('num') or 0) or -abs(hash(sid))
        except Exception:
            _DEX_NUM[sid] = -abs(hash(sid))
    return _DEX_NUM[sid]


def _item(mon) -> str:
    return _sid(getattr(mon, 'item', '') or '')


def species_set(team) -> Set[str]:
    return {_sid(m.species) for m in team.mons}


def max_overlap(team, refs: Iterable) -> int:
    """Máximo de especies en común con algún equipo de ``refs`` (6 = misma combinación)."""
    s = species_set(team)
    return max((len(s & species_set(r)) for r in refs), default=0)


def _legal_swap(mons: Sequence, idx: int, donor) -> bool:
    others = [m for i, m in enumerate(mons) if i != idx]
    if dex_num(donor.species) in {dex_num(m.species) for m in others}:
        return False
    it = _item(donor)
    return not it or it not in {_item(m) for m in others}


def recombine(base, donors: Sequence, k: int, rng: random.Random, *, tries: int = 60):
    """Variante de ``base`` con ``k`` Pokémon reemplazados por donantes (sets completos)."""
    from copy import deepcopy

    from showdown.inference_sim.team_factory import TeamMeta
    mons = [deepcopy(m) for m in base.mons]
    slots = rng.sample(range(len(mons)), k)
    swapped = []
    for idx in slots:
        for _ in range(tries):
            d = donors[rng.randrange(len(donors))]
            if dex_num(d.species) == dex_num(mons[idx].species):
                continue
            if _legal_swap(mons, idx, d):
                swapped.append({'sale': mons[idx].species, 'entra': d.species})
                mons[idx] = deepcopy(d)
                break
        else:
            return None, []
    return TeamMeta(name=f'synth:{base.name}', source_url='', mons=mons), swapped


def generate(fit: List, forbidden_clones: Set[str], *, n: int, seed: int = DEFAULT_SEED,
             weights: Sequence[float] = DEFAULT_WEIGHTS, max_attempts: Optional[int] = None):
    """Hasta ``n`` sintéticos únicos (sin validar todavía). Devuelve (equipos, metadatos)."""
    from showdown.inference_sim.team_factory import to_showdown_export
    from showdown.inference_sim.team_split import clone_key, team_key
    rng = random.Random(int(seed))
    donors = [m for t in fit for m in t.mons]
    seen_exports: Set[str] = set()
    seen_clones = set(forbidden_clones)
    out, meta = [], []
    attempts = 0
    max_attempts = max_attempts or 50 * n
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        base = fit[rng.randrange(len(fit))]
        k = rng.choices((1, 2, 3), weights=weights)[0]
        team, swapped = recombine(base, donors, k, rng)
        if team is None:
            continue
        ck = clone_key(team)
        if ck in seen_clones:
            continue
        exp = to_showdown_export(team)
        if exp in seen_exports:
            continue
        seen_exports.add(exp)
        seen_clones.add(ck)
        i = len(out)
        team.name = f'synth{i:05d}'
        team.source_url = f'synth:{seed}:{i}'
        out.append(team)
        meta.append({'base': team_key(base), 'k': k, 'cambios': swapped})
    return out, meta, attempts


def usage(teams: Iterable) -> Counter:
    c = Counter()
    n = 0
    for t in teams:
        n += 1
        c.update({m.species for m in t.mons})
    return Counter({k: v / max(1, n) for k, v in c.items()})


def report(fit: List, val: List, ev: List, synth: List, meta: List[dict], rejected: int,
           attempts: int, top: int = 15) -> str:
    """Análisis de los sintéticos frente al metajuego real (para la tesis)."""
    L = ['# Equipos sintéticos: análisis', '',
         f'- Generados válidos: {len(synth)}; descartados por el validador de Showdown: {rejected}; '
         f'intentos totales: {attempts}.',
         f"- Reemplazos por equipo: " + ', '.join(f'k={k}: {c}' for k, c in sorted(Counter(m['k'] for m in meta).items())),
         '']
    uf, us = usage(fit), usage(synth)
    L += ['## Uso de especies (fracción de equipos)', '', '| Especie | Reales (`fit`) | Sintéticos | Diferencia |',
          '|---|---|---|---|']
    for sp, _ in (uf + us).most_common(top):
        L.append(f'| {sp} | {100 * uf.get(sp, 0):.1f} % | {100 * us.get(sp, 0):.1f} % | '
                 f'{100 * (us.get(sp, 0) - uf.get(sp, 0)):+.1f} |')
    nf, ns = len(uf), len(us)
    L += ['', f'Especies distintas: {nf} en `fit`, {ns} en los sintéticos (no puede haber nuevas).', '']

    def hist(teams, refs):
        h = Counter(max_overlap(t, refs) for t in teams)
        return ' | '.join(f'{100 * h.get(i, 0) / max(1, len(teams)):.0f} %' for i in range(7))

    L += ['## Parecido con el equipo real más cercano (especies en común, de 0 a 6)', '',
          '| Grupo | Referencia | 0 | 1 | 2 | 3 | 4 | 5 | 6 |', '|---|---|---|---|---|---|---|---|---|']
    L.append(f'| sintéticos | `fit` | {hist(synth, fit)} |')
    L.append(f'| `eval` (real) | `fit` | {hist(ev, fit)} |')
    if val:
        L.append(f'| `val` (real) | `fit` | {hist(val, fit)} |')
    L.append(f'| sintéticos | `eval` | {hist(synth, ev)} |')
    L += ['', 'Lectura: si los sintéticos se parecen a `fit` tanto como `eval` se parece a `fit`, cubren el '
          'tipo de novedad que el agente encuentra en la evaluación (equipos que comparten casi todas las '
          'piezas del metajuego pero en otra combinación).', '']
    return '\n'.join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--weights', default=','.join(str(w) for w in DEFAULT_WEIGHTS),
                    help='pesos de k=1,2,3 reemplazos')
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--report-out', default='docs/equipos_sinteticos_analisis.md')
    ap.add_argument('--no-validate', action='store_true', help='sin validador de Showdown (solo pruebas)')
    args = ap.parse_args(argv)

    from showdown.inference_sim.team_factory import active_teams
    from showdown.inference_sim.team_split import clone_key, team_key
    fit, val, ev = active_teams('fit'), active_teams('val'), active_teams('eval')
    forbidden = {clone_key(t) for t in fit + val + ev}
    weights = tuple(float(x) for x in args.weights.split(','))
    teams, meta, attempts = generate(fit, forbidden, n=args.n, seed=args.seed, weights=weights)
    rejected = 0
    if not args.no_validate:
        from showdown.inference_sim.validate_regmc_teams import validate_team_metas
        ok, bad = validate_team_metas(teams)
        ok_keys = {team_key(t) for t in ok}
        keep = [i for i, t in enumerate(teams) if team_key(t) in ok_keys]
        rejected = len(teams) - len(keep)
        for t, problems in bad[:10]:
            print(f'[synth] inválido {t.name}: {problems[:2]}')
        teams, meta = [teams[i] for i in keep], [meta[i] for i in keep]
    rows = [{'name': t.name, 'source_url': t.source_url, **m, 'mons': [asdict(x) for x in t.mons]}
            for t, m in zip(teams, meta)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'seed': args.seed, 'weights': list(weights), 'base_split': 'fit',
                               'validated': not args.no_validate, 'n': len(rows), 'teams': rows},
                              ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    rep = report(fit, val, ev, teams, meta, rejected, attempts)
    if args.report_out:
        Path(args.report_out).write_text(rep, encoding='utf-8')
    print(rep)
    print(f'OK {out}: {len(rows)} equipos')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
