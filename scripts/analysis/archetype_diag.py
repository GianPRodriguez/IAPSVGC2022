"""Diagnósticos D1 y D2 de PPO v2 / DDQN v1 por arquetipo del rival (protocolo §4.16, antes de entrenar).

Motivo (docs/online_web_partidas.md): contra personas, el control de velocidad del rival (Trick Room en lluvia,
Tailwind en sol) decidió las dos series, Staraptor nunca usó Tailwind y el preview trajo siempre a Arcanine-Hisui.
La observación sí tiene el orden de velocidad efectivo y esos arquetipos están entre los equipos de
entrenamiento. Hipótesis a comprobar: los rivales heurísticos casi nunca *ejecutan* esos planes.

Lee JSON de ``policy_diag`` con ``--context`` (``partidas`` con ``equipo``, ``res``, ``formacion`` y ``eventos``;
en la cola v3 son los ``p1_<fam>_s<k>_off.json``) y escribe un informe con:

- **D1a** tasa de victorias por arquetipo del rival (con / sin la etiqueta);
- **D1b** ejecución de los planes: en partidas contra equipos con Trick Room / Tailwind, qué fracción los activó,
  y la tasa de victorias cuando lo activó frente a cuando no;
- **D1c** control de velocidad propio: cuántas veces trajo a Staraptor y, de esas, cuántas usó Tailwind;
- **D2** preview real por arquetipo: frecuencia de traer cada Pokémon propio, abrir con Staraptor contra
  Defiant/Competitive y victorias con / sin Arcanine-Hisui contra equipos con ≥ 2 atacantes de Tierra/Agua.

    python scripts/analysis/archetype_diag.py --out <OUT>/d1_report.md <OUT>/diag/p1_ppo_s*_off.json ...
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import archetypes as A  # noqa: E402

HEUR = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')
RIVALS = HEUR   # --rivales cambia qué rivales cuentan (p. ej. greedy_joint_arq3, protocolo §4.17)


def _pct(w: int, n: int) -> str:
    return f'{100 * w / n:.1f} % ({n})' if n else '–'


def load_games(paths: Sequence[str]) -> Dict[str, List[dict]]:
    """familia ('ppo' / 'ddqn', del nombre del archivo) -> partidas contra heurísticos con etiquetas."""
    out: Dict[str, List[dict]] = defaultdict(list)
    for pat in paths:
        for f in sorted(glob.glob(pat)):
            name = Path(f).name.lower()
            fam = 'ddqn' if 'ddqn' in name else ('ppo' if 'ppo' in name else 'otro')
            try:
                d = json.loads(Path(f).read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            for g in d.get('partidas', []):
                if g.get('rival') not in RIVALS:
                    continue
                tags = A.tags_for_key(g.get('equipo'))
                if tags is None:
                    continue
                out[fam].append({**g, '_tags': tags, '_win': g.get('res') == 'win'})
    return out


def report(games: Dict[str, List[dict]], names: Sequence[str]) -> List[str]:
    staraptor = next((i + 1 for i, n in enumerate(names) if n.lower().startswith('staraptor')), None)
    arcanine = next((i + 1 for i, n in enumerate(names) if n.lower().startswith('arcanine')), None)
    lines: List[str] = []
    for fam, gs in games.items():
        n_ctx = sum(1 for g in gs if isinstance(g.get('eventos'), dict))
        lines += [f'## {fam.upper()} ({len(gs)} partidas contra heurísticos; {n_ctx} con eventos)', '',
                  '### D1a. Victorias por arquetipo del rival', '',
                  '| Arquetipo | Con la etiqueta | Sin la etiqueta | Diferencia |', '|---|---|---|---|']
        for t in A.TAGS:
            a = [g['_win'] for g in gs if t in g['_tags']]
            b = [g['_win'] for g in gs if t not in g['_tags']]
            diff = (100 * (sum(a) / len(a) - sum(b) / len(b))) if a and b else float('nan')
            lines.append(f'| {t} | {_pct(sum(a), len(a))} | {_pct(sum(b), len(b))} | {diff:+.1f} pts |')

        lines += ['', '### D1b. ¿El rival ejecuta su control de velocidad?', '',
                  '| Plan | Partidas con el plan en el equipo | Lo activó | Victorias si lo activó | Victorias si no |',
                  '|---|---|---|---|---|']
        for tag, key in (('trick_room', 'trick_room_activado'), ('tailwind', 'tailwind_activado'),
                         ('lluvia', 'lluvia_activado'), ('sol', 'sol_activado'), ('arena', 'arena_activado'),
                         ('nieve', 'nieve_activado')):
            sub = [g for g in gs if tag in g['_tags'] and isinstance(g.get('eventos'), dict)]
            act = [g for g in sub if int((g['eventos'].get('rival') or {}).get(key, 0)) > 0]
            no = [g for g in sub if g not in act]
            lines.append(f'| {tag} | {len(sub)} | {_pct(len(act), len(sub))} | '
                         f'{_pct(sum(g["_win"] for g in act), len(act))} | {_pct(sum(g["_win"] for g in no), len(no))} |')

        lines += ['', '### D1c. Control de velocidad propio (Staraptor lleva Tailwind)', '']
        if staraptor:
            brought = [g for g in gs if str(staraptor) in str(g.get('formacion', ''))[:4]]
            ctx = [g for g in brought if isinstance(g.get('eventos'), dict)]
            used = [g for g in ctx if int((g['eventos'].get('nuestro') or {}).get('tailwind', 0)) > 0]
            lines.append(f'Trajo a Staraptor en {_pct(len(brought), len(gs))} partidas; usó Tailwind en '
                         f'{_pct(len(used), len(ctx))} de ellas. Victorias cuando lo usó: '
                         f'{_pct(sum(g["_win"] for g in used), len(used))}.')

        lines += ['', '### D2. Preview real por arquetipo (formación jugada)', '',
                  '| Arquetipo | Partidas | ' + ' | '.join(names) + ' |', '|---|---|' + '---|' * len(names)]
        for t in ('(todos)',) + A.TAGS:
            sub = gs if t == '(todos)' else [g for g in gs if t in g['_tags']]
            if not sub:
                continue
            cells = []
            for i in range(1, len(names) + 1):
                k = sum(1 for g in sub if str(i) in str(g.get('formacion', ''))[:4])
                cells.append(f'{100 * k / len(sub):.0f} %')
            lines.append(f'| {t} | {len(sub)} | ' + ' | '.join(cells) + ' |')
        if staraptor:
            for lab, cond in (('contra Defiant/Competitive', True), ('contra el resto', False)):
                sub = [g for g in gs if ('anti_intimidate' in g['_tags']) == cond]
                lead = [g for g in sub if str(staraptor) in str(g.get('formacion', ''))[:2]]
                lines.append(f'\nAbre con Staraptor (Intimidate) {lab}: {_pct(len(lead), len(sub))}; victorias '
                             f'abriendo con él {_pct(sum(g["_win"] for g in lead), len(lead))}.')
        if arcanine:
            sub = [g for g in gs if 'tierra_agua_fuerte' in g['_tags']]
            yes = [g for g in sub if str(arcanine) in str(g.get('formacion', ''))[:4]]
            no = [g for g in sub if g not in yes]
            lines.append(f'\nContra ≥ 2 atacantes de Tierra/Agua: con Arcanine-Hisui '
                         f'{_pct(sum(g["_win"] for g in yes), len(yes))}, sin él {_pct(sum(g["_win"] for g in no), len(no))}.')
        lines.append('')
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('paths', nargs='+', help='JSON de policy_diag (admite comodines)')
    ap.add_argument('--agent-team', default=str(REPO / 'showdown/rl/data/ddqn_team.txt'))
    ap.add_argument('--out', default='')
    ap.add_argument('--rivales', default=','.join(HEUR), help='rivales que cuentan (p. ej. greedy_joint_arq3)')
    ap.add_argument('--titulo', default='D1 / D2: PPO v2 y DDQN v1 por arquetipo del rival (protocolo §4.16)')
    args = ap.parse_args(argv)
    global RIVALS
    RIVALS = tuple(x for x in args.rivales.split(',') if x)
    from showdown.inference_sim.team_factory import load_team_file
    names = [str(m.species) for m in load_team_file(args.agent_team).mons]
    games = load_games(args.paths)
    lines = [f'# {args.titulo}', '',
             f'Equipos `eval`, rivales deterministas: {", ".join(RIVALS)}. Formación = índices del equipo propio '
             '(los dos primeros abren).', ''] + report(games, names)
    text = '\n'.join(lines) + '\n'
    if args.out:
        Path(args.out).write_text(text, encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
