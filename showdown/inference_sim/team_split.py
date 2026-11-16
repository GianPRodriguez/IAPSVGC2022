"""Separación de equipos: 200 para entrenar y 100 para evaluar (checklist DDQN, punto 11).

Si los agentes RL entrenan con los mismos equipos con que se evalúan, la evaluación mide en
parte memorización (cada rival se ve ~800 veces en 2M pasos) y la red aprende de hecho los
SP reales de esos equipos, información que los heurísticos no tienen (prior leave-one-out).
Por eso el pool se divide en dos grupos fijos, guardados en ``team_split.json``:

- ``train`` (200): entrenamiento de todos los agentes RL y previews, y base del "meta
  conocido" (prior de SP y mundos de expectimax) cuando se usa la separación.
- ``eval`` (100): solo evaluación final (round-robin de la Conclusión); ningún agente los ve
  antes.

``PS_TEAM_SPLIT=all|train|eval`` elige el grupo (por defecto ``all``: comportamiento previo,
con el que se midieron los baselines). Los entornos RL usan ``train`` por defecto.

    python -m showdown.inference_sim.team_split --make     # genera team_split.json (una vez)
    python -m showdown.inference_sim.team_split --report   # balance entre grupos
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

SPLIT_PATH = Path(__file__).resolve().parent / 'team_split.json'
SPLIT_SEED = 20261115
N_TRAIN = 200
SPLITS = ('all', 'train', 'eval')


def team_key(team) -> str:
    """Clave estable de un equipo (TeamMeta o fila del JSON): URL de origen o nombre."""
    if isinstance(team, dict):
        return str(team.get('source_url') or team.get('slug') or team.get('name') or '')
    return str(getattr(team, 'source_url', '') or getattr(team, 'name', '') or '')


def current_split(default: str = 'all') -> str:
    s = os.environ.get('PS_TEAM_SPLIT', '').strip().lower() or default
    if s not in SPLITS:
        raise ValueError(f'PS_TEAM_SPLIT inválido: {s!r} (opciones: {", ".join(SPLITS)})')
    return s


def prior_split(default: str = 'all') -> str:
    """Grupo del que sale el prior de SP / mundos: con separación, siempre ``train``."""
    return 'all' if current_split(default) == 'all' else 'train'


@lru_cache(maxsize=1)
def _load() -> Dict[str, List[str]]:
    if not SPLIT_PATH.is_file():
        return {}
    return json.loads(SPLIT_PATH.read_text(encoding='utf-8'))


def split_keys(which: str) -> Optional[Set[str]]:
    """Claves del grupo; ``None`` = todos (``all`` o sin archivo de separación)."""
    if which == 'all':
        return None
    data = _load()
    if not data:
        raise RuntimeError(f'Falta {SPLIT_PATH.name}: genera la separación con --make')
    return set(data[which])


def filter_teams(teams: Iterable, which: Optional[str] = None) -> List:
    which = current_split() if which is None else which
    keys = split_keys(which)
    teams = list(teams)
    return teams if keys is None else [t for t in teams if team_key(t) in keys]


def make_split(teams: List, *, seed: int = SPLIT_SEED, n_train: int = N_TRAIN) -> Dict[str, List[str]]:
    keys = sorted({team_key(t) for t in teams})
    rng = random.Random(int(seed))
    rng.shuffle(keys)
    return {'seed': seed, 'train': sorted(keys[:n_train]), 'eval': sorted(keys[n_train:])}


def balance_report(teams: List, split: Dict[str, List[str]], top: int = 15) -> str:
    """Frecuencia de las especies más comunes en cada grupo (para revisar el balance)."""
    by = {'train': set(split['train']), 'eval': set(split['eval'])}
    cnt = {k: Counter() for k in by}
    n = {k: 0 for k in by}
    for t in teams:
        k = team_key(t)
        for g, ks in by.items():
            if k in ks:
                n[g] += 1
                cnt[g].update({m.species for m in t.mons})
    allc = cnt['train'] + cnt['eval']
    lines = [f"equipos: train={n['train']} eval={n['eval']}",
             f"{'especie':20s} {'train %':>8s} {'eval %':>8s}"]
    for sp, _c in allc.most_common(top):
        lines.append(f"{sp:20s} {100 * cnt['train'][sp] / max(1, n['train']):8.1f} "
                     f"{100 * cnt['eval'][sp] / max(1, n['eval']):8.1f}")
    return '\n'.join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--make', action='store_true')
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args(argv)
    from showdown.inference_sim.team_factory import active_teams
    os.environ['PS_TEAM_SPLIT'] = 'all'
    teams = active_teams()
    if args.make:
        split = make_split(teams)
        SPLIT_PATH.write_text(json.dumps(split, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        _load.cache_clear()
        print(f'OK {SPLIT_PATH.name}: train={len(split["train"])} eval={len(split["eval"])}')
    data = _load()
    if args.report or args.make:
        print(balance_report(teams, data))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
