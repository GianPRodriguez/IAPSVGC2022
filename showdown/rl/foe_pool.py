"""Pool de equipos rivales del entrenamiento RL (16-11-2026, respuesta a la memorización).

La cuarta cola mostró que DDQN memoriza los equipos rivales de entrenamiento: 84 % de
victorias con equipos ``train`` frente a 50 % con equipos ``eval`` no vistos. Tres opciones,
todas apagadas por defecto (sin variables, el muestreo es idéntico al anterior):

- ``RL_HOLDOUT_VAL=1``: el entrenamiento usa ``fit`` (``train`` sin el subgrupo ``val``) y la
  validación periódica usa ``val``. Así ``best.pt`` se elige con equipos no vistos.
- ``RL_TRAIN_SUBSET=N``: solo los primeros N equipos del grupo de entrenamiento, en un orden
  fijo (subconjuntos anidados: los 50 están dentro de los 100). Sirve para la curva de escala:
  ¿la brecha train/eval baja al tener más equipos?
- ``RL_FOE_SYNTH=p``: con probabilidad p el rival juega un equipo sintético (ver
  ``showdown/rl/synth_teams.py``). Archivo: ``RL_FOE_SYNTH_FILE`` (por defecto
  ``showdown/rl/data/synth_teams.json``).

Ninguna opción toca la evaluación final (Conclusión, grupo ``eval``): solo el entrenamiento.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

SYNTH_DEFAULT = Path(__file__).resolve().parent / 'data' / 'synth_teams.json'
TRAINING_SPLITS = ('train', 'fit')


def _flag(name: str) -> bool:
    return os.environ.get(name, '').strip().lower() in ('1', 'true', 'yes', 'on')


def holdout_val() -> bool:
    return _flag('RL_HOLDOUT_VAL')


def train_subset() -> int:
    try:
        return max(0, int(os.environ.get('RL_TRAIN_SUBSET', '0') or 0))
    except ValueError:
        raise ValueError(f"RL_TRAIN_SUBSET inválido: {os.environ.get('RL_TRAIN_SUBSET')!r}")


def synth_prob() -> float:
    try:
        p = float(os.environ.get('RL_FOE_SYNTH', '0') or 0)
    except ValueError:
        raise ValueError(f"RL_FOE_SYNTH inválido: {os.environ.get('RL_FOE_SYNTH')!r}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f'RL_FOE_SYNTH debe estar en [0, 1]: {p}')
    return p


def training_split() -> str:
    """Grupo de los rivales de entrenamiento: ``fit`` con validación reservada; si no, el de siempre."""
    from showdown.rl.pokeenv_backend import rl_team_split
    split = rl_team_split()
    return 'fit' if holdout_val() and split == 'train' else split


def validation_split() -> Optional[str]:
    """Grupo de la validación periódica: ``val`` con validación reservada; ``None`` = el de entrenamiento."""
    return 'val' if holdout_val() else None


def _order_key(team) -> str:
    from showdown.inference_sim.team_split import team_key
    return hashlib.sha256(('subset|' + team_key(team)).encode('utf-8')).hexdigest()


def subset(teams: List, n: int) -> List:
    """Primeros ``n`` equipos en un orden fijo por hash (anidados; no depende del orden del JSON)."""
    if n <= 0 or n >= len(teams):
        return list(teams)
    return sorted(teams, key=_order_key)[:n]


def base_pool(split: Optional[str] = None) -> List:
    """Equipos reales del grupo; ``RL_TRAIN_SUBSET`` solo recorta grupos de entrenamiento."""
    from showdown.inference_sim.team_factory import active_teams
    split = training_split() if split is None else split
    teams = list(active_teams(split))
    if split in TRAINING_SPLITS and train_subset():
        teams = subset(teams, train_subset())
    return teams


@lru_cache(maxsize=4)
def _load_synth(path: str):
    from showdown.inference_sim.team_factory import MonSet, TeamMeta
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    out = []
    for row in data['teams']:
        mons = [MonSet(**m) for m in row['mons']]
        out.append(TeamMeta(name=row['name'], source_url=row['source_url'], mons=mons))
    return out


def synth_pool() -> List:
    path = os.environ.get('RL_FOE_SYNTH_FILE', '').strip() or str(SYNTH_DEFAULT)
    if not Path(path).is_file():
        raise FileNotFoundError(f'RL_FOE_SYNTH activo pero falta {path} (python -m showdown.rl.synth_teams)')
    return _load_synth(path)


def maybe_synthetic_foe(rng, split: str, foe):
    """Con probabilidad ``RL_FOE_SYNTH`` reemplaza al rival real por uno sintético.

    Solo en grupos de entrenamiento. Sin la variable no consume números aleatorios (el muestreo
    queda idéntico al de antes).
    """
    p = synth_prob()
    if p <= 0.0 or split not in TRAINING_SPLITS:
        return foe
    if float(rng.random()) >= p:
        return foe
    pool = synth_pool()
    return pool[int(rng.integers(0, len(pool)))]
