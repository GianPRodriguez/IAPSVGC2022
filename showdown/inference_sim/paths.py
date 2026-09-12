"""Rutas a repos hermanos (pokemon-showdown / poke-env / damage-calc)."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GITHUB = REPO_ROOT.parent

POKEMON_SHOWDOWN_ROOT = Path(
    os.environ.get('POKEMON_SHOWDOWN_ROOT', str(GITHUB / 'pokemon-showdown'))
)
POKE_ENV_ROOT = Path(os.environ.get('POKE_ENV_ROOT', str(GITHUB / 'poke-env')))
DAMAGE_CALC_ROOT = Path(os.environ.get('DAMAGE_CALC_ROOT', str(GITHUB / 'damage-calc')))

BATTLE_FORMAT = os.environ.get(
    'INFERENCE_SIM_FORMAT',
    'gen9championsvgc2026regmc',
)
PS_HOST = os.environ.get('PS_HOST', 'localhost')
PS_PORT = int(os.environ.get('PS_PORT', '8000'))
REPORT_PATH = REPO_ROOT / 'logs' / 'inference_sim_report.txt'
