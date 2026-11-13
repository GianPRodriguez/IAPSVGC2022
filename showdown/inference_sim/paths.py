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


def _version_key(p: Path):
    parts = []
    for tok in p.name.lstrip('v').split('.'):
        parts.append(int(tok) if tok.isdigit() else 0)
    return parts


def node_executable() -> str:
    """Ruta del ejecutable de Node.js.

    Orden: ``NODE_BINARY`` > ``node`` en PATH > la versión más reciente de nvm
    (``~/.nvm/versions/node/*/bin/node``). nvm solo entra al PATH en shells
    interactivos (``~/.bashrc``), así que procesos lanzados por SSH, cron o la
    cola de entrenamiento no lo verían sin este respaldo.
    """
    import shutil

    explicit = os.environ.get('NODE_BINARY', '').strip()
    if explicit:
        _prepend_to_path(Path(explicit).parent)
        return explicit
    found = shutil.which('node')
    if found:
        return found
    nvm_dir = Path(os.environ.get('NVM_DIR') or (Path.home() / '.nvm'))
    candidates = sorted(
        (d for d in (nvm_dir / 'versions' / 'node').glob('v*') if (d / 'bin' / 'node').is_file()),
        key=_version_key,
    )
    if candidates:
        _prepend_to_path(candidates[-1] / 'bin')
        return str(candidates[-1] / 'bin' / 'node')
    return 'node'


def _prepend_to_path(bin_dir: Path) -> None:
    """Pone ``bin_dir`` al inicio del PATH de este proceso (lo heredan los hijos).

    El lanzador ``pokemon-showdown`` ejecuta ``node build`` por su cuenta: no
    basta con invocarlo por ruta absoluta, ``node`` debe estar en el PATH.
    """
    d = str(bin_dir)
    if not d or d == '.':
        return
    parts = os.environ.get('PATH', '').split(os.pathsep)
    if d not in parts:
        os.environ['PATH'] = os.pathsep.join([d] + [p for p in parts if p])

BATTLE_FORMAT = os.environ.get(
    'INFERENCE_SIM_FORMAT',
    'gen9championsvgc2026regmc',
)
PS_HOST = os.environ.get('PS_HOST', 'localhost')
PS_PORT = int(os.environ.get('PS_PORT', '8000'))
# Paralelismo: PS_PORTS=8000,8001 (si vacío → PS_PORT..+workers-1)
PS_AUTH_URL = os.environ.get(
    'PS_AUTH_URL',
    'https://play.pokemonshowdown.com/action.php?',
)
REPORT_PATH = REPO_ROOT / 'logs' / 'inference_sim_report.txt'


def server_configuration_for_port(port: int):
    """poke-env ServerConfiguration apuntando a un puerto local concreto."""
    from poke_env.ps_client import ServerConfiguration

    host = PS_HOST
    return ServerConfiguration(
        f'ws://{host}:{int(port)}/showdown/websocket',
        PS_AUTH_URL,
    )
