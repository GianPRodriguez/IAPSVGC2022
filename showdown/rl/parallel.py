"""Recursos de cómputo para entrenamiento paralelo (servidor multi-core / multi-GPU).

Un solo método se entrena a la vez; el paralelismo vive *dentro* del entrenamiento:

- N entornos en procesos separados (``vec_env.SubprocVecVGCEnv``), cada uno con
  su propio cliente poke-env apuntando a un servidor Showdown (varios entornos
  pueden compartir un servidor: ``envs_per_server``).
- La red neuronal decide en lote para los N entornos (GPU si hay).

Este módulo no importa torch al cargarse: los workers de entorno no lo necesitan.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional

# Hilos de BLAS/OpenMP por proceso worker. Con 24 workers en 32 hilos, dejar el
# default (= todos los núcleos) produce sobre-suscripción y frena todo.
WORKER_THREAD_VARS = (
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
)

# Núcleos que se reservan para el proceso learner + servidores Showdown.
_RESERVED_CORES = 4
_MAX_DEFAULT_ENVS = 28


def cpu_count() -> int:
    """Núcleos lógicos utilizables por este proceso (respeta afinidad/cgroups)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def default_num_envs(cpus: Optional[int] = None) -> int:
    """Entornos paralelos por defecto: núcleos - reservados, con tope."""
    n = int(cpus if cpus is not None else cpu_count())
    return max(1, min(_MAX_DEFAULT_ENVS, n - _RESERVED_CORES))


def raise_nofile_limit(target: int = 65536) -> Optional[int]:
    """Sube el límite blando de archivos abiertos hasta ``target`` (o el duro).

    Ubuntu trae ``ulimit -n 1024``: con decenas de clientes websocket, servidores
    Showdown y la caché SQLite se agota. No requiere sudo (blando <= duro).
    Devuelve el nuevo límite blando, o None si la plataforma no lo soporta.
    """
    try:
        import resource  # POSIX
    except ImportError:
        return None
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = int(target)
        if hard != resource.RLIM_INFINITY:
            want = min(want, int(hard))
        if soft == resource.RLIM_INFINITY or soft >= want:
            return int(soft) if soft != resource.RLIM_INFINITY else None
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
        return want
    except (ValueError, OSError):
        return None


def limit_worker_threads(n: int = 1) -> None:
    """Fija los hilos BLAS/OpenMP del proceso actual (llamar antes de importar numpy/torch)."""
    for var in WORKER_THREAD_VARS:
        os.environ[var] = str(int(n))


def resolve_device(spec: Optional[str]) -> str:
    """``auto`` → ``cuda`` si hay GPU visible, si no ``cpu``. Respeta ``cuda:N``/``cpu``."""
    key = str(spec or 'cpu').strip().lower()
    if key != 'auto':
        return key
    try:
        import torch
    except ImportError:
        return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def gpu_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    try:
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


@dataclass(frozen=True)
class EnvLayout:
    """Reparto de entornos sobre servidores Showdown."""

    num_envs: int
    ports: List[int]
    env_ports: List[int]  # puerto asignado a cada entorno (len == num_envs)

    @property
    def num_servers(self) -> int:
        return len(self.ports)


def plan_env_layout(
    num_envs: int,
    *,
    envs_per_server: int = 4,
    base_port: Optional[int] = None,
) -> EnvLayout:
    """Asigna entornos a puertos consecutivos desde ``base_port``.

    Un servidor Showdown ya reparte las batallas en sus propios subprocesos de
    simulación, así que 3-4 entornos por servidor equilibran CPU y memoria.
    """
    n = max(1, int(num_envs))
    per = max(1, int(envs_per_server))
    if base_port is None:
        from showdown.inference_sim.paths import PS_PORT
        base_port = int(PS_PORT)
    n_servers = int(math.ceil(n / per))
    ports = [int(base_port) + i for i in range(n_servers)]
    env_ports = [ports[i // per] for i in range(n)]
    return EnvLayout(num_envs=n, ports=ports, env_ports=env_ports)


def describe_resources() -> str:
    """Línea legible para logs: núcleos, GPUs, límite de archivos."""
    parts = [f'cpus={cpu_count()}', f'gpus={gpu_count()}']
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        parts.append(f'nofile={soft}/{hard}')
    except ImportError:
        pass
    return ' '.join(parts)
