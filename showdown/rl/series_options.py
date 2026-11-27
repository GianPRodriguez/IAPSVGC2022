"""Opciones de PPO v3 / DDQN v2 (protocolo §4.16), comunes a los dos entrenadores por juego limpio.

- ``--series-memory 1``: bloque de memoria de serie BO3 en la obs (``cfg.series_memory``; rl_env_spec v1.35).
- ``--series-bo3 1``: los entornos agrupan las partidas en series BO3 contra el mismo rival y equipos
  (``RL_SERIES_BO3``). Por defecto sigue a ``--series-memory`` (sin series la memoria siempre estaría vacía).
- ``--mask-protect-streak 1``: la máscara quita el Protect consecutivo (``cfg.mask_protect_streak``).
- ``--preview-series 1``: el preview congelado ve en la partida 2–3 solo lo que el rival trajo antes
  (``PREVIEW_SERIES_CONDITION``; sin reentrenar el preview).
- ``--init-from <ckpt>``: arranque en caliente desde la versión anterior (PPO v2 / DDQN v1): copia los
  pesos y amplía la primera capa del torso con columnas en cero para la memoria.

Todo queda en variables de entorno que heredan los procesos de los entornos vectorizados, y en
``train_meta.json`` (``serie_bo3``). Con ``--resume`` las opciones del modelo salen del checkpoint.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


def add_series_cli_args(p) -> None:
    g = p.add_argument_group('PPO v3 / DDQN v2 (protocolo §4.16)')
    g.add_argument('--series-memory', type=int, choices=(0, 1), default=None,
                   help='memoria de serie BO3 en la obs (1 = PPO v3 / DDQN v2)')
    g.add_argument('--series-bo3', type=int, choices=(0, 1), default=None,
                   help='entornos en series BO3 (por defecto = --series-memory)')
    g.add_argument('--mask-protect-streak', type=int, choices=(0, 1), default=None,
                   help='máscara del Protect consecutivo (1 = PPO v3 / DDQN v2)')
    g.add_argument('--preview-series', type=int, choices=(0, 1), default=0,
                   help='preview condicionado a lo que el rival trajo en la serie (sin reentrenar)')
    g.add_argument('--init-from', default='',
                   help='arranque en caliente desde un checkpoint sin memoria (PPO v2 / DDQN v1)')


def series_opts_from_args(args) -> Dict[str, Any]:
    return {
        'series_memory': getattr(args, 'series_memory', None),
        'series_bo3': getattr(args, 'series_bo3', None),
        'mask_protect_streak': getattr(args, 'mask_protect_streak', None),
        'preview_series': getattr(args, 'preview_series', 0),
        'init_from': getattr(args, 'init_from', '') or '',
    }


def _rebuild(agent) -> None:
    if hasattr(agent, '_rebuild_net'):      # PPO
        agent._rebuild_net()
    else:                                   # DDQN: _ensure_torch reconstruye redes, optimizador y replay
        agent.online = None
        agent._ensure_torch()


def apply_series_options(agent, opts: Optional[Dict[str, Any]], *, resume: bool = False,
                         tag: str = 'rl') -> Dict[str, Any]:
    """Fija las opciones en el agente y en el entorno del proceso; devuelve el resumen para train_meta."""
    o = dict(opts or {})
    cfg = agent.cfg
    if not resume:
        if o.get('series_memory') is not None:
            cfg.series_memory = bool(o['series_memory'])
        if o.get('mask_protect_streak') is not None:
            cfg.mask_protect_streak = bool(o['mask_protect_streak'])
    net = getattr(agent, 'net', None) or getattr(agent, 'online', None)
    if net is not None and int(net.encoder.obs_dim) != int(agent.obs_dim):
        _rebuild(agent)
    init_info: Optional[Dict[str, Any]] = None
    if o.get('init_from') and not resume:
        init_info = agent.init_from_previous(o['init_from'])
        if hasattr(cfg, 'eps_start') and hasattr(cfg, 'eps_end'):
            # DDQN: el ε del programa empieza en 1,0 (política aleatoria) porque env_steps vuelve a 0; con una
            # política ya entrenada eso la destruiría. Se continúa con el ε final (0,05) desde el principio.
            init_info['eps_start'] = cfg.eps_start = float(cfg.eps_end)
        print(f'[{tag}] arranque en caliente desde {o["init_from"]}: {init_info}', flush=True)
    series_bo3 = bool(cfg.series_memory) if o.get('series_bo3') is None else bool(o['series_bo3'])
    preview_series = bool(o.get('preview_series') or 0)
    os.environ['RL_OBS_SERIES_MEMORY'] = '1' if cfg.series_memory else '0'
    os.environ['RL_MASK_PROTECT_STREAK'] = '1' if cfg.mask_protect_streak else '0'
    os.environ['RL_SERIES_BO3'] = '1' if series_bo3 else '0'
    os.environ['PREVIEW_SERIES_CONDITION'] = '1' if preview_series else '0'
    meta = {'series_memory': bool(cfg.series_memory), 'series_bo3': series_bo3,
            'mask_protect_streak': bool(cfg.mask_protect_streak), 'preview_series': preview_series,
            'obs_dim': int(agent.obs_dim), 'init_from': o.get('init_from') or None, 'init_info': init_info}
    print(f'[{tag}] PPO v3 / DDQN v2: memoria de serie={meta["series_memory"]}, series BO3={series_bo3}, '
          f'máscara Protect consecutivo={meta["mask_protect_streak"]}, preview condicionado={preview_series}, '
          f'obs={meta["obs_dim"]}', flush=True)
    return meta
