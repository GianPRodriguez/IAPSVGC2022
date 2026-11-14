"""Métricas de estrategia del Team Preview por update (tesis).

Para cada lote de formaciones elegidas registra:

- ``rollout_wr``: WR de las acciones muestreadas (on-policy) en ese update.
- ``lead_pos``: frecuencia de cada posición (1–6) como lead → sesgo posicional
  (p. ej. imitar el orden de exportación ``/team 1234``).
- ``same_as_fixed``: fracción de formaciones idénticas a ``/team 1234``.
- Por rol interpretable (Fake Out, Tailwind, Trick Room, redirección,
  Intimidate, megas…):
  - ``lead_rate``: fracción de formaciones que lo llevan de lead, entre los
    contextos donde el equipo lo tiene disponible;
  - ``wr_with`` / ``wr_without``: WR cuando se lleva de lead vs cuando,
    estando disponible, no se lleva. Muestra qué ventajas va descubriendo el
    agente (y cuáles no le sirven con este piloto).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from showdown.rl.preview_eval import wr_from_rewards
from showdown.rl.team_preview_action import decode_preview_action
from showdown.rl.team_preview_obs import ROLE_NAMES, role_flags


def formation_stats(bundles: Sequence, actions: Sequence[int], rewards: Sequence[float]) -> Dict:
    n = len(actions)
    if n == 0:
        return {}
    w = wr_from_rewards(np.asarray(rewards, dtype=np.float64))
    lead_pos = np.zeros(6, dtype=np.float64)
    roles: Dict[str, Dict[str, list]] = {r: {'with': [], 'without': []} for r in ROLE_NAMES}
    for k, (bundle, a) in enumerate(zip(bundles, actions)):
        order = decode_preview_action(int(a))
        leads = order[:2]
        brought = order[:4]
        for p in leads:
            lead_pos[int(p)] += 1.0
        flags = [role_flags(m) for m in list(bundle.our)[:6]]
        for r in ROLE_NAMES:
            # Mega: se mira si se TRAE (cualquiera de los 4); el resto, si se lidera.
            chosen = brought if r == 'mega' else leads
            avail = any(f[r] for f in flags)
            if not avail:
                continue
            used = any(flags[i][r] for i in chosen if i < len(flags))
            roles[r]['with' if used else 'without'].append(w[k])
    out: Dict = {
        'n': n,
        'rollout_wr': float(w.mean()),
        'lead_pos': (lead_pos / max(1.0, 2.0 * n)).tolist(),
        'same_as_fixed': float(np.mean([int(a) == 0 for a in actions])),
        'roles': {},
    }
    for r, d in roles.items():
        n_with, n_without = len(d['with']), len(d['without'])
        if n_with + n_without == 0:
            continue
        out['roles'][r] = {
            'available': n_with + n_without,
            'lead_rate': n_with / (n_with + n_without),
            'wr_with': float(np.mean(d['with'])) if n_with else None,
            'wr_without': float(np.mean(d['without'])) if n_without else None,
        }
    return out
