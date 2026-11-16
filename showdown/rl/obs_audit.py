"""Auditoría de la observación de combate con batallas reales (checklist DDQN, punto 7).

Juega episodios contra Showdown con acciones legales al azar (mismo camino que el
entrenamiento, equipo fijo del agente) y resume los 6332 valores de la observación por
sección (``obs_layout``):

- cuántos valores **nunca cambian** (constantes: no aportan información a la red);
- cuántos salen del rango esperado (|x| > 1,5 o > 5): escalas que dominarían el primer
  lineal y justificarían LayerNorm;
- fracción de valores distintos de cero (densidad) y media / desviación por sección.

    python -m showdown.rl.obs_audit --episodes 25 --port 8360 --out logs/obs_audit.json
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np

FOES = ('random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian')


def collect(*, episodes: int, port: int, seed: int, agent_team: str, max_steps: int = 60) -> np.ndarray:
    from showdown.rl.action_space import decode_joint_action
    from showdown.rl.ddqn.train import _joint_masks
    from showdown.rl.env import VGCEnv
    from showdown.rl.observation import build_observation
    from showdown.rl.pokeenv_backend import PokeEnvBattleBackend
    from showdown.rl.reward import RewardCandidate

    our_team = None
    if agent_team:
        from showdown.inference_sim.team_factory import load_team_file
        our_team = load_team_file(agent_team)
    rng = np.random.default_rng(seed)
    rows: List[np.ndarray] = []
    for foe in FOES:
        backend = PokeEnvBattleBackend(opponent=foe, start_ps=True, port=port,
                                       account_prefix=f'Aud{foe[:4].title()}',
                                       combat_family='ddqn', our_team=our_team)
        env = VGCEnv(reward_candidate=RewardCandidate.DENSE, backend=backend, max_episode_turns=40)
        try:
            for ep in range(episodes):
                env.reset(seed=int(seed) + 1000 * ep)
                for _ in range(max_steps):
                    rows.append(build_observation(env._snap).astype(np.float32))
                    legal = np.flatnonzero(np.asarray(_joint_masks(env), dtype=bool))
                    joint = int(rng.choice(legal)) if legal.size else 0
                    _o, _r, term, trunc, _i = env.step(decode_joint_action(joint))
                    if term or trunc:
                        break
        finally:
            env.close()
    return np.stack(rows)


def summarize(x: np.ndarray) -> Dict[str, object]:
    from showdown.rl.observation import obs_layout
    lay = obs_layout()
    const = np.isclose(x.max(axis=0), x.min(axis=0))
    absmax = np.abs(x).max(axis=0)
    sections = {}
    for name, (a, b) in lay.items():
        seg = x[:, a:b]
        sections[name] = {
            'dims': int(b - a),
            'constantes': int(const[a:b].sum()),
            'fuera_1_5': int((absmax[a:b] > 1.5).sum()),
            'fuera_5': int((absmax[a:b] > 5.0).sum()),
            'max_abs': float(absmax[a:b].max()) if b > a else 0.0,
            'densidad': float((seg != 0).mean()) if seg.size else 0.0,
            'media': float(seg.mean()) if seg.size else 0.0,
            'desv': float(seg.std()) if seg.size else 0.0,
        }
    top = np.argsort(-absmax)[:15]
    return {
        'muestras': int(x.shape[0]),
        'dims': int(x.shape[1]),
        'constantes_total': int(const.sum()),
        'fuera_1_5_total': int((absmax > 1.5).sum()),
        'fuera_5_total': int((absmax > 5.0).sum()),
        'max_abs_global': float(absmax.max()),
        'top_max_abs': [{'idx': int(i), 'max_abs': float(absmax[i]),
                         'seccion': next((n for n, (a, b) in lay.items() if a <= i < b), '?')}
                        for i in top],
        'secciones': sections,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--episodes', type=int, default=25, help='episodios por rival')
    ap.add_argument('--port', type=int, default=8360)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--agent-team', default='showdown/rl/data/ddqn_team.txt')
    ap.add_argument('--out', default='logs/obs_audit.json')
    args = ap.parse_args(argv)
    x = collect(episodes=args.episodes, port=args.port, seed=args.seed, agent_team=args.agent_team)
    rep = summarize(x)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: rep[k] for k in rep if k != 'secciones'}, indent=1))
    for name, s in rep['secciones'].items():
        print(f"{name:14s} dims={s['dims']:5d} const={s['constantes']:5d} >1.5={s['fuera_1_5']:4d} "
              f">5={s['fuera_5']:4d} max={s['max_abs']:8.2f} dens={s['densidad']:.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
