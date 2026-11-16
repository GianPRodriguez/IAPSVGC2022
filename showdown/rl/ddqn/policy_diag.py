"""Desglose de las jugadas de una política DDQN sin exploración (diagnóstico previo al kick-off).

Juega partidas reales (rivales deterministas, equipo fijo del agente) con ε = 0 y clasifica la
decisión de cada hueco:

- ``pass_obligatorio``: la máscara solo permitía pasar (espacio vacío o cambio forzado de otro
  hueco). ``pass_con_opciones`` debería ser 0: la máscara quita ``pass`` cuando Showdown no lo
  acepta ("Can't pass: Your Pokémon must make a move").
- ``cambio_forzado`` / ``cambio_voluntario``.
- ``ataque_rival`` (un objetivo), ``area_rivales`` (todos los rivales), ``area_con_aliado``
  (daña también al aliado, p. ej. Earthquake), ``ataque_aliado`` (movimiento de daño dirigido al
  aliado), ``apoyo_aliado`` (de estado al aliado, p. ej. Helping Hand), ``protect`` y
  ``otro_estado`` (estado propio o de campo). ``mega`` cuenta aparte.

    python -m showdown.rl.ddqn.policy_diag --checkpoint <dir>/best.pt --games 50 --port 8370 \\
        --preview-checkpoint <preview>.pt --out logs/policy_diag.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Dict

import numpy as np

FOES = ('random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian')
_PROTECT = {'protect', 'detect', 'spikyshield', 'banefulbunker', 'kingsshield', 'silktrap',
            'obstruct', 'burningbulwark', 'wideguard', 'quickguard', 'maxguard'}


def _norm(s) -> str:
    return ''.join(ch for ch in str(s or '').lower() if ch.isalnum())


def classify(battle, slot: int, act, mask_slot: np.ndarray) -> str:
    """Categoría de la decisión ``act`` (SlotAction) del hueco ``slot``."""
    from showdown.rl.constants import PASS_ID
    if act.kind == 'pass':
        others = np.flatnonzero(np.asarray(mask_slot) > 0.5)
        return 'pass_obligatorio' if set(int(x) for x in others) <= {int(PASS_ID)} else 'pass_con_opciones'
    if act.kind == 'switch':
        fs = list(getattr(battle, 'force_switch', None) or [])
        return 'cambio_forzado' if (slot < len(fs) and bool(fs[slot])) else 'cambio_voluntario'
    moves = list(battle.available_moves[slot]) if slot < len(battle.available_moves) else []
    if not moves:
        return 'sin_movimientos'
    move = moves[int(max(0, min(len(moves) - 1, act.move_index)))]
    mid = _norm(getattr(move, 'id', ''))
    if mid in _PROTECT:
        return 'protect'
    tgt = _norm(getattr(getattr(move, 'target', None), 'name', ''))
    damaging = float(getattr(move, 'base_power', 0) or 0) > 0
    if tgt in ('alladjacentfoes',):
        return 'area_rivales'
    if tgt in ('alladjacent',):
        return 'area_con_aliado' if damaging else 'otro_estado'
    if int(act.target) == 2:
        return 'ataque_aliado' if damaging else 'apoyo_aliado'
    if int(act.target) in (0, 1) and tgt in ('normal', 'adjacentfoe', 'any', 'randomnormal'):
        return 'ataque_rival' if damaging else 'estado_rival'
    return 'otro_estado'


def run(*, checkpoint: str, games: int, port: int, seed: int, agent_team: str,
        preview_checkpoint: str = '', agent_kind: str = 'ddqn') -> Dict[str, object]:
    from showdown.inference_sim.team_factory import load_team_file
    from showdown.rl.action_space import decode_joint_action, decode_slot_action
    from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
    from showdown.rl.ddqn.train import _joint_masks
    from showdown.rl.env import VGCEnv
    from showdown.rl.frozen_preview import configure_preview
    from showdown.rl.observation import build_observation_ddqn
    from showdown.rl.pe_adapter import pe_autoregressive_masks
    from showdown.rl.pokeenv_backend import PokeEnvBattleBackend
    from showdown.rl.reward import RewardCandidate

    print('[diag] ' + configure_preview(agent_kind, checkpoint=preview_checkpoint or None), flush=True)
    if agent_kind == 'ppo':
        from showdown.rl.observation import build_observation_ppo
        from showdown.rl.ppo.agent import PPOAgent, PPOConfig
        agent = PPOAgent(PPOConfig(device='cpu'))
    else:
        agent = DDQNAgent(DDQNConfig(device='cpu', replay_device='cpu', replay_capacity=64))
    agent.load(checkpoint, map_location='cpu')
    our_team = load_team_file(agent_team) if agent_team else None
    by_foe: Dict[str, Counter] = defaultdict(Counter)
    results: Dict[str, Counter] = defaultdict(Counter)
    for fi, foe in enumerate(FOES):
        backend = PokeEnvBattleBackend(opponent=foe, start_ps=True, port=port, account_prefix='DdqnDiag',
                                       combat_family='ddqn', our_team=our_team, foe_tie=False)
        env = VGCEnv(reward_candidate=RewardCandidate.SPARSE, backend=backend, max_episode_turns=40)
        try:
            for g in range(games):
                env.reset(seed=int(seed) + 300_000 + fi * 10_000 + g)
                info = {}
                for _ in range(80):
                    battle = env.backend._battle
                    if agent_kind == 'ppo':
                        obs = build_observation_ppo(env._snap)
                        p1, _ = pe_autoregressive_masks(battle, slot1_action_id=None)
                        a1, a2, _lp, _v = agent.act(
                            obs, p1, lambda x, _b=battle: pe_autoregressive_masks(_b, slot1_action_id=int(x))[1],
                            greedy=True)
                    else:
                        obs = build_observation_ddqn(env._snap)
                        a = agent.act(obs, _joint_masks(env), epsilon=0.0)
                        a1, a2 = decode_joint_action(int(a))
                    m1, _ = pe_autoregressive_masks(battle, slot1_action_id=None)
                    _, m2 = pe_autoregressive_masks(battle, slot1_action_id=int(a1))
                    for slot, (aid, m) in enumerate(((a1, m1), (a2, m2))):
                        sa = decode_slot_action(int(aid))
                        by_foe[foe][classify(battle, slot, sa, m)] += 1
                        if sa.kind == 'mega':
                            by_foe[foe]['mega'] += 1
                        by_foe[foe]['decisiones'] += 1
                    _o, _r, term, trunc, info = env.step((a1, a2))
                    if term or trunc:
                        break
                ret = float(info.get('episode_return', 0.0))
                results[foe]['win' if ret > 0.05 else ('loss' if ret < -0.05 else 'tie')] += 1
        finally:
            env.close()
    total = Counter()
    for c in by_foe.values():
        total.update(c)
    return {'checkpoint': checkpoint, 'games_per_foe': games,
            'por_rival': {f: dict(c) for f, c in by_foe.items()},
            'resultados': {f: dict(c) for f, c in results.items()},
            'total': dict(total)}


def format_table(rep: Dict[str, object]) -> str:
    tot = rep['total']
    n = max(1, int(tot.get('decisiones', 0)))
    cats = ['ataque_rival', 'area_rivales', 'area_con_aliado', 'ataque_aliado', 'apoyo_aliado',
            'estado_rival', 'protect', 'otro_estado', 'cambio_voluntario', 'cambio_forzado',
            'pass_obligatorio', 'pass_con_opciones', 'sin_movimientos', 'mega']
    lines = ['| Categoría | Decisiones | % |', '|---|---|---|']
    for c in cats:
        lines.append(f"| {c} | {int(tot.get(c, 0))} | {100 * tot.get(c, 0) / n:.1f} % |")
    lines.append(f'| **total de decisiones de hueco** | {n} | |')
    lines += ['', '| Rival | Victorias | Derrotas |', '|---|---|---|']
    for f, r in rep['resultados'].items():
        lines.append(f"| {f} | {r.get('win', 0)} | {r.get('loss', 0)} |")
    return '\n'.join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--preview-checkpoint', default='')
    ap.add_argument('--games', type=int, default=50, help='partidas por rival')
    ap.add_argument('--port', type=int, default=8370)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--agent-team', default='showdown/rl/data/ddqn_team.txt')
    ap.add_argument('--out', default='logs/policy_diag.json')
    ap.add_argument('--agent', choices=('ddqn', 'ppo'), default='ddqn')
    args = ap.parse_args(argv)
    rep = run(checkpoint=args.checkpoint, games=args.games, port=args.port, seed=args.seed,
              agent_team=args.agent_team, preview_checkpoint=args.preview_checkpoint,
              agent_kind=args.agent)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, indent=1)
    print(format_table(rep))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
