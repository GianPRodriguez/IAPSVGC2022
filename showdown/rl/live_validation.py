"""Validación en vivo del entorno RL antes del entrenamiento oficial (checklist DDQN, punto B).

Juega episodios reales contra Showdown por el MISMO camino que el entrenamiento de DDQN
(``VGCEnv`` + ``PokeEnvBattleBackend`` + máscara ``_joint_masks``) con acciones legales al
azar, y en cada paso verifica:

- **máscara**: nunca vacía; la de entrenamiento (``pe_autoregressive_masks``) coincide con la
  de inferencia (``legal_joint_mask``, la que usa ``DDQNPlayer``);
- **paridad de la obs** (punto 2): la obs del entorno es idéntica a la que arma el jugador
  de inferencia sobre la misma batalla (``snapshot_from_battle(..., stat_beliefs)``);
- **alineación**: el movimiento k de la obs (``avail_moves``) es el que ejecuta la acción k;
- **historial**: cada movimiento elegido en el turno anterior (protocolo crudo) aparece como
  ``last_move`` del Pokémon que lo usó, si sigue activo;
- **simetrías**: para cada elemento g del grupo Z2³, permutar la obs real equivale a la obs
  del combate espejado;
- **fallbacks y caídas**: acciones reemplazadas por el adaptador y truncados por excepción;
- **costo**: ms de snapshot + obs y del forward de la red de DDQN.

    PS_RL_STRICT=1 python -m showdown.rl.live_validation --episodes 12 --port 8350 \\
        --out logs/live_validation.json
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

FOES = ('random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian')


def _segment(idx: int, layout: Dict[str, tuple]) -> str:
    for name, (a, b) in layout.items():
        if a <= idx < b:
            return name
    return '?'


def mirror_snapshot(s, g: int):
    """Combate espejado para el elemento g (misma convención que tests/test_rl_symmetry)."""
    from showdown.rl import symmetry as SYM
    s = copy.deepcopy(s)
    if g & SYM.BIT_OUR:
        s.our.active = [s.our.active[1], s.our.active[0]]
        if s.avail_moves is not None:
            s.avail_moves = (s.avail_moves[1], s.avail_moves[0])
        s.force_switch = (s.force_switch[1], s.force_switch[0])
    if g & SYM.BIT_FOE:
        s.foe.active = [s.foe.active[1], s.foe.active[0]]
    if g & SYM.BIT_BENCH:
        s.our.reserve = [s.our.reserve[1], s.our.reserve[0]] + list(s.our.reserve[2:])
        s.our.bench = list(s.our.reserve[:2])
    return s


def _last_moves(battle) -> Dict[str, tuple]:
    """Último movimiento elegido por cada Pokémon según el protocolo crudo.

    Devuelve ``{clave 'p1: Nombre': (id del movimiento, turnos desde entonces)}``. Incluye
    los movimientos posteriores al último ``|turn|`` (pasos de cambio forzado tras un KO).
    Los movimientos llamados (``[from]``: Instruct, Dancer…) no cuentan.
    """
    from showdown.rl.battle_history import _id, ident_key
    turn = 0
    last: Dict[str, tuple] = {}
    for m in list(getattr(battle, '_replay_data', None) or []):
        if len(m) > 2 and m[1] == 'turn':
            try:
                turn = int(m[2])
            except ValueError:
                pass
        elif len(m) > 3 and m[1] == 'move' and not any(str(p).startswith('[from]') for p in m[4:]):
            k = ident_key(m[2])
            if k:
                last[k] = (_id(m[3]), turn)
    return {k: (mid, turn - t) for k, (mid, t) in last.items()}


def _active_slot_of(battle, key: str):
    """(lado 'our'/'foe', slot) del Pokémon con esa clave si está activo, si no None."""
    role = getattr(battle, 'player_role', None) or 'p1'
    ours = key.startswith(role)
    team = battle.team if ours else battle.opponent_team
    mon = (team or {}).get(key)
    if mon is None:
        return None
    act = battle.active_pokemon if ours else battle.opponent_active_pokemon
    for i, a in enumerate(act or []):
        if a is mon:
            return ('our' if ours else 'foe', i)
    return None


class Validation:
    def __init__(self, *, device: str = 'cpu'):
        from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
        from showdown.rl.observation import obs_layout
        self.layout = obs_layout()
        self.agent = DDQNAgent(DDQNConfig(device=device))
        self.c: Counter = Counter()
        self.ms: Dict[str, List[float]] = defaultdict(list)
        self.sym_bad: Counter = Counter()      # (g, segmento) → nº de pasos con diferencia
        self.parity_bad: Counter = Counter()   # segmento → nº de pasos con diferencia
        self.examples: Dict[str, List[Any]] = defaultdict(list)
        self.by_foe: Dict[str, Counter] = defaultdict(Counter)

    def _ex(self, kind: str, item: Any, n: int = 5) -> None:
        if len(self.examples[kind]) < n:
            self.examples[kind].append(item)

    # ------------------------------------------------------------------ checks
    def check_step(self, env, foe: str) -> np.ndarray:
        import torch
        from showdown.rl import symmetry as SYM
        from showdown.rl.action_space import decode_slot_action
        from showdown.rl.ddqn.train import _joint_masks
        from showdown.rl.observation import build_observation
        from showdown.rl.pe_adapter import (
            legal_joint_mask, player_stat_beliefs, slot_action_to_pe_order, snapshot_from_battle,
        )

        battle = env.backend._battle
        snap = env._snap
        self.c['steps'] += 1
        self.by_foe[foe]['steps'] += 1

        # máscara
        jm = np.asarray(_joint_masks(env), dtype=bool)
        if not jm.any():
            self.c['mask_empty'] += 1
            self._ex('mask_empty', {'turn': snap.turn, 'force_switch': snap.force_switch})
        im = np.asarray(legal_joint_mask(battle), dtype=bool)
        if jm.shape != im.shape or not np.array_equal(jm, im):
            self.c['mask_train_vs_infer'] += 1
            only_t = int((jm & ~im).sum()) if jm.shape == im.shape else -1
            only_i = int((im & ~jm).sum()) if jm.shape == im.shape else -1
            self._ex('mask_train_vs_infer', {'turn': snap.turn, 'solo_entrenamiento': only_t,
                                             'solo_inferencia': only_i})

        # paridad de la obs (entorno frente a jugador de inferencia)
        t0 = time.perf_counter()
        snap_inf = snapshot_from_battle(battle, stat_beliefs=player_stat_beliefs(env.backend._agent))
        obs_inf = build_observation(snap_inf)
        self.ms['snapshot+obs'].append(1000 * (time.perf_counter() - t0))
        obs = build_observation(snap)
        diff = np.flatnonzero(~np.isclose(obs, obs_inf, atol=1e-6))
        if diff.size:
            self.c['parity_steps_bad'] += 1
            for seg in {_segment(int(i), self.layout) for i in diff}:
                self.parity_bad[seg] += 1

        # costo del forward de la red (lote 1)
        with torch.no_grad():
            x = torch.as_tensor(obs[None], dtype=torch.float32, device=self.agent.device)
            t0 = time.perf_counter()
            self.agent.online(x)
            if self.agent.device.type == 'cuda':
                torch.cuda.synchronize()
            self.ms['forward'].append(1000 * (time.perf_counter() - t0))

        # alineación movimiento k de la obs ↔ acción k
        fs = tuple(snap.force_switch or (False, False))
        for s in range(2):
            if s < len(fs) and fs[s]:
                continue
            avail = list((snap.avail_moves or ((), ()))[s])
            m_s = np.asarray(env.backend.action_masks(slot1_action_id=None)[s], dtype=bool) \
                if s == 0 else None
            for k, mid in enumerate(avail[:4]):
                cand = None
                for aid in range(len(m_s) if m_s is not None else 40):
                    a = decode_slot_action(aid)
                    if a.kind == 'move' and a.move_index == k and (m_s is None or m_s[aid]):
                        cand = a
                        break
                if cand is None:
                    continue
                order = slot_action_to_pe_order(battle, s, cand)
                got = getattr(getattr(order, 'order', None), 'id', None)
                self.c['align_checks'] += 1
                if got != mid:
                    self.c['align_bad'] += 1
                    self._ex('align_bad', {'slot': s, 'k': k, 'obs': mid, 'orden': got})

        # historial: último movimiento de cada Pokémon activo (protocolo crudo) → obs
        for key, (mid, ago) in _last_moves(battle).items():
            where = _active_slot_of(battle, key)
            if where is None:
                continue
            side, slot = where
            mon = (snap.our if side == 'our' else snap.foe).active[slot]
            if mon is None:
                continue
            self.c['history_checks'] += 1
            if mon.last_move != mid or mon.last_move_ago != ago:
                self.c['history_bad'] += 1
                self._ex('history_bad', {'key': key, 'protocolo': mid, 'hace_protocolo': ago,
                                         'obs': mon.last_move, 'hace_obs': mon.last_move_ago,
                                         'turn': snap.turn})

        # simetrías con el combate real
        for g in range(1, SYM.N_ELEMENTS):
            if g & SYM.BIT_BENCH and len([m for m in snap.our.reserve if m is not None]) < 2:
                continue
            self.c['sym_checks'] += 1
            perm = SYM.augment_obs(obs[None], np.array([g]))[0]
            mir = build_observation(mirror_snapshot(snap, g))
            bad = np.flatnonzero(~np.isclose(perm, mir, atol=1e-5))
            if bad.size:
                self.c['sym_bad'] += 1
                for seg in {_segment(int(i), self.layout) for i in bad}:
                    self.sym_bad[(g, seg)] += 1
        return jm

    # ------------------------------------------------------------------ run
    def run(self, *, foes, episodes: int, port: int, seed: int, max_steps: int = 60) -> None:
        from showdown.rl.action_space import decode_joint_action, decode_slot_action
        from showdown.rl.env import VGCEnv
        from showdown.rl.pokeenv_backend import PokeEnvBattleBackend
        from showdown.rl.reward import RewardCandidate

        rng = np.random.default_rng(seed)
        for foe in foes:
            backend = PokeEnvBattleBackend(opponent=foe, start_ps=True, port=port,
                                           account_prefix=f'Val{foe[:4].title()}',
                                           combat_family='ddqn')
            env = VGCEnv(reward_candidate=RewardCandidate.DENSE, backend=backend,
                         max_episode_turns=40)
            try:
                for ep in range(episodes):
                    env.reset(seed=int(seed) + 1000 * ep)
                    self.c['episodes'] += 1
                    self.by_foe[foe]['episodes'] += 1
                    for _ in range(max_steps):
                        jm = self.check_step(env, foe)
                        legal = np.flatnonzero(jm)
                        joint = int(rng.choice(legal)) if legal.size else 0
                        a1, a2 = decode_joint_action(joint)
                        vo = getattr(env.backend._battle, 'valid_orders', None) or []
                        pre_valid = [[str(o) for o in (side or [])][:12] for side in vo[:2]]
                        _obs, _r, term, trunc, info = env.step((a1, a2))
                        eng = info.get('engine') or {}
                        if isinstance(eng, dict) and eng.get('used_fallback'):
                            self.c['order_fallbacks'] += 1
                            self.by_foe[foe]['order_fallbacks'] += 1
                            self._ex('order_fallback', {
                                'a1': repr(decode_slot_action(a1)), 'a2': repr(decode_slot_action(a2)),
                                'orden_enviada': eng.get('order'), 'turn': info.get('turn'),
                                'force_switch': list(env._snap.force_switch or ()) if env._snap else None,
                                'valid_orders': pre_valid,
                            }, n=10)
                        if info.get('trunc_reason') == 'crash':
                            self.c['crashes'] += 1
                            self._ex('crash', info.get('error'))
                        if term or trunc:
                            break
            finally:
                env.close()

    def report(self) -> Dict[str, Any]:
        def stats(v):
            if not v:
                return None
            a = np.asarray(v)
            return {'n': int(a.size), 'mean': float(a.mean()), 'p50': float(np.median(a)),
                    'p95': float(np.percentile(a, 95))}
        return {
            'counts': dict(self.c),
            'by_foe': {k: dict(v) for k, v in self.by_foe.items()},
            'ms': {k: stats(v) for k, v in self.ms.items()},
            'parity_segments': dict(self.parity_bad),
            'symmetry_bad': {f'g{g}:{seg}': n for (g, seg), n in sorted(self.sym_bad.items())},
            'examples': dict(self.examples),
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--foes', default=','.join(FOES))
    ap.add_argument('--episodes', type=int, default=12, help='episodios por rival')
    ap.add_argument('--port', type=int, default=8350)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='logs/live_validation.json')
    args = ap.parse_args(argv)
    v = Validation(device=args.device)
    v.run(foes=[f.strip() for f in args.foes.split(',') if f.strip()],
          episodes=args.episodes, port=args.port, seed=args.seed)
    rep = v.report()
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, indent=2, default=str)
    print(json.dumps({k: rep[k] for k in ('counts', 'ms', 'parity_segments', 'symmetry_bad')},
                     indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
