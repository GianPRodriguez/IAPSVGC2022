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

``--series`` (protocolo §4.16; PPO v3 / DDQN v2): ``--games`` pasa a ser el número de SERIES BO3 por rival
(mismo rival y mismos equipos hasta que alguien gane 2); se informa la tasa de victorias por partida 1/2/3 y
por serie. La obs lleva la memoria de serie si el checkpoint la usa, la máscara del Protect consecutivo sigue
al checkpoint y ``--preview-series`` condiciona el preview a lo que el rival trajo. ``--foe-tie`` activa el
sorteo de casi-empates de los rivales (por defecto deterministas, como siempre).

``--context`` (memorizacion_equipos.md §12) agrega el desglose de decisiones según el plan del
rival (Trick Room o Tailwind rival activos, Fake Out disponible, redirección, rival que puede
activar Trick Room; ver ``showdown/rl/context_diag.py``) y, por partida, los usos reales de Trick
Room, Tailwind, Fake Out y redirección de cada lado. ``--agent-team ''`` sortea el equipo del
agente en cada partida (del mismo grupo que el rival) y lo registra.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict

import numpy as np

FOES = ('random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian')
# Rivales opcionales (solo con --foes; van al final para no cambiar las semillas de los anteriores):
# greedy_joint_arq = arquetipo nivel 2; ppo_v1 = PPO v1 congelado en espejo (PPO_FOE_CHECKPOINTS).
EXTRA_FOES = ('greedy_joint_arq', 'ppo_v1', 'greedy_joint_arq3')  # al final: no cambia las semillas de los demás
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
        preview_checkpoint: str = '', agent_kind: str = 'ddqn', context: bool = False,
        foes=FOES, sample: bool = False, temperature: float = 1.0,
        family: str = '', series: bool = False, preview_series: bool = False,
        foe_tie: bool = False, mask_protect: str = 'auto') -> Dict[str, object]:
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

    # family: familia del preview congelado (p. ej. qd_ppo = preview QD-PGA con combate PPO, I 2.0)
    fam = family or agent_kind
    print('[diag] ' + configure_preview(fam, checkpoint=preview_checkpoint or None), flush=True)
    if agent_kind == 'ppo':
        from showdown.rl.observation import build_observation_ppo
        from showdown.rl.ppo.agent import PPOAgent, PPOConfig
        agent = PPOAgent(PPOConfig(device='cpu'))
    else:
        agent = DDQNAgent(DDQNConfig(device='cpu', replay_device='cpu', replay_capacity=64))
    agent.load(checkpoint, map_location='cpu')
    # PPO v3 / DDQN v2: la máscara del Protect consecutivo y la memoria de serie siguen al checkpoint.
    mem_on = bool(getattr(agent.cfg, 'series_memory', False))
    # mask_protect: 'auto' = la del checkpoint; '1' / '0' la fuerza (P1 de §4.16: máscara sin reentrenar).
    mask_on = bool(getattr(agent.cfg, 'mask_protect_streak', False)) if mask_protect == 'auto' else mask_protect == '1'
    os.environ['RL_MASK_PROTECT_STREAK'] = '1' if mask_on else '0'
    protect_streak: Dict[str, Counter] = defaultdict(Counter)  # situaciones de Protect repetible y repeticiones
    os.environ['PREVIEW_SERIES_CONDITION'] = '1' if preview_series else '0'
    our_team = load_team_file(agent_team) if agent_team else None
    from showdown.rl.train_metrics import SeriesTally
    series_by_foe: Dict[str, SeriesTally] = defaultdict(SeriesTally)
    series_out: Dict[str, Dict] = {}
    by_foe: Dict[str, Counter] = defaultdict(Counter)
    results: Dict[str, Counter] = defaultdict(Counter)
    games_log = []  # equipo rival y resultado de cada partida (análisis de similitud con train)
    ctx_counts: Dict[str, Counter] = defaultdict(Counter)
    div: Dict[str, Counter] = defaultdict(Counter)  # sorteo frente a determinista (P1)
    if context:
        from showdown.rl import context_diag as CD
    if sample:  # P1: política estocástica reproducible
        import torch
        torch.manual_seed(int(seed))
    for fi, foe in enumerate(FOES + EXTRA_FOES):
        if foe not in foes:
            continue
        backend = PokeEnvBattleBackend(opponent=foe, start_ps=True, port=port, account_prefix='DdqnDiag',
                                       combat_family=family or 'ddqn', our_team=our_team, foe_tie=bool(foe_tie),
                                       series=bool(series))
        env = VGCEnv(reward_candidate=RewardCandidate.SPARSE, backend=backend, max_episode_turns=40)
        # En modo serie, ``games`` = series; cada serie juega hasta 3 partidas (semilla de la 1.ª partida).
        plan = []
        for g in range(games):
            plan.extend([(g, k) for k in range(3)] if series else [(g, 0)])
        try:
            skip_series = -1
            for g, k in plan:
                if series and (g == skip_series or (k > 0 and backend._series.decided)):
                    skip_series = g
                    continue
                if series and k == 0 and backend._series.records and not backend._series.decided:
                    backend._series_broken = True  # la serie anterior quedó cortada: se empieza otra
                # Sin --series, las mismas semillas de siempre (reproducibilidad de evaluaciones anteriores).
                env.reset(seed=int(seed) + 300_000 + fi * 10_000 + ((g * 3 + k) if series else g))
                info = {}
                game_div = False
                for _ in range(80):
                    battle = env.backend._battle
                    if agent_kind == 'ppo':
                        obs = build_observation_ppo(env._snap, abilities=bool(agent.cfg.obs_abilities),
                                                    memory=mem_on)
                        p1, _ = pe_autoregressive_masks(battle, slot1_action_id=None)
                        m2f = (lambda x, _b=battle: pe_autoregressive_masks(_b, slot1_action_id=int(x))[1])
                        a1, a2, _lp, _v = agent.act(obs, p1, m2f, greedy=not sample, temperature=temperature)
                        if sample:  # ¿cuánto se aparta el sorteo del modo determinista? (P1)
                            g1, g2, _l, _vv = agent.act(obs, p1, m2f, greedy=True)
                            free = int((np.asarray(p1) > 0.5).sum()) > 1
                            t1 = int(getattr(battle, 'turn', 0) or 0) <= 1
                            for key, on in (('todas', True), ('turno_1', t1)):
                                if not (on and free):
                                    continue
                                div[foe][key + '_decisiones'] += 1
                                div[foe][key + '_hueco1_distinto'] += int(a1 != g1)
                                div[foe][key + '_jugada_distinta'] += int((a1, a2) != (g1, g2))
                            if free and (a1, a2) != (g1, g2):
                                game_div = True
                    else:
                        obs = build_observation_ddqn(env._snap, abilities=bool(agent.cfg.obs_abilities),
                                                     memory=mem_on)
                        a = agent.act(obs, _joint_masks(env), epsilon=0.0)
                        a1, a2 = decode_joint_action(int(a))
                    m1, _ = pe_autoregressive_masks(battle, slot1_action_id=None)
                    _, m2 = pe_autoregressive_masks(battle, slot1_action_id=int(a1))
                    ctxs = CD.contexts(env._snap, battle) if context else ()
                    for slot, (aid, m) in enumerate(((a1, m1), (a2, m2))):
                        sa = decode_slot_action(int(aid))
                        cat = classify(battle, slot, sa, m)
                        # ¿Venía de un Protect exitoso y tenía otro disponible? (lo que quita la máscara)
                        act_mons = list(getattr(battle, 'active_pokemon', None) or [])
                        mon_s = act_mons[slot] if slot < len(act_mons) else None
                        if mon_s is not None and int(getattr(mon_s, 'protect_counter', 0) or 0) >= 1 and any(
                                bool(getattr(mv, 'is_protect_counter', False))
                                for mv in (battle.available_moves[slot] if slot < len(battle.available_moves) else [])):
                            protect_streak[foe]['situaciones'] += 1
                            protect_streak[foe]['repitio_protect'] += int(cat == 'protect')
                        by_foe[foe][cat] += 1
                        if sa.kind == 'mega':
                            by_foe[foe]['mega'] += 1
                        by_foe[foe]['decisiones'] += 1
                        if foe == 'random' or cat in ('pass_obligatorio', 'cambio_forzado', 'sin_movimientos'):
                            continue  # contexto: solo decisiones libres contra los heurísticos
                        for c in ctxs:
                            ctx_counts[c][cat] += 1
                            ctx_counts[c]['decisiones'] += 1
                            if cat == 'ataque_rival':
                                ctx_counts[c]['ataques_un_rival'] += 1
                                for k, v in CD.target_flags(env._snap, int(sa.target)).items():
                                    ctx_counts[c][k] += v
                    _o, _r, term, trunc, info = env.step((a1, a2))
                    if term or trunc:
                        break
                ret = float(info.get('episode_return', 0.0))
                res = 'win' if ret > 0.05 else ('loss' if ret < -0.05 else 'tie')
                results[foe][res] += 1
                eng = info.get('engine') if isinstance(info, dict) else None
                ser = eng.get('series') if isinstance(eng, dict) else None
                if series:
                    series_by_foe[foe].add(ser, res == 'win')
                    if ser is None or not backend._series.records:
                        skip_series = g  # partida abortada: la serie se corta
                if sample and agent_kind == 'ppo':
                    div[foe]['partidas'] += 1
                    div[foe]['partidas_con_alguna_distinta'] += int(game_div)
                fs = env.backend.foe_state_episode() if hasattr(env.backend, 'foe_state_episode') else None
                rec = {'rival': foe, 'equipo': getattr(env.backend, 'last_foe_team', None), 'res': res,
                       **({'foe_state': fs} if fs else {})}
                if our_team is None:
                    rec['nuestro_equipo'] = getattr(env.backend, 'last_our_team', None)
                pp = getattr(env.backend, '_preview_policy', None)
                if getattr(pp, 'last_order', None):
                    rec['formacion'] = ''.join(str(x) for x in pp.last_order)  # 1-based, orden del equipo
                if getattr(pp, 'last_greedy_order', None):  # preview no determinista (P1b)
                    rec['formacion_determinista'] = ''.join(str(x) for x in pp.last_greedy_order)
                if context:
                    rec['eventos'] = CD.count_events(env.backend._battle)
                if series:
                    rec['serie'] = g
                    rec['partida'] = (ser or {}).get('partida', k + 1)
                    if getattr(pp, 'last_foe_keep', None):
                        rec['preview_vio'] = list(pp.last_foe_keep)
                games_log.append(rec)
        finally:
            env.close()
        if series:
            series_out[foe] = series_by_foe[foe].flush()
    total = Counter()
    for c in by_foe.values():
        total.update(c)
    return {'checkpoint': checkpoint, 'games_per_foe': games,
            'por_rival': {f: dict(c) for f, c in by_foe.items()},
            'resultados': {f: dict(c) for f, c in results.items()},
            'partidas': games_log,
            'preview_forzado': os.environ.get('PREVIEW_FORCE_ORDER', '') or None,
            'joint_arquetipo': os.environ.get('PS_JOINT_ARCHETYPE', '') or None,
            'obs_habilidades': bool(agent.cfg.obs_abilities),
            'muestreo': {'activo': bool(sample), 'temperatura': float(temperature)} if sample else None,
            'preview_muestreo': ({'temperatura': float(os.environ.get('PREVIEW_TEMPERATURE', '1.0') or 1.0)}
                                 if os.environ.get('PREVIEW_SAMPLE', '').strip().lower() in ('1', 'true', 'yes')
                                 else None),
            'divergencia': {f: dict(c) for f, c in div.items()} if sample else None,
            'mascara_protect_consecutivo': {'activa': mask_on,
                                            'por_rival': {f: dict(c) for f, c in protect_streak.items()}},
            'series': ({'por_rival': series_out, 'preview_condicionado': bool(preview_series),
                        'rivales_con_sorteo': bool(foe_tie), 'memoria': mem_on,
                        'mascara_protect': bool(getattr(agent.cfg, 'mask_protect_streak', False))}
                       if series else None),
            'total': dict(total),
            **({'contexto': {c: dict(v) for c, v in ctx_counts.items()}} if context else {})}


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
    ser = rep.get('series')
    if ser:
        lines += ['', '| Rival | Partida 1 | Partida 2 | Partida 3 | Series ganadas |', '|---|---|---|---|---|']
        for f, r in (ser.get('por_rival') or {}).items():
            cells = []
            for g in (1, 2, 3):
                n_g = r.get(f'n_partida_{g}')
                cells.append(f"{100 * r[f'wr_partida_{g}']:.1f} % ({n_g})" if n_g else '–')
            sw = f"{100 * r['wr_series']:.1f} % ({r['n_series']})" if r.get('n_series') else '–'
            lines.append(f'| {f} | ' + ' | '.join(cells) + f' | {sw} |')
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
    ap.add_argument('--family', default='', help='familia del preview congelado (p. ej. qd_ppo); defecto: --agent')
    # Por defecto también el nivel 3 de arquetipos (métrica secundaria, protocolo §2; 27-11-2026). Va al final de
    # FOES + EXTRA_FOES: no cambia las semillas de los demás rivales.
    ap.add_argument('--foes', default=','.join(FOES + ('greedy_joint_arq3',)),
                    help='rivales a jugar (lista separada por comas)')
    ap.add_argument('--sample', action='store_true',
                    help='PPO: sortear la acción de la política en vez de tomar la más probable (P1)')
    ap.add_argument('--temperature', type=float, default=1.0, help='temperatura del sorteo (con --sample)')
    ap.add_argument('--series', action='store_true',
                    help='series BO3 (protocolo §4.16): --games = series por rival; resultados por partida y serie')
    ap.add_argument('--preview-series', action='store_true',
                    help='preview condicionado a lo que el rival trajo en la serie (con --series)')
    ap.add_argument('--foe-tie', action='store_true', help='rivales con sorteo de casi-empates (no deterministas)')
    ap.add_argument('--mask-protect', choices=('auto', '0', '1'), default='auto',
                    help='máscara del Protect consecutivo: auto = la del checkpoint; 1/0 la fuerza (P1, §4.16)')
    ap.add_argument('--context', action='store_true',
                    help='desglose por plan del rival y usos de movimientos clave (§12)')
    args = ap.parse_args(argv)
    rep = run(checkpoint=args.checkpoint, games=args.games, port=args.port, seed=args.seed,
              agent_team=args.agent_team, preview_checkpoint=args.preview_checkpoint,
              agent_kind=args.agent, context=args.context, sample=args.sample, temperature=args.temperature,
              foes=tuple(f.strip() for f in args.foes.split(',') if f.strip()), family=args.family,
              series=bool(args.series), preview_series=bool(args.preview_series), foe_tie=bool(args.foe_tie),
              mask_protect=args.mask_protect)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, indent=1)
    print(format_table(rep))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
