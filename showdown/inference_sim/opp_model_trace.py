"""Mide qué tan bien predice el modelo del rival de expectimax (``PS_OPPMODEL_LOG``).

En cada decisión de expectimax, ``opp_policy_targeted`` calcula una distribución
sobre las acciones del rival. Aquí se guarda esa predicción y, en la decisión
siguiente, se compara con los movimientos que el rival realmente usó (leídos del
protocolo del turno ya jugado). Cada movimiento observado produce una línea JSON.

    PS_OPPMODEL_LOG=logs/oppmodel.jsonl python -m showdown.inference_sim.run_robust_e2e ...
    python -m showdown.inference_sim.analyze_opp_model logs/oppmodel.jsonl
"""
from __future__ import annotations

import contextvars
import inspect
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

_BUF: contextvars.ContextVar[Optional[List[dict]]] = contextvars.ContextVar('oppmodel_buf', default=None)


def _norm(name: str) -> str:
    return str(name or '').strip().lower().replace(' ', '').replace('-', '')


def _pred_sink(ev: dict) -> None:
    buf = _BUF.get()
    if buf is not None:
        buf.append(ev)


def parse_foe_moves(lines: Sequence[str], turn: int, opp_prefix: str,
                    our_prefix: str) -> List[Tuple[str, str, int]]:
    """Movimientos del rival durante ``turn`` (entre ``|turn|N`` y ``|turn|N+1``).

    Devuelve ``(especie, movimiento, objetivo)``; objetivo 1/2 = nuestro slot a/b,
    0 = sin objetivo nuestro (propio, aliado, campo, múltiple).
    """
    start = None
    for i, line in enumerate(lines):
        if line.startswith('|turn|'):
            try:
                n = int(line.split('|')[2])
            except (IndexError, ValueError):
                continue
            if n == turn:
                start = i
            elif start is not None and n > turn:
                lines = lines[:i]
                break
    if start is None:
        return []
    out: List[Tuple[str, str, int]] = []
    for line in lines[start + 1:]:
        parts = line.split('|')
        if len(parts) < 5 or parts[1] != 'move':
            continue
        if '[from]' in line:  # movimiento llamado, no elegido
            continue
        actor = parts[2]
        if not actor.startswith(opp_prefix):
            continue
        target = parts[4].strip()
        slot = 0
        if target.startswith(our_prefix + 'a'):
            slot = 1
        elif target.startswith(our_prefix + 'b'):
            slot = 2
        species = actor.split(':', 1)[-1].strip()
        out.append((species, parts[3].strip(), slot))
    return out


def score_event(preds: Sequence[dict], move: str, target: int) -> Optional[Dict[str, Any]]:
    """Compara un movimiento observado con la predicción que lo contiene."""
    mid = _norm(move)
    cands = [p for p in preds if mid in {_norm(n) for n in p['moves']}]
    if len(cands) != 1:
        return {'status': 'sin_prediccion' if not cands else 'ambiguo'}
    merged: Dict[Tuple[str, int, str], float] = cands[0]['merged']
    by_move: Dict[str, float] = {}
    by_pair: Dict[Tuple[str, int], float] = {}
    mass_kind: Dict[str, float] = {}
    for (m, t, kind), p in merged.items():
        by_move[m] = by_move.get(m, 0.0) + p
        by_pair[(m, t)] = by_pair.get((m, t), 0.0) + p
        mass_kind[kind] = mass_kind.get(kind, 0.0) + p
    total = sum(by_move.values()) or 1.0
    top_move = max(by_move, key=by_move.get)
    top_pair = max(by_pair, key=by_pair.get)
    n_moves = len({m for m in by_move if m != '__switch__'}) or 1
    return {
        'status': 'ok',
        'p_move': by_move.get(mid, 0.0) / total,
        'p_exact': by_pair.get((mid, target), 0.0) / total,
        'top1_move_hit': top_move == mid,
        'top1_exact_hit': top_pair == (mid, target),
        'pred_top_move': top_move,
        'uniform_p': 1.0 / n_moves,
        'n_options': n_moves,
        'mass_control': mass_kind.get('control', 0.0) / total,
        'mass_switch': mass_kind.get('switch', 0.0) / total,
        'mass_greedy': mass_kind.get('greedy', 0.0) / total,
    }


def _is_control(move: str) -> bool:
    from showdown.battle_bots.expectimax_bayesian.opp_policy import is_control_move
    return is_control_move(move)


def _evaluate(player, state: dict, path: str) -> None:
    role = getattr(player, '_our_role', 'p1')
    opp_prefix = 'p2' if role == 'p1' else 'p1'
    lines = list(getattr(player, '_game_lines', []) or [])
    events = parse_foe_moves(lines, state['turn'], opp_prefix, role)
    for species, move, target in events:
        sc = score_event(state['preds'], move, target)
        if sc is None:
            continue
        rec = {'species': species, 'move': move, 'target': target, 'control': _is_control(move),
               'turn': state['turn'], 'tag': state['tag'], 'pid': os.getpid()}
        rec.update(sc)
        try:
            with open(path, 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(rec) + chr(10))
        except OSError:
            pass


def install(player, path: Optional[str] = None) -> bool:
    """Envuelve ``choose_move`` de un jugador expectimax. No-op sin ``PS_OPPMODEL_LOG``."""
    path = path or os.environ.get('PS_OPPMODEL_LOG', '').strip()
    if not path:
        return False
    from showdown.battle_bots.expectimax_bayesian.opp_policy import set_pred_sink
    set_pred_sink(_pred_sink)
    orig = player.choose_move
    state: Dict[str, Any] = {'tag': None, 'turn': None, 'preds': None}

    def _before(battle) -> None:
        tag, turn = str(getattr(battle, 'battle_tag', '')), int(getattr(battle, 'turn', 0) or 0)
        if state['tag'] != tag:
            state.update(tag=tag, turn=None, preds=None)
        elif state['preds'] and state['turn'] is not None and turn > state['turn']:
            _evaluate(player, state, path)
        _BUF.set([])

    def _after(battle) -> None:
        buf = _BUF.get() or []
        uniq: Dict[frozenset, dict] = {}
        for ev in buf:  # la última predicción por conjunto de movimientos (con daño real)
            uniq[frozenset(_norm(n) for n in ev['moves'])] = ev
        state.update(turn=int(getattr(battle, 'turn', 0) or 0), preds=list(uniq.values()))
        if os.environ.get('PS_OPPMODEL_DEBUG') == '1':
            try:
                with open(path + '.dbg', 'a', encoding='utf-8') as fh:
                    fh.write(json.dumps({'turn': state['turn'], 'n_buf': len(buf), 'n_preds': len(uniq),
                                         'n_lines': len(getattr(player, '_game_lines', []) or [])}) + chr(10))
            except OSError:
                pass

    def wrapped(battle):
        res = orig(battle)  # una corrutina aún no ha empezado a ejecutarse
        if inspect.isawaitable(res):
            async def _aw():
                _before(battle)  # el búfer se fija en el contexto de la tarea que sí calcula
                order = await res
                _after(battle)
                return order
            return _aw()
        return res

    player.choose_move = wrapped
    return True
