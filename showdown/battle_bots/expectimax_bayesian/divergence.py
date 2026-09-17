"""Agregado phase1 vs phase2 (diagnostic; no afecta la decisión).

Escribe resumen en logs/expectimax_divergence.txt al finalizar un batch robusto.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from showdown.inference_sim.paths import REPO_ROOT

DIVERGENCE_TXT = REPO_ROOT / 'logs' / 'expectimax_divergence.txt'
DIVERGENCE_JSONL = REPO_ROOT / 'logs' / 'expectimax_divergence.jsonl'
OUTCOME_JSONL = REPO_ROOT / 'logs' / 'expectimax_outcome.jsonl'

_lock = threading.Lock()


@dataclass
class DivergenceStats:
    n_choose: int = 0
    n_phase1_only: int = 0
    n_phase2: int = 0
    n_diverged: int = 0
    n_timeout: int = 0
    n_gated: int = 0
    sum_delta_u: float = 0.0  # phase2_u - phase1_u when diverged
    samples: List[Dict[str, Any]] = field(default_factory=list)
    max_samples: int = 40
    # joint (phase1/chosen) vs greedy_solo factored
    n_solo_cmp: int = 0
    n_neq_solo: int = 0
    samples_solo: List[Dict[str, Any]] = field(default_factory=list)
    # Outcome calibration (game-level + turn-level labels).
    games: int = 0
    games_w: int = 0
    games_with_diverge: int = 0
    games_with_diverge_w: int = 0
    games_no_diverge: int = 0
    games_no_diverge_w: int = 0
    # Turn-level: diverged turns labeled by eventual game win/loss
    div_turns_in_wins: int = 0
    div_turns_in_losses: int = 0
    sum_delta_v_wins: float = 0.0  # bayes V_p2 - V_p1
    sum_delta_v_losses: float = 0.0
    n_delta_v_wins: int = 0
    n_delta_v_losses: int = 0
    # Turns where chosen ≠ solo, labeled by game W/L
    solo_neq_turns_in_wins: int = 0
    solo_neq_turns_in_losses: int = 0
    games_with_solo_neq: int = 0
    games_with_solo_neq_w: int = 0
    games_eq_solo: int = 0
    games_eq_solo_w: int = 0


_STATS = DivergenceStats()


def reset_divergence_stats() -> None:
    global _STATS
    with _lock:
        _STATS = DivergenceStats()


def action_sig(action: Any) -> Optional[Tuple]:
    if action is None:
        return None
    return (
        str(getattr(action, 'kind', 'move') or 'move'),
        int(getattr(action, 'move_slot', 0) or 0),
        int(getattr(action, 'target', 0) or 0),
        int(getattr(action, 'mega', 0) or 0),
        str(getattr(action, 'move_name', '') or '').lower().replace(' ', ''),
    )


def pair_sig(left: Any, right: Any) -> Tuple:
    return (action_sig(left), action_sig(right))


def record_choose(
    *,
    phase: str,
    used_fallback: bool,
    diverged: bool,
    phase1_u: float,
    phase2_u: float,
    phase1_sig: Tuple,
    phase2_sig: Tuple,
    elapsed_ms: float,
    gated: bool = False,
) -> None:
    """Incrementa contadores in-process (workers async = mismo proceso)."""
    with _lock:
        st = _STATS
        st.n_choose += 1
        timed_out = '+timeout' in str(phase)
        if timed_out:
            st.n_timeout += 1
        if gated or 'gated' in str(phase):
            st.n_gated += 1
        base_phase = str(phase).split('+')[0]
        if used_fallback or base_phase == 'phase1':
            st.n_phase1_only += 1
        else:
            st.n_phase2 += 1
        if diverged:
            st.n_diverged += 1
            st.sum_delta_u += float(phase2_u) - float(phase1_u)
            if len(st.samples) < st.max_samples:
                st.samples.append({
                    'phase': phase,
                    'phase1_u': round(float(phase1_u), 3),
                    'phase2_u': round(float(phase2_u), 3),
                    'delta_u': round(float(phase2_u) - float(phase1_u), 3),
                    'phase1': phase1_sig,
                    'phase2': phase2_sig,
                    'elapsed_ms': round(float(elapsed_ms), 1),
                    'gated': bool(gated),
                })


def record_solo_compare(
    *,
    chosen_sig: Tuple,
    solo_sig: Tuple,
    joint_u: float,
    solo_u: float,
) -> None:
    """Compara la acción elegida (joint/expectimax) vs greedy_solo factored."""
    with _lock:
        st = _STATS
        st.n_solo_cmp += 1
        neq = chosen_sig != solo_sig
        if neq:
            st.n_neq_solo += 1
            if len(st.samples_solo) < st.max_samples:
                st.samples_solo.append({
                    'chosen': chosen_sig,
                    'solo': solo_sig,
                    'joint_u': round(float(joint_u), 3),
                    'solo_u': round(float(solo_u), 3),
                    'delta_u': round(float(joint_u) - float(solo_u), 3),
                })


def record_battle_outcome(
    won: bool,
    decisions: Sequence[Dict[str, Any]],
) -> None:
    """Cierra una partida: asocia divergences del turn con W/L."""
    decisions = list(decisions or [])
    n_div = sum(1 for d in decisions if d.get('diverged'))
    n_solo_neq = sum(1 for d in decisions if d.get('neq_solo'))
    with _lock:
        st = _STATS
        st.games += 1
        if won:
            st.games_w += 1
        if n_div > 0:
            st.games_with_diverge += 1
            if won:
                st.games_with_diverge_w += 1
        else:
            st.games_no_diverge += 1
            if won:
                st.games_no_diverge_w += 1
        if n_solo_neq > 0:
            st.games_with_solo_neq += 1
            if won:
                st.games_with_solo_neq_w += 1
        else:
            st.games_eq_solo += 1
            if won:
                st.games_eq_solo_w += 1
        for d in decisions:
            if d.get('neq_solo'):
                if won:
                    st.solo_neq_turns_in_wins += 1
                else:
                    st.solo_neq_turns_in_losses += 1
            if not d.get('diverged'):
                continue
            dv = float(d.get('delta_v', 0.0) or 0.0)
            if won:
                st.div_turns_in_wins += 1
                st.sum_delta_v_wins += dv
                st.n_delta_v_wins += 1
            else:
                st.div_turns_in_losses += 1
                st.sum_delta_v_losses += dv
                st.n_delta_v_losses += 1
    try:
        OUTCOME_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with OUTCOME_JSONL.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps({
                'won': bool(won),
                'n_decisions': len(decisions),
                'n_diverged': n_div,
                'n_neq_solo': n_solo_neq,
                'n_gated': sum(1 for d in decisions if d.get('gated')),
                'mean_delta_v_div': (
                    sum(float(d.get('delta_v') or 0) for d in decisions if d.get('diverged'))
                    / n_div
                ) if n_div else 0.0,
            }, ensure_ascii=False) + '\n')
    except Exception:
        pass


def snapshot_stats() -> Dict[str, Any]:
    with _lock:
        st = _STATS
        n_div = st.n_diverged
        n2 = st.n_phase2
        g_div = st.games_with_diverge
        g_nod = st.games_no_diverge
        return {
            'n_choose': st.n_choose,
            'n_phase1_only': st.n_phase1_only,
            'n_phase2': st.n_phase2,
            'n_diverged': st.n_diverged,
            'n_timeout': st.n_timeout,
            'n_gated': st.n_gated,
            'diverge_rate_all': (n_div / st.n_choose) if st.n_choose else 0.0,
            'diverge_rate_phase2': (n_div / n2) if n2 else 0.0,
            'gate_rate': (st.n_gated / st.n_choose) if st.n_choose else 0.0,
            'mean_delta_u_diverged': (st.sum_delta_u / n_div) if n_div else 0.0,
            'samples': list(st.samples),
            'games': st.games,
            'games_wr': (st.games_w / st.games) if st.games else 0.0,
            'games_with_diverge': g_div,
            'wr_with_diverge': (st.games_with_diverge_w / g_div) if g_div else 0.0,
            'games_no_diverge': g_nod,
            'wr_no_diverge': (st.games_no_diverge_w / g_nod) if g_nod else 0.0,
            'div_turns_in_wins': st.div_turns_in_wins,
            'div_turns_in_losses': st.div_turns_in_losses,
            'mean_delta_v_div_wins': (
                st.sum_delta_v_wins / st.n_delta_v_wins
            ) if st.n_delta_v_wins else 0.0,
            'mean_delta_v_div_losses': (
                st.sum_delta_v_losses / st.n_delta_v_losses
            ) if st.n_delta_v_losses else 0.0,
            'n_solo_cmp': st.n_solo_cmp,
            'n_neq_solo': st.n_neq_solo,
            'solo_neq_rate': (st.n_neq_solo / st.n_solo_cmp) if st.n_solo_cmp else 0.0,
            'samples_solo': list(st.samples_solo),
            'solo_neq_turns_in_wins': st.solo_neq_turns_in_wins,
            'solo_neq_turns_in_losses': st.solo_neq_turns_in_losses,
            'games_with_solo_neq': st.games_with_solo_neq,
            'wr_with_solo_neq': (
                st.games_with_solo_neq_w / st.games_with_solo_neq
            ) if st.games_with_solo_neq else 0.0,
            'games_eq_solo': st.games_eq_solo,
            'wr_eq_solo': (
                st.games_eq_solo_w / st.games_eq_solo
            ) if st.games_eq_solo else 0.0,
        }


def format_divergence_summary(stats: Optional[Dict[str, Any]] = None) -> List[str]:
    s = stats if stats is not None else snapshot_stats()
    n = int(s.get('n_choose') or 0)
    if n <= 0:
        return [
            '--- expectimax divergence (phase1 vs phase2) ---',
            'divergence: no expectimax chooses recorded',
        ]
    lines = [
        '--- expectimax divergence (phase1 vs phase2) ---',
        f'div_n_choose={n}',
        f'div_phase1_only={s.get("n_phase1_only", 0)} '
        f'div_phase2={s.get("n_phase2", 0)} '
        f'div_diverged={s.get("n_diverged", 0)} '
        f'div_gated={s.get("n_gated", 0)} '
        f'div_timeout={s.get("n_timeout", 0)}',
        f'div_rate_all={100.0 * float(s.get("diverge_rate_all") or 0):.1f}% '
        f'div_rate_phase2={100.0 * float(s.get("diverge_rate_phase2") or 0):.1f}% '
        f'gate_rate={100.0 * float(s.get("gate_rate") or 0):.1f}%',
        f'div_mean_delta_u={float(s.get("mean_delta_u_diverged") or 0):+.2f} '
        f'(phase2_u - phase1_u | diverged only)',
    ]
    n_solo = int(s.get('n_solo_cmp') or 0)
    if n_solo > 0:
        lines.extend([
            '--- joint/chosen vs greedy_solo ---',
            f'solo_cmp={n_solo} neq={s.get("n_neq_solo", 0)} '
            f'neq_rate={100.0 * float(s.get("solo_neq_rate") or 0):.1f}%',
            f'solo_neq_turns W={s.get("solo_neq_turns_in_wins")} '
            f'L={s.get("solo_neq_turns_in_losses")}',
            f'wr_with_solo_neq={100.0 * float(s.get("wr_with_solo_neq") or 0):.1f}% '
            f'(n={s.get("games_with_solo_neq")}) '
            f'wr_eq_solo={100.0 * float(s.get("wr_eq_solo") or 0):.1f}% '
            f'(n={s.get("games_eq_solo")})',
        ])
        solo_samples = list(s.get('samples_solo') or [])
        if solo_samples:
            lines.append(f'solo_neq_samples (first {min(8, len(solo_samples))}):')
            for i, row in enumerate(solo_samples[:8], start=1):
                lines.append(
                    f'  {i}. Δu={row.get("delta_u"):+.1f} '
                    f'chosen={row.get("chosen")} solo={row.get("solo")}'
                )
    if int(s.get('games') or 0) > 0:
        lines.extend([
            '--- outcome calibration (V vs W/L) ---',
            f'out_games={s.get("games")} wr={100.0 * float(s.get("games_wr") or 0):.1f}%',
            f'out_wr_with_diverge={100.0 * float(s.get("wr_with_diverge") or 0):.1f}% '
            f'(n={s.get("games_with_diverge")}) '
            f'out_wr_no_diverge={100.0 * float(s.get("wr_no_diverge") or 0):.1f}% '
            f'(n={s.get("games_no_diverge")})',
            f'out_div_turns W={s.get("div_turns_in_wins")} L={s.get("div_turns_in_losses")}',
            f'out_mean_delta_v_div_W={float(s.get("mean_delta_v_div_wins") or 0):+.2f} '
            f'out_mean_delta_v_div_L={float(s.get("mean_delta_v_div_losses") or 0):+.2f}',
        ])
    samples = list(s.get('samples') or [])
    if samples:
        lines.append(f'div_samples (first {len(samples)}):')
        for i, row in enumerate(samples[:12], start=1):
            lines.append(
                f'  {i}. Δu={row.get("delta_u"):+.1f} '
                f'p1={row.get("phase1")} → p2={row.get("phase2")} '
                f'ms={row.get("elapsed_ms")}'
            )
    return lines


def write_divergence_summary(
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Escribe txt + append jsonl; return path del txt."""
    stats = snapshot_stats()
    if extra:
        stats = {**stats, **extra}
    DIVERGENCE_TXT.parent.mkdir(parents=True, exist_ok=True)
    lines = format_divergence_summary(stats)
    DIVERGENCE_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    payload = {
        **{k: v for k, v in stats.items() if k != 'samples'},
        'n_samples_kept': len(stats.get('samples') or []),
    }
    with DIVERGENCE_JSONL.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + '\n')
    return DIVERGENCE_TXT
