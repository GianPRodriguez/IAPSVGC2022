"""Expectimax bayesiano 1-ply anytime (filtro Greedy-Joint Top-K × Top-K).

Arquitectura A: phase1=joint; phase2 solo sobre Top-K×Top-K.
Arquitectura C: daño = EV (expected_damage), sin ramificar rolls.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from showdown.battle_bots.greedy_joint.joint import (
    JointAtom,
    _atoms_from_side,
    _norm_move,
    _score_pair,
    _strip_mega,
    best_joint_double_action,
)
from showdown.battle_bots.greedy_solo.calc_async import calc_damage_batch
from showdown.battle_bots.greedy_solo.greedy import (
    CalcPoke,
    FieldContext,
    MoveOption,
    ScoredAction,
    _collect_ally_hit_requests,
    _collect_side_requests,
    _stats_or_default,
    apply_pending_mega,
    best_double_action,
)
from showdown.battle_bots.greedy_solo.type_chart import matchup_vs_actives
from showdown.battle_bots.expectimax_bayesian.belief import (
    belief_nodes_for_foes,
    iter_worlds,
    profile_to_calc,
    resolve_opp_move_names,
)
from showdown.battle_bots.expectimax_bayesian.residual import project_residual_into_v
from showdown.battle_bots.expectimax_bayesian.chance import (
    SOFT_KO_BONUS,
    combined_expected_damage,
    combined_p_ko,
    expected_damage,
    p_ko,
)
from showdown.battle_bots.expectimax_bayesian.divergence import (
    pair_sig,
    record_choose,
    record_solo_compare,
)
from showdown.battle_bots.expectimax_bayesian.opp_policy import (
    OppAction,
    emit_pred,
    is_control_move,
    is_switch_action,
    move_options_from_names,
    opp_policy_distribution,
    opp_policy_targeted,
)
from showdown.battle_bots.expectimax_bayesian.speed_order import (
    acts_before,
    move_prio,
    poke_effective_spe,
    race_scale,
)
from showdown.battle_bots.expectimax_bayesian.utility import (
    FieldPositionBonus,
    JointActionUtility,
    field_control_bonus,
    switch_positioning_value,
    switch_preservation_value,
)


# Cobertura Top-K sin inyección FO (rank = daño greedy). Activo = 75%.
# 100% queda mapeado para activar después vía apply_search_coverage('100').
SEARCH_COVERAGE_PRESETS: Dict[str, Dict[str, float]] = {
    '50': {
        'topk_per_side': 5,
        'max_our_pairs': 25,
        'deadline_ms': 1000.0,
        'phase1_budget_ms': 50.0,
        'phase2_end_ms': 950.0,
    },
    '75': {
        'topk_per_side': 7,
        'max_our_pairs': 49,
        'deadline_ms': 1600.0,
        'phase1_budget_ms': 80.0,
        'phase2_end_ms': 1520.0,
    },
    '100': {
        'topk_per_side': 10,
        'max_our_pairs': 100,
        'deadline_ms': 3000.0,
        'phase1_budget_ms': 100.0,
        'phase2_end_ms': 2900.0,
    },
}
ACTIVE_SEARCH_COVERAGE = '75'

# Overlay opcional para tune autónomo (no toca defaults del dataclass).
# Escrito por `python -m showdown.inference_sim.run_expectimax_tune`.
ACTIVE_OVERRIDES_PATH = Path(__file__).resolve().parent / 'ACTIVE_OVERRIDES.json'


def apply_search_coverage(
    cfg: 'ExpectimaxConfig',
    level: Optional[str] = None,
) -> 'ExpectimaxConfig':
    """Aplica preset de cobertura Top-K (mutates + returns cfg)."""
    key = str(level or ACTIVE_SEARCH_COVERAGE)
    preset = SEARCH_COVERAGE_PRESETS.get(key)
    if not preset:
        raise KeyError(f'unknown search coverage {key!r}; use {sorted(SEARCH_COVERAGE_PRESETS)}')
    for name, val in preset.items():
        setattr(cfg, name, type(getattr(cfg, name))(val))
    return cfg


def read_active_overrides() -> Dict[str, Any]:
    if not ACTIVE_OVERRIDES_PATH.is_file():
        return {}
    try:
        raw = json.loads(ACTIVE_OVERRIDES_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {}
    return dict(raw) if isinstance(raw, dict) else {}


def write_active_overrides(overrides: Dict[str, Any]) -> Path:
    """Persiste overlay; claves desconocidas se ignoran al cargar."""
    clean = {str(k): v for k, v in (overrides or {}).items()}
    ACTIVE_OVERRIDES_PATH.write_text(
        json.dumps(clean, indent=2, ensure_ascii=False, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return ACTIVE_OVERRIDES_PATH


def clear_active_overrides() -> None:
    if ACTIVE_OVERRIDES_PATH.is_file():
        ACTIVE_OVERRIDES_PATH.unlink()


def load_expectimax_config(**kwargs: Any) -> 'ExpectimaxConfig':
    """ExpectimaxConfig efectivo = defaults + ACTIVE_OVERRIDES.json + kwargs."""
    known = {f.name for f in fields(ExpectimaxConfig)}
    base = asdict(ExpectimaxConfig())
    file_ov = {k: v for k, v in read_active_overrides().items() if k in known}
    # Diagnóstico de ablación: EXPECTIMAX_OVERRIDES="opp_policy_control_weight=0.1;opp_policy_greedy_weight=0.9"
    import os as _os
    for item in _os.environ.get('EXPECTIMAX_OVERRIDES', '').split(';'):
        if '=' in item:
            name, value = (x.strip() for x in item.split('=', 1))
            if name not in known:
                raise ValueError(f'EXPECTIMAX_OVERRIDES: campo desconocido {name!r}')
            file_ov[name] = value
    merged = {**base, **file_ov, **{k: v for k, v in kwargs.items() if k in known}}
    # Coerce types from JSON (bool/int/float/str).
    typed: Dict[str, Any] = {}
    for f in fields(ExpectimaxConfig):
        val = merged[f.name]
        default = getattr(ExpectimaxConfig(), f.name)
        if isinstance(default, bool):
            typed[f.name] = bool(val) if not isinstance(val, str) else val.strip().lower() in (
                '1', 'true', 'yes', 'on',
            )
        elif isinstance(default, int) and not isinstance(default, bool):
            typed[f.name] = int(val)
        elif isinstance(default, float):
            typed[f.name] = float(val)
        else:
            typed[f.name] = val
    return ExpectimaxConfig(**typed)


@dataclass
class ExpectimaxConfig:
    # Cobertura 75%: K=7, pairs=49, ~1.6s (sin inyección FO). Ver SEARCH_COVERAGE_PRESETS.
    deadline_ms: float = 1600.0
    phase1_budget_ms: float = 80.0
    phase2_end_ms: float = 1520.0
    k_profiles: int = 2
    max_worlds: int = 4
    # Top-K por daño greedy → K×K; FO entra si rank≤K (75% ≈ K=7).
    topk_per_side: int = 7
    max_our_pairs: int = 49
    soft_ko_bonus: float = SOFT_KO_BONUS
    # Cuevas: off (A/B seed10 teams40: volOn scale=0.25 empeoró pair 50%→33%).
    enable_voluntary_switch: bool = False
    max_switch_hybrids: int = 0
    enable_niche_pairs: bool = False
    loss_weight: float = 650.0
    save_weight: float = 0.0
    chip_soft_cap: float = 30.0
    switch_min_stay_p_ko: float = 0.50
    overkill_waste_weight: float = 140.0
    ko_free_switch_bonus: float = 55.0
    entry_punish_weight: float = 1.15
    threat_weight: float = 0.0
    hp_rel_weight: float = 0.15
    syn_weight: float = 0.0
    condemned_chip_discount: float = 0.80
    condemned_lasting_boost: float = 1.35
    threat_relief_weight: float = 1.0
    entry_ability_weight: float = 1.0
    opp_policy_greedy_weight: float = 1.0
    opp_policy_control_weight: float = 0.0
    opp_policy_switch_weight: float = 0.0
    opp_policy_use_real_damage: bool = True
    opp_policy_split_targets: bool = False
    opp_policy_mirror_solo: bool = False
    # Slots de movimiento distintos por rival (sin esto, los dos rivales se pisan el daño por slot).
    opp_policy_unique_slots: bool = True
    # Fake Out rival: solo en su primer turno en campo; entonces masa ~p (si puede afectar).
    opp_policy_fakeout_rule: bool = True
    opp_policy_fakeout_first_p: float = 0.7
    # V anclada a offense (EV de daño = chance abstraction C).
    v_solo_isomorphic: bool = True
    phase2_retarget_only: bool = False
    # Phase1 = greedy_joint (referencia); phase2 elige entre Top-K×Top-K.
    phase1_use_joint: bool = True
    use_topk_abstraction: bool = True
    phase2_gate_enabled: bool = True
    # 'beat': exigir V_bayes >= V_p1 + margin (ancla a joint; div~0).
    # 'indifference': aceptar si V_bayes >= V_p1 - eps (diverge sin exigir mejora).
    # 'off': phase2 argmax libre (gate_enabled puede quedar True y se ignora).
    phase2_gate_mode: str = 'indifference'
    phase2_gate_margin: float = 80.0
    phase2_indifference_eps: float = 40.0
    # 0 = off. Antes 320 cortaba Fake Out con deny≈soft_KO (delta~700+).
    # La seguridad queda en indifference (no aceptar V << phase1).
    phase2_max_delta_v: float = 0.0
    enable_phase2: bool = True
    # False = cualquier par Top-K puede ganar (objetivo: jugar distinto a greedy).
    phase2_surgical_only: bool = False
    switch_v_scale: float = 0.65


@dataclass
class ExpectimaxMeta:
    used_fallback: bool = True
    worlds: int = 0
    pairs_scored: int = 0
    elapsed_ms: float = 0.0
    phase: str = 'fallback'
    utility: float = 0.0
    errors: List[str] = field(default_factory=list)
    # Diagnostic: phase2 eligió distinto a phase1/joint.
    diverged: bool = False
    phase1_utility: float = 0.0
    phase2_utility: float = 0.0
    phase2_gated: bool = False  # True si phase2 se rechazó por el gate
    phase1_v: float = 0.0  # V bayesiana del par phase1 (mismo espacio que phase2)
    phase2_v: float = 0.0
    # Diagnostic: chosen ≠ greedy_solo factored
    neq_solo: bool = False
    solo_utility: float = 0.0
    # Reporte online (logs/reporte.txt): Top-K candidatos + resumen.
    topk_lines: List[str] = field(default_factory=list)
    decision_summary: str = ''
    decision_why: str = ''
    topk_k: int = 0
    max_pairs: int = 0


@dataclass(frozen=True)
class ReserveOption:
    """Bench candidato a switch voluntario."""

    slot: int  # 3 o 4 (IAP) / índice poke-env mapeado
    poke: CalcPoke
    types: Tuple[Optional[str], ...] = ()


def _elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _speed_control(name: str) -> bool:
    return _norm_move(name) in (
        'tailwind', 'trickroom', 'icywind', 'electroweb', 'bulldoze', 'glaciate',
    )


def _atom_key(a: JointAtom) -> Tuple[int, int, int, str]:
    return (a.move_slot, a.target, a.mega, a.utility)


def _pair_key(l: JointAtom, r: JointAtom) -> Tuple:
    return (_atom_key(l), _atom_key(r))


def _atom_label(a: Optional[JointAtom]) -> str:
    if a is None:
        return '—'
    if a.utility == 'switch':
        return f'switch{int(a.move_slot)}'
    name = str(a.move_name or f'slot{a.move_slot}').strip() or f'slot{a.move_slot}'
    name = name.replace(' ', '')
    mega = {1: '+M', 2: '+T'}.get(int(a.mega or 0), '')
    tgt = f'>{int(a.target)}' if int(a.target or 0) else ''
    return f'{name}{mega}{tgt}'


def _scored_label(a: Optional[ScoredAction]) -> str:
    if a is None:
        return '—'
    kind = str(getattr(a, 'kind', 'move') or 'move')
    if kind == 'switch':
        return f'sw{int(a.move_slot)}'
    return f'slot{int(a.move_slot)}>t{int(getattr(a, "target", 0) or 0)}'


def _fill_decision_report(
    meta: ExpectimaxMeta,
    *,
    pairs: Sequence[Tuple[JointAtom, JointAtom]],
    utilities: Dict,
    best_key,
    phase1_l: Optional[ScoredAction],
    phase1_r: Optional[ScoredAction],
    best_l: Optional[ScoredAction],
    best_r: Optional[ScoredAction],
    cfg: ExpectimaxConfig,
) -> None:
    """Rellena topk_lines + decision_summary para logs/reporte.txt."""
    meta.topk_k = int(cfg.topk_per_side or 0)
    meta.max_pairs = int(cfg.max_our_pairs or 0)
    key_to_pair = {_pair_key(pl, pr): (pl, pr) for pl, pr in pairs}
    ranked = sorted(
        ((k, ju) for k, ju in utilities.items() if getattr(ju, 'samples', 0) > 0),
        key=lambda kv: -float(kv[1].mean),
    )
    lines: List[str] = []
    for i, (key, ju) in enumerate(ranked[: max(1, int(cfg.max_our_pairs or 12))], 1):
        pl, pr = key_to_pair.get(key, (None, None))
        mark = '>' if key == best_key else ' '
        lines.append(
            f'{mark}{i:2d}. V={float(ju.mean):8.1f}  '
            f'L={_atom_label(pl)}  R={_atom_label(pr)}'
        )
    meta.topk_lines = lines
    if best_key is not None and best_key in key_to_pair:
        bl, br = key_to_pair[best_key]
        chosen = f'{_atom_label(bl)} | {_atom_label(br)}'
    else:
        chosen = f'{_scored_label(best_l)} | {_scored_label(best_r)}'
    p1 = f'{_scored_label(phase1_l)} | {_scored_label(phase1_r)}'
    # Mejorar phase1 si hay match en utilities/pairs
    for pl, pr in pairs:
        if _scored_matches_atom(phase1_l, pl) and _scored_matches_atom(phase1_r, pr):
            p1 = f'{_atom_label(pl)} | {_atom_label(pr)}'
            break
    gate = 'YES' if meta.phase2_gated else 'NO'
    div = 'YES' if meta.diverged else 'NO'
    meta.decision_summary = (
        f'phase={meta.phase}  diverged={div}  gated={gate}  '
        f'K={meta.topk_k} pairs={meta.pairs_scored}/{meta.max_pairs}  '
        f'ms={meta.elapsed_ms:.0f}\n'
        f'  phase1 (joint): {p1}  V={meta.phase1_v:.1f}\n'
        f'  chosen:         {chosen}  V={meta.utility:.1f}  '
        f'(phase2_best V={meta.phase2_v:.1f})'
    )

    # Por qué esa combinación: margen vs runner-up + gate/phase1.
    why: List[str] = []
    why.append(
        f'criterio: argmax V bayesiana 1-ply sobre Top-K×Top-K '
        f'(K={meta.topk_k}, max_pairs={meta.max_pairs})'
    )
    if ranked:
        top_v = float(ranked[0][1].mean)
        if len(ranked) >= 2:
            second_v = float(ranked[1][1].mean)
            margin = top_v - second_v
            pl2, pr2 = key_to_pair.get(ranked[1][0], (None, None))
            why.append(
                f'mejor V={top_v:.1f}  vs 2º {_atom_label(pl2)}|{_atom_label(pr2)} '
                f'V={second_v:.1f}  (margen={margin:+.1f})'
            )
        else:
            why.append(f'unico par con samples; V={top_v:.1f}')
    if meta.phase2_gated:
        why.append(
            f'gate rechazó phase2_best (V={meta.phase2_v:.1f}); '
            f'se conserva phase1 joint ({p1})'
        )
    elif meta.diverged:
        why.append(
            f'divergió de phase1 ({p1} V={meta.phase1_v:.1f}) → '
            f'chosen ({chosen} V={meta.utility:.1f})'
        )
    elif meta.used_fallback or str(meta.phase).startswith('phase1'):
        why.append(f'sin refine phase2 útil; floor phase1/joint ({chosen})')
    else:
        why.append(f'phase2 confirmó/eligió ({chosen}); phase={meta.phase}')
    if meta.errors:
        why.append(f'notas: {"; ".join(meta.errors[:3])}')
    meta.decision_why = '\n'.join(why)


def _scored_matches_atom(scored: Optional[ScoredAction], atom: JointAtom) -> bool:
    """¿ScoredAction de phase1 corresponde a este JointAtom de phase2?"""
    if scored is None:
        # Sin acción en ese lado (faint/pass): no exigimos match estricto.
        return True
    kind = str(getattr(scored, 'kind', 'move') or 'move')
    if kind == 'switch' or atom.utility == 'switch':
        return kind == 'switch' and atom.utility == 'switch' and int(scored.move_slot) == int(atom.move_slot)
    return (
        atom.utility != 'switch'
        and int(scored.move_slot) == int(atom.move_slot)
        and int(getattr(scored, 'target', 0) or 0) == int(atom.target or 0)
        and int(getattr(scored, 'mega', 0) or 0) == int(atom.mega or 0)
    )


def _phase1_bayes_v(
    utilities: Dict[Tuple, JointActionUtility],
    pairs: Sequence[Tuple[JointAtom, JointAtom]],
    phase1_l: Optional[ScoredAction],
    phase1_r: Optional[ScoredAction],
) -> Optional[float]:
    """V bayesiana del par phase1 en el mismo espacio que el best de phase2."""
    for pl, pr in pairs:
        if not (_scored_matches_atom(phase1_l, pl) and _scored_matches_atom(phase1_r, pr)):
            continue
        ju = utilities.get(_pair_key(pl, pr))
        if ju is not None and ju.samples > 0:
            return float(ju.mean)
    # Match flojo: solo slot+mega (target a veces difiere entre joint/atoms).
    for pl, pr in pairs:
        ok_l = phase1_l is None or (
            int(phase1_l.move_slot) == int(pl.move_slot)
            and int(getattr(phase1_l, 'mega', 0) or 0) == int(pl.mega or 0)
            and (
                (getattr(phase1_l, 'kind', 'move') == 'switch') == (pl.utility == 'switch')
            )
        )
        ok_r = phase1_r is None or (
            int(phase1_r.move_slot) == int(pr.move_slot)
            and int(getattr(phase1_r, 'mega', 0) or 0) == int(pr.mega or 0)
            and (
                (getattr(phase1_r, 'kind', 'move') == 'switch') == (pr.utility == 'switch')
            )
        )
        if not (ok_l and ok_r):
            continue
        ju = utilities.get(_pair_key(pl, pr))
        if ju is not None and ju.samples > 0:
            return float(ju.mean)
    return None


def _is_protect_move(name: str) -> bool:
    return _norm_move(name) in (
        'protect', 'detect', 'spikyshield', 'banefulbunker', 'silktrap',
        'burningbulwark', 'obstruct',
    )


def _is_niche_atom(atom: JointAtom) -> bool:
    """Cuevas ortogonales a greedy daño: Protect / pivot / speed / priority."""
    if atom.utility in ('switch', 'protect'):
        return True
    if _is_protect_move(atom.move_name):
        return True
    if _speed_control(atom.move_name):
        return True
    try:
        if int(move_prio(atom.move_name) or 0) > 0:
            return True
    except Exception:
        pass
    return False


def _is_surgical_pair(pl: JointAtom, pr: JointAtom) -> bool:
    """Diverge phase2 solo en cuevas teacher (Protect / switch / speed / prio)."""
    return _is_niche_atom(pl) or _is_niche_atom(pr)


def _switch_atom(slot: int) -> JointAtom:
    return JointAtom(
        move_slot=int(slot),
        target=0,
        mega=0,
        utility='switch',
        move_name=f'Switch {slot}',
    )


def _retarget_pairs_only(
    pairs: Sequence[Tuple[JointAtom, JointAtom]],
    phase1_l: Optional[ScoredAction],
    phase1_r: Optional[ScoredAction],
) -> List[Tuple[JointAtom, JointAtom]]:
    """Hipótesis C: conservar move slots de phase1; solo variar targets."""
    out: List[Tuple[JointAtom, JointAtom]] = []
    for pl, pr in pairs:
        if pl.utility == 'switch' or pr.utility == 'switch':
            continue
        ok_l = phase1_l is None or (
            int(pl.move_slot) == int(phase1_l.move_slot)
            and int(pl.mega or 0) == int(getattr(phase1_l, 'mega', 0) or 0)
        )
        ok_r = phase1_r is None or (
            int(pr.move_slot) == int(phase1_r.move_slot)
            and int(pr.mega or 0) == int(getattr(phase1_r, 'mega', 0) or 0)
        )
        if ok_l and ok_r:
            out.append((pl, pr))
    return out


def _candidate_pairs(
    left_atoms: Sequence[JointAtom],
    right_atoms: Sequence[JointAtom],
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    max_pairs: int,
    switch_left: Sequence[JointAtom] = (),
    switch_right: Sequence[JointAtom] = (),
    max_switch_hybrids: int = 8,
    left_poke: Optional[CalcPoke] = None,
    right_poke: Optional[CalcPoke] = None,
    topk_per_side: int = 0,
    enable_niche_pairs: bool = True,
) -> List[Tuple[JointAtom, JointAtom]]:
    """Arquitectura A: si topk_per_side>0, solo Top-K × Top-K (no todo el producto)."""
    move_left = [a for a in left_atoms if a.utility != 'switch']
    move_right = [a for a in right_atoms if a.utility != 'switch']

    if not move_left and not move_right and not switch_left and not switch_right:
        return []

    k = int(topk_per_side or 0)
    if k > 0 and move_left and move_right:
        # Top-K por lado: daño + utilidad estructurada (Fake Out deny, Knock Off item).
        def _rank_key(a: JointAtom) -> float:
            return float(a.total_damage_pct) + float(getattr(a, 'effect_bonus', 0.0) or 0.0)

        left_ranked = sorted(move_left, key=_rank_key, reverse=True)[:k]
        right_ranked = sorted(move_right, key=_rank_key, reverse=True)[:k]
        scored: List[Tuple[float, JointAtom, JointAtom]] = []
        for a1 in left_ranked:
            for a2 in right_ranked:
                pl, pr = a1, a2
                if a1.mega and a2.mega:
                    if a1.total_damage_pct >= a2.total_damage_pct:
                        pr = _strip_mega(a2)
                    else:
                        pl = _strip_mega(a1)
                s = _score_pair(
                    pl, pr, foes,
                    left_poke=left_poke, right_poke=right_poke,
                )
                scored.append((s, pl, pr))
        # Sin inyección FO: entra solo si rank≤K (cobertura via topk_per_side).
        # Cuevas: Protect/speed/prio no entran al Top-K por daño → inyectar
        # híbridos niche×top (presupuesto acotado anti-timeout).
        if enable_niche_pairs:
            niches_l = [a for a in move_left if _is_niche_atom(a)][:4]
            niches_r = [a for a in move_right if _is_niche_atom(a)][:4]
            for nl in niches_l:
                for a2 in right_ranked[:2]:
                    s = _score_pair(
                        nl, a2, foes,
                        left_poke=left_poke, right_poke=right_poke,
                    )
                    scored.append((s, nl, a2))
            for nr in niches_r:
                for a1 in left_ranked[:2]:
                    s = _score_pair(
                        a1, nr, foes,
                        left_poke=left_poke, right_poke=right_poke,
                    )
                    scored.append((s, a1, nr))
        scored.sort(key=lambda x: -x[0])
        out: List[Tuple[JointAtom, JointAtom]] = []
        seen = set()
        for _, pl, pr in scored:
            key = _pair_key(pl, pr)
            if key in seen:
                continue
            seen.add(key)
            out.append((pl, pr))
            if len(out) >= max(1, int(max_pairs)):
                break
        # Switches: híbridos acotados
        hybrids = 0
        pad = JointAtom(0, 0, utility='status')
        top_r = list(right_ranked[:2]) or [pad]
        top_l = list(left_ranked[:2]) or [pad]
        for sw in switch_left:
            for atk in top_r:
                key = _pair_key(sw, atk)
                if key in seen:
                    continue
                seen.add(key)
                out.append((sw, atk))
                hybrids += 1
                if hybrids >= max_switch_hybrids or len(out) >= max(1, int(max_pairs)):
                    break
            if hybrids >= max_switch_hybrids or len(out) >= max(1, int(max_pairs)):
                break
        if hybrids < max_switch_hybrids and len(out) < max(1, int(max_pairs)):
            for sw in switch_right:
                for atk in top_l:
                    key = _pair_key(atk, sw)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((atk, sw))
                    hybrids += 1
                    if hybrids >= max_switch_hybrids or len(out) >= max(1, int(max_pairs)):
                        break
                if hybrids >= max_switch_hybrids or len(out) >= max(1, int(max_pairs)):
                    break
        return out[: max(1, int(max_pairs))]

    scored = []
    if move_left and move_right:
        for a1 in move_left:
            for a2 in move_right:
                pl, pr = a1, a2
                if a1.mega and a2.mega:
                    if a1.total_damage_pct >= a2.total_damage_pct:
                        pr = _strip_mega(a2)
                    else:
                        pl = _strip_mega(a1)
                s = _score_pair(
                    pl, pr, foes,
                    left_poke=left_poke, right_poke=right_poke,
                )
                scored.append((s, pl, pr))
    elif move_left:
        for a1 in move_left:
            scored.append((a1.total_damage_pct, a1, JointAtom(0, 0, utility='status')))
    elif move_right:
        for a2 in move_right:
            scored.append((a2.total_damage_pct, JointAtom(0, 0, utility='status'), a2))

    scored.sort(key=lambda x: -x[0])
    budget = max(1, int(max_pairs) - int(max_switch_hybrids))
    out = []
    seen = set()
    for _, pl, pr in scored:
        key = _pair_key(pl, pr)
        if key in seen:
            continue
        seen.add(key)
        out.append((pl, pr))
        if len(out) >= budget:
            break

    top_l: List[JointAtom] = []
    top_r: List[JointAtom] = []
    for _, pl, pr in scored[:6]:
        if pl.utility != 'switch' and pl.move_slot > 0 and len(top_l) < 2:
            top_l.append(pl)
        if pr.utility != 'switch' and pr.move_slot > 0 and len(top_r) < 2:
            top_r.append(pr)
    if not top_l:
        top_l = list(move_left[:2])
    if not top_r:
        top_r = list(move_right[:2])

    hybrids = 0
    pad = JointAtom(0, 0, utility='status')
    for sw in switch_left:
        for atk in (top_r or [pad]):
            key = _pair_key(sw, atk)
            if key in seen:
                continue
            seen.add(key)
            out.append((sw, atk))
            hybrids += 1
            if hybrids >= max_switch_hybrids:
                break
        if hybrids >= max_switch_hybrids:
            break
    if hybrids < max_switch_hybrids:
        for sw in switch_right:
            for atk in (top_l or [pad]):
                key = _pair_key(atk, sw)
                if key in seen:
                    continue
                seen.add(key)
                out.append((atk, sw))
                hybrids += 1
                if hybrids >= max_switch_hybrids:
                    break
            if hybrids >= max_switch_hybrids:
                break
    return out[: max(1, int(max_pairs))]


def _hit_map_from_batch(
    meta: Sequence[Tuple[str, MoveOption, int, CalcPoke, int]],
    ranges: Sequence[Optional[Tuple[int, int]]],
    *,
    attackers: Optional[Dict[str, Optional[CalcPoke]]] = None,
    weather: Optional[str] = None,
) -> Dict[Tuple[str, int, int, int], Optional[Tuple[int, int]]]:
    from showdown.battle_bots.move_mechanics import gate_damage_range

    out: Dict[Tuple[str, int, int, int], Optional[Tuple[int, int]]] = {}
    atk_map = attackers or {}
    for (side, move, tid, _foe, special), rng in zip(meta, ranges):
        atk = atk_map.get(side)
        gated = gate_damage_range(
            rng,
            move.name,
            weather=weather,
            item=atk.item if atk is not None else None,
            already_charging=bool(atk.is_two_turn_charging) if atk is not None else False,
            must_recharge=bool(atk.must_recharge) if atk is not None else False,
        )
        out[(side, move.slot, tid, special)] = gated
    return out


def _atom_ranges_on_foe(
    side: str,
    atom: JointAtom,
    tid: int,
    hit_map: Dict[Tuple[str, int, int, int], Optional[Tuple[int, int]]],
) -> Optional[Tuple[int, int]]:
    if atom.utility == 'switch' or atom.move_slot <= 0:
        return None
    if atom.utility not in ('damage',) and not atom.damage_pct and not atom.max_roll:
        return None
    special = 2 if atom.mega == 2 else 0
    rng = hit_map.get((side, atom.move_slot, tid, special))
    if rng is None and special == 2:
        rng = hit_map.get((side, atom.move_slot, tid, 0))
    if rng is not None:
        return rng
    mx = int(atom.max_roll.get(tid, 0))
    if mx > 0:
        return (max(1, int(round(mx * 0.85))), mx)
    return None


def _our_pair_offense_detail(
    left: JointAtom,
    right: JointAtom,
    foes: Sequence[Tuple[int, CalcPoke]],
    hit_map: Dict[Tuple[str, int, int, int], Optional[Tuple[int, int]]],
    *,
    soft_ko_bonus: float,
) -> Tuple[float, float, Dict[int, float], Dict[str, float]]:
    """
    Returns offense, mean_foe_hp_after, p_ko_by_foe_tid, offense_share_by_side.
    """
    score = 0.0
    foe_remain = 0.0
    n_foe = 0
    p_ko_foe: Dict[int, float] = {}
    share = {'L': 0.0, 'R': 0.0}
    for tid, foe in foes:
        max_hp = int((_stats_or_default(foe).get('hp') or 1))
        cur_hp = int(foe.cur_hp if foe.cur_hp is not None else max_hp)
        ranges: List[Optional[Tuple[int, int]]] = []
        side_ranges: Dict[str, List[Optional[Tuple[int, int]]]] = {'L': [], 'R': []}
        for side, atom in (('L', left), ('R', right)):
            rng = _atom_ranges_on_foe(side, atom, tid, hit_map)
            if rng is not None:
                ranges.append(rng)
                side_ranges[side].append(rng)

        remain = max(0.0, 100.0 * cur_hp / max(max_hp, 1))
        if foe.has_substitute:
            e_dmg = combined_expected_damage(ranges)
            pct = 100.0 * e_dmg / float(max_hp) if max_hp else 0.0
            score += min(pct, remain)
            p_ko_foe[tid] = 0.0
            foe_remain += remain
            n_foe += 1
            continue

        e_dmg = combined_expected_damage(ranges)
        pk = combined_p_ko(ranges, cur_hp)
        pct = 100.0 * e_dmg / float(max_hp) if max_hp else 0.0
        part = min(pct, remain) + pk * soft_ko_bonus
        score += part
        p_ko_foe[tid] = pk
        foe_remain += remain * (1.0 - pk)
        n_foe += 1
        # Rough share by side expected dmg
        for side in ('L', 'R'):
            share[side] += combined_expected_damage(side_ranges[side])

    mean_foe = foe_remain / float(n_foe) if n_foe else 0.0
    return score, mean_foe, p_ko_foe, share


def _opp_threat_detail(
    our_left: Optional[CalcPoke],
    our_right: Optional[CalcPoke],
    left_atom: JointAtom,
    right_atom: JointAtom,
    opp_hit_map: Dict[Tuple[str, str, str], Optional[Tuple[int, int]]],
    opp_dists: Dict[str, List[Tuple[OppAction, float]]],
    *,
    soft_ko_bonus: float,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Returns threat_total, p_ko_on_side (OL/OR), e_dmg_pct_on_side.
    Switch/Protect → 0 amenaza sobre ese activo.
    """
    threat = 0.0
    p_ko_side: Dict[str, float] = {'OL': 0.0, 'OR': 0.0}
    pct_side: Dict[str, float] = {'OL': 0.0, 'OR': 0.0}
    sides = [
        ('OL', our_left, left_atom),
        ('OR', our_right, right_atom),
    ]
    for side_key, our, our_atom in sides:
        if our is None:
            continue
        if our_atom.utility in ('switch', 'protect') or our.is_protected:
            continue
        max_hp = int((_stats_or_default(our).get('hp') or 1))
        cur_hp = int(our.cur_hp if our.cur_hp is not None else max_hp)
        dist = opp_dists.get(side_key) or []
        side_threat = 0.0
        side_pk = 0.0
        side_pct = 0.0
        for action, p_b in dist:
            if is_control_move(action.move_name) or is_switch_action(action):
                continue
            best_rng = None
            for atk_tag in ('F1', 'F2'):
                rng = opp_hit_map.get((atk_tag, _norm_move(action.move_name), side_key))
                if rng is not None:
                    best_rng = rng
                    break
            if best_rng is None:
                continue
            # Respetar target si la política lo fijó (1=OL, 2=OR).
            if int(action.target) in (1, 2):
                want = 1 if side_key == 'OL' else 2
                if int(action.target) != want:
                    continue
            e_pct = 100.0 * expected_damage(best_rng) / float(max_hp) if max_hp else 0.0
            pk = p_ko(best_rng, cur_hp)
            contrib = float(p_b) * (min(e_pct, 100.0 * cur_hp / max(max_hp, 1)) + pk * soft_ko_bonus)
            side_threat += contrib
            side_pk += float(p_b) * pk
            side_pct += float(p_b) * e_pct
        threat += side_threat
        p_ko_side[side_key] = min(1.0, side_pk)
        pct_side[side_key] = side_pct
    return threat, p_ko_side, pct_side


def _threat_by_foe_tid(
    world_foes: Sequence[Tuple[int, CalcPoke]],
    our_left: Optional[CalcPoke],
    our_right: Optional[CalcPoke],
    left_atom: JointAtom,
    right_atom: JointAtom,
    opp_hit_map: Dict[Tuple[str, str, str], Optional[Tuple[int, int]]],
    opp_dists: Dict[str, List[Tuple[OppAction, float]]],
    *,
    soft_ko_bonus: float,
) -> Dict[int, float]:
    """Amenaza atribuible a cada foe (F1→tid0, F2→tid1) bajo π_opp actual."""
    out: Dict[int, float] = {}
    tag_for_tid = {
        int(tid): ('F1' if i == 0 else 'F2')
        for i, (tid, _) in enumerate(world_foes)
    }
    sides = [
        ('OL', our_left, left_atom),
        ('OR', our_right, right_atom),
    ]
    for tid, foe_tag in tag_for_tid.items():
        threat_f = 0.0
        for side_key, our, our_atom in sides:
            if our is None:
                continue
            if our_atom.utility in ('switch', 'protect') or our.is_protected:
                continue
            max_hp = int((_stats_or_default(our).get('hp') or 1))
            cur_hp = int(our.cur_hp if our.cur_hp is not None else max_hp)
            for action, p_b in (opp_dists.get(side_key) or []):
                if is_control_move(action.move_name) or is_switch_action(action):
                    continue
                if int(action.target) in (1, 2):
                    want = 1 if side_key == 'OL' else 2
                    if int(action.target) != want:
                        continue
                rng = opp_hit_map.get((foe_tag, _norm_move(action.move_name), side_key))
                if rng is None:
                    continue
                e_pct = 100.0 * expected_damage(rng) / float(max_hp) if max_hp else 0.0
                pk = p_ko(rng, cur_hp)
                threat_f += float(p_b) * (
                    min(e_pct, 100.0 * cur_hp / max(max_hp, 1)) + pk * soft_ko_bonus
                )
        out[int(tid)] = float(threat_f)
    return out


_FLINCH_IMMUNE_ABILITIES = frozenset({'innerfocus', 'shielddust'})
_ABILITY_IGNORING = frozenset({'moldbreaker', 'teravolt', 'turboblaze'})


def _flinch_immune(attacker: Optional[CalcPoke], foe: CalcPoke) -> bool:
    """El foe no se achanta: Inner Focus / Shield Dust (salvo Mold Breaker), Covert Cloak o Substitute."""
    if bool(getattr(foe, 'has_substitute', False)):
        return True
    if _norm_move(getattr(foe, 'item', None) or '') == 'covertcloak':
        return True
    atk_abi = _norm_move(getattr(attacker, 'ability', None) or '') if attacker is not None else ''
    if atk_abi in _ABILITY_IGNORING:
        return False
    return _norm_move(getattr(foe, 'ability', None) or '') in _FLINCH_IMMUNE_ABILITIES


def _foe_beats_fakeout(
    attacker: Optional[CalcPoke],
    foe: CalcPoke,
    foe_moves: Sequence[str],
    *,
    trick_room: bool,
    our_tailwind: bool,
    opp_tailwind: bool,
) -> bool:
    """True si el foe actúa antes (o a la vez) con su propio Fake Out (+3): el nuestro llega tarde.

    Solo cuenta si el foe está en su primer turno y conoce Fake Out; empate de Spe = no seguro.
    """
    if attacker is None or not bool(getattr(foe, 'just_switched', False)):
        return False
    if not any(_norm_move(n) == 'fakeout' for n in (foe_moves or ())):
        return False
    return not acts_before(
        poke_effective_spe(attacker, tailwind=our_tailwind), 3,
        poke_effective_spe(foe, tailwind=opp_tailwind), 3,
        trick_room=trick_room,
    )


def _fakeout_fails_by_tid(
    attacker: Optional[CalcPoke],
    world_foes: Sequence[Tuple[int, CalcPoke]],
    *,
    terrain: Optional[str],
    gravity: bool = False,
    known_moves: Optional[Dict[int, Sequence[str]]] = None,
    trick_room: bool = False,
    our_tailwind: bool = False,
    opp_tailwind: bool = False,
) -> Dict[int, bool]:
    """True si Fake Out no anularía a ese foe: Psychic Terrain / Armor Tail / Ghost, no se achanta
    (Inner Focus, Shield Dust, Covert Cloak, Substitute) o el foe actúa antes con su propio Fake Out."""
    from showdown.battle_bots.move_mechanics import priority_move_fails
    from showdown.battle_bots.greedy_joint.joint import fake_out_can_affect

    foe_mons = [f for _, f in world_foes]
    out: Dict[int, bool] = {}
    for tid, foe in world_foes:
        fails = False
        if not fake_out_can_affect(attacker, foe):
            fails = True
        elif priority_move_fails(
            'Fake Out',
            attacker=attacker,
            defender=foe,
            defender_side=foe_mons,
            terrain=terrain,
            gravity=gravity,
        ):
            fails = True
        elif _flinch_immune(attacker, foe):
            fails = True
        elif _foe_beats_fakeout(
            attacker, foe, (known_moves or {}).get(int(tid)) or (),
            trick_room=trick_room, our_tailwind=our_tailwind, opp_tailwind=opp_tailwind,
        ):
            fails = True
        out[int(tid)] = fails
    return out


def _direct_foe_threat(
    foe: CalcPoke,
    our_left: Optional[CalcPoke],
    our_right: Optional[CalcPoke],
    left_atom: JointAtom,
    right_atom: JointAtom,
    move_names: Sequence[str],
    *,
    weather: Optional[str],
    terrain: Optional[str],
    soft_ko_bonus: float,
) -> float:
    """Amenaza del foe con calc síncrono (sin deadline) — para Fake Out deny fiable."""
    from showdown.damage_calc_bridge import damage_ranges_batch
    from showdown.battle_bots.greedy_solo.greedy import _collect_side_requests

    if not move_names:
        return 0.0
    ctx = FieldContext(weather=weather, terrain=terrain)
    moves = move_options_from_names(list(move_names)[:4])
    threat = 0.0
    sides = [
        ('OL', our_left, left_atom, 1),
        ('OR', our_right, right_atom, 2),
    ]
    for _sk, our, our_atom, tid in sides:
        if our is None:
            continue
        if our_atom.utility in ('switch', 'protect') or our.is_protected:
            continue
        reqs: List[dict] = []
        for m in moves:
            if is_control_move(m.name):
                continue
            r, _ = _collect_side_requests(
                'F', foe, None, [(tid, our)], [m],
                field_ctx=ctx, can_tera=False, tera_type=None,
            )
            reqs.extend(r)
        if not reqs:
            continue
        try:
            ranges = damage_ranges_batch(reqs)
        except Exception:
            continue
        max_hp = int((_stats_or_default(our).get('hp') or 1))
        cur_hp = int(our.cur_hp if our.cur_hp is not None else max_hp)
        best = 0.0
        for rng in ranges:
            if rng is None:
                continue
            e_pct = 100.0 * expected_damage(rng) / float(max_hp) if max_hp else 0.0
            pk = p_ko(rng, cur_hp)
            best = max(
                best,
                min(e_pct, 100.0 * cur_hp / max(max_hp, 1)) + pk * soft_ko_bonus,
            )
        threat += best
    return float(threat)


def _opp_dists_greedy_mirror(
    opp_moves_by_tag: Dict[str, List[MoveOption]],
    opp_hit_map: Dict[Tuple[str, str, str], Optional[Tuple[int, int]]],
    *,
    left_atk: Optional[CalcPoke],
    right_atk: Optional[CalcPoke],
) -> Dict[str, List[Tuple[OppAction, float]]]:
    """π_opp Dirac: cada foe elige argmax daño (espejo greedy_solo factored)."""
    out: Dict[str, List[Tuple[OppAction, float]]] = {'OL': [], 'OR': []}
    available: List[Tuple[int, str]] = []
    if left_atk is not None:
        available.append((1, 'OL'))
    if right_atk is not None:
        available.append((2, 'OR'))
    if not available:
        return out

    for tag in ('F1', 'F2'):
        moves = opp_moves_by_tag.get(tag) or []
        best_score = -1.0
        best_action: Optional[OppAction] = None
        best_side: Optional[str] = None
        for m in moves:
            if is_control_move(m.name):
                continue
            mid = _norm_move(m.name)
            for tgt, side in available:
                rng = opp_hit_map.get((tag, mid, side))
                if rng is None:
                    continue
                sc = float(expected_damage(rng))
                if sc > best_score:
                    best_score = sc
                    best_action = OppAction(m.name, int(tgt), 'greedy')
                    best_side = side
        if best_action is not None and best_side is not None:
            out[best_side].append((best_action, 1.0))
            emit_pred(
                [m.name for m in moves],
                {(_norm_move(best_action.move_name), int(best_action.target), 'greedy'): 1.0},
            )
    return out


def _rebuild_opp_dists_from_hits(
    opp_moves: Sequence[MoveOption],
    opp_hit_map: Dict[Tuple[str, str, str], Optional[Tuple[int, int]]],
    *,
    left_atk: Optional[CalcPoke],
    right_atk: Optional[CalcPoke],
    w_greedy: float,
    w_control: float,
    use_real_damage: bool,
    w_switch: float = 0.10,
    split_targets: bool = True,
    fakeout_first_p: float = 0.0,
) -> Dict[str, List[Tuple[OppAction, float]]]:
    """π_opp por lado con daño real post-calc.

    Con split_targets=True: masa greedy ∝ daño por (move, target); sin doble
    conteo OL+OR. w_switch reserva masa de pivot rival (0 dmg).
    """
    out: Dict[str, List[Tuple[OppAction, float]]] = {'OL': [], 'OR': []}
    if not opp_moves:
        return out

    available: List[int] = []
    if left_atk is not None:
        available.append(1)
    if right_atk is not None:
        available.append(2)
    if not available:
        return out

    # daño por slot → {target: dmg}
    dmg_st: Dict[int, Dict[int, float]] = {}
    for m in opp_moves:
        per: Dict[int, float] = {t: 0.0 for t in available}
        if is_control_move(m.name):
            dmg_st[m.slot] = per
            continue
        mid = _norm_move(m.name)
        for t, def_tag in ((1, 'OL'), (2, 'OR')):
            if t not in available:
                continue
            if not use_real_damage:
                per[t] = 1.0
                continue
            best = 0.0
            for atk_tag in ('F1', 'F2'):
                rng = opp_hit_map.get((atk_tag, mid, def_tag))
                if rng is None:
                    continue
                best = max(best, float(expected_damage(rng)))
            per[t] = best if best > 0.0 else 0.0
        # Si ningún target tiene calc, fallback 1.0 uniforme (legacy agresivo)
        if use_real_damage and all(v <= 0.0 for v in per.values()):
            for t in available:
                per[t] = 1.0
        dmg_st[m.slot] = per

    if fakeout_first_p > 0.0:
        p_fo = min(0.95, float(fakeout_first_p))
        ratio = p_fo / (1.0 - p_fo)
        fo_slots = {m.slot for m in opp_moves if _norm_move(m.name) == 'fakeout'}
        for t in available:
            others = sum(float((dmg_st.get(m.slot) or {}).get(t, 0.0)) for m in opp_moves
                         if m.slot not in fo_slots and not is_control_move(m.name))
            for s in fo_slots:
                if float((dmg_st.get(s) or {}).get(t, 0.0)) > 0.0 and others > 0.0:
                    dmg_st[s][t] = ratio * others

    w_uniform = max(
        0.0,
        1.0 - float(w_greedy) - float(w_control) - float(w_switch),
    )

    if split_targets:
        return opp_policy_targeted(
            opp_moves,
            damage_by_slot_target=dmg_st,
            available_targets=available,
            w_greedy=float(w_greedy),
            w_control=float(w_control),
            w_uniform=w_uniform,
            w_switch=float(w_switch),
        )

    # Legacy: dist independiente por lado (puede sobrecontar amenaza).
    for def_tag, our, tgt in (('OL', left_atk, 1), ('OR', right_atk, 2)):
        if our is None:
            continue
        dmg_slots = {m.slot: float((dmg_st.get(m.slot) or {}).get(tgt, 0.0)) for m in opp_moves}
        out[def_tag] = opp_policy_distribution(
            opp_moves,
            damage_by_slot=dmg_slots,
            default_target=tgt,
            w_greedy=w_greedy,
            w_control=w_control,
            w_uniform=max(0.0, 1.0 - float(w_greedy) - float(w_control)),
        )
    merged_legacy: Dict[Tuple[str, int, str], float] = {}
    for def_tag, tgt in (('OL', 1), ('OR', 2)):
        for action, p in out.get(def_tag) or []:
            key = (_norm_move(action.move_name), tgt, action.kind)
            merged_legacy[key] = merged_legacy.get(key, 0.0) + float(p)
    if merged_legacy:
        emit_pred([m.name for m in opp_moves], merged_legacy, norm='per_target')
    return out


def _bench_incoming_pct(
    reserve: ReserveOption,
    foes: Sequence[Tuple[int, CalcPoke]],
    opp_hit_map: Dict[Tuple[str, str, str], Optional[Tuple[int, int]]],
    opp_move_names: Sequence[str],
) -> float:
    """Proxy: max E[dmg%] de moves rival vs bench (si hay calc); si no, tipado."""
    max_hp = int((_stats_or_default(reserve.poke).get('hp') or 1))
    best = 0.0
    for name in opp_move_names:
        if is_control_move(name):
            continue
        mid = _norm_move(name)
        for atk_tag in ('F1', 'F2'):
            # Reuse calcs vs OL/OR as rough proxy if same bulk order — weak.
            # Prefer type proxy below if no dedicated bench keys.
            for def_tag in ('OL', 'OR'):
                rng = opp_hit_map.get((atk_tag, mid, def_tag))
                if rng is None:
                    continue
                pct = 100.0 * expected_damage(rng) / float(max_hp) if max_hp else 0.0
                # Scale by HP ratio active→bench if stats differ
                best = max(best, pct * 0.85)
    if best > 0:
        return best
    foe_types = [tuple(f.types or ()) for _, f in foes]
    mu = matchup_vs_actives(reserve.types or reserve.poke.types, foe_types or [tuple()])
    # mu alto = buen matchup defensivo → menos chip
    return max(0.0, 35.0 - float(mu) * 12.0)


def _solo_p_ko_on_target(
    side: str,
    atom: JointAtom,
    tid: int,
    foe: CalcPoke,
    hit_map: Dict[Tuple[str, int, int, int], Optional[Tuple[int, int]]],
) -> float:
    if atom.utility == 'switch' or tid <= 0:
        return 0.0
    rng = _atom_ranges_on_foe(side, atom, tid, hit_map)
    if rng is None:
        return 0.0
    max_hp = int((_stats_or_default(foe).get('hp') or 1))
    cur_hp = int(foe.cur_hp if foe.cur_hp is not None else max_hp)
    return float(p_ko(rng, cur_hp))


def _overkill_and_free_switch(
    left: JointAtom,
    right: JointAtom,
    foes: Sequence[Tuple[int, CalcPoke]],
    hit_map: Dict[Tuple[str, int, int, int], Optional[Tuple[int, int]]],
    *,
    overkill_waste_weight: float,
    ko_free_switch_bonus: float,
) -> Tuple[float, float]:
    """
    Returns (overkill_penalty, free_switch_bonus).

    - Si ambos atacan el mismo foe y uno solo ya tiene P(KO) alto → desperdicio.
    - Si un lado ataca con P(KO) alto y el otro switch → bonus de reposicionar.
    """
    foe_by_tid = {int(tid): foe for tid, foe in foes}
    pk_l = 0.0
    pk_r = 0.0
    if left.utility != 'switch' and int(left.target) in foe_by_tid:
        pk_l = _solo_p_ko_on_target('L', left, int(left.target), foe_by_tid[int(left.target)], hit_map)
    if right.utility != 'switch' and int(right.target) in foe_by_tid:
        pk_r = _solo_p_ko_on_target('R', right, int(right.target), foe_by_tid[int(right.target)], hit_map)

    overkill = 0.0
    if (
        left.utility != 'switch'
        and right.utility != 'switch'
        and int(left.target) > 0
        and int(left.target) == int(right.target)
    ):
        strong = max(pk_l, pk_r)
        weak = min(pk_l, pk_r)
        if strong >= 0.55 and weak >= 0.08:
            overkill = float(overkill_waste_weight) * strong

    free_sw = 0.0
    if left.utility == 'switch' and right.utility != 'switch' and pk_r >= 0.55:
        free_sw = float(ko_free_switch_bonus) * pk_r
    elif right.utility == 'switch' and left.utility != 'switch' and pk_l >= 0.55:
        free_sw = float(ko_free_switch_bonus) * pk_l
    return overkill, free_sw


def _entry_punish_score(
    reserve: ReserveOption,
    *,
    opp_dists: Dict[str, List[Tuple[OppAction, float]]],
    opp_hit_map: Dict[Tuple[str, str, str], Optional[Tuple[int, int]]],
    foes: Sequence[Tuple[int, CalcPoke]],
    soft_ko_bonus: float,
    entry_punish_weight: float,
) -> float:
    """
    2-ply barato: ¿el rival castiga al entrante este turno?
    Usa la política π_opp ya calculada + tipado si no hay calc de banca.
    """
    max_hp = int((_stats_or_default(reserve.poke).get('hp') or 1))
    cur_hp = int(reserve.poke.cur_hp if reserve.poke.cur_hp is not None else max_hp)
    threat = 0.0
    for side_key in ('OL', 'OR'):
        dist = opp_dists.get(side_key) or []
        for action, p_b in dist:
            if is_control_move(action.move_name) or is_switch_action(action):
                continue
            best_rng = None
            mid = _norm_move(action.move_name)
            for atk_tag in ('F1', 'F2'):
                rng = opp_hit_map.get((atk_tag, mid, side_key))
                if rng is not None:
                    best_rng = rng
                    break
            if best_rng is None:
                continue
            # Proxy: daño vs activo ≈ vs bench (bulk distinto → 0.9)
            e_pct = 100.0 * expected_damage(best_rng) / float(max_hp) if max_hp else 0.0
            pk = p_ko(best_rng, cur_hp)
            threat += float(p_b) * (
                min(e_pct * 0.9, 100.0 * cur_hp / max(max_hp, 1))
                + pk * soft_ko_bonus * 0.9
            )
    if threat <= 1e-9:
        # Sin calc: tipado del entrante vs foes
        foe_types = [tuple(f.types or ()) for _, f in foes]
        mu = matchup_vs_actives(reserve.types or reserve.poke.types, foe_types or [tuple()])
        threat = max(0.0, 40.0 - float(mu) * 14.0)
    return float(entry_punish_weight) * threat


def _apply_speed_race(
    *,
    offense: float,
    threat: float,
    share: Dict[str, float],
    p_ko_foe: Dict[int, float],
    p_ko_side: Dict[str, float],
    left: JointAtom,
    right: JointAtom,
    left_atk: Optional[CalcPoke],
    right_atk: Optional[CalcPoke],
    world_foes: Sequence[Tuple[int, CalcPoke]],
    pos: FieldPositionBonus,
    terrain: Optional[str],
) -> Tuple[float, float]:
    """Escala offense/threat según quién actúa primero (Spe + prioridad)."""
    from showdown.battle_bots.move_mechanics import priority_move_fails

    tr = bool(pos.our_trick_room or pos.opp_trick_room)
    our_tw = bool(pos.our_tailwind)
    opp_tw = bool(pos.opp_tailwind)
    foe_mons = [f for _, f in world_foes]

    spe_l = poke_effective_spe(left_atk, tailwind=our_tw) if left_atk else 0
    spe_r = poke_effective_spe(right_atk, tailwind=our_tw) if right_atk else 0

    def _our_prio(atom: JointAtom, atk: Optional[CalcPoke]) -> int:
        if atom.utility == 'switch' or atk is None:
            return 0
        if _norm_move(atom.move_name) == 'fakeout' and not atk.just_switched:
            return 0
        defender = None
        for tid, foe in world_foes:
            if tid == atom.target:
                defender = foe
                break
        if defender is None and world_foes:
            defender = world_foes[0][1]
        if priority_move_fails(
            atom.move_name,
            attacker=atk,
            defender=defender,
            defender_side=foe_mons,
            terrain=terrain,
        ):
            return 0
        return move_prio(atom.move_name, ability=atk.ability, terrain=terrain)

    prio_l = _our_prio(left, left_atk)
    prio_r = _our_prio(right, right_atk)

    threat_scale = 1.0
    for i, (tid, foe) in enumerate(world_foes):
        foe_spe = poke_effective_spe(foe, tailwind=opp_tw)
        # Prioridad rival: asume 0 (daño estándar); Fake Out rival raro sin just_switched
        foe_prio = 0
        if foe.just_switched:
            foe_prio = 1  # leve sesgo first-turn
        # ¿Algún nuestro que pega a este foe va antes?
        we_first = False
        for atom, spe, prio in ((left, spe_l, prio_l), (right, spe_r, prio_r)):
            if atom.utility == 'switch':
                continue
            if atom.utility == 'damage' or atom.damage_pct or atom.target == tid:
                if acts_before(spe, prio, foe_spe, foe_prio, trick_room=tr):
                    we_first = True
                    break
            elif acts_before(spe, prio, foe_spe, foe_prio, trick_room=tr):
                we_first = True
                break
        if we_first:
            threat_scale *= race_scale(p_ko_foe.get(tid, 0.0))

    # Si un foe nos KO-ea yendo primero, reduce share ofensivo del slot
    off_l = share.get('L', 0.0)
    off_r = share.get('R', 0.0)
    total_share = off_l + off_r
    if total_share <= 1e-9:
        return offense * 1.0, threat * threat_scale

    scale_l = 1.0
    scale_r = 1.0
    for tid, foe in world_foes:
        foe_spe = poke_effective_spe(foe, tailwind=opp_tw)
        foe_prio = 0
        if left_atk and acts_before(foe_spe, foe_prio, spe_l, prio_l, trick_room=tr):
            scale_l = min(scale_l, race_scale(p_ko_side.get('OL', 0.0)))
        if right_atk and acts_before(foe_spe, foe_prio, spe_r, prio_r, trick_room=tr):
            scale_r = min(scale_r, race_scale(p_ko_side.get('OR', 0.0)))

    offense_scale = (off_l * scale_l + off_r * scale_r) / total_share
    return offense * offense_scale, threat * threat_scale


def _scored_from_atom(atom: JointAtom, score: float, foes: Sequence[Tuple[int, CalcPoke]]) -> ScoredAction:
    if atom.utility == 'switch':
        return ScoredAction(
            score=score,
            move_slot=atom.move_slot,
            target=0,
            mega=0,
            damage_pct=0.0,
            ko=False,
            kind='switch',
        )
    return ScoredAction(
        score=score,
        move_slot=atom.move_slot,
        target=atom.target if atom.target else (foes[0][0] if foes else 0),
        mega=atom.mega,
        damage_pct=atom.total_damage_pct,
        ko=atom.has_guaranteed_ko or atom.mega == 2,
        kind='move',
    )


async def _build_atoms(
    left_attacker: Optional[CalcPoke],
    right_attacker: Optional[CalcPoke],
    left_moves: Sequence[MoveOption],
    right_moves: Sequence[MoveOption],
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    field_ctx: FieldContext,
    can_mega_left: bool,
    can_mega_right: bool,
    can_tera_left: bool,
    can_tera_right: bool,
    tera_left: Optional[str],
    tera_right: Optional[str],
    deadline_monotonic: Optional[float] = None,
) -> Tuple[List[JointAtom], List[JointAtom]]:
    left_atk = apply_pending_mega(left_attacker) if (left_attacker and can_mega_left) else left_attacker
    right_atk = apply_pending_mega(right_attacker) if (right_attacker and can_mega_right) else right_attacker
    requests: List[dict] = []
    meta: List[Tuple[str, MoveOption, int, CalcPoke, int]] = []
    ally_meta: List[Tuple[str, int]] = []
    if left_atk and left_moves:
        r, m = _collect_side_requests(
            'L', left_atk, right_atk, foes, left_moves,
            field_ctx=field_ctx, can_tera=can_tera_left, tera_type=tera_left,
        )
        requests.extend(r)
        meta.extend(m)
    if right_atk and right_moves:
        r, m = _collect_side_requests(
            'R', right_atk, left_atk, foes, right_moves,
            field_ctx=field_ctx, can_tera=can_tera_right, tera_type=tera_right,
        )
        requests.extend(r)
        meta.extend(m)
    if left_atk and left_moves:
        ar, am = _collect_ally_hit_requests(
            'L', left_atk, right_atk, left_moves, field_ctx=field_ctx,
        )
        requests.extend(ar)
        ally_meta.extend(am)
    if right_atk and right_moves:
        ar, am = _collect_ally_hit_requests(
            'R', right_atk, left_atk, right_moves, field_ctx=field_ctx,
        )
        requests.extend(ar)
        ally_meta.extend(am)
    try:
        ranges = await calc_damage_batch(
            requests, deadline_monotonic=deadline_monotonic,
        ) if requests else []
    except Exception:
        ranges = [None] * len(requests)

    foe_count = len(meta)
    foe_ranges = ranges[:foe_count]
    ally_ranges = ranges[foe_count:]
    ally_hit_l: Dict[int, Optional[Tuple[int, int]]] = {}
    ally_hit_r: Dict[int, Optional[Tuple[int, int]]] = {}
    for (side, slot), rng in zip(ally_meta, ally_ranges):
        if side == 'L':
            ally_hit_l[int(slot)] = rng
        else:
            ally_hit_r[int(slot)] = rng

    left_meta = [m for m in meta if m[0] == 'L']
    right_meta = [m for m in meta if m[0] == 'R']
    left_ranges = [rng for m, rng in zip(meta, foe_ranges) if m[0] == 'L']
    right_ranges = [rng for m, rng in zip(meta, foe_ranges) if m[0] == 'R']
    left_atoms = (
        _atoms_from_side(
            left_moves, left_meta, left_ranges, foes,
            can_mega=can_mega_left, attacker=left_atk,
            weather=field_ctx.weather, terrain=field_ctx.terrain,
            gravity=field_ctx.gravity,
            ally=right_atk, ally_hit_by_slot=ally_hit_l,
        )
        if left_atk and left_moves else []
    )
    right_atoms = (
        _atoms_from_side(
            right_moves, right_meta, right_ranges, foes,
            can_mega=can_mega_right, attacker=right_atk,
            weather=field_ctx.weather, terrain=field_ctx.terrain,
            gravity=field_ctx.gravity,
            ally=left_atk, ally_hit_by_slot=ally_hit_r,
        )
        if right_atk and right_moves else []
    )
    return left_atoms, right_atoms


async def best_expectimax_bayesian_action(
    left_attacker: Optional[CalcPoke],
    right_attacker: Optional[CalcPoke],
    left_moves: Sequence[MoveOption],
    right_moves: Sequence[MoveOption],
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    weather: Optional[str] = None,
    terrain: Optional[str] = None,
    can_mega_left: bool = False,
    can_mega_right: bool = False,
    can_tera_left: bool = False,
    can_tera_right: bool = False,
    tera_left: Optional[str] = None,
    tera_right: Optional[str] = None,
    field_ctx: Optional[FieldContext] = None,
    beliefs=None,
    position: Optional[FieldPositionBonus] = None,
    config: Optional[ExpectimaxConfig] = None,
    foe_known_moves: Optional[Dict[int, Sequence[str]]] = None,
    reserves: Optional[Sequence[ReserveOption]] = None,
    ots_sheet: Optional[Dict[str, object]] = None,
) -> Tuple[Optional[ScoredAction], Optional[ScoredAction], float, ExpectimaxMeta]:
    """Anytime Bayesian expectimax 1-ply; nunca falla sin acción (fallback joint).

    ots_sheet: species_id → OpponentPublicOTS (sin SP). Clava item/ability/moves/nature.
    """
    cfg = config or ExpectimaxConfig()
    pos = position or FieldPositionBonus(
        tera_available_left=can_tera_left,
        tera_available_right=can_tera_right,
        mega_available_left=can_mega_left,
        mega_available_right=can_mega_right,
    )
    ctx = field_ctx or FieldContext(weather=weather, terrain=terrain)
    meta = ExpectimaxMeta()
    t0 = time.perf_counter()
    known_moves = foe_known_moves or {}
    calc_deadline = t0 + (float(cfg.phase2_end_ms) / 1000.0)

    # --- Phase 1: greedy floor (joint por defecto = filtro A) ---
    try:
        if bool(cfg.phase1_use_joint):
            fb_l, fb_r, fb_score = await best_joint_double_action(
                left_attacker, right_attacker, left_moves, right_moves, foes,
                weather=weather, terrain=terrain,
                can_mega_left=can_mega_left, can_mega_right=can_mega_right,
                can_tera_left=can_tera_left, can_tera_right=can_tera_right,
                tera_left=tera_left, tera_right=tera_right,
                field_ctx=ctx,
            )
        else:
            fb_l, fb_r, fb_score = await best_double_action(
                left_attacker, right_attacker, left_moves, right_moves, foes,
                weather=weather, terrain=terrain,
                can_mega_left=can_mega_left, can_mega_right=can_mega_right,
                can_tera_left=can_tera_left, can_tera_right=can_tera_right,
                tera_left=tera_left, tera_right=tera_right,
                field_ctx=ctx,
            )
    except Exception as exc:
        meta.errors.append(f'phase1:{exc!r}')
        meta.elapsed_ms = _elapsed_ms(t0)
        return None, None, 0.0, meta

    best_l, best_r, best_u = fb_l, fb_r, float(fb_score)
    phase1_l, phase1_r = fb_l, fb_r
    phase1_u = float(fb_score)
    meta.utility = best_u
    meta.phase = 'phase1'
    meta.used_fallback = True
    meta.phase1_utility = phase1_u
    meta.phase2_utility = phase1_u
    pairs: List[Tuple[JointAtom, JointAtom]] = []
    utilities: Dict[Tuple, JointActionUtility] = {}
    best_key = None
    # Diagnostic vs solo: un calc extra solo si queda presupuesto.
    solo_l, solo_r, solo_u = fb_l, fb_r, float(fb_score)
    meta.solo_utility = solo_u
    if bool(cfg.phase1_use_joint) and _elapsed_ms(t0) < float(cfg.phase1_budget_ms):
        try:
            solo_l, solo_r, solo_u = await best_double_action(
                left_attacker, right_attacker, left_moves, right_moves, foes,
                weather=weather, terrain=terrain,
                can_mega_left=can_mega_left, can_mega_right=can_mega_right,
                can_tera_left=can_tera_left, can_tera_right=can_tera_right,
                tera_left=tera_left, tera_right=tera_right,
                field_ctx=ctx,
            )
            meta.solo_utility = float(solo_u)
        except Exception:
            pass

    def _finalize_decision() -> None:
        meta.elapsed_ms = _elapsed_ms(t0)
        if meta.elapsed_ms >= cfg.deadline_ms and '+timeout' not in str(meta.phase):
            meta.phase = meta.phase + '+timeout'
        _log_divergence(meta, phase1_l, phase1_r, best_l, best_r, solo_l, solo_r)
        try:
            _fill_decision_report(
                meta,
                pairs=pairs,
                utilities=utilities,
                best_key=best_key,
                phase1_l=phase1_l,
                phase1_r=phase1_r,
                best_l=best_l,
                best_r=best_r,
                cfg=cfg,
            )
        except Exception as exc:
            meta.errors.append(f'decision_report:{exc!r}')

    if not cfg.enable_phase2:
        meta.phase = 'phase1+ablation'
        _finalize_decision()
        return best_l, best_r, best_u, meta

    if _elapsed_ms(t0) >= cfg.phase2_end_ms:
        _finalize_decision()
        return best_l, best_r, best_u, meta

    # --- Phase 2: Bayesian refinement ---
    try:
        left_atoms, right_atoms = await _build_atoms(
            left_attacker, right_attacker, left_moves, right_moves, foes,
            field_ctx=ctx,
            can_mega_left=can_mega_left, can_mega_right=can_mega_right,
            can_tera_left=can_tera_left, can_tera_right=can_tera_right,
            tera_left=tera_left, tera_right=tera_right,
            deadline_monotonic=calc_deadline,
        )

        switch_left: List[JointAtom] = []
        switch_right: List[JointAtom] = []
        reserve_list = list(reserves or [])
        if cfg.enable_voluntary_switch and reserve_list:
            for res in reserve_list[:4]:
                sw = _switch_atom(res.slot)
                if left_attacker is not None:
                    switch_left.append(sw)
                if right_attacker is not None:
                    switch_right.append(sw)

        left_atk = apply_pending_mega(left_attacker) if (left_attacker and can_mega_left) else left_attacker
        right_atk = apply_pending_mega(right_attacker) if (right_attacker and can_mega_right) else right_attacker

        pairs = _candidate_pairs(
            left_atoms, right_atoms, foes,
            max_pairs=cfg.max_our_pairs,
            switch_left=switch_left,
            switch_right=switch_right,
            max_switch_hybrids=cfg.max_switch_hybrids if cfg.enable_voluntary_switch else 0,
            left_poke=left_atk,
            right_poke=right_atk,
            topk_per_side=(
                int(cfg.topk_per_side) if bool(cfg.use_topk_abstraction) else 0
            ),
            enable_niche_pairs=bool(getattr(cfg, 'enable_niche_pairs', True)),
        )
        if bool(cfg.phase2_retarget_only):
            pairs = _retarget_pairs_only(pairs, phase1_l, phase1_r)
        if not pairs:
            _finalize_decision()
            return best_l, best_r, best_u, meta

        nodes = belief_nodes_for_foes(
            foes, beliefs=beliefs, k=cfg.k_profiles, foe_known_moves=known_moves,
            ots_sheet=ots_sheet,  # type: ignore[arg-type]
        )
        foe_order = [tid for tid, _ in foes]
        worlds = iter_worlds(nodes, foe_order, max_worlds=cfg.max_worlds)
        meta.worlds = len(worlds)

        # Rellenar el dict outer (no rebind) para que _finalize_decision lo vea.
        utilities.clear()
        for pl, pr in pairs:
            utilities[_pair_key(pl, pr)] = JointActionUtility(
                left_slot=pl.move_slot,
                right_slot=pr.move_slot,
                left_target=pl.target,
                right_target=pr.target,
                left_mega=pl.mega,
                right_mega=pr.mega,
                left_move=pl.move_name,
                right_move=pr.move_name,
            )

        reserve_by_slot = {int(r.slot): r for r in reserve_list}

        for world_profiles, p_world in worlds:
            if _elapsed_ms(t0) >= cfg.phase2_end_ms:
                break

            world_foes: List[Tuple[int, CalcPoke]] = []
            for tid, foe in foes:
                prof = world_profiles.get(tid)
                if prof is None:
                    world_foes.append((tid, foe))
                    continue
                base_stats = None
                try:
                    if beliefs is not None and hasattr(beliefs, 'beliefs'):
                        b = beliefs.beliefs.get(prof.species) or beliefs.beliefs.get(foe.species)
                        if b is not None:
                            base_stats = getattr(b, 'base_stats', None)
                except Exception:
                    base_stats = None
                try:
                    world_foes.append((tid, profile_to_calc(foe, prof, base_stats=base_stats)))
                except Exception:
                    world_foes.append((tid, foe))

            # Proyectar mega rival (stats/ability) + clima de Drought/etc. este turno.
            from showdown.battle_bots.move_mechanics.vgc_utility import (
                project_foe_active_forme,
                weather_from_projected_foes,
            )
            world_foes = [
                (tid, project_foe_active_forme(foe)) for tid, foe in world_foes
            ]
            world_weather = weather_from_projected_foes(
                [f for _, f in world_foes],
                current_weather=ctx.weather,
            )
            world_ctx = ctx
            if world_weather and world_weather != ctx.weather:
                try:
                    from dataclasses import replace as _dc_replace
                    world_ctx = _dc_replace(ctx, weather=world_weather)
                except Exception:
                    world_ctx = FieldContext(weather=world_weather, terrain=ctx.terrain)

            reqs: List[dict] = []
            meta_hits: List[Tuple[str, MoveOption, int, CalcPoke, int]] = []
            if left_atk and left_moves:
                r, m = _collect_side_requests(
                    'L', left_atk, right_atk, world_foes, left_moves,
                    field_ctx=world_ctx, can_tera=can_tera_left, tera_type=tera_left,
                )
                reqs.extend(r)
                meta_hits.extend(m)
            if right_atk and right_moves:
                r, m = _collect_side_requests(
                    'R', right_atk, left_atk, world_foes, right_moves,
                    field_ctx=world_ctx, can_tera=can_tera_right, tera_type=tera_right,
                )
                reqs.extend(r)
                meta_hits.extend(m)

            opp_reqs: List[dict] = []
            opp_meta: List[Tuple[str, str, str]] = []
            # Moves de TODOS los foes (antes π_opp solo veía el primero).
            opp_moves_all: List[MoveOption] = []
            opp_moves_by_tag: Dict[str, List[MoveOption]] = {}
            seen_opp_slots: Set[Tuple[str, int]] = set()
            all_opp_move_names: List[str] = []
            fakeout_first_any = False

            for i, (tid, foe) in enumerate(world_foes):
                tag = 'F1' if i == 0 else 'F2'
                prof = world_profiles.get(tid)
                move_names = list(resolve_opp_move_names(prof, known_moves.get(tid) or ()))
                if bool(cfg.opp_policy_fakeout_rule):
                    has_fo = any(_norm_move(n) == 'fakeout' for n in move_names)
                    if has_fo and not foe.just_switched:
                        # Showdown: Fake Out falla tras el primer turno en campo
                        move_names = [n for n in move_names if _norm_move(n) != 'fakeout']
                    elif has_fo:
                        fakeout_first_any = True
                if not move_names:
                    continue
                all_opp_move_names.extend(move_names)
                opp_moves = move_options_from_names(
                    move_names,
                    start_slot=(1 if i == 0 else 5) if bool(cfg.opp_policy_unique_slots) else 1,
                )
                opp_moves_by_tag[tag] = list(opp_moves)
                for m in opp_moves:
                    key = (_norm_move(m.name), int(m.slot))
                    if key in seen_opp_slots:
                        continue
                    seen_opp_slots.add(key)
                    opp_moves_all.append(m)
                for def_tag, our in (('OL', left_atk), ('OR', right_atk)):
                    if our is None:
                        continue
                    for m in opp_moves:
                        if is_control_move(m.name):
                            continue
                        r, _mta = _collect_side_requests(
                            tag, foe, None, [(1 if def_tag == 'OL' else 2, our)], [m],
                            field_ctx=world_ctx, can_tera=False, tera_type=None,
                        )
                        for _req in r:
                            opp_reqs.append(_req)
                            opp_meta.append((tag, _norm_move(m.name), def_tag))

            try:
                our_ranges = await calc_damage_batch(
                    reqs, deadline_monotonic=calc_deadline,
                ) if reqs else []
            except Exception as exc:
                meta.errors.append(f'calc_our:{exc!r}')
                our_ranges = [None] * len(reqs)
            try:
                opp_ranges = await calc_damage_batch(
                    opp_reqs, deadline_monotonic=calc_deadline,
                ) if opp_reqs else []
            except Exception as exc:
                meta.errors.append(f'calc_opp:{exc!r}')
                opp_ranges = [None] * len(opp_reqs)

            hit_map = _hit_map_from_batch(
                meta_hits,
                our_ranges,
                attackers={'L': left_atk, 'R': right_atk},
                weather=world_ctx.weather,
            )
            opp_hit_map: Dict[Tuple[str, str, str], Optional[Tuple[int, int]]] = {}
            from showdown.battle_bots.move_mechanics import gate_damage_range, priority_move_fails
            foe_by_tag = {
                'F1': world_foes[0][1] if len(world_foes) > 0 else None,
                'F2': world_foes[1][1] if len(world_foes) > 1 else None,
            }
            our_by_def = {'OL': left_atk, 'OR': right_atk}
            our_side = [p for p in (left_atk, right_atk) if p is not None]
            for key, rng in zip(opp_meta, opp_ranges):
                tag, move_id, def_tag = key
                foe = foe_by_tag.get(tag)
                our = our_by_def.get(def_tag)
                gated = gate_damage_range(
                    rng,
                    move_id,
                    weather=world_ctx.weather,
                    item=foe.item if foe is not None else None,
                    already_charging=bool(foe.is_two_turn_charging) if foe is not None else False,
                    must_recharge=bool(foe.must_recharge) if foe is not None else False,
                )
                if gated is not None and priority_move_fails(
                    move_id,
                    attacker=foe,
                    defender=our,
                    defender_side=our_side,
                    terrain=ctx.terrain,
                    gravity=ctx.gravity,
                ):
                    gated = None
                opp_hit_map[key] = gated

            # Estructura: π_opp tras calc (espejo solo o mezcla legacy).
            if bool(cfg.opp_policy_mirror_solo):
                opp_dists = _opp_dists_greedy_mirror(
                    opp_moves_by_tag,
                    opp_hit_map,
                    left_atk=left_atk,
                    right_atk=right_atk,
                )
            else:
                opp_dists = _rebuild_opp_dists_from_hits(
                    opp_moves_all,
                    opp_hit_map,
                    left_atk=left_atk,
                    right_atk=right_atk,
                    w_greedy=float(cfg.opp_policy_greedy_weight),
                    w_control=float(cfg.opp_policy_control_weight),
                    use_real_damage=bool(cfg.opp_policy_use_real_damage),
                    w_switch=float(cfg.opp_policy_switch_weight),
                    split_targets=bool(cfg.opp_policy_split_targets),
                    fakeout_first_p=(float(cfg.opp_policy_fakeout_first_p)
                                     if (fakeout_first_any and bool(cfg.opp_policy_fakeout_rule)) else 0.0),
                )

            foe_types = [tuple(f.types or ()) for _, f in world_foes]

            for pl, pr in pairs:
                if _elapsed_ms(t0) >= cfg.phase2_end_ms:
                    break

                offense, foe_hp, p_ko_foe, share = _our_pair_offense_detail(
                    pl, pr, world_foes, hit_map, soft_ko_bonus=cfg.soft_ko_bonus,
                )
                threat, p_ko_side, _pct_side = _opp_threat_detail(
                    left_atk, right_atk, pl, pr, opp_hit_map, opp_dists,
                    soft_ko_bonus=cfg.soft_ko_bonus,
                )

                # Friendly fire (EQ/Surf/…): threat propio + P(KO) al partner.
                from showdown.battle_bots.move_mechanics import (
                    friendly_fire_expectimax_adjust,
                )
                threat, p_ko_side = friendly_fire_expectimax_adjust(
                    threat=threat,
                    p_ko_side=p_ko_side,
                    left=pl,
                    right=pr,
                    left_poke=left_atk,
                    right_poke=right_atk,
                    soft_ko_bonus=cfg.soft_ko_bonus,
                )

                # B: orden de turno
                offense, threat = _apply_speed_race(
                    offense=offense,
                    threat=threat,
                    share=share,
                    p_ko_foe=p_ko_foe,
                    p_ko_side=p_ko_side,
                    left=pl,
                    right=pr,
                    left_atk=left_atk,
                    right_atk=right_atk,
                    world_foes=world_foes,
                    pos=pos,
                    terrain=terrain or ctx.terrain,
                )

                # Proyección residual: status → ↓threat; condenado → chip↓ lasting↑
                offense, threat, lasting_extra = project_residual_into_v(
                    offense=offense,
                    threat=threat,
                    share=share,
                    left=pl,
                    right=pr,
                    left_atk=left_atk,
                    right_atk=right_atk,
                    foes=world_foes,
                    p_ko_side=p_ko_side,
                    p_ko_foe=p_ko_foe,
                    terrain=terrain or ctx.terrain,
                    chip_discount=cfg.condemned_chip_discount,
                    lasting_boost=cfg.condemned_lasting_boost,
                    threat_relief_weight=cfg.threat_relief_weight,
                )

                # C: switches voluntarios (preservación KO + reposicionamiento)
                switch_v = 0.0
                turn_n = int(getattr(pos, 'turn', 0) or 0) if pos is not None else 0
                for atom, side_key in ((pl, 'OL'), (pr, 'OR')):
                    if atom.utility != 'switch':
                        continue
                    res = reserve_by_slot.get(int(atom.move_slot))
                    if res is None:
                        continue
                    stay_pk = float(p_ko_side.get(side_key, 0.0))
                    active_types = ()
                    if side_key == 'OL' and left_atk is not None:
                        active_types = tuple(left_atk.types or ())
                    elif side_key == 'OR' and right_atk is not None:
                        active_types = tuple(right_atk.types or ())
                    mu_active = matchup_vs_actives(
                        active_types or (tuple(),), foe_types or [tuple()],
                    )
                    # Si no hay amenaza calc (dists vacíos), estimar por tipado del activo
                    if stay_pk < 1e-9:
                        stay_pk = max(0.0, min(0.9, 0.20 - float(mu_active) * 0.10))
                    mu_bench = matchup_vs_actives(
                        res.types or res.poke.types, foe_types or [tuple()],
                    )
                    improvement = float(mu_bench) - float(mu_active)
                    # Frívolo solo si no mejora el asiento y no hay amenaza real.
                    if (
                        stay_pk < cfg.switch_min_stay_p_ko
                        and improvement < 0.2
                        and stay_pk < 0.12
                    ):
                        switch_v -= 22.0
                        continue
                    chip = _bench_incoming_pct(res, world_foes, opp_hit_map, all_opp_move_names)
                    switch_v += switch_preservation_value(
                        stay_p_ko=max(stay_pk, cfg.switch_min_stay_p_ko * 0.5),
                        bench_incoming_pct=chip,
                        matchup=mu_bench,
                        loss_weight=cfg.loss_weight,
                        save_weight=cfg.save_weight,
                        chip_soft_cap=cfg.chip_soft_cap,
                        soft_ko_bonus=cfg.soft_ko_bonus,
                    )
                    switch_v += switch_positioning_value(
                        active_matchup=mu_active,
                        bench_matchup=mu_bench,
                        turn=turn_n,
                    )
                    # Habilidades al entrar (Intimidate, Surges, Screen Cleaner…).
                    from showdown.battle_bots.move_mechanics import (
                        entry_ability_switch_value,
                    )
                    other_atom = pr if side_key == 'OL' else pl
                    stay_ally = (
                        right_atk if side_key == 'OL' else left_atk
                    )
                    if other_atom.utility == 'switch':
                        stay_ally = None
                    switch_v += entry_ability_switch_value(
                        res.poke,
                        [f for _, f in world_foes],
                        ally=stay_ally,
                        weather=ctx.weather,
                        terrain=ctx.terrain or terrain,
                        opp_reflect=bool(ctx.reflect),
                        opp_light_screen=bool(ctx.light_screen),
                        opp_aurora_veil=bool(ctx.aurora_veil),
                        weight=cfg.entry_ability_weight,
                    )
                    # 2-ply barato: ¿el rival castiga al entrante?
                    switch_v -= _entry_punish_score(
                        res,
                        opp_dists=opp_dists,
                        opp_hit_map=opp_hit_map,
                        foes=world_foes,
                        soft_ko_bonus=cfg.soft_ko_bonus,
                        entry_punish_weight=cfg.entry_punish_weight,
                    )

                overkill_pen, free_sw = _overkill_and_free_switch(
                    pl, pr, world_foes, hit_map,
                    overkill_waste_weight=cfg.overkill_waste_weight,
                    ko_free_switch_bonus=cfg.ko_free_switch_bonus,
                )
                switch_v += free_sw
                switch_v *= float(cfg.switch_v_scale)

                syn = 0.0
                if pl.utility != 'switch' and pr.utility != 'switch':
                    try:
                        # Sin FF aquí: ya entra en threat vía friendly_fire_expectimax_adjust.
                        syn = _score_pair(
                            pl, pr, world_foes,
                            left_poke=left_atk, right_poke=right_atk,
                            include_friendly_fire=False,
                        ) - (
                            pl.total_damage_pct + pr.total_damage_pct
                        )
                        if syn > cfg.soft_ko_bonus:
                            syn = min(syn, 80.0)
                    except Exception:
                        syn = 0.0

                used_sc = _speed_control(pl.move_name) or _speed_control(pr.move_name)
                used_tera = pl.mega == 2 or pr.mega == 2
                field_b = field_control_bonus(pos, used_speed_control=used_sc, used_tera=used_tera)

                our_hp = max(0.0, 100.0 - min(threat / max(cfg.soft_ko_bonus / 100.0, 1.0), 100.0))
                from showdown.battle_bots.move_mechanics.vgc_utility import (
                    pair_vgc_utility_extra,
                )
                threat_by_tid = _threat_by_foe_tid(
                    world_foes, left_atk, right_atk, pl, pr,
                    opp_hit_map, opp_dists,
                    soft_ko_bonus=cfg.soft_ko_bonus,
                )
                # Si el batch opp cortó por deadline (Heat Wave→OR=None), recalc
                # amenaza directa para targets de Fake Out del par.
                for atom in (pl, pr):
                    if atom is None:
                        continue
                    if not (
                        _norm_move(atom.move_name) == 'fakeout'
                        or atom.utility == 'fakeout'
                    ):
                        continue
                    tid = int(atom.target or 0)
                    if tid <= 0 or threat_by_tid.get(tid, 0.0) >= 0.45 * float(cfg.soft_ko_bonus):
                        continue
                    foe = next((f for t, f in world_foes if int(t) == tid), None)
                    if foe is None:
                        continue
                    moves = list(known_moves.get(tid) or ())
                    if not moves:
                        continue
                    direct = _direct_foe_threat(
                        foe, left_atk, right_atk, pl, pr, moves,
                        weather=world_ctx.weather,
                        terrain=terrain or ctx.terrain,
                        soft_ko_bonus=cfg.soft_ko_bonus,
                    )
                    if direct > threat_by_tid.get(tid, 0.0):
                        threat_by_tid[tid] = direct
                fo_fails_l = _fakeout_fails_by_tid(
                    left_atk, world_foes,
                    terrain=terrain or ctx.terrain,
                    gravity=bool(getattr(ctx, 'gravity', False)),
                    known_moves=known_moves,
                    trick_room=bool(pos.our_trick_room or pos.opp_trick_room),
                    our_tailwind=bool(pos.our_tailwind),
                    opp_tailwind=bool(pos.opp_tailwind),
                )
                fo_fails_r = _fakeout_fails_by_tid(
                    right_atk, world_foes,
                    terrain=terrain or ctx.terrain,
                    gravity=bool(getattr(ctx, 'gravity', False)),
                    known_moves=known_moves,
                    trick_room=bool(pos.our_trick_room or pos.opp_trick_room),
                    our_tailwind=bool(pos.our_tailwind),
                    opp_tailwind=bool(pos.opp_tailwind),
                )
                # Fails si CUALQUIER atacante del par con FO fallaría vs ese tid.
                fo_fails = {
                    tid: bool(fo_fails_l.get(tid) and fo_fails_r.get(tid))
                    for tid in set(fo_fails_l) | set(fo_fails_r)
                }
                # Si solo un lado tiene Fake Out, usar fails de ese atacante.
                for atom, fails_map in ((pl, fo_fails_l), (pr, fo_fails_r)):
                    if atom is None:
                        continue
                    if _norm_move(atom.move_name) == 'fakeout' or atom.utility == 'fakeout':
                        tid = int(atom.target or 0)
                        if tid:
                            fo_fails[tid] = bool(fails_map.get(tid, False))
                vgc_u = pair_vgc_utility_extra(
                    pl, pr,
                    left_atk=left_atk,
                    right_atk=right_atk,
                    foes=world_foes,
                    threat_by_tid=threat_by_tid,
                    fakeout_fails_by_tid=fo_fails,
                )
                if bool(cfg.v_solo_isomorphic):
                    # Offense + utilidad VGC (Fake Out deny=amenaza / Knock Off item).
                    v = float(offense) + float(vgc_u)
                else:
                    v = (
                        offense
                        - float(cfg.threat_weight) * threat
                        + float(cfg.hp_rel_weight) * (our_hp - foe_hp)
                        + float(cfg.syn_weight) * syn
                        + lasting_extra
                        + field_b
                        + switch_v
                        - overkill_pen
                        + float(vgc_u)
                    )

                utilities[_pair_key(pl, pr)].add(v, weight=p_world)

            meta.pairs_scored = len(utilities)

        best_key = None
        best_mean = float('-inf')
        for key, ju in utilities.items():
            if ju.samples <= 0:
                continue
            m = ju.mean
            if m > best_mean:
                best_mean = m
                best_key = key

        if best_key is not None and best_mean > float('-inf'):
            p1_v = _phase1_bayes_v(utilities, pairs, phase1_l, phase1_r)
            best_pl = best_pr = None
            for pl, pr in pairs:
                if _pair_key(pl, pr) == best_key:
                    best_pl, best_pr = pl, pr
                    break

            accept_p2 = best_pl is not None
            if accept_p2 and cfg.phase2_surgical_only and best_pl is not None:
                # Misma acción que phase1 → ok; diverge solo en nicho quirúrgico
                # o retarget (mismos slots, distinto target) si phase2_retarget_only.
                same_as_p1 = (
                    _scored_matches_atom(phase1_l, best_pl)
                    and _scored_matches_atom(phase1_r, best_pr)
                )
                same_slots = (
                    (phase1_l is None or int(phase1_l.move_slot) == int(best_pl.move_slot))
                    and (phase1_r is None or int(phase1_r.move_slot) == int(best_pr.move_slot))
                    and best_pl.utility != 'switch'
                    and best_pr.utility != 'switch'
                )
                if not same_as_p1:
                    allow = False
                    if bool(cfg.phase2_retarget_only) and same_slots:
                        allow = True
                    elif _is_surgical_pair(best_pl, best_pr):
                        allow = True
                    if not allow:
                        accept_p2 = False
                        meta.phase2_gated = True

            gate_mode = str(getattr(cfg, 'phase2_gate_mode', 'beat') or 'beat').lower()
            if accept_p2 and cfg.phase2_gate_enabled and gate_mode != 'off':
                meta.phase1_v = float(p1_v) if p1_v is not None else 0.0
                meta.phase2_v = float(best_mean)
                if p1_v is None:
                    # Sin V de phase1 en el set: no anclar; Top-K Bayes puede diverger.
                    if gate_mode == 'beat':
                        accept_p2 = False
                        meta.phase2_gated = True
                else:
                    delta = float(best_mean) - float(p1_v)
                    max_dv = float(cfg.phase2_max_delta_v)
                    if max_dv > 0.0 and delta > max_dv:
                        accept_p2 = False
                        meta.phase2_gated = True
                    elif gate_mode == 'indifference':
                        eps = float(getattr(cfg, 'phase2_indifference_eps', 40.0) or 0.0)
                        if float(best_mean) < float(p1_v) - eps:
                            accept_p2 = False
                            meta.phase2_gated = True
                    else:
                        # beat: exigir mejora mínima vs phase1.
                        if delta < float(cfg.phase2_gate_margin):
                            accept_p2 = False
                            meta.phase2_gated = True
            elif accept_p2:
                meta.phase1_v = float(p1_v) if p1_v is not None else 0.0
                meta.phase2_v = float(best_mean)

            if accept_p2 and best_pl is not None and best_pr is not None:
                best_l = _scored_from_atom(best_pl, best_mean, foes)
                best_r = _scored_from_atom(best_pr, best_mean, foes)
                best_u = best_mean
                meta.used_fallback = False
                meta.phase = 'phase2'
                meta.utility = best_u
                meta.phase2_utility = best_u
            else:
                # Conserva phase1/joint; anota V de phase2 rechazada.
                meta.phase = 'phase1+gated'
                meta.phase2_utility = float(best_mean)
                meta.used_fallback = True

    except Exception as exc:
        meta.errors.append(f'phase2:{exc!r}')

    _finalize_decision()
    return best_l, best_r, best_u, meta


def _log_divergence(
    meta: ExpectimaxMeta,
    phase1_l: Optional[ScoredAction],
    phase1_r: Optional[ScoredAction],
    best_l: Optional[ScoredAction],
    best_r: Optional[ScoredAction],
    solo_l: Optional[ScoredAction] = None,
    solo_r: Optional[ScoredAction] = None,
) -> None:
    p1 = pair_sig(phase1_l, phase1_r)
    p2 = pair_sig(best_l, best_r)
    diverged = (not meta.used_fallback) and p1 != p2
    meta.diverged = diverged
    try:
        record_choose(
            phase=meta.phase,
            used_fallback=bool(meta.used_fallback),
            diverged=diverged,
            phase1_u=float(meta.phase1_utility),
            phase2_u=float(meta.phase2_utility),
            phase1_sig=p1,
            phase2_sig=p2,
            elapsed_ms=float(meta.elapsed_ms),
            gated=bool(meta.phase2_gated),
        )
    except Exception:
        pass
    try:
        solo_sig = pair_sig(solo_l, solo_r)
        chosen_sig = p2
        neq = chosen_sig != solo_sig
        meta.neq_solo = neq
        record_solo_compare(
            chosen_sig=chosen_sig,
            solo_sig=solo_sig,
            joint_u=float(meta.utility),
            solo_u=float(meta.solo_utility),
        )
    except Exception:
        pass
