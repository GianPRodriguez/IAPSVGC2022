"""Proyección residual 1-ply: status → amenaza baja; mon condenado → priorizar lasting.

Generaliza jugadas tipo «Sneasler muere igual → Dire Claw > chip Close Combat»:
no hardcodea especies; usa P(KO nuestro) y E[status] sobre el rival.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from showdown.battle_bots.greedy_solo.greedy import CalcPoke, _stats_or_default
from showdown.battle_bots.greedy_joint.joint import JointAtom
from showdown.battle_bots.move_mechanics.status_effects import expected_threat_relief


def _norm_move(name: str) -> str:
    return str(name or '').strip().lower().replace(' ', '').replace('-', '')


def _foe_by_tid(
    foes: Sequence[Tuple[int, CalcPoke]],
) -> Dict[int, CalcPoke]:
    return {int(tid): foe for tid, foe in foes}


def atom_threat_relief(
    atom: JointAtom,
    attacker: Optional[CalcPoke],
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    terrain: Optional[str] = None,
    we_outspeed: Optional[bool] = None,
) -> float:
    """E[alivio] de un átomo sobre su target (0 si switch/protect sin efecto)."""
    if atom is None or atom.utility in ('switch', 'protect'):
        return 0.0
    tid = int(atom.target or 0)
    foe_map = _foe_by_tid(foes)
    foe = foe_map.get(tid)
    if foe is None and len(foes) == 1:
        foe = foes[0][1]
    if foe is None:
        # Status sin tid: mejor alivio entre foes.
        best = 0.0
        for _ft, f in foes:
            best = max(
                best,
                expected_threat_relief(
                    atom.move_name,
                    attacker=attacker,
                    defender=f,
                    terrain=terrain,
                    we_outspeed=we_outspeed,
                ),
            )
        return best
    return expected_threat_relief(
        atom.move_name,
        attacker=attacker,
        defender=foe,
        terrain=terrain,
        we_outspeed=we_outspeed,
    )


def combined_threat_relief(
    left: JointAtom,
    right: JointAtom,
    left_atk: Optional[CalcPoke],
    right_atk: Optional[CalcPoke],
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    terrain: Optional[str] = None,
) -> float:
    """
    Alivio total sin doble-contar el mismo foe.

    Si ambos pegan al mismo tid: 1 - (1-p_approx) vía max + complemento suave.
    """
    rel_l = atom_threat_relief(left, left_atk, foes, terrain=terrain)
    rel_r = atom_threat_relief(right, right_atk, foes, terrain=terrain)
    tid_l = int(left.target or 0) if left is not None else 0
    tid_r = int(right.target or 0) if right is not None else 0
    if tid_l and tid_r and tid_l == tid_r:
        # Independientes aprox.: r = a + b - a*b/cap
        a, b = rel_l, rel_r
        cap = max(THREAT_RELIEF_CAP, 1.0)
        return min(cap, a + b - (a * b) / cap)
    return rel_l + rel_r


# Tope de alivio por foe (≈ sleep relief).
THREAT_RELIEF_CAP = 78.0


def condemned_reweight(
    *,
    offense: float,
    share: Dict[str, float],
    left: JointAtom,
    right: JointAtom,
    p_ko_side: Dict[str, float],
    p_ko_foe: Dict[int, float],
    chip_discount: float = 0.70,
    lasting_boost: float = 1.35,
) -> Tuple[float, float]:
    """
    Si un activo nuestro tiene alta P(KO), el chip que no KO-ea vale menos y
    los efectos que sobreviven (effect_bonus) valen más.

    Returns (offense_ajustado, lasting_extra).
    """
    total_share = float(share.get('L', 0.0)) + float(share.get('R', 0.0))
    if total_share <= 1e-9 and offense <= 1e-9:
        lasting = (
            float(getattr(left, 'effect_bonus', 0.0) or 0.0)
            * (1.0 + lasting_boost * float(p_ko_side.get('OL', 0.0)))
            + float(getattr(right, 'effect_bonus', 0.0) or 0.0)
            * (1.0 + lasting_boost * float(p_ko_side.get('OR', 0.0)))
        )
        # effect_bonus ya entra en syn; lasting_extra = solo el boost incremental
        base = (
            float(getattr(left, 'effect_bonus', 0.0) or 0.0)
            + float(getattr(right, 'effect_bonus', 0.0) or 0.0)
        )
        return offense, max(0.0, lasting - base)

    def _side_scale(side: str, atom: JointAtom) -> float:
        p_dead = max(0.0, min(1.0, float(p_ko_side.get(side, 0.0))))
        if p_dead <= 1e-9 or atom is None or atom.utility in ('switch', 'protect'):
            return 1.0
        tid = int(atom.target or 0)
        pk_foe = max(0.0, min(1.0, float(p_ko_foe.get(tid, 0.0)))) if tid else 0.0
        # Si ya vamos a KO-ear al rival, el daño sigue valiendo; si no, descontar chip.
        return 1.0 - float(chip_discount) * p_dead * (1.0 - pk_foe)

    scale_l = _side_scale('OL', left)
    scale_r = _side_scale('OR', right)
    sh_l = float(share.get('L', 0.0))
    sh_r = float(share.get('R', 0.0))
    if total_share > 1e-9:
        blend = (sh_l * scale_l + sh_r * scale_r) / total_share
    else:
        blend = 0.5 * (scale_l + scale_r)
    new_offense = float(offense) * blend

    # Boost incremental de lasting (syn ya trae effect_bonus a 1×).
    extra = 0.0
    extra += float(getattr(left, 'effect_bonus', 0.0) or 0.0) * lasting_boost * float(
        p_ko_side.get('OL', 0.0)
    )
    extra += float(getattr(right, 'effect_bonus', 0.0) or 0.0) * lasting_boost * float(
        p_ko_side.get('OR', 0.0)
    )
    return new_offense, extra


def project_residual_into_v(
    *,
    offense: float,
    threat: float,
    share: Dict[str, float],
    left: JointAtom,
    right: JointAtom,
    left_atk: Optional[CalcPoke],
    right_atk: Optional[CalcPoke],
    foes: Sequence[Tuple[int, CalcPoke]],
    p_ko_side: Dict[str, float],
    p_ko_foe: Dict[int, float],
    terrain: Optional[str] = None,
    chip_discount: float = 0.70,
    lasting_boost: float = 1.35,
    threat_relief_weight: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Ajusta (offense, threat) y devuelve lasting_extra para sumar a V.

    - threat baja con E[status] proyectado (S').
    - offense de chip cae si el atacante está condenado.
    - lasting_extra amplifica effect_bonus bajo condena.
    """
    new_off, lasting_extra = condemned_reweight(
        offense=offense,
        share=share,
        left=left,
        right=right,
        p_ko_side=p_ko_side,
        p_ko_foe=p_ko_foe,
        chip_discount=chip_discount,
        lasting_boost=lasting_boost,
    )
    relief = combined_threat_relief(
        left, right, left_atk, right_atk, foes, terrain=terrain,
    )
    # También amplificar relief si quien aplica el status está condenado.
    p_dead_l = float(p_ko_side.get('OL', 0.0))
    p_dead_r = float(p_ko_side.get('OR', 0.0))
    rel_l = atom_threat_relief(left, left_atk, foes, terrain=terrain)
    rel_r = atom_threat_relief(right, right_atk, foes, terrain=terrain)
    relief_boosted = relief + rel_l * lasting_boost * p_dead_l + rel_r * lasting_boost * p_dead_r
    new_threat = max(0.0, float(threat) - float(threat_relief_weight) * relief_boosted)
    return new_off, new_threat, lasting_extra
