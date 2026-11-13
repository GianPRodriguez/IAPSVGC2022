"""ExpectimaxBayesianPlayer: poke-env con expectimax bayesiano anytime."""
from __future__ import annotations

from showdown.battle_bots.helpers import enforce_single_mega

import asyncio
from typing import Dict, List, Optional, Sequence, Tuple

from poke_env.battle import AbstractBattle, DoubleBattle, Move, Pokemon
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)
from poke_env.player import Player

from showdown.battle_bots.greedy_joint.pokeenv_player import (
    GreedyJointPlayer,
    _count_pe_fainted,
    _field_context_from_pe,
    _field_key,
    _move_option,
    _poke_to_calc,
    _side_has,
    _species_name,
    _terrain_str,
    _weather_str,
)
from showdown.battle_bots.greedy_solo.greedy import apply_belief_stats, types_from_species
from showdown.battle_bots.greedy_solo.team_preview import (
    mon_from_species,
    order_to_team_string,
)
from showdown.battle_bots.expectimax_bayesian.expectimax import (
    ExpectimaxConfig,
    ReserveOption,
    best_expectimax_bayesian_action,
    load_expectimax_config,
)
from showdown.battle_bots.expectimax_bayesian.team_preview import (
    best_expectimax_preview_with_meta,
)
from showdown.battle_bots.expectimax_bayesian.utility import FieldPositionBonus


def _position_from_pe(
    battle: DoubleBattle,
    *,
    can_tera_l: bool,
    can_tera_r: bool,
    can_mega_l: bool,
    can_mega_r: bool,
) -> FieldPositionBonus:
    our_sc = getattr(battle, 'side_conditions', None) or {}
    opp_sc = getattr(battle, 'opponent_side_conditions', None) or {}
    fields = getattr(battle, 'fields', None) or {}
    field_ids = {_field_key(k) for k in fields.keys()}
    return FieldPositionBonus(
        our_tailwind=_side_has(our_sc, 'TAILWIND'),
        our_trick_room='TRICKROOM' in field_ids,
        opp_tailwind=_side_has(opp_sc, 'TAILWIND'),
        opp_trick_room=False,
        tera_available_left=can_tera_l,
        tera_available_right=can_tera_r,
        mega_available_left=can_mega_l,
        mega_available_right=can_mega_r,
        turn=int(getattr(battle, 'turn', 0) or 0),
    )


def _pe_known_moves(mon: Pokemon) -> Tuple[str, ...]:
    out: List[str] = []
    moves = getattr(mon, 'moves', None) or {}
    if isinstance(moves, dict):
        for mv in moves.values():
            if mv is None:
                continue
            name = getattr(mv, 'id', None) or getattr(mv, 'name', None)
            if name:
                out.append(str(name).replace('-', ' ').title() if '-' in str(name) else str(name))
    elif isinstance(moves, (list, tuple)):
        for mv in moves:
            if not mv:
                continue
            name = getattr(mv, 'id', None) or getattr(mv, 'name', None) or mv
            if name:
                out.append(str(name))
    return tuple(out)


class ExpectimaxBayesianPlayer(GreedyJointPlayer):
    """
    Expectimax bayesiano 1-ply (anytime):
      Phase1 Greedy-Joint → Phase2 belief/meta mundos + Spe + switches → select.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expectimax_config = load_expectimax_config()
        self.last_expectimax_meta = None
        self.last_preview_meta = None
        # species_id → OpponentPublicOTS (Champions OTS sin SP)
        self.opp_ots_sheet = {}
        # Outcome calibration: metas del game en curso.
        self._game_decision_log: List[dict] = []

    def set_opp_ots_sheet(self, sheet) -> None:
        """Instala hoja pública del rival (nunca incluir true_sp)."""
        self.opp_ots_sheet = dict(sheet or {})

    def reset_game_decision_log(self) -> None:
        self._game_decision_log = []

    def note_expectimax_decision(self, emeta) -> None:
        """Acumula un choose para calibración V↔outcome al fin del game."""
        if emeta is None:
            return
        try:
            p1v = float(getattr(emeta, 'phase1_v', 0.0) or 0.0)
            p2v = float(getattr(emeta, 'phase2_v', 0.0) or 0.0)
            self._game_decision_log.append({
                'phase': str(getattr(emeta, 'phase', '') or ''),
                'diverged': bool(getattr(emeta, 'diverged', False)),
                'neq_solo': bool(getattr(emeta, 'neq_solo', False)),
                'gated': bool(getattr(emeta, 'phase2_gated', False)),
                'delta_v': p2v - p1v,
                'phase1_v': p1v,
                'phase2_v': p2v,
            })
        except Exception:
            pass

    def flush_battle_outcome(self, won: bool) -> None:
        """Cierra el game: escribe outcome + limpia log."""
        try:
            from showdown.battle_bots.expectimax_bayesian.divergence import (
                record_battle_outcome,
            )
            record_battle_outcome(bool(won), list(self._game_decision_log))
        except Exception:
            pass
        self._game_decision_log = []

    def teampreview(self, battle) -> str:
        from poke_env.battle import AbstractBattle, DoubleBattle

        assert isinstance(battle, DoubleBattle)
        our: List = []
        for i, mon in enumerate(battle.teampreview_team or list(battle.team.values()), start=1):
            our.append(mon_from_species(i, _species_name(mon)))

        foe_src = battle.teampreview_opponent_team or list(battle.opponent_team.values())
        foes = [mon_from_species(i, _species_name(m)) for i, m in enumerate(foe_src, start=1)]

        scored, pmeta = best_expectimax_preview_with_meta(our, foes)
        order = scored.order
        self.last_preview_meta = pmeta
        selected = set(order)
        for i, mon in enumerate(battle.teampreview_team or [], start=1):
            try:
                mon._selected_in_teampreview = i in selected  # type: ignore[attr-defined]
            except Exception:
                pass
        self.last_preview_order = order_to_team_string(order)
        return f'/team {self.last_preview_order}'

    def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        return self._choose_move_guarded(battle)

    async def _choose_move_guarded(self, battle: DoubleBattle) -> BattleOrder:
        """Wall-clock duro: un turno lento no debe tumbar el BO3 (fail_timeout)."""
        cfg = getattr(self, 'expectimax_config', None) or load_expectimax_config()
        budget_s = max(2.5, float(cfg.deadline_ms) / 1000.0 + 2.0)
        try:
            return await asyncio.wait_for(
                self._choose_move_async(battle),
                timeout=budget_s,
            )
        except Exception:
            return DefaultBattleOrder()

    async def _choose_move_async(self, battle: DoubleBattle) -> BattleOrder:
        if any(battle.force_switch):
            return self._forced_switch_order(battle)

        actives = list(battle.active_pokemon)
        foes_raw = list(battle.opponent_active_pokemon)
        foes: List[Tuple[int, object]] = []
        foe_known: Dict[int, Sequence[str]] = {}
        for idx, foe in enumerate(foes_raw):
            if foe is None or foe.fainted:
                continue
            tid = battle.OPPONENT_1_POSITION if idx == 0 else battle.OPPONENT_2_POSITION
            foes.append((tid, _poke_to_calc(foe)))
            foe_known[tid] = _pe_known_moves(foe)

        iap = getattr(self, 'iap_battle', None)
        beliefs = getattr(iap, 'stat_beliefs', None) if iap is not None else None
        if iap is not None:
            enriched = []
            for tid, calc in foes:
                matched = None
                for attr in ('active1', 'active2'):
                    mon = getattr(getattr(iap, 'opponent', None), attr, None)
                    if mon is None:
                        continue
                    if str(getattr(mon, 'name', '')).lower().replace('-', '') == (
                        calc.species.lower().replace('-', '').replace(' ', '')
                    ):
                        matched = mon
                        break
                if matched is not None:
                    if beliefs is not None:
                        try:
                            apply_belief_stats(calc, beliefs.belief_for(matched))
                        except Exception:
                            pass
                    vs = getattr(matched, 'volatile_statuses', None) or {}
                    calc.has_substitute = bool(vs.get('substitute'))
                    calc.is_redirecting = bool(getattr(matched, 'is_redirecting', False))
                    calc.is_protected = bool(getattr(matched, 'is_protected', False))
                    from showdown.battle_bots.expectimax_bayesian.belief import extract_known_moves
                    iap_moves = extract_known_moves(matched)
                    if iap_moves:
                        foe_known[tid] = iap_moves
                enriched.append((tid, calc))
            foes = enriched
            # OTS plantilla
            for opp in getattr(getattr(iap, 'opponent', None), 'opponents', None) or []:
                from showdown.battle_bots.expectimax_bayesian.belief import extract_known_moves
                known = extract_known_moves(opp)
                if not known:
                    continue
                for tid, calc in foes:
                    if foe_known.get(tid):
                        continue
                    if str(getattr(opp, 'name', '')).lower().replace('-', '') == (
                        calc.species.lower().replace('-', '').replace(' ', '')
                    ):
                        foe_known[tid] = known

        weather = _weather_str(battle)
        terrain = _terrain_str(battle)
        field_ctx = _field_context_from_pe(battle)

        left_mon = actives[0] if len(actives) > 0 else None
        right_mon = actives[1] if len(actives) > 1 else None
        left_moves_raw = battle.available_moves[0] if len(battle.available_moves) > 0 else []
        right_moves_raw = battle.available_moves[1] if len(battle.available_moves) > 1 else []

        left_opts = [_move_option(m, i + 1) for i, m in enumerate(left_moves_raw)]
        right_opts = [_move_option(m, i + 1) for i, m in enumerate(right_moves_raw)]

        left_calc = _poke_to_calc(left_mon) if left_mon and not left_mon.fainted else None
        right_calc = _poke_to_calc(right_mon) if right_mon and not right_mon.fainted else None
        our_fainted = _count_pe_fainted(list(battle.team.values()) if getattr(battle, 'team', None) else [])
        if left_calc is not None:
            left_calc.allies_fainted = our_fainted
        if right_calc is not None:
            right_calc.allies_fainted = our_fainted
        foe_fainted = _count_pe_fainted(
            list(battle.opponent_team.values()) if getattr(battle, 'opponent_team', None) else []
        )
        for _tid, foe in foes:
            foe.allies_fainted = foe_fainted

        can_mega_l = bool(battle.can_mega_evolve[0]) if battle.can_mega_evolve else False
        can_mega_r = bool(battle.can_mega_evolve[1]) if len(battle.can_mega_evolve) > 1 else False
        from showdown.format_rules import allow_tera
        can_tera_l = allow_tera(bool(battle.can_tera[0]) if battle.can_tera else False)
        can_tera_r = allow_tera(
            bool(battle.can_tera[1]) if battle.can_tera and len(battle.can_tera) > 1 else False
        )
        tera_l = str(left_mon.tera_type.name) if left_mon and left_mon.tera_type else None
        tera_r = str(right_mon.tera_type.name) if right_mon and right_mon.tera_type else None

        # Reserves: map available switches to stable slots 3+
        reserves: List[ReserveOption] = []
        switch_by_slot: Dict[int, Pokemon] = {}
        seen = set()
        slot = 3
        for i in range(2):
            for mon in (battle.available_switches[i] if i < len(battle.available_switches) else []):
                if mon is None or mon in seen or getattr(mon, 'fainted', False):
                    continue
                seen.add(mon)
                types = tuple(
                    str(t.name).capitalize() if hasattr(t, 'name') else str(t)
                    for t in (mon.types or []) if t
                ) or types_from_species(_species_name(mon))
                reserves.append(ReserveOption(slot=slot, poke=_poke_to_calc(mon), types=types))
                switch_by_slot[slot] = mon
                slot += 1

        best_l, best_r, total, emeta = await best_expectimax_bayesian_action(
            left_calc, right_calc, left_opts, right_opts, foes,
            weather=weather, terrain=terrain,
            can_mega_left=can_mega_l, can_mega_right=can_mega_r,
            can_tera_left=can_tera_l, can_tera_right=can_tera_r,
            tera_left=tera_l, tera_right=tera_r,
            field_ctx=field_ctx,
            beliefs=beliefs,
            position=_position_from_pe(
                battle,
                can_tera_l=can_tera_l, can_tera_r=can_tera_r,
                can_mega_l=can_mega_l, can_mega_r=can_mega_r,
            ),
            config=self.expectimax_config,
            foe_known_moves=foe_known,
            reserves=reserves,
            ots_sheet=getattr(self, 'opp_ots_sheet', None) or {},
        )
        self.last_decision_score = total
        self.last_expectimax_meta = emeta
        self.note_expectimax_decision(emeta)

        def to_order(
            mon: Optional[Pokemon],
            moves: Sequence[Move],
            action,
            can_mega: bool,
            can_tera: bool,
        ):
            if mon is None or mon.fainted:
                return PassBattleOrder()
            if action is not None and getattr(action, 'kind', 'move') == 'switch':
                target = switch_by_slot.get(int(action.move_slot))
                if target is not None:
                    return SingleBattleOrder(target)
                switches = []
                idx = 0 if mon is left_mon else 1
                if idx < len(battle.available_switches):
                    switches = list(battle.available_switches[idx])
                if switches:
                    return SingleBattleOrder(switches[0])
                return DefaultBattleOrder()
            if action is None or not moves:
                return DefaultBattleOrder()
            move = moves[action.move_slot - 1] if 0 < action.move_slot <= len(moves) else moves[0]
            targets = battle.get_possible_showdown_targets(move, mon)
            target = action.target if action.target in targets else (targets[0] if targets else 0)
            use_tera = allow_tera(action.mega == 2 and can_tera and action.ko)
            use_mega = can_mega and (action is None or action.mega != 2)
            return Player.create_order(
                move,
                mega=use_mega,
                terastallize=use_tera,
                move_target=target,
            )

        o1 = to_order(left_mon, left_moves_raw, best_l, can_mega_l, can_tera_l)
        o2 = to_order(right_mon, right_moves_raw, best_r, can_mega_r, can_tera_r)
        if (
            best_l is not None and best_r is not None
            and getattr(best_l, 'kind', '') == 'switch'
            and getattr(best_r, 'kind', '') == 'switch'
            and int(best_l.move_slot) == int(best_r.move_slot)
        ):
            o2 = DefaultBattleOrder()
        o1, o2 = enforce_single_mega(
            o1, o2,
            float(getattr(best_l, 'total_damage_pct', None) or getattr(best_l, 'damage_pct', 0.0) or 0.0), float(getattr(best_r, 'total_damage_pct', None) or getattr(best_r, 'damage_pct', 0.0) or 0.0),
        )
        return DoubleBattleOrder(o1, o2)
