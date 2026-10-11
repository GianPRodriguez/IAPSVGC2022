"""Greedy-Joint: Team Preview + 1-ply joint action search (A1 × A2)."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import constants
from API.MoveData import getColorMove
from showdown.battle import Battle
from showdown.battle_bots.helpers import getMoveName, getToMovePokemon
from showdown.battle_bots.greedy_solo.greedy import (
    CalcPoke,
    MoveOption,
    any_positive_damage,
    apply_belief_stats,
    best_forced_switch_slot,
    field_context_from_iap,
    types_from_species,
)
from showdown.battle_bots.greedy_joint.joint import best_joint_double_action
from showdown.battle_bots.greedy_solo.team_preview import (
    best_team_preview_order,
    mon_from_details,
    mon_from_species,
)
from showdown.battle_bots.move_mechanics import (
    is_must_recharge_from_volatiles,
    is_two_turn_charging_from_volatiles,
)


def _count_iap_fainted(side) -> int:
    n = 0
    for p in (
        getattr(side, 'active1', None),
        getattr(side, 'active2', None),
        *(getattr(side, 'reserved', None) or []),
    ):
        if p is None:
            continue
        if getattr(p, 'is_fainted', False) or int(getattr(p, 'HPA_battle_actual', 1) or 0) <= 0:
            n += 1
    return n


def _iap_to_calc(pkmn, battle=None) -> CalcPoke:
    max_hp = int(getattr(pkmn, 'HPB_battle_base', 100) or 100)
    cur_hp = int(getattr(pkmn, 'HPA_battle_actual', max_hp) or max_hp)
    stats = {
        'hp': max_hp,
        'atk': int(getattr(pkmn, 'ATK_battle_actual', 0) or getattr(pkmn, 'ATK_battle_base', 100) or 100),
        'def': int(getattr(pkmn, 'DEF_battle_actual', 0) or getattr(pkmn, 'DEF_battle_base', 100) or 100),
        'spa': int(getattr(pkmn, 'SPA_battle_actual', 0) or getattr(pkmn, 'SPA_battle_base', 100) or 100),
        'spd': int(getattr(pkmn, 'SPD_battle_actual', 0) or getattr(pkmn, 'SPD_battle_base', 100) or 100),
        'spe': int(getattr(pkmn, 'VEL_battle_actual', 0) or getattr(pkmn, 'VEL_battle_base', 100) or 100),
    }
    boosts = {
        'atk': int(pkmn.boosts.get('atk', 0) or 0),
        'def': int(pkmn.boosts.get('def', 0) or 0),
        'spa': int(pkmn.boosts.get('spa', 0) or 0),
        'spd': int(pkmn.boosts.get('spd', 0) or 0),
        'spe': int(pkmn.boosts.get('spe', 0) or 0),
    }
    status = ''
    for st, val in (pkmn.non_volatile_statuses or {}).items():
        if val:
            status = str(st)
            break
    types = tuple(t for t in (pkmn.type1, pkmn.type2) if t)
    volatiles = getattr(pkmn, 'volatile_statuses', None) or {}
    allies_fainted = 0
    if battle is not None:
        side = battle.opponent if getattr(pkmn, 'is_opponent', False) else battle.user
        allies_fainted = _count_iap_fainted(side)
    poke = CalcPoke(
        species=str(pkmn.name),
        nature=str(pkmn.nature or 'Serious'),
        item=pkmn.objectPkmn,
        ability=pkmn.abilityPkmn,
        status=status,
        boosts=boosts,
        stats=stats,
        cur_hp=max(cur_hp, 1),
        types=types or types_from_species(str(pkmn.name)),
        ability_on=bool(getattr(pkmn, 'ability_on', False)),
        just_switched=bool(getattr(pkmn, 'just_switched', False)),
        tera_type=getattr(pkmn, 'tera_type', None),
        is_terastallized=bool(getattr(pkmn, 'is_terastallized', False)),
        is_helping_hand=bool(getattr(pkmn, 'is_helping_hand', False)),
        is_charged=bool(getattr(pkmn, 'is_charged', False)),
        is_protected=bool(getattr(pkmn, 'is_protected', False)),
        has_substitute=bool(volatiles.get('substitute')),
        is_redirecting=bool(getattr(pkmn, 'is_redirecting', False)),
        allies_fainted=allies_fainted,
        is_two_turn_charging=is_two_turn_charging_from_volatiles(volatiles),
        must_recharge=bool(getattr(pkmn, 'must_recharge', False))
        or is_must_recharge_from_volatiles(volatiles),
    )
    if battle is not None and getattr(pkmn, 'is_opponent', False):
        beliefs = getattr(battle, 'stat_beliefs', None)
        if beliefs is not None:
            try:
                apply_belief_stats(poke, beliefs.belief_for(pkmn))
            except Exception:
                pass
    return poke


class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_preview_order: Tuple[int, ...] = (1, 2, 3, 4)

    def _choose_switch_slot(self, battle: Battle) -> int:
        reserved = []
        foe_types = []
        for foe in (battle.opponent.active1, battle.opponent.active2):
            if foe and not foe.is_fainted:
                foe_types.append(tuple(t for t in (foe.type1, foe.type2) if t))
        for slot, mon in enumerate(battle.user.reserved, start=3):
            if mon and not mon.is_fainted:
                types = tuple(t for t in (mon.type1, mon.type2) if t) or types_from_species(mon.name)
                reserved.append((slot, types))
        if not reserved:
            return 3
        return best_forced_switch_slot(reserved, foe_types or [tuple()])

    def _get_available_moves(self, battle: Battle, active_index: int) -> list[int]:
        active_pokemon_data = battle.request_json.get('active', [])
        if active_index >= len(active_pokemon_data):
            return []
        moves_data = active_pokemon_data[active_index].get('moves', [])
        available = []
        for i, move_info in enumerate(moves_data):
            if not move_info.get('disabled', False) and move_info.get('pp', 1) > 0:
                available.append(i + 1)
        return available

    def _move_options(self, battle: Battle, active_index: int) -> List[MoveOption]:
        active_pokemon_data = battle.request_json.get('active', [])
        if active_index >= len(active_pokemon_data):
            return []
        moves_data = active_pokemon_data[active_index].get('moves', [])
        out: List[MoveOption] = []
        for i, info in enumerate(moves_data):
            if info.get('disabled', False) or info.get('pp', 1) <= 0:
                continue
            name = info.get('move') or info.get('id') or ''
            out.append(MoveOption(
                slot=i + 1,
                name=str(name),
                target_type=str(info.get('target', 'normal')),
                category='Physical',  # el calc resuelve categoría real
            ))
        return out

    async def action_team_preview(self, battle: Battle) -> tuple[int, ...]:
        pokemon_list = battle.request_json.get('side', {}).get('pokemon', [])
        our = [
            mon_from_details(i, p.get('details', ''))
            for i, p in enumerate(pokemon_list[:6], start=1)
        ]
        foes = []
        for i, opp in enumerate(getattr(battle.opponent, 'opponents', []) or [], start=1):
            foes.append(mon_from_species(i, getattr(opp, 'name', f'foe{i}')))

        scored = best_team_preview_order(our, foes)
        order = scored.order
        self.last_preview_order = order

        names = []
        for idx in order:
            if 0 <= idx - 1 < len(pokemon_list):
                names.append(pokemon_list[idx - 1].get('details', '').partition(',')[0])
            else:
                names.append('Unknown')
        while len(names) < 4:
            names.append('Unknown')

        if getattr(battle, 'q', None) is not None:
            battle.q.put([1, names[0], names[1], names[2], names[3], 0, 0])

        return order

    async def action_battle(self, battle: Battle, _) -> tuple:
        moveSwitch1 = 'pass'
        nMoveSwitch1 = 0
        megaEvo1 = 0
        toMove1 = 0
        moveSwitch2 = 'pass'
        nMoveSwitch2 = 0
        megaEvo2 = 0
        toMove2 = 0

        request_active = battle.request_json.get('active', [])

        if battle.force_switch:
            force_switches = battle.request_json.get(constants.FORCE_SWITCH, [False, False])
            izquierda = force_switches[0] if len(force_switches) > 0 else False
            derecha = force_switches[1] if len(force_switches) > 1 else False

            if izquierda and not derecha:
                moveSwitch1, moveSwitch2 = 'switch', 'pass'
                nMoveSwitch1 = self._choose_switch_slot(battle)
            elif not izquierda and derecha:
                moveSwitch1, moveSwitch2 = 'pass', 'switch'
                nMoveSwitch2 = self._choose_switch_slot(battle)
            else:
                moveSwitch1 = moveSwitch2 = 'switch'
                s3 = battle.user.reserved[0].is_fainted
                s4 = battle.user.reserved[1].is_fainted
                if s3 and not s4:
                    nMoveSwitch1, moveSwitch2 = 4, 'pass'
                elif not s3 and s4:
                    nMoveSwitch1, moveSwitch2 = 3, 'pass'
                elif not s3 and not s4:
                    # Mejor tipado primero; el otro ocupa el slot restante
                    best = self._choose_switch_slot(battle)
                    nMoveSwitch1 = best
                    nMoveSwitch2 = 4 if best == 3 else 3
        else:
            foes: List[Tuple[int, CalcPoke]] = []
            if battle.opponent.active1 and not battle.opponent.active1.is_fainted:
                foes.append((1, _iap_to_calc(battle.opponent.active1, battle)))
            if battle.opponent.active2 and not battle.opponent.active2.is_fainted:
                foes.append((2, _iap_to_calc(battle.opponent.active2, battle)))

            left = battle.user.active1 if not battle.user.active1.is_fainted else None
            right = battle.user.active2 if not battle.user.active2.is_fainted else None
            left_opts = self._move_options(battle, 0) if left else []
            right_opts = self._move_options(battle, 1) if right else []

            weather = battle.weather
            terrain = None
            for k, v in (battle.terrain or {}).items():
                if v:
                    terrain = str(k)
                    break
            field_ctx = field_context_from_iap(battle)

            a1 = request_active[0] if len(request_active) > 0 else {}
            a2 = request_active[1] if len(request_active) > 1 else {}
            can_mega_l = bool(a1.get('canMegaEvo', False))
            can_mega_r = bool(a2.get('canMegaEvo', False))
            can_tera_l = bool(a1.get('canTerastallize', False) or a1.get('canTera', False))
            can_tera_r = bool(a2.get('canTerastallize', False) or a2.get('canTera', False))

            best_l, best_r, _ = await best_joint_double_action(
                _iap_to_calc(left, battle) if left else None,
                _iap_to_calc(right, battle) if right else None,
                left_opts,
                right_opts,
                foes,
                weather=weather,
                terrain=terrain,
                can_mega_left=can_mega_l,
                can_mega_right=can_mega_r,
                can_tera_left=can_tera_l,
                can_tera_right=can_tera_r,
                field_ctx=field_ctx,
            )

            # Switches voluntarios: 0% (1-ply miope). Solo force_switch arriba.
            attack_ok = any_positive_damage([best_l, best_r]) or bool(left_opts or right_opts)

            def apply_side(is_left: bool, mon, action, active_info):
                nonlocal moveSwitch1, nMoveSwitch1, megaEvo1, toMove1
                nonlocal moveSwitch2, nMoveSwitch2, megaEvo2, toMove2
                if mon is None or mon.is_fainted:
                    if is_left:
                        moveSwitch1 = 'pass'
                    else:
                        moveSwitch2 = 'pass'
                    return
                if action is None or not attack_ok:
                    # Sin acción joint: move disponible o pass — nunca switch voluntario.
                    avail = self._get_available_moves(battle, 0 if is_left else 1)
                    if avail:
                        if is_left:
                            moveSwitch1, nMoveSwitch1, toMove1 = 'move', avail[0], 1 if foes else 0
                        else:
                            moveSwitch2, nMoveSwitch2, toMove2 = 'move', avail[0], 1 if foes else 0
                    else:
                        if is_left:
                            moveSwitch1 = 'pass'
                        else:
                            moveSwitch2 = 'pass'
                    return

                # Target según tipo de movimiento
                moves_data = active_info.get('moves', [])
                move_info = moves_data[action.move_slot - 1] if 0 < action.move_slot <= len(moves_data) else {}
                target_type = move_info.get('target', 'normal')
                if 'target' not in move_info:
                    tgt = 0
                elif target_type in constants.MOVETYPES_WITHOUT_PARAMETERS:
                    tgt = 0
                elif target_type in constants.MOVETYPES_SINGLE_TARGET or target_type == constants.MOVETYPES_ADJACENT_FOE:
                    tgt = action.target if action.target in (1, 2) else (foes[0][0] if foes else 1)
                elif target_type == constants.MOVETYPES_ADJACENT_ALLY:
                    tgt = -1 if is_left else -1
                elif target_type == constants.MOVETYPES_ALLY_OR_SELF:
                    tgt = 0
                else:
                    tgt = action.target if action.target else 0

                special = 0
                if action.mega == 2 and (can_tera_l if is_left else can_tera_r) and action.ko:
                    special = 2
                elif action.mega == 1 and (can_mega_l if is_left else can_mega_r):
                    special = 1
                elif (can_mega_l if is_left else can_mega_r):
                    # Mega por defecto aunque el ScoredAction no lo trajera (status, etc.)
                    special = 1

                if is_left:
                    moveSwitch1, nMoveSwitch1, megaEvo1, toMove1 = 'move', action.move_slot, special, tgt
                else:
                    moveSwitch2, nMoveSwitch2, megaEvo2, toMove2 = 'move', action.move_slot, special, tgt

            if left:
                apply_side(True, left, best_l, a1)
            else:
                moveSwitch1 = 'pass'
            if right:
                apply_side(False, right, best_r, a2)
            else:
                moveSwitch2 = 'pass'

            # Solo un mega/tera por turno
            if megaEvo1 and megaEvo2:
                megaEvo2 = 0

        # UI / rewards (mismo contrato que random)
        if moveSwitch1 == 'move':
            text1 = await getMoveName(battle, 0, nMoveSwitch1 - 1)
            color1 = await getColorMove(text1.lower().replace(' ', '-'))
            text3 = await getToMovePokemon(battle, toMove1)
            color3 = '#FFFFFF'
        elif moveSwitch1 == 'pass':
            text1, color1, text3, color3 = 'Pass', '#FFFFFF', 'Pass', '#FFFFFF'
        else:
            text1, color1 = 'Switch to', '#FFFFFF'
            text3 = battle.user.reserved[nMoveSwitch1 - 3].name if len(battle.user.reserved) > (nMoveSwitch1 - 3) else 'Pkmn'
            color3 = '#FFFFFF'

        if moveSwitch2 == 'move':
            text2 = await getMoveName(battle, 1, nMoveSwitch2 - 1)
            color2 = await getColorMove(text2.lower().replace(' ', '-'))
            text4 = await getToMovePokemon(battle, toMove2)
            color4 = '#FFFFFF'
        elif moveSwitch2 == 'pass':
            text2, color2, text4, color4 = 'Pass', '#FFFFFF', 'Pass', '#FFFFFF'
        else:
            text2, color2 = 'Switch to', '#FFFFFF'
            text4 = battle.user.reserved[nMoveSwitch2 - 3].name if len(battle.user.reserved) > (nMoveSwitch2 - 3) else 'Pkmn'
            color4 = '#FFFFFF'

        lastReward, reward = self.calculateRewards()
        delta = round(float(lastReward), 1)
        total = round(float(reward), 1)
        color5 = '#325928' if delta > 0 else ('#565828' if delta >= 0 else '#582831')
        color6 = '#325928' if total > 0 else '#582831'

        if getattr(battle, 'q', None) is not None:
            battle.q.put([
                2, text1, color1, text2, color2, text3, color3, text4, color4,
                delta, color5, total, color6,
            ])

        return (
            moveSwitch1, int(nMoveSwitch1), int(megaEvo1), int(toMove1),
            moveSwitch2, int(nMoveSwitch2), int(megaEvo2), int(toMove2),
            float(lastReward), float(reward),
        )
