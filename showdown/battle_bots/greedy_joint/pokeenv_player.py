"""GreedyJointPlayer: poke-env con búsqueda conjunta 1-ply (Greedy-Joint)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from poke_env.battle import AbstractBattle, DoubleBattle, Move, Pokemon
from poke_env.player import Player
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)
from poke_env.ps_client import LocalhostServerConfiguration

from showdown.battle_bots.greedy_solo.calc_async import warm_damage_calc
from showdown.battle_bots.greedy_solo.species_names import resolve_species_name
from showdown.battle_bots.greedy_solo.greedy import (
    CalcPoke,
    FieldContext,
    MoveOption,
    apply_belief_stats,
    best_forced_switch_slot,
    types_from_species,
)
from showdown.battle_bots.greedy_joint.joint import best_joint_double_action
from showdown.battle_bots.greedy_solo.team_preview import (
    best_team_preview_order,
    mon_from_species,
    order_to_team_string,
)

_MOVES_PATH = Path(__file__).resolve().parents[3] / 'API' / 'JSONs' / 'moves.json'


@lru_cache(maxsize=1)
def _moves_dex() -> dict:
    try:
        return json.loads(_MOVES_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {}


@lru_cache(maxsize=2048)
def _move_name_cached(move_id: str) -> str:
    mid = str(move_id)
    data = _moves_dex()
    entry = data.get(mid) or data.get(mid.replace('-', ''))
    if entry and entry.get('name'):
        return entry['name']
    return mid.replace('-', ' ').title()


def _species_name(mon: Pokemon) -> str:
    for attr in ('species', 'name', 'nickname'):
        val = getattr(mon, attr, None)
        if val:
            return resolve_species_name(str(val))
    return 'Magikarp'


def _move_name(move: Move) -> str:
    if getattr(move, 'id', None):
        return _move_name_cached(str(move.id))
    return str(getattr(move, 'name', move))


def _effect_names(mon: Pokemon) -> set:
    out = set()
    effects = getattr(mon, 'effects', None) or {}
    for key in effects.keys():
        name = key.name if hasattr(key, 'name') else str(key)
        out.add(name.upper().replace('_', '').replace(' ', ''))
    return out


def _poke_to_calc(mon: Pokemon) -> CalcPoke:
    stats = {
        'hp': int(mon.max_hp or 100),
        'atk': int((mon.stats or {}).get('atk') or mon.base_stats.get('atk') or 100),
        'def': int((mon.stats or {}).get('def') or mon.base_stats.get('def') or 100),
        'spa': int((mon.stats or {}).get('spa') or mon.base_stats.get('spa') or 100),
        'spd': int((mon.stats or {}).get('spd') or mon.base_stats.get('spd') or 100),
        'spe': int((mon.stats or {}).get('spe') or mon.base_stats.get('spe') or 100),
    }
    boosts = {
        'atk': int(mon.boosts.get('atk', 0) or 0),
        'def': int(mon.boosts.get('def', 0) or 0),
        'spa': int(mon.boosts.get('spa', 0) or 0),
        'spd': int(mon.boosts.get('spd', 0) or 0),
        'spe': int(mon.boosts.get('spe', 0) or 0),
    }
    item = None
    if mon.item and mon.item not in ('unknown_item', 'unknownitem'):
        item = str(mon.item)
    ability = None
    if mon.ability:
        ability = str(mon.ability)
    status = ''
    if mon.status:
        status = str(mon.status.name if hasattr(mon.status, 'name') else mon.status).lower()
        if status in ('none', 'empty'):
            status = ''
    types = tuple(
        str(t.name).capitalize() if hasattr(t, 'name') else str(t)
        for t in (mon.types or []) if t
    )
    species = _species_name(mon)
    effects = _effect_names(mon)
    tera = None
    is_tera = bool(getattr(mon, 'terastallized', False) or getattr(mon, 'is_terastallized', False))
    if getattr(mon, 'tera_type', None):
        tera = str(mon.tera_type.name if hasattr(mon.tera_type, 'name') else mon.tera_type)
    return CalcPoke(
        species=species,
        nature='Serious',
        item=item,
        ability=ability,
        status=status,
        boosts=boosts,
        stats=stats,
        cur_hp=int(mon.current_hp or stats['hp']),
        types=types or types_from_species(species),
        ability_on=bool(effects & {'FLASHFIRE', 'SLOWSTART', 'ELECTROMORPHOSIS'}),
        just_switched=bool(getattr(mon, 'first_turn', False)),
        tera_type=tera,
        is_terastallized=is_tera,
        is_helping_hand='HELPINGHAND' in effects,
        is_charged='CHARGE' in effects,
        is_protected=bool(effects & {
            'PROTECT', 'DETECT', 'SPIKYSHIELD', 'BANEFULBUNKER', 'KINGSSHIELD',
            'OBSTRUCT', 'SILKTRAP', 'BURNINGBULWARK', 'MAXGUARD',
        }),
        has_substitute='SUBSTITUTE' in effects,
        is_redirecting=bool(effects & {'FOLLOWME', 'RAGEPOWDER', 'SPOTLIGHT'}),
        allies_fainted=0,
        is_two_turn_charging=bool(getattr(mon, 'preparing', False)),
        must_recharge=bool(getattr(mon, 'must_recharge', False))
        or ('MUSTRECHARGE' in effects),
    )


def _count_pe_fainted(mons) -> int:
    n = 0
    for mon in mons or []:
        if mon is None:
            continue
        if getattr(mon, 'fainted', False) or int(getattr(mon, 'current_hp', 1) or 0) <= 0:
            n += 1
    return n


def _move_option(move: Move, slot_hint: int) -> MoveOption:
    cat = 'Status'
    if move.category:
        cat = str(move.category.name if hasattr(move.category, 'name') else move.category)
    return MoveOption(
        slot=slot_hint,
        name=_move_name(move),
        target_type=str(move.target.name if hasattr(move.target, 'name') else (move.target or 'normal')),
        category=cat,
    )


def _weather_str(battle: DoubleBattle) -> Optional[str]:
    w = battle.weather
    if not w:
        return None
    if isinstance(w, dict):
        # {Weather.SUNNYDAY: turn}
        for key in w.keys():
            name = key.name if hasattr(key, 'name') else str(key)
            return name.lower().replace('_', '')
        return None
    return str(getattr(w, 'name', w)).lower().replace('_', '')


def _terrain_str(battle: DoubleBattle) -> Optional[str]:
    fields = getattr(battle, 'fields', None) or {}
    for key in fields.keys():
        name = key.name if hasattr(key, 'name') else str(key)
        low = name.lower().replace('_', '')
        if 'terrain' in low or low in ('electricterrain', 'grassyterrain', 'mistyterrain', 'psychicterrain'):
            return low
    return None


def _field_key(key) -> str:
    name = key.name if hasattr(key, 'name') else str(key)
    return name.upper().replace('_', '').replace(' ', '')


def _side_has(conds, *names: str) -> bool:
    want = {n.upper().replace('_', '') for n in names}
    for key in (conds or {}).keys():
        if _field_key(key) in want:
            return True
    return False


def _field_context_from_pe(battle: DoubleBattle) -> FieldContext:
    weather = _weather_str(battle)
    terrain = _terrain_str(battle)
    fields = getattr(battle, 'fields', None) or {}
    field_ids = {_field_key(k) for k in fields.keys()}
    opp_sc = getattr(battle, 'opponent_side_conditions', None) or {}

    actives = []
    for mon in list(battle.active_pokemon or []) + list(battle.opponent_active_pokemon or []):
        if mon is not None and not getattr(mon, 'fainted', False):
            actives.append(mon)

    def _has_abi(ab: str) -> bool:
        for mon in actives:
            raw = getattr(mon, 'ability', None)
            if not raw:
                continue
            name = raw.name if hasattr(raw, 'name') else str(raw)
            if name.lower().replace(' ', '').replace('-', '').replace('_', '') == ab:
                return True
        return False

    return FieldContext(
        weather=weather,
        terrain=terrain,
        wonder_room='WONDERROOM' in field_ids,
        magic_room='MAGICROOM' in field_ids,
        gravity='GRAVITY' in field_ids,
        reflect=_side_has(opp_sc, 'REFLECT'),
        light_screen=_side_has(opp_sc, 'LIGHTSCREEN'),
        aurora_veil=_side_has(opp_sc, 'AURORAVEIL'),
        is_beads_of_ruin=_has_abi('beadsofruin'),
        is_sword_of_ruin=_has_abi('swordofruin'),
        is_tablets_of_ruin=_has_abi('tabletsofruin'),
        is_vessel_of_ruin=_has_abi('vesselofruin'),
    )


class GreedyJointPlayer(Player):
    """
    Greedy-Joint (poke-env):
      - Team Preview: cobertura tipada + BST
      - Combate: 1-ply sobre A1×A2 (daño combinado + redirect + utilidades)
      - Switches solo en force_switch (nunca voluntarios)
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('server_configuration', LocalhostServerConfiguration)
        kwargs.setdefault('accept_open_team_sheet', True)
        kwargs.setdefault('max_concurrent_battles', 1)
        super().__init__(*args, **kwargs)
        self.last_preview_order: str = '1234'
        self.last_decision_score: float = 0.0
        self._left_rooms: set = set()
        # Precalienta calc (TCP/stdio) fuera del primer turno crítico
        try:
            warm_damage_calc()
        except Exception:
            pass
        # Precarga dex de movimientos / species index
        try:
            _moves_dex()
        except Exception:
            pass

    def _battle_finished_callback(self, battle: AbstractBattle):
        tag = getattr(battle, 'battle_tag', None)
        if tag:
            self._left_rooms.add(str(tag).lstrip('>'))
        return super()._battle_finished_callback(battle)

    async def _handle_battle_request(
        self, battle: AbstractBattle, maybe_default_order: bool = False
    ):
        """Evita /choose a salas BO3 ya cerradas (popup 'not in that room')."""
        if getattr(battle, 'finished', False):
            return
        tag = getattr(battle, 'battle_tag', None)
        if tag and str(tag).lstrip('>') in self._left_rooms:
            return
        battles = getattr(self, 'battles', None) or {}
        if tag and battles and tag not in battles and maybe_default_order:
            return
        try:
            await super()._handle_battle_request(battle, maybe_default_order=maybe_default_order)
        except Exception:
            return

    def teampreview(self, battle: AbstractBattle) -> str:
        assert isinstance(battle, DoubleBattle)
        our: List = []
        for i, mon in enumerate(battle.teampreview_team or list(battle.team.values()), start=1):
            our.append(mon_from_species(i, _species_name(mon)))

        foe_src = battle.teampreview_opponent_team or list(battle.opponent_team.values())
        foes = [mon_from_species(i, _species_name(m)) for i, m in enumerate(foe_src, start=1)]

        scored = best_team_preview_order(our, foes)
        order = scored.order
        # Marcar seleccionados (requisito poke-env)
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
        return self._choose_move_async(battle)

    async def _choose_move_async(self, battle: DoubleBattle) -> BattleOrder:
        if any(battle.force_switch):
            return self._forced_switch_order(battle)

        actives = list(battle.active_pokemon)
        foes_raw = list(battle.opponent_active_pokemon)
        foes: List[Tuple[int, CalcPoke]] = []
        for idx, foe in enumerate(foes_raw):
            if foe is None or foe.fainted:
                continue
            tid = battle.OPPONENT_1_POSITION if idx == 0 else battle.OPPONENT_2_POSITION
            foes.append((tid, _poke_to_calc(foe)))

        # Si hay espejo IAP (ValidatedGreedy), enriquecer foes con belief SP.
        iap = getattr(self, 'iap_battle', None)
        if iap is not None:
            enriched: List[Tuple[int, CalcPoke]] = []
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
                    beliefs = getattr(iap, 'stat_beliefs', None)
                    if beliefs is not None:
                        try:
                            apply_belief_stats(calc, beliefs.belief_for(matched))
                        except Exception:
                            pass
                    vs = getattr(matched, 'volatile_statuses', None) or {}
                    calc.has_substitute = bool(vs.get('substitute'))
                    calc.is_redirecting = bool(getattr(matched, 'is_redirecting', False))
                    calc.is_protected = bool(getattr(matched, 'is_protected', False))
                enriched.append((tid, calc))
            foes = enriched

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

        best_l, best_r, total = await best_joint_double_action(
            left_calc, right_calc, left_opts, right_opts, foes,
            weather=weather, terrain=terrain,
            can_mega_left=can_mega_l, can_mega_right=can_mega_r,
            can_tera_left=can_tera_l, can_tera_right=can_tera_r,
            tera_left=tera_l, tera_right=tera_r,
            field_ctx=field_ctx,
        )
        self.last_decision_score = total

        # Greedy-Joint: no voluntary switches (0%). Only force_switch path above.
        def to_order(
            mon: Optional[Pokemon],
            moves: Sequence[Move],
            action,
            can_mega: bool,
            can_tera: bool,
        ) -> SingleBattleOrder:
            if mon is None or mon.fainted:
                return PassBattleOrder()
            if action is None or not moves:
                # Prefer Default/Pass over voluntary switch (1-ply has no switch value).
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
        return DoubleBattleOrder(o1, o2)

    def _forced_switch_order(self, battle: DoubleBattle) -> BattleOrder:
        foe_types = []
        for foe in battle.opponent_active_pokemon:
            if foe and not foe.fainted:
                foe_types.append(
                    tuple(
                        str(t.name).capitalize() if hasattr(t, 'name') else str(t)
                        for t in foe.types if t
                    )
                )
        orders: List[SingleBattleOrder] = []
        used = set()
        for i, need in enumerate(battle.force_switch):
            if not need:
                mon = battle.active_pokemon[i] if i < len(battle.active_pokemon) else None
                if mon is None or mon.fainted:
                    orders.append(PassBattleOrder())
                else:
                    orders.append(DefaultBattleOrder())
                continue
            switches = [s for s in battle.available_switches[i] if s not in used]
            if not switches:
                orders.append(DefaultBattleOrder())
                continue
            ranked = []
            for s in switches:
                types = tuple(
                    str(t.name).capitalize() if hasattr(t, 'name') else str(t)
                    for t in s.types if t
                ) or types_from_species(_species_name(s))
                # fake slot index
                ranked.append((s, types))
            # Reusar scorer: map to slots
            reserved = [(idx, types) for idx, (_s, types) in enumerate(ranked)]
            best_i = 0
            if reserved and foe_types:
                best_slot = best_forced_switch_slot(reserved, foe_types)
                best_i = int(best_slot)
            pick = ranked[best_i][0]
            used.add(pick)
            orders.append(SingleBattleOrder(pick))
        while len(orders) < 2:
            orders.append(PassBattleOrder())
        return DoubleBattleOrder(orders[0], orders[1])
