"""Bootstrap Battle IAPSVGC y feed de líneas de protocolo PS."""
from __future__ import annotations

from typing import Iterable, List, Optional, Set

from showdown.battle import Battle, Pokemon
from showdown.battle_modifier import BATTLE_MODIFIERS_LOOKUP
from showdown.inference_sim.team_factory import MonSet, TeamMeta
from showdown.stat_inference import STAT_KEYS, calibrate_from_known_team, reset_sp_limits
from teams.team_converter import normalize_name

# Eventos de combate / campo que la inferencia necesita.
_FEED_ACTIONS = frozenset(BATTLE_MODIFIERS_LOOKUP.keys())


class HarnessBattle(Battle):
    async def action_team_preview(self, battle):
        return None

    async def action_battle(self, battle):
        return None


def _build_mon(mon: MonSet, *, is_opponent: bool) -> Pokemon:
    from showdown.stat_inference import sanitize_ability_for_forme

    moves = list(mon.moves) + [None, None, None, None]
    name = normalize_name(mon.species)
    item = normalize_name(mon.item)
    ability = normalize_name(sanitize_ability_for_forme(name, mon.ability) or mon.ability)
    nature = normalize_name(mon.nature)
    # Opponent preview: HP base=-1 → getStatsBattle pone bases de especie y HP% = 100.
    hp_arg = -1 if is_opponent else mon.finals()['hp']
    p = Pokemon(
        name,
        hp_arg,
        is_opponent,
        item,
        ability,
        normalize_name(moves[0]) if moves[0] else None,
        normalize_name(moves[1]) if moves[1] else None,
        normalize_name(moves[2]) if moves[2] else None,
        normalize_name(moves[3]) if moves[3] else None,
        nature,
    )
    if not is_opponent:
        finals = mon.finals()
        p.HPB_battle_base = finals['hp']
        p.HPA_battle_actual = finals['hp']
        p.ATK_battle_actual = finals['atk']
        p.DEF_battle_actual = finals['def']
        p.SPA_battle_actual = finals['spa']
        p.SPD_battle_actual = finals['spd']
        p.VEL_battle_actual = finals['spe']
        # bases de especie también
        p.ATK_battle_base = p.ATK_battle_base if p.ATK_battle_base > 0 else finals['atk']
    else:
        p.is_opponent_confirmed = True
    return p


def bootstrap_battle(user_team: TeamMeta, opp_team: TeamMeta) -> HarnessBattle:
    reset_sp_limits()
    battle = HarnessBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    user_mons = [_build_mon(m, is_opponent=False) for m in user_team.mons]
    opp_mons = [_build_mon(m, is_opponent=True) for m in opp_team.mons]

    # Preview OTS
    battle.opponent.opponents = list(opp_mons)
    # Solo banquillo en reserved: si también están los leads, |switch| puede
    # matchear la misma referencia que active1/2 y dejar el espejo desfasado.
    battle.user.reserved = list(user_mons[2:])
    battle.user.active1 = user_mons[0] if user_mons else None
    battle.user.active2 = user_mons[1] if len(user_mons) > 1 else None
    battle.opponent.active1 = None
    battle.opponent.active2 = None

    calibrate_from_known_team(user_mons)
    # Prefirmar creencias del rival con naturaleza OTS
    for mon in opp_mons:
        battle.stat_beliefs.belief_for(mon)
    return battle


def feed_line(battle: Battle, line: str) -> Optional[str]:
    """Procesa una línea PS. Devuelve la action si se manejó."""
    line = line.strip()
    if not line or line.startswith('>'):
        return None
    if not line.startswith('|'):
        line = '|' + line
    parts = line.split('|')
    if len(parts) < 2:
        return None
    action = parts[1].strip()
    fn = BATTLE_MODIFIERS_LOOKUP.get(action)
    if fn is None:
        return None
    fn(battle, parts)
    return action


def feed_messages(battle: Battle, lines: Iterable[str]) -> List[str]:
    handled = []
    for line in lines:
        action = feed_line(battle, line)
        if action:
            handled.append(action)
    return handled


def snapshot_candidates(battle: Battle) -> dict:
    out = {}
    seen: Set[int] = set()
    for side_attr in ('active1', 'active2'):
        pkmn = getattr(battle.opponent, side_attr, None)
        if pkmn is None or id(pkmn) in seen:
            continue
        seen.add(id(pkmn))
        belief = battle.stat_beliefs.belief_for(pkmn)
        if belief is None:
            continue
        out[normalize_name(pkmn.name)] = {
            k: sorted(belief.candidates[k]) for k in STAT_KEYS
        }
    return out
