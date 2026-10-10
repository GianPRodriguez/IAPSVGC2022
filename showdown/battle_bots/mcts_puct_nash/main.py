"""IAP BattleBot for MCTS–PUCT/Nash — Fixed Export Order + SM-MCTS B1."""
from __future__ import annotations

import math
import os

from API.MoveData import getColorMove
import constants
from showdown.battle import Battle
from showdown.battle_bots.ddqn.iap_obs import joint_to_iap_action, snapshot_from_iap
from showdown.battle_bots.helpers import getMoveName, getToMovePokemon
from showdown.rl.action_space import encode_joint_action
from showdown.rl.constants import PASS_ID
from showdown.rl.mcts_puct_nash.agent import MCTSPuctNashAgent, MCTSPuctNashConfig


class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sims = int(os.environ.get('MCTS_NASH_SIMS', os.environ.get('MCTS_SIMS', '50')))
        seed = int(os.environ.get('MCTS_NASH_SEED', os.environ.get('MCTS_SEED', '0')))
        self._agent = MCTSPuctNashAgent(MCTSPuctNashConfig(n_sims=sims, seed=seed))

    async def action_team_preview(self, battle: Battle) -> tuple:
        pokemon_list = battle.request_json.get('side', {}).get('pokemon', [])
        names = []
        for idx in (1, 2, 3, 4):
            if 0 <= idx - 1 < len(pokemon_list):
                name = pokemon_list[idx - 1].get('details', '').partition(',')[0]
            else:
                name = 'Unknown'
            names.append(name)
        if getattr(battle, 'q', None) is not None:
            battle.q.put([1, names[0], names[1], names[2], names[3], 0, 0])
        return (1, 2, 3, 4)

    def _choose_switch_slot(self, battle: Battle) -> int:
        import random
        switch3_fainted = battle.user.reserved[0].is_fainted
        switch4_fainted = battle.user.reserved[1].is_fainted
        if not switch3_fainted and switch4_fainted:
            return 3
        if switch3_fainted and not switch4_fainted:
            return 4
        return random.choice([3, 4])

    async def action_battle(self, battle: Battle, _) -> tuple:
        if battle.force_switch:
            return await self._force_switch_action(battle)
        try:
            snap = snapshot_from_iap(battle)
            joint = self._agent.act(snap)
            k1, n1, m1, t1, k2, n2, m2, t2 = joint_to_iap_action(joint, battle)
        except Exception:
            j = encode_joint_action(PASS_ID, PASS_ID)
            k1, n1, m1, t1, k2, n2, m2, t2 = joint_to_iap_action(j, battle)

        text1, color1, text3, color3 = await self._describe(battle, 0, k1, n1, t1)
        text2, color2, text4, color4 = await self._describe(battle, 1, k2, n2, t2)
        last_reward, reward = self.calculateRewards()
        delta = math.floor(last_reward)
        total = math.floor(reward)
        color5 = '#325928' if delta > 0 else ('#565828' if delta >= 0 else '#582831')
        color6 = '#325928' if total > 0 else '#582831'
        if getattr(battle, 'q', None) is not None:
            battle.q.put([
                2, text1, color1, text2, color2, text3, color3, text4, color4,
                delta, color5, total, color6,
            ])
        return (
            k1, int(n1), int(m1), int(t1),
            k2, int(n2), int(m2), int(t2),
            float(last_reward), float(reward),
        )

    async def _describe(self, battle, active_index, kind, n, target):
        if kind == 'move':
            text = await getMoveName(battle, active_index, max(0, n - 1))
            color = await getColorMove(text.lower().replace(' ', '-'))
            t3 = await getToMovePokemon(battle, target)
            return text, color, t3, '#FFFFFF'
        if kind == 'pass':
            return 'Pass', '#FFFFFF', 'Pass', '#FFFFFF'
        if kind == 'switch':
            name = (
                battle.user.reserved[n - 3].name
                if len(battle.user.reserved) > (n - 3)
                else 'Pkmn'
            )
            return 'Switch to', '#FFFFFF', name, '#FFFFFF'
        return 'Pass', '#FFFFFF', 'Pass', '#FFFFFF'

    async def _force_switch_action(self, battle: Battle) -> tuple:
        force_switches = battle.request_json.get(constants.FORCE_SWITCH, [False, False])
        left = force_switches[0] if len(force_switches) > 0 else False
        right = force_switches[1] if len(force_switches) > 1 else False
        move1, n1, move2, n2 = 'pass', 0, 'pass', 0
        if left and not right:
            move1, n1, move2 = 'switch', self._choose_switch_slot(battle), 'pass'
        elif not left and right:
            move1, move2, n2 = 'pass', 'switch', self._choose_switch_slot(battle)
        else:
            move1, move2 = 'switch', 'switch'
            s3 = battle.user.reserved[0].is_fainted
            s4 = battle.user.reserved[1].is_fainted
            if s3 and not s4:
                n1, move2 = 4, 'pass'
            elif not s3 and s4:
                n1, move2 = 3, 'pass'
            elif not s3 and not s4:
                n1, n2 = 3, 4
        last_reward, reward = self.calculateRewards()
        return (
            move1, int(n1), 0, 0,
            move2, int(n2), 0, 0,
            float(last_reward), float(reward),
        )
