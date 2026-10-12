"""IAP BattleBot for QD-PPO — family frozen preview + best elite π*."""
from __future__ import annotations

import math
import os
from pathlib import Path

from API.MoveData import getColorMove
import constants
from showdown.battle import Battle
from showdown.battle_bots.helpers import getMoveName, getToMovePokemon
from showdown.battle_bots.ddqn.iap_obs import joint_to_iap_action, snapshot_from_iap
from showdown.rl.action_space import build_autoregressive_masks, encode_joint_action
from showdown.rl.observation import build_observation_ppo


class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._agent = None
        self._agent_failed = False

    def _get_agent(self):
        if self._agent_failed:
            return None
        if self._agent is not None:
            return self._agent
        try:
            from showdown.rl.qd_ppo.agent import QDPPOAgent
            agent = QDPPOAgent(
                checkpoint=os.environ.get('QD_PPO_CHECKPOINT') or os.environ.get('PPO_CHECKPOINT'),
                archive_path=os.environ.get('QD_PPO_ARCHIVE'),
            )
            self._agent = agent
            return agent
        except Exception:
            self._agent_failed = True
            return None

    async def action_team_preview(self, battle: Battle) -> tuple:
        from showdown.battle_bots._family_preview_iap import iap_family_team_preview
        return iap_family_team_preview('qd_ppo', battle)

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

        agent = self._get_agent()
        try:
            snap = snapshot_from_iap(battle)
            obs = build_observation_ppo(snap)
            m1, _ = build_autoregressive_masks(snap, slot1_action_id=None)

            def m2_fn(a1: int):
                _m1, m2 = build_autoregressive_masks(snap, slot1_action_id=a1)
                return m2

            if agent is not None:
                greedy = bool(
                    os.environ.get('QD_PPO_CHECKPOINT')
                    or os.environ.get('QD_PPO_ARCHIVE')
                    or os.environ.get('PPO_CHECKPOINT')
                )
                a1, a2, _lp, _v = agent.act(obs, m1, m2_fn, greedy=greedy)
            else:
                import random
                legal1 = [i for i, v in enumerate(m1) if v > 0.5]
                a1 = random.choice(legal1) if legal1 else 20
                m2 = m2_fn(a1)
                legal2 = [i for i, v in enumerate(m2) if v > 0.5]
                a2 = random.choice(legal2) if legal2 else 20
            joint = encode_joint_action(a1, a2)
            k1, n1, m1f, t1, k2, n2, m2f, t2 = joint_to_iap_action(joint, battle)
        except Exception:
            k1, n1, m1f, t1 = 'pass', 0, 0, 0
            k2, n2, m2f, t2 = 'pass', 0, 0, 0

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
            k1, int(n1), int(m1f), int(t1),
            k2, int(n2), int(m2f), int(t2),
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
