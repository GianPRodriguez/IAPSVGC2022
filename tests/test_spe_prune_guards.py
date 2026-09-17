"""Regresión Spe: no expulsar true_sp=0 / true_sp=32 por podas agresivas (fixplan)."""
from __future__ import annotations

from showdown import stat_inference as si
from showdown.battle import Battle, Pokemon
from showdown.stat_inference import SpeedContext, champions_stat, dex_base_stats


class _HarnessBattle(Battle):
    async def action_team_preview(self, battle):
        pass

    async def action_battle(self, battle):
        pass


def _mon(name: str, *, nature: str, is_opponent: bool = True) -> Pokemon:
    p = Pokemon(name, 100, is_opponent, '', '', None, None, None, None, None)
    p.nature = nature
    p.HPA_battle_actual = 100
    return p


def test_speed_order_skips_large_low_chop_volcarona_sp0():
    """Volcarona Timid: poda SP0..6 → 7..32 no debe estrechar (true_sp=0)."""
    si.reset_sp_limits()
    battle = _HarnessBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    foe = _mon('volcarona', nature='timid')
    battle.opponent.active1 = foe
    belief = battle.stat_beliefs.belief_for(foe)
    assert 0 in belief.candidates['spe']

    # user_speed=160 → umbral 136: SP0..6 (≤135) caen; SP7..32 quedan (26).
    user_speed = 160
    opp_ctx = SpeedContext(boost=0, item=None)
    user_ctx = SpeedContext(boost=0, item=None)

    changed = battle.stat_beliefs.observe_speed_order(
        foe, opp_ctx, user_speed, user_ctx, opponent_first=True, trick_room=False,
    )
    assert changed is False
    assert belief.skipped >= 1
    assert 0 in belief.candidates['spe']
    assert len(belief.candidates['spe']) == 33
    si.reset_sp_limits()


def test_speed_order_skips_large_high_chop_lucario_sp32():
    """Lucario: expulsar SP32 (y tramo alto) no debe estrechar."""
    si.reset_sp_limits()
    battle = _HarnessBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    foe = _mon('lucario', nature='jolly')
    battle.opponent.active1 = foe
    belief = battle.stat_beliefs.belief_for(foe)
    assert 32 in belief.candidates['spe']

    # opponent_first=False → Spe <= user+margin. user=106 → techo 130:
    # SP0..18 Jolly (≤130) quedan; SP19..32 caen (incluye true_sp=32).
    user_speed = 106
    opp_ctx = SpeedContext(boost=0, item=None)
    user_ctx = SpeedContext(boost=0, item=None)

    changed = battle.stat_beliefs.observe_speed_order(
        foe, opp_ctx, user_speed, user_ctx, opponent_first=False, trick_room=False,
    )
    assert changed is False
    assert belief.skipped >= 1
    assert 32 in belief.candidates['spe']
    si.reset_sp_limits()


def test_volcarona_spe_range_matches_fixplan_band():
    """Sample fixplan: Timid Spe 136-149 = SP 7..32 (26 cands), SP0=132."""
    base = dex_base_stats('volcarona')['spe']
    band = [
        sp for sp in range(33)
        if 136 <= champions_stat(base, sp, 'timid', 'spe') <= 149
    ]
    assert band[0] == 7 and band[-1] == 32
    assert len(band) == 26
    assert champions_stat(base, 0, 'timid', 'spe') == 132
