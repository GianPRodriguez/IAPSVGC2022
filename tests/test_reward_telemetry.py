"""Unit tests for robust E2E reward telemetry."""
from __future__ import annotations

from types import SimpleNamespace

from showdown.inference_sim.reward_telemetry import (
    BattleRewardAccumulator,
    BattleRewardSummary,
    format_reward_report_section,
    summarize_rewards,
)
from showdown.rl.reward import RewardCandidate, compute_reward
from showdown.rl.snapshot import TeamMaterial


def _mon(species: str, hp: float, *, fainted: bool = False):
    return SimpleNamespace(
        species=species,
        name=species,
        base_species=species,
        types=('Normal',),
        current_hp_fraction=0.0 if fainted else hp / 100.0,
        current_hp=0 if fainted else hp,
        max_hp=100,
        fainted=fainted,
        boosts={},
        status=None,
        effects={},
        moves={},
        ability='',
        item='',
        tera_type=None,
        first_turn=False,
        active=True,
        is_dynamaxed=False,
    )


def _battle(*, turn: int, our_hp, foe_hp, finished=False, won=None):
    a0 = _mon('A', our_hp[0], fainted=our_hp[0] <= 0)
    a1 = _mon('B', our_hp[1], fainted=our_hp[1] <= 0)
    b0 = _mon('C', foe_hp[0], fainted=foe_hp[0] <= 0)
    b1 = _mon('D', foe_hp[1], fainted=foe_hp[1] <= 0)
    # Full 4-mon sides so padding is a no-op in this test
    a2 = _mon('E', our_hp[2] if len(our_hp) > 2 else 100)
    a3 = _mon('F', our_hp[3] if len(our_hp) > 3 else 100)
    b2 = _mon('G', foe_hp[2] if len(foe_hp) > 2 else 100)
    b3 = _mon('H', foe_hp[3] if len(foe_hp) > 3 else 100)
    for m, hp in ((a2, our_hp[2] if len(our_hp) > 2 else 100), (a3, our_hp[3] if len(our_hp) > 3 else 100)):
        if hp <= 0:
            m.fainted = True
            m.current_hp_fraction = 0.0
    for m, hp in ((b2, foe_hp[2] if len(foe_hp) > 2 else 100), (b3, foe_hp[3] if len(foe_hp) > 3 else 100)):
        if hp <= 0:
            m.fainted = True
            m.current_hp_fraction = 0.0
    return SimpleNamespace(
        turn=turn,
        team={'a': a0, 'b': a1, 'c': a2, 'd': a3},
        opponent_team={'a': b0, 'b': b1, 'c': b2, 'd': b3},
        active_pokemon=[a0, a1],
        opponent_active_pokemon=[b0, b1],
        can_mega_evolve=[False, False],
        can_tera=[False, False],
        finished=finished,
        won=won,
        force_switch=[False, False],
        player_role='p1',
        weather={},
        fields={},
        side_conditions={},
        opponent_side_conditions={},
        battle_tag='battle-test-1',
    )


def test_accumulator_dense_win_path():
    acc = BattleRewardAccumulator()
    acc.observe(_battle(turn=1, our_hp=[100, 100, 100, 100], foe_hp=[100, 100, 100, 100]))
    acc.observe(_battle(turn=2, our_hp=[100, 100, 100, 100], foe_hp=[0, 100, 100, 100]))
    # faint +100hp → 0.15 + 0.05 = 0.20 mid
    assert abs(acc.c2_return - 0.20) < 1e-6
    summary = acc.finalize(
        _battle(turn=3, our_hp=[100, 100, 100, 100], foe_hp=[0, 0, 0, 0], finished=True, won=True)
    )
    assert summary.won is True
    assert summary.c1_return == 1.0
    # prior 0.20 + faint*2 (0.30) + hp drop remaining ~200/100*0.05=0.10 + win 1.0
    # foe: from [0,100,100,100] hp=300 → [0,0,0,0] hp=0; faint 1→4 => +3 faints
    assert summary.c2_return > 1.0
    # Clip is per-step (Spec), so episode Σ can exceed R_CLIP_HI.


def test_sparse_matches_spec():
    z = TeamMaterial(0, 400.0)
    r = compute_reward(
        candidate=RewardCandidate.SPARSE,
        prev_our=z, prev_foe=z, cur_our=z, cur_foe=z,
        finished=True, won=False, tied=False,
    )
    assert r.reward == -1.0


def test_report_section_nonempty():
    acc = BattleRewardAccumulator()
    acc.observe(_battle(turn=1, our_hp=[100, 100, 100, 100], foe_hp=[100, 100, 100, 100]))
    s = acc.finalize(
        _battle(turn=2, our_hp=[50, 100, 100, 100], foe_hp=[0, 0, 0, 0], finished=True, won=True)
    )
    lines = format_reward_report_section([s], player_a='greedy_solo')
    text = '\n'.join(lines)
    assert 'reward_c2_mean:' in text
    assert 'c2=' in text
    stats = summarize_rewards([s])
    assert stats['n'] == 1.0


def test_reward_analysis_trend_and_history(tmp_path):
    from showdown.inference_sim import reward_telemetry as rt

    hist_path = tmp_path / 'hist.jsonl'
    ana_path = tmp_path / 'ana.txt'
    rt.REWARD_HISTORY_PATH = hist_path
    rt.REWARD_ANALYSIS_PATH = ana_path

    rows = [
        BattleRewardSummary(turns=5, won=True, tied=False, c1_return=1.0, c2_return=1.0),
        BattleRewardSummary(turns=5, won=False, tied=False, c1_return=-1.0, c2_return=-0.8),
    ]
    r1 = rt.build_reward_run_record(
        player_a='expectimax_bayesian',
        player_b='greedy_joint',
        teams=4,
        seed=1,
        wins=1,
        losses=1,
        errors=0,
        reward_rows=rows,
    )
    rt.write_reward_analysis(r1, history_before=[])
    assert hist_path.is_file()
    assert 'BASELINE' in ana_path.read_text(encoding='utf-8')

    rows2 = [
        BattleRewardSummary(turns=5, won=True, tied=False, c1_return=1.0, c2_return=1.3),
        BattleRewardSummary(turns=5, won=True, tied=False, c1_return=1.0, c2_return=1.2),
    ]
    r2 = rt.build_reward_run_record(
        player_a='expectimax_bayesian',
        player_b='greedy_joint',
        teams=4,
        seed=2,
        wins=2,
        losses=0,
        errors=0,
        reward_rows=rows2,
    )
    before = rt.load_reward_history(matchup=r2['matchup'])
    rt.write_reward_analysis(r2, history_before=before)
    text = ana_path.read_text(encoding='utf-8')
    assert 'IMPROVED' in text
    assert 'vs_prev_c2_mean:' in text
    assert 'history_same_matchup' in text
