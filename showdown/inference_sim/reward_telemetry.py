"""Reward telemetry for robust E2E (player_a perspective, Spec v1.1).

Informative only — does not affect NEED_FIX / sync / inference gates.

Reports both:
- Candidate 1 (sparse): episode return ∈ {-1, 0, +1}
- Candidate 2 (dense): clipped step-sum; shows *how much* was won/lost
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from showdown.inference_sim.paths import REPO_ROOT
from showdown.rl.snapshot import TeamMaterial

REWARD_HISTORY_PATH = REPO_ROOT / 'logs' / 'robust_e2e_reward_history.jsonl'
REWARD_ANALYSIS_PATH = REPO_ROOT / 'logs' / 'robust_e2e_reward_analysis.txt'

def _pad_team_material(
    mat: TeamMaterial,
    *,
    n_known: int,
    party_size: int = 4,
) -> TeamMaterial:
    """Treat unknown party slots as alive at 100% HP (avoids reveal spikes)."""
    missing = max(0, int(party_size) - max(0, int(n_known)))
    if missing <= 0:
        return mat
    return TeamMaterial(
        faint_count=int(mat.faint_count),
        hp_pct_total=float(mat.hp_pct_total) + 100.0 * missing,
    )


def _iap_mon_hp_pct(pkmn) -> Optional[float]:
    """HP% [0,100] from IAP Pokemon, or None if unknown."""
    if pkmn is None:
        return None
    if bool(getattr(pkmn, 'is_fainted', False)):
        return 0.0
    raw_cur = getattr(pkmn, 'HPA_battle_actual', None)
    raw_mx = getattr(pkmn, 'HPB_battle_base', None)
    if raw_cur is None or raw_mx is None:
        return None
    try:
        cur = float(str(raw_cur).split()[0])
        mx = float(str(raw_mx).split()[0])
    except (TypeError, ValueError, IndexError):
        return None
    if mx <= 0:
        return None
    # Init glitch: foe preview sometimes sets HPA=100 with HPB=base-stat (< max HP).
    # Treat as full bar until a real max HP (>= cur) is known from protocol.
    if cur > mx and mx < 150 and cur <= 100:
        return 100.0
    return max(0.0, min(100.0, 100.0 * cur / mx))


def material_from_iap_side(battler, *, party_size: int = 4) -> TeamMaterial:
    """TeamMaterial from IAP Battler (user actives+bench or opp actives+revealed)."""
    slots: List[Any] = []
    seen: set = set()

    def _add(m) -> None:
        if m is None:
            return
        mid = id(m)
        if mid in seen:
            return
        seen.add(mid)
        slots.append(m)

    if bool(getattr(battler, 'isUser', False)):
        _add(getattr(battler, 'active1', None))
        _add(getattr(battler, 'active2', None))
        for m in list(getattr(battler, 'reserved', None) or []):
            _add(m)
    else:
        _add(getattr(battler, 'active1', None))
        _add(getattr(battler, 'active2', None))
        for m in list(getattr(battler, 'opponents', None) or []):
            if m is None:
                continue
            if _iap_mon_hp_pct(m) is not None or bool(getattr(m, 'is_fainted', False)):
                _add(m)

    faint = 0
    hp = 0.0
    n_known = 0
    for m in slots[: int(party_size)]:
        if bool(getattr(m, 'is_fainted', False)):
            n_known += 1
            faint += 1
            continue
        pct = _iap_mon_hp_pct(m)
        if pct is None:
            continue
        n_known += 1
        if pct <= 0.0:
            faint += 1
        else:
            hp += float(pct)
    return _pad_team_material(
        TeamMaterial(faint_count=faint, hp_pct_total=hp),
        n_known=n_known,
        party_size=int(party_size),
    )


def push_iap_reward_to_ui(battle, q=None, *, label: str = 'Turno') -> None:
    """Enqueue Spec-C2 UI payload ``[2, …, delta, color, total, color]`` if ``q`` set."""
    if battle is None or q is None:
        return
    try:
        delta_f, total_f = battle.calculateRewards()
    except Exception:
        return
    delta = round(float(delta_f), 1)
    total = round(float(total_f), 1)
    color5 = '#325928' if delta > 0 else ('#565828' if delta >= 0 else '#582831')
    color6 = '#325928' if total > 0 else '#582831'
    try:
        q.put([
            2, label, '#FFFFFF', '', '#FFFFFF', '', '#FFFFFF', '', '#FFFFFF',
            delta, color5, total, color6,
        ])
    except Exception:
        pass


def finalize_iap_game_reward(battle, winner: str, q=None) -> None:
    """Apply terminal Cand.1/2 for an individual game ``|win|`` and push UI."""
    if battle is None:
        return
    user = str(getattr(getattr(battle, 'user', None), 'account_name', '') or '')
    bot_won = _ps_names_match(winner, user)
    battle.finished = True
    battle.won = bool(bot_won)
    try:
        observe_iap_spec_reward(battle)
    except Exception as exc:
        print(f'[WARN]     observe_iap_spec_reward (fin juego): {exc}')
    push_iap_reward_to_ui(battle, q, label='Fin juego')
    # Listo para el siguiente juego del BO3 (initialize_team_preview también resetea).
    battle.finished = False
    battle.won = None


def _ps_names_match(a: str, b: str) -> bool:
    def _norm(s: str) -> str:
        return ''.join(ch for ch in str(s or '').lower() if ch.isalnum())
    na, nb = _norm(a), _norm(b)
    return bool(na) and na == nb


def observe_iap_spec_reward(battle) -> Tuple[float, float]:
    """Update Spec C2 (and C1) returns on an IAP Battle; return (c2_delta, c2_return).

    Call once per |request| after environment update. First call only baselines.
    Stores last_c2_reward / last_c1_reward / spec_c2_return / spec_c1_return on battle.
    """
    from showdown.rl.reward import RewardCandidate, compute_reward

    if battle is None:
        return 0.0, 0.0
    cur_our = material_from_iap_side(getattr(battle, 'user', None))
    cur_foe = material_from_iap_side(getattr(battle, 'opponent', None))
    prev_our = getattr(battle, '_spec_prev_our', None)
    prev_foe = getattr(battle, '_spec_prev_foe', None)
    if prev_our is None or prev_foe is None:
        battle._spec_prev_our = cur_our
        battle._spec_prev_foe = cur_foe
        battle.last_c2_reward = 0.0
        battle.last_c1_reward = 0.0
        if not hasattr(battle, 'spec_c2_return') or battle.spec_c2_return is None:
            battle.spec_c2_return = 0.0
        if not hasattr(battle, 'spec_c1_return') or battle.spec_c1_return is None:
            battle.spec_c1_return = 0.0
        return 0.0, float(battle.spec_c2_return)

    finished = False
    won: Optional[bool] = None
    tied = False
    # Online IAP rarely sets finished mid-report; terminal handled if present.
    if bool(getattr(battle, 'finished', False)):
        finished = True
        won_raw = getattr(battle, 'won', None)
        if won_raw is True:
            won = True
        elif won_raw is False:
            won = False
        else:
            tied = True

    r1 = compute_reward(
        candidate=RewardCandidate.SPARSE,
        prev_our=prev_our,
        prev_foe=prev_foe,
        cur_our=cur_our,
        cur_foe=cur_foe,
        finished=finished,
        won=won,
        tied=tied,
    )
    r2 = compute_reward(
        candidate=RewardCandidate.DENSE,
        prev_our=prev_our,
        prev_foe=prev_foe,
        cur_our=cur_our,
        cur_foe=cur_foe,
        finished=finished,
        won=won,
        tied=tied,
    )
    c2 = float(r2.reward)
    c1 = float(r1.reward)
    battle.spec_c2_return = float(getattr(battle, 'spec_c2_return', 0.0) or 0.0) + c2
    battle.spec_c1_return = float(getattr(battle, 'spec_c1_return', 0.0) or 0.0) + c1
    battle.last_c2_reward = c2
    battle.last_c1_reward = c1
    battle._spec_prev_our = cur_our
    battle._spec_prev_foe = cur_foe
    return c2, float(battle.spec_c2_return)


@dataclass
class BattleRewardSummary:
    turns: int
    won: Optional[bool]
    tied: bool
    c1_return: float
    c2_return: float
    n_steps: int = 0
    tag: str = ''

    def compact(self, *, index: Optional[int] = None) -> str:
        prefix = f'g{index}:' if index is not None else ''
        if self.tied or self.won is None:
            wl = 'T'
        else:
            wl = 'W' if self.won else 'L'
        return (
            f'{prefix}t{self.turns}{wl} '
            f'c2={self.c2_return:+.3f} c1={self.c1_return:+.0f}'
        )


@dataclass
class BattleRewardAccumulator:
    """Online Σ R_t from poke-env DoubleBattle snapshots."""

    c1_return: float = 0.0
    c2_return: float = 0.0
    n_steps: int = 0
    _prev_our: Any = None
    _prev_foe: Any = None
    _done: bool = False
    _tag: str = ''

    def reset(self, *, tag: str = '') -> None:
        self.c1_return = 0.0
        self.c2_return = 0.0
        self.n_steps = 0
        self._prev_our = None
        self._prev_foe = None
        self._done = False
        self._tag = tag or ''

    def observe(self, battle) -> None:
        if self._done or battle is None:
            return
        from showdown.rl.pe_adapter import snapshot_from_battle
        from showdown.rl.reward import RewardCandidate, compute_reward
        from showdown.rl.snapshot import material_from_side

        snap = snapshot_from_battle(battle)
        cur_our = material_from_side(snap.our)
        cur_foe = material_from_side(snap.foe)
        # Pad unseen party slots as 100% HP (PE reveals foe gradually).
        cur_our = _pad_team_material(cur_our, n_known=len(snap.our.all_mons()))
        cur_foe = _pad_team_material(cur_foe, n_known=len(snap.foe.all_mons()))
        if self._prev_our is None:
            self._prev_our = cur_our
            self._prev_foe = cur_foe
            if not snap.finished:
                return

        finished = bool(snap.finished)
        won = snap.won
        tied = bool(snap.tied)
        if finished and won is None and not tied:
            # poke-env: won=False on loss; None usually means tie / unknown
            if getattr(battle, 'won', None) is False:
                won = False
            elif getattr(battle, 'won', None) is True:
                won = True
            else:
                tied = True

        r1 = compute_reward(
            candidate=RewardCandidate.SPARSE,
            prev_our=self._prev_our,
            prev_foe=self._prev_foe,
            cur_our=cur_our,
            cur_foe=cur_foe,
            finished=finished,
            won=won,
            tied=tied,
        )
        r2 = compute_reward(
            candidate=RewardCandidate.DENSE,
            prev_our=self._prev_our,
            prev_foe=self._prev_foe,
            cur_our=cur_our,
            cur_foe=cur_foe,
            finished=finished,
            won=won,
            tied=tied,
        )
        self.c1_return += float(r1.reward)
        self.c2_return += float(r2.reward)
        self.n_steps += 1
        self._prev_our = cur_our
        self._prev_foe = cur_foe
        if finished:
            self._done = True

    def finalize(self, battle) -> BattleRewardSummary:
        self.observe(battle)
        turns = int(getattr(battle, 'turn', 0) or 0) if battle is not None else 0
        won_raw = getattr(battle, 'won', None) if battle is not None else None
        finished = bool(getattr(battle, 'finished', False)) if battle is not None else False
        tied = finished and won_raw is None
        won: Optional[bool]
        if won_raw is True:
            won = True
        elif won_raw is False:
            won = False
        else:
            won = None
        tag = ''
        if battle is not None:
            tag = str(getattr(battle, 'battle_tag', '') or '')
        self._done = True
        return BattleRewardSummary(
            turns=turns,
            won=won,
            tied=tied,
            c1_return=float(self.c1_return),
            c2_return=float(self.c2_return),
            n_steps=int(self.n_steps),
            tag=tag,
        )


def summarize_rewards(rows: Sequence[BattleRewardSummary]) -> Dict[str, float]:
    """Aggregate stats for the report header."""
    if not rows:
        return {
            'n': 0.0,
            'c2_mean': 0.0,
            'c2_mean_w': 0.0,
            'c2_mean_l': 0.0,
            'c1_mean': 0.0,
            'c2_per_turn_mean': 0.0,
        }
    c2 = [r.c2_return for r in rows]
    c1 = [r.c1_return for r in rows]
    wins = [r.c2_return for r in rows if r.won is True]
    losses = [r.c2_return for r in rows if r.won is False]
    per_turn = [
        r.c2_return / max(1, r.turns) for r in rows if r.turns > 0
    ]

    def _mean(xs: Sequence[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    return {
        'n': float(len(rows)),
        'c2_mean': _mean(c2),
        'c2_mean_w': _mean(wins),
        'c2_mean_l': _mean(losses),
        'c1_mean': _mean(c1),
        'c2_per_turn_mean': _mean(per_turn),
    }


def format_reward_report_section(
    rows: Sequence[BattleRewardSummary],
    *,
    player_a: str = 'player_a',
    cap: int = 80,
) -> List[str]:
    stats = summarize_rewards(rows)
    lines = [
        '--- reward telemetry (player_a perspective, Spec v1.1) ---',
        f'reward_perspective: {player_a}',
        'reward_note: informative only (not a NEED_FIX gate)',
        'reward_c1: sparse terminal {-1,0,+1}',
        'reward_c2: dense clipped episode return (Candidate 2)',
        f'reward_games: {int(stats["n"])}',
        f'reward_c2_mean: {stats["c2_mean"]:+.4f}',
        f'reward_c2_mean_wins: {stats["c2_mean_w"]:+.4f}',
        f'reward_c2_mean_losses: {stats["c2_mean_l"]:+.4f}',
        f'reward_c2_per_turn_mean: {stats["c2_per_turn_mean"]:+.4f}',
        f'reward_c1_mean: {stats["c1_mean"]:+.4f}',
    ]
    if not rows:
        lines.append('reward_per_game: (none)')
        return lines
    lines.append('')
    lines.append('--- reward per game (capped) ---')
    for i, row in enumerate(rows[:cap], start=1):
        lines.append(f'{i}. {row.compact(index=i)}')
    if len(rows) > cap:
        lines.append(f'... ({len(rows) - cap} more games omitted)')
    return lines


# ---------------------------------------------------------------------------
# Cross-run reward analysis (auto; no Cursor credits needed)
# ---------------------------------------------------------------------------

def _matchup_key(player_a: str, player_b: str) -> str:
    return f'{player_a}|{player_b}'


def build_reward_run_record(
    *,
    player_a: str,
    player_b: str,
    teams: int,
    seed: Optional[int],
    wins: int,
    losses: int,
    errors: int,
    reward_rows: Sequence[BattleRewardSummary],
    elapsed_s: float = 0.0,
) -> Dict[str, Any]:
    stats = summarize_rewards(reward_rows)
    decided = wins + losses
    wr = (100.0 * wins / decided) if decided else 0.0
    return {
        'ts_utc': datetime.now(timezone.utc).isoformat(),
        'player_a': player_a,
        'player_b': player_b,
        'matchup': _matchup_key(player_a, player_b),
        'teams': int(teams),
        'seed': seed,
        'series_w': int(wins),
        'series_l': int(losses),
        'series_e': int(errors),
        'wr_raw_pct': round(wr, 2),
        'elapsed_s': round(float(elapsed_s), 1),
        'reward_games': int(stats['n']),
        'c2_mean': round(stats['c2_mean'], 4),
        'c2_mean_wins': round(stats['c2_mean_w'], 4),
        'c2_mean_losses': round(stats['c2_mean_l'], 4),
        'c2_per_turn': round(stats['c2_per_turn_mean'], 4),
        'c1_mean': round(stats['c1_mean'], 4),
    }


def append_reward_history(record: Dict[str, Any], path: Optional[Path] = None) -> Path:
    out = path or REWARD_HISTORY_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    return out


def load_reward_history(
    *,
    matchup: Optional[str] = None,
    path: Optional[Path] = None,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    src = path or REWARD_HISTORY_PATH
    if not src.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        text = src.read_text(encoding='utf-8')
    except Exception:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if matchup and obj.get('matchup') != matchup:
            continue
        rows.append(obj)
    return rows[-max(1, int(limit)):]


def _delta_str(cur: float, prev: Optional[float]) -> str:
    if prev is None:
        return 'n/a (first run)'
    d = float(cur) - float(prev)
    return f'{d:+.4f}'


def format_reward_analysis(
    record: Dict[str, Any],
    *,
    history: Optional[Sequence[Dict[str, Any]]] = None,
    history_n: int = 8,
) -> List[str]:
    """Informational analysis + vs previous same matchup (for expectimax tuning)."""
    matchup = str(record.get('matchup') or '')
    hist = list(history) if history is not None else load_reward_history(
        matchup=matchup, limit=max(history_n + 1, 12),
    )
    # history may already include current if appended first — drop trailing duplicate ts
    prev_runs = [h for h in hist if h.get('ts_utc') != record.get('ts_utc')]
    prev = prev_runs[-1] if prev_runs else None

    c2 = float(record.get('c2_mean') or 0.0)
    c2w = float(record.get('c2_mean_wins') or 0.0)
    c2l = float(record.get('c2_mean_losses') or 0.0)
    c2t = float(record.get('c2_per_turn') or 0.0)
    wr = float(record.get('wr_raw_pct') or 0.0)

    # Simple signal for tuning: prefer c2_mean, then wr, then c2_per_turn
    teams_cur = record.get('teams')
    teams_prev = prev.get('teams') if prev else None
    teams_mismatch = (
        prev is not None
        and teams_prev is not None
        and int(teams_prev) != int(teams_cur or 0)
    )
    if prev is None:
        trend = 'BASELINE (no prior run for this matchup)'
    else:
        dc2 = c2 - float(prev.get('c2_mean') or 0.0)
        dwr = wr - float(prev.get('wr_raw_pct') or 0.0)
        if dc2 > 0.05 or (abs(dc2) <= 0.05 and dwr > 2.0):
            trend = 'IMPROVED vs previous'
        elif dc2 < -0.05 or (abs(dc2) <= 0.05 and dwr < -2.0):
            trend = 'REGRESSED vs previous'
        else:
            trend = 'FLAT vs previous'
        if teams_mismatch:
            trend += f' (NOTE: teams {teams_prev}→{teams_cur}; prefer same --teams)'

    lines = [
        '--- reward analysis (auto; compare across expectimax param runs) ---',
        f'analysis_matchup: {record.get("player_a")} vs {record.get("player_b")}',
        f'analysis_teams: {record.get("teams")} seed={record.get("seed")}',
        f'analysis_series: W={record.get("series_w")} L={record.get("series_l")} '
        f'E={record.get("series_e")} WR_raw={wr:.1f}%',
        f'analysis_games: {record.get("reward_games")}',
        f'analysis_c2_mean: {c2:+.4f}',
        f'analysis_c2_mean_wins: {c2w:+.4f}',
        f'analysis_c2_mean_losses: {c2l:+.4f}',
        f'analysis_c2_per_turn: {c2t:+.4f}',
        f'analysis_c1_mean: {float(record.get("c1_mean") or 0.0):+.4f}',
        '',
        f'vs_prev_c2_mean: {_delta_str(c2, float(prev["c2_mean"]) if prev else None)}',
        f'vs_prev_c2_wins: {_delta_str(c2w, float(prev["c2_mean_wins"]) if prev else None)}',
        f'vs_prev_c2_losses: {_delta_str(c2l, float(prev["c2_mean_losses"]) if prev else None)}',
        f'vs_prev_c2_per_turn: {_delta_str(c2t, float(prev["c2_per_turn"]) if prev else None)}',
        f'vs_prev_wr_raw: {_delta_str(wr, float(prev["wr_raw_pct"]) if prev else None)}',
        f'trend: {trend}',
        '',
        f'history_file: {REWARD_HISTORY_PATH}',
        f'analysis_file: {REWARD_ANALYSIS_PATH}',
        f'history_same_matchup (last {history_n}):',
    ]
    show = (prev_runs + [record])[-history_n:]
    if len(show) <= 1 and prev is None:
        lines.append(
            f'  1. {record.get("ts_utc", "?")[:19]} teams={record.get("teams")} '
            f'c2={c2:+.3f} wr={wr:.1f}%  ← current'
        )
    else:
        for i, h in enumerate(show, start=1):
            mark = ' ← current' if h.get('ts_utc') == record.get('ts_utc') else ''
            lines.append(
                f'  {i}. {str(h.get("ts_utc", "?"))[:19]} teams={h.get("teams")} '
                f'c2={float(h.get("c2_mean") or 0):+.3f} '
                f'c2W={float(h.get("c2_mean_wins") or 0):+.2f} '
                f'c2L={float(h.get("c2_mean_losses") or 0):+.2f} '
                f'wr={float(h.get("wr_raw_pct") or 0):.1f}%{mark}'
            )
    lines.append('')
    lines.append(
        'tip: after changing expectimax params, re-run the same --teams/--player-* '
        'and read trend / vs_prev_* (no Cursor needed).'
    )
    return lines


def write_reward_analysis(
    record: Dict[str, Any],
    *,
    history_before: Optional[Sequence[Dict[str, Any]]] = None,
) -> Path:
    """Append history + write standalone analysis txt; return analysis path."""
    hist = list(history_before) if history_before is not None else load_reward_history(
        matchup=str(record.get('matchup') or ''),
        limit=40,
    )
    append_reward_history(record)
    lines = format_reward_analysis(record, history=hist + [record])
    text = '\n'.join(lines) + '\n'
    REWARD_ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_ANALYSIS_PATH.write_text(text, encoding='utf-8')
    return REWARD_ANALYSIS_PATH
