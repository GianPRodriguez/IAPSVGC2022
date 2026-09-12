"""Asserciones de soundness de la inferencia SP / stats finales."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from showdown.battle import Battle
from showdown.inference_sim.team_factory import TeamMeta
from showdown.stat_inference import STAT_KEYS
from teams.team_converter import normalize_name


@dataclass
class SoundnessFailure:
    battle_tag: str
    species: str
    stat: str
    true_sp: int
    true_final: int
    candidates: List[int]
    stat_range: Optional[Tuple[int, int]]
    detail: str
    recent_lines: List[str] = field(default_factory=list)


@dataclass
class SoundnessReport:
    checks: int = 0
    failures: List[SoundnessFailure] = field(default_factory=list)
    observations: int = 0
    skipped: int = 0
    narrowing_events: int = 0
    battle_errors: int = 0
    # Especie -> lista de snapshots {turn, event, ranges, true_*}
    timelines: Dict[str, List[dict]] = field(default_factory=dict)
    _failure_cap: int = 60
    _total_failures: int = 0

    @property
    def ok(self) -> bool:
        return self._total_failures == 0 and not self.failures

    def add_failure(self, fail: SoundnessFailure) -> None:
        self._total_failures += 1
        if len(self.failures) < self._failure_cap:
            self.failures.append(fail)

    @property
    def failure_count(self) -> int:
        return self._total_failures or len(self.failures)


def snapshot_beliefs(battle: Battle, opp_meta: TeamMeta, *, turn: int, event: str) -> Dict[str, dict]:
    """Captura rangos actuales de cada rival activo (para timeline)."""
    from showdown.stat_inference import champions_stat

    out: Dict[str, dict] = {}
    for pkmn in _iter_opponent_active(battle):
        sid = normalize_name(getattr(pkmn, 'name', '') or '')
        if not sid:
            continue
        belief = battle.stat_beliefs.belief_for(pkmn)
        if belief is None:
            continue
        true_sp, _ = _lookup_truth(opp_meta, sid)
        nature = getattr(belief, 'nature', None) or getattr(pkmn, 'nature', None)
        ranges = {}
        for stat in STAT_KEYS:
            rng = belief.stat_range(stat)
            sp_rng = belief.sp_range(stat)
            tsp = (true_sp or {}).get(stat)
            base = int(belief.base_stats.get(stat) or 0)
            tf = (
                champions_stat(base, int(tsp), nature, stat)
                if (base > 0 and tsp is not None) else None
            )
            ranges[stat] = {
                'final': rng,
                'sp': sp_rng,
                'n': len(belief.candidates.get(stat) or []),
                'true_sp': tsp,
                'true_final': tf,
            }
        out[sid] = {
            'turn': turn,
            'event': event,
            'observations': int(getattr(belief, 'observations', 0) or 0),
            'skipped': int(getattr(belief, 'skipped', 0) or 0),
            'ranges': ranges,
        }
    return out


def record_timeline(report: SoundnessReport, snaps: Dict[str, dict]) -> None:
    for sid, snap in snaps.items():
        hist = report.timelines.setdefault(sid, [])
        # Evitar duplicar el mismo estado consecutivo
        if hist:
            prev = hist[-1]
            same = (
                prev.get('ranges') == snap.get('ranges')
                and prev.get('turn') == snap.get('turn')
                and prev.get('event') == snap.get('event')
            )
            if same:
                continue
        hist.append(snap)


def _iter_opponent_active(battle: Battle):
    for attr in ('active1', 'active2'):
        p = getattr(battle.opponent, attr, None)
        if p is not None:
            yield p


def _lookup_truth(opp_meta: TeamMeta, sid: str):
    true_sp = opp_meta.true_sp.get(sid)
    true_final = opp_meta.true_final.get(sid)
    if true_sp and true_final:
        return true_sp, true_final
    # Mega / forme: salamencemega → salamence, floettemega → floette
    for key in (
        sid,
        sid.replace('-', ''),
        sid.split('-')[0],
        sid.replace('mega', '').replace('-', '') or sid,
    ):
        if key in opp_meta.true_sp:
            return opp_meta.true_sp[key], opp_meta.true_final[key]
    # quitar sufijo mega
    if sid.endswith('mega'):
        base = sid[: -len('mega')]
        if base in opp_meta.true_sp:
            return opp_meta.true_sp[base], opp_meta.true_final[base]
    if 'mega' in sid:
        base = sid.replace('mega', '').replace('-', '')
        # try original keys containing base
        for k in opp_meta.true_sp:
            if k.replace('-', '') == base or k == base:
                return opp_meta.true_sp[k], opp_meta.true_final[k]
    return None, None


def check_soundness(
    battle: Battle,
    opp_meta: TeamMeta,
    *,
    battle_tag: str = '',
    recent_lines: Optional[List[str]] = None,
    report: Optional[SoundnessReport] = None,
) -> SoundnessReport:
    """
    Propiedad clave: el SP verdadero (y el stat final) del rival NUNCA debe
    quedar fuera de los candidatos / rango estimado.
    """
    from showdown.stat_inference import champions_stat

    report = report or SoundnessReport()
    recent = list(recent_lines or [])[-40:]

    for pkmn in _iter_opponent_active(battle):
        sid = normalize_name(getattr(pkmn, 'name', '') or '')
        if not sid:
            continue
        belief = battle.stat_beliefs.belief_for(pkmn)
        if belief is None:
            continue
        report.observations += int(getattr(belief, 'observations', 0) or 0)
        report.skipped += int(getattr(belief, 'skipped', 0) or 0)

        true_sp, _meta_final = _lookup_truth(opp_meta, sid)
        if not true_sp:
            continue

        nature = getattr(belief, 'nature', None) or getattr(pkmn, 'nature', None)
        for stat in STAT_KEYS:
            report.checks += 1
            cands: Set[int] = set(belief.candidates.get(stat) or set())
            tsp = int(true_sp.get(stat, 0))
            # Final verdadero con las BASES ACTUALES (Mega/Blade): SP son del mon,
            # no de la forma; el meta precomputado es de la forma base.
            base = int(belief.base_stats.get(stat) or 0)
            tf = champions_stat(base, tsp, nature, stat) if base > 0 else int((_meta_final or {}).get(stat) or 0)
            rng = belief.stat_range(stat)

            if tsp not in cands:
                report.add_failure(SoundnessFailure(
                    battle_tag=battle_tag,
                    species=sid,
                    stat=stat,
                    true_sp=tsp,
                    true_final=tf,
                    candidates=sorted(cands),
                    stat_range=rng,
                    detail=f'true_sp={tsp} excluido de candidatos',
                    recent_lines=recent,
                ))
                continue
            if rng is not None and not (rng[0] <= tf <= rng[1]):
                report.add_failure(SoundnessFailure(
                    battle_tag=battle_tag,
                    species=sid,
                    stat=stat,
                    true_sp=tsp,
                    true_final=tf,
                    candidates=sorted(cands),
                    stat_range=rng,
                    detail=f'true_final={tf} fuera de rango {rng}',
                    recent_lines=recent,
                ))
    return report


def _fmt_rng(pair) -> str:
    if not pair:
        return '?'
    return f'{pair[0]}-{pair[1]}'


def format_timeline_example(report: SoundnessReport) -> List[str]:
    """Elige el mon con más estrechamiento y muestra su evolución."""
    best_sid = None
    best_score = 0
    best_stat = None
    for sid, hist in (report.timelines or {}).items():
        if len(hist) < 2:
            continue
        for stat in STAT_KEYS:
            first_n = hist[0]['ranges'].get(stat, {}).get('n') or 33
            last_n = hist[-1]['ranges'].get(stat, {}).get('n') or 33
            score = first_n - last_n
            if score > best_score:
                best_score = score
                best_sid = sid
                best_stat = stat
    if not best_sid or not best_stat:
        # fallback: cualquier especie con historial
        if not report.timelines:
            return ['(Sin timeline capturada)']
        best_sid = max(report.timelines.keys(), key=lambda s: len(report.timelines[s]))
        best_stat = 'spe'
        hist = report.timelines[best_sid]
        for stat in STAT_KEYS:
            if hist[0]['ranges'].get(stat, {}).get('n', 33) > hist[-1]['ranges'].get(stat, {}).get('n', 33):
                best_stat = stat
                break

    hist = report.timelines[best_sid]
    lines = [
        '',
        '-' * 72,
        f'EJEMPLO DE INFERENCIA EN EL TIEMPO: {best_sid} (stat={best_stat})',
        f'True SP={hist[0]["ranges"][best_stat].get("true_sp")}  '
        f'True final={hist[0]["ranges"][best_stat].get("true_final")}',
        '',
    ]
    prev_n = None
    for snap in hist:
        r = snap['ranges'].get(best_stat) or {}
        n = r.get('n')
        # Solo imprimir cuando cambia el # de candidatos o es un |turn|
        if prev_n is not None and n == prev_n and snap.get('event') != 'turn':
            continue
        mark = ''
        if prev_n is not None and n is not None and n < prev_n:
            mark = f'  << estrechó {prev_n}->{n}'
        lines.append(
            f"  T{snap['turn']:>2} [{snap['event']:<7}]  "
            f"SP {_fmt_rng(r.get('sp')):>7}  "
            f"final {_fmt_rng(r.get('final')):>9}  "
            f"cand={n}  obs={snap.get('observations', 0)}{mark}"
        )
        prev_n = n

    # Resumen de todos los stats al final
    last = hist[-1]
    lines.append('')
    lines.append(f'Estado final de {best_sid}:')
    for stat in STAT_KEYS:
        r = last['ranges'].get(stat) or {}
        lines.append(
            f"  {stat:>3}: SP {_fmt_rng(r.get('sp')):>7} | "
            f"final {_fmt_rng(r.get('final')):>9} | "
            f"cand={r.get('n')} | true_sp={r.get('true_sp')} true_final={r.get('true_final')}"
        )
    return lines


def format_report(report: SoundnessReport, *, battles: int, teams: List[str]) -> str:
    battle_errors = int(getattr(report, 'battle_errors', 0) or 0)
    lines = [
        '=' * 72,
        'INFERENCE SIM REPORT (pokemon-showdown + poke-env + MetaVGC Reg M-C)',
        '=' * 72,
        f'Battles requested: {battles}',
        f'Teams: {", ".join(teams)}',
        f'Soundness checks: {report.checks}',
        f'Soundness failures: {report.failure_count}',
        f'Battle infra errors/timeouts: {battle_errors}',
        f'Belief observations (sum): {report.observations}',
        f'Belief skips (sum): {report.skipped}',
        '',
    ]
    if report.ok:
        lines.append('RESULT: PASS — true SP/final siempre dentro de candidatos/rango.')
        if battle_errors:
            lines.append(
                f'Nota: {battle_errors} batalla(s) no completaron (timeout/team reject); '
                f'no invalidan soundness de las que sí corrieron.'
            )
    else:
        lines.append('RESULT: FAIL — se excluyó el valor verdadero en al menos un caso.')
        for i, fail in enumerate(report.failures[:30], 1):
            lines.append('-' * 72)
            lines.append(
                f'#{i} {fail.species}.{fail.stat}: {fail.detail} '
                f'(true_sp={fail.true_sp}, true_final={fail.true_final}, '
                f'range={fail.stat_range}, n_cand={len(fail.candidates)})'
            )
            if fail.recent_lines:
                lines.append('  recent protocol:')
                for ln in fail.recent_lines[-12:]:
                    lines.append(f'    {ln}')
    lines.extend(format_timeline_example(report))
    lines.append('=' * 72)
    return '\n'.join(lines)
