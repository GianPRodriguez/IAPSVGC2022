"""Métricas de entrenamiento para la tesis: curva de aprendizaje y diagnóstico.

Escribe un JSONL (una línea por evento) que permite reconstruir, después del
run, **cómo** mejora el agente y **qué** lo frena o lo ayuda:

- ``kind='episode'``: cada episodio de combate — rival, resultado
  (win/loss/tie/turn_cap/crash), retorno, turnos, KOs a favor/en contra, daño
  infligido (ΔHP% rival), mezcla de acciones (move/switch/mega/pass), fallbacks
  y transiciones descartadas.
- ``kind='window'``: cada ``log_every`` pasos — WR móvil por rival y global,
  retorno/turnos/KOs medios, mezcla de acciones, tasas de fallback/descartes y
  las métricas del learner (loss, TD, Q medio, ε, composición del replay…).
- ``kind='update'`` / ``kind='eval'``: para agentes por lotes (preview PPO):
  estadísticas por update y evaluaciones periódicas sobre contextos fijos.

Lectura: ``python -m showdown.rl.train_metrics <ruta.jsonl>`` imprime la curva
resumida por ventana. Los JSONL son la fuente de las figuras de la tesis.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

ACTION_KINDS = ('move', 'switch', 'mega', 'pass')


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


class MetricsLog:
    """Escritor JSONL con flush por línea (sobrevive a cortes del run)."""

    def __init__(self, path: Path | str, *, run_meta: Optional[Mapping[str, Any]] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open('a', encoding='utf-8')
        self.t0 = time.time()
        if run_meta:
            self.write('run', **dict(run_meta))

    def write(self, kind: str, **fields: Any) -> None:
        rec = {'kind': kind, 't': round(time.time() - self.t0, 3), **fields}
        self._fh.write(json.dumps(_jsonable(rec), ensure_ascii=False) + '\n')
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def action_kind(slot_action_id: int) -> str:
    from showdown.rl.action_space import decode_slot_action

    kind = decode_slot_action(int(slot_action_id)).kind
    return kind if kind in ACTION_KINDS else 'pass'


class FoeStateTally:
    """Contraste de estado de los rivales heurísticos (``battle_state_check``) durante el entrenamiento.

    Los rivales validated comparan cada turno su estado interno (clases del repo) con poke-env.
    Si el espejo se desincroniza, el rival juega con información equivocada (más débil) sin
    que nada falle. Se suma por rival y se escribe con cada ventana / actualización; las
    primeras discrepancias se avisan en consola con detalle.
    """

    WARN_N = 5

    def __init__(self) -> None:
        self.total = Counter()
        self._win: Dict[str, Counter] = defaultdict(Counter)
        self._samples: List[str] = []
        self._warned = 0

    def add(self, fs: Optional[Mapping[str, Any]], opponent: str = '') -> None:
        if not fs:
            return
        c = Counter({k: int(fs.get(k, 0) or 0) for k in ('checks', 'mismatches', 'feed_errors')})
        c['episodes'] = 1
        self.total.update(c)
        self._win[str(opponent or 'unknown')].update(c)
        bad = c['mismatches'] + c['feed_errors']
        if bad:
            for smp in list(fs.get('samples') or [])[:3]:
                if len(self._samples) < 20:
                    self._samples.append(f'{opponent}: {smp}')
            if self._warned < self.WARN_N:
                self._warned += 1
                print(f'[foe_state] AVISO rival {opponent}: {c["mismatches"]} discrepancias y '
                      f'{c["feed_errors"]} errores de espejo en un episodio '
                      f'({c["checks"]} revisiones). {list(fs.get("samples") or [])[:2]}', flush=True)

    def flush(self) -> dict:
        """Resumen desde el último ``flush`` (por rival) y totales de la corrida."""
        def _fmt(c: Counter) -> dict:
            ch = max(1, c['checks'])
            return {'episodes': c['episodes'], 'checks': c['checks'], 'mismatches': c['mismatches'],
                    'feed_errors': c['feed_errors'], 'mismatch_rate': c['mismatches'] / ch}
        out = {'by_opp': {o: _fmt(c) for o, c in self._win.items()}, 'total': _fmt(self.total)}
        if self._samples:
            out['samples'] = list(self._samples)
        self._win = defaultdict(Counter)
        self._samples = []
        return out


FOE_STATE = FoeStateTally()


@dataclass
class _EpAcc:
    opponent: str
    ret: float = 0.0
    steps: int = 0
    kos_for: float = 0.0
    kos_against: float = 0.0
    hp_dealt: float = 0.0
    fallbacks: int = 0
    skipped: int = 0
    actions: Counter = field(default_factory=Counter)


class EpisodeTracker:
    """Acumula por entorno y resume en ventanas móviles por rival."""

    def __init__(self, log: Optional[MetricsLog], *, window: int = 200, log_episodes: bool = True):
        self.log = log
        self.window = int(window)
        self.log_episodes = bool(log_episodes)
        self._acc: Dict[int, _EpAcc] = {}
        self._recent: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=self.window))
        self._recent_all: Deque[dict] = deque(maxlen=self.window)
        self.episodes = 0
        self.steps = 0
        self._totals = Counter()

    def start(self, env_id: int, opponent: str) -> None:
        self._acc[int(env_id)] = _EpAcc(opponent=str(opponent))

    def step(
        self,
        env_id: int,
        *,
        actions: Sequence[int],
        reward: float,
        components: Optional[Mapping[str, Any]] = None,
        used_fallback: bool = False,
        skipped: bool = False,
    ) -> None:
        acc = self._acc.get(int(env_id))
        if acc is None:
            acc = self._acc[int(env_id)] = _EpAcc(opponent='unknown')
        acc.ret += float(reward)
        acc.steps += 1
        self.steps += 1
        comp = components or {}
        acc.kos_for += float(comp.get('d_faint_opp', 0.0) or 0.0)
        acc.kos_against += float(comp.get('d_faint_self', 0.0) or 0.0)
        acc.hp_dealt += float(comp.get('d_hp_opp', 0.0) or 0.0)
        acc.fallbacks += int(bool(used_fallback))
        acc.skipped += int(bool(skipped))
        for a in actions:
            acc.actions[action_kind(a)] += 1

    def end(
        self,
        env_id: int,
        *,
        global_step: int,
        terminated: bool,
        truncated: bool,
        components: Optional[Mapping[str, Any]] = None,
        trunc_reason: Optional[str] = None,
        next_opponent: Optional[str] = None,
        foe_state: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        acc = self._acc.pop(int(env_id), None) or _EpAcc(opponent='unknown')
        FOE_STATE.add(foe_state, acc.opponent)
        comp = components or {}
        if truncated:
            outcome = 'crash' if trunc_reason == 'crash' else 'turn_cap'
        elif float(comp.get('d_win', 0.0) or 0.0) > 0:
            outcome = 'win'
        elif float(comp.get('d_loss', 0.0) or 0.0) > 0:
            outcome = 'loss'
        else:
            outcome = 'tie'
        n_act = max(1, sum(acc.actions.values()))
        rec = {
            'step': int(global_step),
            'opp': acc.opponent,
            'outcome': outcome,
            'return': acc.ret,
            'turns': acc.steps,
            'kos_for': acc.kos_for,
            'kos_against': acc.kos_against,
            'hp_dealt': acc.hp_dealt,
            'fallbacks': acc.fallbacks,
            'skipped': acc.skipped,
            'act': {k: acc.actions.get(k, 0) / n_act for k in ACTION_KINDS},
        }
        if foe_state:
            rec['foe_state'] = {k: foe_state.get(k, 0) for k in ('checks', 'mismatches', 'feed_errors')}
        self.episodes += 1
        self._totals[outcome] += 1
        self._recent[acc.opponent].append(rec)
        self._recent_all.append(rec)
        if self.log is not None and self.log_episodes:
            self.log.write('episode', **rec)
        if next_opponent is not None:
            self.start(env_id, next_opponent)
        return rec

    @staticmethod
    def _agg(recs: Iterable[dict]) -> dict:
        recs = list(recs)
        if not recs:
            return {}
        n = len(recs)
        oc = Counter(r['outcome'] for r in recs)
        act = {k: float(np.mean([r['act'][k] for r in recs])) for k in ACTION_KINDS}
        steps = max(1, sum(r['turns'] for r in recs))
        return {
            'n': n,
            'wr': (oc['win'] + 0.5 * oc['tie']) / n,
            'loss_rate': oc['loss'] / n,
            'tie_rate': oc['tie'] / n,
            'turn_cap_rate': oc['turn_cap'] / n,
            'crash_rate': oc['crash'] / n,
            'return': float(np.mean([r['return'] for r in recs])),
            'turns': float(np.mean([r['turns'] for r in recs])),
            'kos_for': float(np.mean([r['kos_for'] for r in recs])),
            'kos_against': float(np.mean([r['kos_against'] for r in recs])),
            'hp_dealt': float(np.mean([r['hp_dealt'] for r in recs])),
            'fallback_rate': sum(r['fallbacks'] for r in recs) / steps,
            'skip_rate': sum(r['skipped'] for r in recs) / steps,
            'act': act,
        }

    def summary(self) -> dict:
        return {
            'all': self._agg(self._recent_all),
            'by_opp': {o: self._agg(d) for o, d in self._recent.items() if d},
            'episodes': self.episodes,
            'outcomes_total': dict(self._totals),
        }

    def write_window(self, global_step: int, *, phase: str = '',
                     learner: Optional[Mapping[str, Any]] = None, **extra: Any) -> dict:
        s = self.summary()
        if self.log is not None:
            self.log.write('window', step=int(global_step), phase=phase,
                           learner=dict(learner or {}), foe_state=FOE_STATE.flush(), **s, **extra)
        return s

    def console_extras(self) -> Dict[str, Any]:
        """Resumen corto para el heartbeat de consola."""
        a = self._agg(self._recent_all)
        if not a:
            return {}
        out: Dict[str, Any] = {'wr': a['wr'], 'ret': a['return'], 'ko+': a['kos_for'],
                               'ko-': a['kos_against']}
        for o, d in self._recent.items():
            if d:
                out[f'wr_{o[:6]}'] = self._agg(d)['wr']
        return out


def read_jsonl(path: Path | str) -> List[dict]:
    out = []
    with Path(path).open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description='Resumen de un JSONL de métricas de entrenamiento')
    ap.add_argument('path')
    args = ap.parse_args(argv)
    recs = read_jsonl(args.path)
    for r in recs:
        k = r.get('kind')
        if k == 'window':
            a = r.get('all') or {}
            by = ' '.join(f"{o}={d.get('wr', 0):.2f}" for o, d in (r.get('by_opp') or {}).items())
            lr = r.get('learner') or {}
            print(f"step={r['step']:>9} wr={a.get('wr', float('nan')):.3f} ret={a.get('return', 0):+.2f} "
                  f"ko+={a.get('kos_for', 0):.2f} ko-={a.get('kos_against', 0):.2f} "
                  f"turns={a.get('turns', 0):.1f} loss={lr.get('loss')} | {by}")
        elif k == 'update':
            print(f"update={r.get('update'):>4} wr_rollout={r.get('rollout_wr', float('nan')):.3f} "
                  f"H={r.get('entropy', 0):.3f} kl={r.get('approx_kl', 0):.4f} "
                  f"clip={r.get('clip_frac', 0):.3f} ev={r.get('explained_var', 0):.3f}")
        elif k == 'eval':
            print(f"EVAL update={r.get('update')} wr={r.get('mean_wr', 0):.3f} "
                  f"d={r.get('delta_wr', 0):+.3f}+-{1.96 * (r.get('delta_se') or 0):.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
