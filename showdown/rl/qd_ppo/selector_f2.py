"""F2 de la familia QD 2.0: selector de élite por rival (protocolo §4.12, exploración).

Idea (repertorio de Cully et al., 2015): en VGC el preview muestra los 6 Pokémon del rival antes de
jugar, así que se puede elegir, entre élites de estilos distintos y rendimiento parecido, la que mejor
le va a ese equipo. El selector solo ve el equipo rival (especies y roles), no el algoritmo del rival.

1. ``play``: repertorio = las ``R`` élites con más victorias en el despliegue de F1 (``deploy.json``,
   1000 partidas ``val``), de los archivos de I, J y K. Cada élite juega los mismos enfrentamientos
   ``fit`` (datos del selector) y, junto al control, los mismos enfrentamientos ``eval`` (prueba). Como
   todas juegan todo, la tabla da el resultado de cualquier elección (bandit con información completa).
2. ``analyze``: por élite, regresión logística con L2 sobre las características del equipo rival;
   λ por validación cruzada agrupada por equipo rival (5 pliegues, en ``fit``). En ``eval`` el selector
   elige la élite con mayor probabilidad estimada de ganar, y su resultado es el de esa élite en ese
   enfrentamiento. Se compara con el control, con la mejor élite única (elegida en ``fit``) y con un
   oráculo por equipo (descriptivo, optimista: elige con los resultados de ``eval``).

    python -m showdown.rl.qd_ppo.selector_f2 play --deploy <f1>/deploy/deploy.json --archive I=... \\
        --control <qd_i2>/control/final.pt --cfg-checkpoint <ctrl>.pt --out <dir>
    python -m showdown.rl.qd_ppo.selector_f2 analyze --results <dir>/results.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from showdown.rl.qd_ppo.confirm_f1 import (
    FOES, _for_env, _kv, _play_all, _setup, balanced_matchups, paired, wilson, wins_of,
)

FIT_SEED_BLOCK = 50_001
EVAL_SEED_BLOCK = 60_001
LAMBDAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


# --------------------------------------------------------------------------- características del rival

_POOLS: Dict[str, list] = {}


def opp_team(seed: int, split: Optional[str]):
    """El rival de ``_default_team_metas(seed, split)`` (mismo sorteo) con el pool en caché.

    ``_default_team_metas`` recarga el pool en cada llamada (~0,35 s); aquí se carga una vez por grupo.
    """
    from showdown.rl.foe_pool import base_pool, maybe_synthetic_foe, training_split
    sp = training_split() if split is None else split
    if sp not in _POOLS:
        _POOLS[sp] = base_pool(sp)
    pool = _POOLS[sp]
    rng = np.random.default_rng(int(seed))
    i, j = int(rng.integers(0, len(pool))), int(rng.integers(0, len(pool) - 1))
    if j >= i:
        j += 1
    return maybe_synthetic_foe(rng, sp, pool[j])


def _sid(x: str) -> str:
    return ''.join(ch for ch in str(x).lower() if ch.isalnum())


def team_key(team) -> str:
    return '|'.join(sorted(_sid(m.species) for m in team.mons))


def build_features(teams: Sequence, vocab: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """Especies (multi-hot, vocabulario de ``teams`` si no se da) + conteos de roles + sesgo."""
    from showdown.rl.team_roles import team_role_counts
    if vocab is None:
        vocab = sorted({_sid(m.species) for t in teams for m in t.mons})
    idx = {s: i for i, s in enumerate(vocab)}
    X = np.zeros((len(teams), len(vocab) + 11), dtype=np.float64)
    for r, t in enumerate(teams):
        for m in t.mons:
            j = idx.get(_sid(m.species))
            if j is not None:
                X[r, j] = 1.0
        X[r, len(vocab):len(vocab) + 10] = np.asarray(team_role_counts(t.mons)) / 6.0
        X[r, -1] = 1.0
    return X, vocab


# --------------------------------------------------------------------------- modelo

def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 25) -> np.ndarray:
    """Logística con L2 (el sesgo, última columna, sin penalizar) por Newton (IRLS)."""
    d = X.shape[1]
    w = np.zeros(d)
    reg = np.full(d, float(lam))
    reg[-1] = 1e-6
    for _ in range(int(iters)):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + reg * w
        h = (X * (p * (1 - p))[:, None]).T @ X + np.diag(reg)
        step = np.linalg.solve(h, g)
        w -= step
        if float(np.max(np.abs(step))) < 1e-6:
            break
    return w


def predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Probabilidad de ganar de cada élite (columnas) para cada enfrentamiento (filas)."""
    return 1.0 / (1.0 + np.exp(-np.clip(X @ W.T, -30, 30)))


def fit_all(X: np.ndarray, Y: np.ndarray, valid: np.ndarray, lam: float) -> np.ndarray:
    W = []
    for e in range(Y.shape[1]):
        ok = valid[:, e]
        W.append(fit_logistic(X[ok], Y[ok, e], lam))
    return np.asarray(W)


def group_folds(groups: Sequence[str], k: int, seed: int = 0) -> List[np.ndarray]:
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    f = np.asarray([fold_of[g] for g in groups])
    return [np.where(f == i)[0] for i in range(k)]


def cv_lambda(X, Y, valid, groups, lambdas=LAMBDAS, k: int = 5) -> Tuple[float, Dict[str, float]]:
    """λ con más victorias del elegido en los pliegues retenidos (agrupados por equipo rival)."""
    folds = group_folds(groups, k)
    scores = {}
    for lam in lambdas:
        got = []
        for te in folds:
            tr = np.setdiff1d(np.arange(len(groups)), te)
            W = fit_all(X[tr], Y[tr], valid[tr], lam)
            choice = np.argmax(predict(W, X[te]), axis=1)
            for r, e in zip(te, choice):
                if valid[r, e]:
                    got.append(Y[r, e])
        scores[str(lam)] = float(np.mean(got)) if got else 0.0
    best = max(lambdas, key=lambda l: scores[str(l)])
    return float(best), scores


def _table(wins: Dict[str, List[Optional[int]]], names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(wins[names[0]])
    Y = np.zeros((n, len(names)))
    V = np.zeros((n, len(names)), dtype=bool)
    for j, nm in enumerate(names):
        for i, w in enumerate(wins[nm]):
            if w is not None:
                Y[i, j], V[i, j] = float(w), True
    return Y, V


def analyze(results: Dict[str, Any], *, control: str = 'control') -> Dict[str, Any]:
    names = list(results['repertoire'])
    fit, ev = results['fit'], results['eval']
    t_fit = [opp_team(s, results.get('fit_split', 'fit')) for s in fit['seeds']]
    t_ev = [opp_team(s, results.get('eval_split', 'eval')) for s in ev['seeds']]
    X_fit, vocab = build_features(t_fit)
    X_ev, _ = build_features(t_ev, vocab)
    Y_fit, V_fit = _table(fit['wins'], names)
    Y_ev, V_ev = _table(ev['wins'], names)
    groups = [team_key(t) for t in t_fit]
    lam, cv = cv_lambda(X_fit, Y_fit, V_fit, groups)
    W = fit_all(X_fit, Y_fit, V_fit, lam)
    choice = np.argmax(predict(W, X_ev), axis=1)
    sel = [int(Y_ev[i, e]) if V_ev[i, e] else None for i, e in enumerate(choice)]
    # mejor élite única (elegida en fit) y oráculo por equipo rival (descriptivo, en eval)
    fit_rate = [float(Y_fit[V_fit[:, j], j].mean()) for j in range(len(names))]
    best_single = int(np.argmax(fit_rate))
    single = [int(Y_ev[i, best_single]) if V_ev[i, best_single] else None for i in range(len(choice))]
    ev_groups = [team_key(t) for t in t_ev]
    oracle_pick = {}
    for g in set(ev_groups):
        rows = [i for i, gg in enumerate(ev_groups) if gg == g]
        rates = [np.mean([Y_ev[i, j] for i in rows if V_ev[i, j]] or [0]) for j in range(len(names))]
        oracle_pick[g] = int(np.argmax(rates))
    oracle = [int(Y_ev[i, oracle_pick[g]]) if V_ev[i, oracle_pick[g]] else None
              for i, g in enumerate(ev_groups)]
    ctrl = ev['wins'][control]
    foes = ev['foes']
    arms = {'selector': sel, f'mejor élite única ({names[best_single]})': single, 'control': ctrl,
            'oráculo por equipo (optimista)': oracle}
    if 'init' in ev['wins']:
        arms['init'] = ev['wins']['init']
    rates = {}
    for arm, w in arms.items():
        row = {}
        for f in list(FOES) + ['3 heur.']:
            v = [x for x, ff in zip(w, foes) if x is not None and (f == '3 heur.' or ff == f)]
            k, n = int(sum(v)), len(v)
            row[f] = {'rate': k / n if n else float('nan'), 'n': n, 'ci95': list(wilson(k, n))}
        rates[arm] = row
    comps = {
        'selector - control (primaria)': paired(sel, ctrl),
        'selector - mejor élite única': paired(sel, single),
        'mejor élite única - control': paired(single, ctrl),
    }
    for f in FOES:
        idx = [i for i, ff in enumerate(foes) if ff == f]
        comps[f'selector - control | {f}'] = paired([sel[i] for i in idx], [ctrl[i] for i in idx])
    counts = {names[e]: int(c) for e, c in zip(*np.unique(choice, return_counts=True))}
    return {'lambda': lam, 'cv': cv, 'fit_rates': dict(zip(names, fit_rate)), 'best_single': names[best_single],
            'rates': rates, 'comparisons': comps, 'chosen_counts': counts, 'n_fit_teams': len(set(groups)),
            'n_eval_teams': len(set(ev_groups)), 'vocab': len(vocab)}


def format_analysis(a: Dict[str, Any]) -> str:
    L = ['=== F2: selector de élite por rival (exploración; protocolo §4.12) ===',
         f"λ = {a['lambda']} (VC agrupada: {a['cv']}); equipos rivales fit {a['n_fit_teams']}, eval "
         f"{a['n_eval_teams']}; vocabulario {a['vocab']} especies", '']
    for arm, row in a['rates'].items():
        L.append(f'{arm:40s} ' + '  '.join(f"{f[:10]:>10s} {100 * row[f]['rate']:5.1f} %" for f in row))
    L += ['', 'Comparaciones pareadas (descriptivas, IC 95 %):']
    for k, t in a['comparisons'].items():
        L.append(f"  {k}: {100 * t['diff']:+.2f} pts (± {100 * 1.96 * t['se']:.2f}, n = {t['n']}, "
                 f"p unilateral = {t['p']:.3f})")
    L += ['', f"Elecciones del selector en eval: {a['chosen_counts']}",
          f"Victorias en fit por élite: " + ', '.join(f'{k} {100 * v:.1f} %' for k, v in a['fit_rates'].items())]
    return '\n'.join(L)


# --------------------------------------------------------------------------- juego

def _repertoire(deploy: Dict[str, Any], archives: Dict[str, str], r: int) -> Dict[str, Dict[str, Any]]:
    from showdown.rl.qd_ppo.archive import Archive
    rows = sorted(deploy['rows'], key=lambda x: (x['win_rate'], x['fitness']), reverse=True)[:int(r)]
    loaded = {}
    out = {}
    for row in rows:
        ag = row['agent']
        if ag not in loaded:
            a = Archive()
            a.load(archives[ag])
            loaded[ag] = a
        cell = tuple(row['cell'])
        out[row['name']] = loaded[ag].cells[cell].state_dict
    return out


def run_play(*, deploy_json: str, archives: Dict[str, str], control: str, init: Optional[str],
             cfg_checkpoint: str, out: Path, r: int = 10, per_foe_fit: int = 1000, per_foe_eval: int = 1000,
             mock: bool = False, num_envs: int = 32, envs_per_server: int = 4, base_port: Optional[int] = None,
             agent_team: str = '', chunk: int = 300, fit_split: str = 'fit', fit_seed_block: int = FIT_SEED_BLOCK,
             eval_from: Optional[str] = None) -> Dict[str, Any]:
    """``fit_split``: grupo de equipos de los datos del selector (F2b: ``val``, equipos no vistos).

    ``eval_from``: reutiliza la parte ``eval`` de otro ``results.json`` con el mismo repertorio (F2b usa la
    prueba de F2, así la comparación es sobre las mismas partidas).
    """
    from showdown.rl.qd_ppo.confirm_f1 import _state_of
    out.mkdir(parents=True, exist_ok=True)
    deploy = json.loads(Path(deploy_json).read_text(encoding='utf-8'))
    cfg, env = _setup(cfg_checkpoint, mock=mock, num_envs=num_envs, envs_per_server=envs_per_server,
                      base_port=base_port, agent_team=agent_team)
    rep = _repertoire(deploy, archives, r)
    prev = json.loads(Path(eval_from).read_text(encoding='utf-8')) if eval_from else None
    if prev is not None and list(prev['repertoire']) != list(rep):
        raise ValueError(f'{eval_from}: repertorio distinto {prev["repertoire"]} != {list(rep)}')
    fit_name = fit_split
    fit_split, ev_split = (None, None) if mock else (fit_split, 'eval')
    m_fit = balanced_matchups(fit_seed_block, per_foe_fit, fit_split)
    m_ev = balanced_matchups(EVAL_SEED_BLOCK, per_foe_eval, ev_split)
    ev_states = dict(rep)
    ev_states['control'] = _state_of(control, cfg)
    if init:
        ev_states['init'] = _state_of(init, cfg)
    print(f'[f2] repertorio {list(rep)}; fit {len(m_fit)}, eval {len(m_ev)} enfrentamientos', flush=True)
    try:
        r_fit = _play_all(cfg, rep, env, _for_env(m_fit, mock), chunk=chunk, tag='fit')
        r_ev = None if prev is not None else _play_all(cfg, ev_states, env, _for_env(m_ev, mock), chunk=chunk,
                                                       tag='eval')
    finally:
        env.close()
    res = {'repertoire': list(rep), 'fit_split': fit_name, 'eval_split': ev_split or 'eval',
           'fit': {'foes': [m['opponent'] for m in m_fit], 'seeds': [m['seed'] for m in m_fit],
                   'wins': {k: wins_of(v) for k, v in r_fit.items()}},
           'eval': prev['eval'] if prev is not None else {
               'foes': [m['opponent'] for m in m_ev], 'seeds': [m['seed'] for m in m_ev],
               'wins': {k: wins_of(v) for k, v in r_ev.items()}}}
    if prev is not None:
        res['eval_from'] = str(eval_from)
    (out / 'results.json').write_text(json.dumps(res), encoding='utf-8')
    return res


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description='F2: selector de élite por rival')
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('play')
    q.add_argument('--deploy', required=True)
    q.add_argument('--archive', action='append', default=[])
    q.add_argument('--control', required=True)
    q.add_argument('--init', default=None)
    q.add_argument('--cfg-checkpoint', required=True)
    q.add_argument('--out', required=True)
    q.add_argument('--r', type=int, default=10)
    q.add_argument('--per-foe-fit', type=int, default=1000)
    q.add_argument('--per-foe-eval', type=int, default=1000)
    q.add_argument('--fit-split', default='fit', choices=('fit', 'val'), help='equipos de los datos del selector')
    q.add_argument('--fit-seed-block', type=int, default=FIT_SEED_BLOCK)
    q.add_argument('--eval-from', default=None, help='results.json cuya parte eval se reutiliza')
    q.add_argument('--mock', action='store_true')
    q.add_argument('--num-envs', type=int, default=32)
    q.add_argument('--envs-per-server', type=int, default=4)
    q.add_argument('--base-port', type=int, default=None)
    q.add_argument('--agent-team', default='')
    from showdown.rl.frozen_preview import add_preview_cli_args
    add_preview_cli_args(q)
    s = sub.add_parser('analyze')
    s.add_argument('--results', required=True)
    a = p.parse_args(argv)
    if a.cmd == 'analyze':
        from showdown.rl.foe_pool import trainer_defaults
        res = json.loads(Path(a.results).read_text(encoding='utf-8'))
        with trainer_defaults():
            an = analyze(res)
        outp = Path(a.results).with_name('selector_analysis.json')
        outp.write_text(json.dumps(an, indent=2), encoding='utf-8')
        txt = format_analysis(an)
        outp.with_suffix('.txt').write_text(txt + '\n', encoding='utf-8')
        print(txt)
        return 0
    from showdown.rl.foe_pool import trainer_defaults
    from showdown.rl.frozen_preview import configure_preview
    with trainer_defaults():
        if not a.mock:
            print('[f2] ' + configure_preview('qd_ppo', checkpoint=a.preview_checkpoint,
                                              required=a.require_preview), flush=True)
        run_play(deploy_json=a.deploy, archives=_kv(a.archive), control=a.control, init=a.init,
                 cfg_checkpoint=a.cfg_checkpoint, out=Path(a.out), r=a.r, per_foe_fit=a.per_foe_fit,
                 per_foe_eval=a.per_foe_eval, mock=a.mock, num_envs=a.num_envs,
                 envs_per_server=a.envs_per_server, base_port=a.base_port, agent_team=a.agent_team,
                 fit_split=a.fit_split, fit_seed_block=a.fit_seed_block, eval_from=a.eval_from)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
