"""F1 de la familia QD 2.0 (protocolo §4.11, decisión del usuario del 26-11-2026).

1a. ``deploy``: las ``top`` élites de cada archivo (mejor estimación del archivo) juegan los mismos
    ``games`` enfrentamientos con equipos ``val`` contra los tres heurísticos (por igual). Se despliega
    la de mayor tasa de victorias (desempate: retorno medio). Reemplaza la reevaluación de 300 partidas,
    que inflaba al elegido (maldición del ganador: 90–93 % en ``val`` frente a ~80 % en eval).
1b. ``confirm``: los agentes juegan los mismos ``per_foe`` enfrentamientos por heurístico con equipos
    ``eval`` (bloque de semillas nuevo); se guardan los resultados por partida y ``stats`` calcula las
    pruebas pre-registradas:
      N_I, N_J, N_K  no inferioridad frente al control (diferencia pareada de victorias; H0: d ≤ −margen),
      S_J            J > control solo contra expectimax (H0: d ≤ 0),
    con Holm sobre las cuatro. Descriptivo: tasas por rival con IC de Wilson, diferencias por rival y
    frente al arranque.

    python -m showdown.rl.qd_ppo.confirm_f1 deploy --archive I=<dir>/archive.pt ... --cfg-checkpoint <ppo>.pt
    python -m showdown.rl.qd_ppo.confirm_f1 confirm --agent I=<pt> --agent control=<pt> ... --out <dir>
    python -m showdown.rl.qd_ppo.confirm_f1 stats --results <dir>/results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FOES: Tuple[str, ...] = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')
MARGIN = 0.015          # −1,5 pts (el mismo margen de H 2.2, decisión del usuario)
ALPHA = 0.05
DEPLOY_SEED_BLOCK = 30_001
CONFIRM_SEED_BLOCK = 40_001


# --------------------------------------------------------------------------- enfrentamientos y estadística

def balanced_matchups(seed: int, per_foe: int, split: Optional[str],
                      foes: Sequence[str] = FOES) -> List[Dict[str, Any]]:
    """``per_foe`` enfrentamientos por rival, semillas de equipos de un bloque fijo (reproducible)."""
    rng = np.random.default_rng(int(seed))
    return [{'opponent': f, 'seed': int(rng.integers(0, 2**31 - 1)), 'split': split}
            for f in foes for _ in range(int(per_foe))]


def _for_env(matches: List[Dict[str, Any]], mock: bool) -> List[Dict[str, Any]]:
    """En el simulador de prueba el rival no se cambia (los heurísticos reales necesitan Showdown)."""
    return [dict(m, opponent=None) for m in matches] if mock else matches


def wins_of(returns: Sequence[Optional[float]]) -> List[Optional[int]]:
    """Victoria = 1, empate o derrota = 0, partida caída = None (no cuenta)."""
    return [None if r is None else int(float(r) > 0) for r in returns]


def wilson(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired(a: Sequence[Optional[int]], b: Sequence[Optional[int]], *, shift: float = 0.0
           ) -> Dict[str, float]:
    """Diferencia pareada de victorias a − b sobre los enfrentamientos válidos en los dos.

    p unilateral de H0: d ≤ −shift por aproximación normal (equivale a McNemar con n grande); con
    ``shift`` = 0 es la prueba de superioridad.
    """
    d = np.asarray([x - y for x, y in zip(a, b) if x is not None and y is not None], dtype=np.float64)
    n = int(d.size)
    if n < 2:
        return {'n': n, 'diff': float('nan'), 'se': float('nan'), 'z': float('nan'), 'p': 1.0,
                'lower95': float('nan')}
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    z = (mean + shift) / se if se > 0 else (math.inf if mean + shift > 0 else -math.inf)
    return {'n': n, 'diff': mean, 'se': se, 'z': z, 'p': 1.0 - _phi(z) if math.isfinite(z) else
            (0.0 if z > 0 else 1.0), 'lower95': mean - 1.644854 * se,
            'discordant': int(np.count_nonzero(d))}


def holm(pvals: Dict[str, float], alpha: float = ALPHA) -> Dict[str, Dict[str, Any]]:
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    out: Dict[str, Dict[str, Any]] = {}
    running = 0.0
    stop = False
    for i, k in enumerate(order):
        adj = min(1.0, (m - i) * pvals[k])
        running = max(running, adj)
        rej = (not stop) and pvals[k] <= alpha / (m - i)
        stop = stop or not rej
        out[k] = {'p': pvals[k], 'p_holm': running, 'rejected': bool(rej)}
    return out


def compute_stats(results: Dict[str, Any], *, control: str = 'control', margin: float = MARGIN,
                  alpha: float = ALPHA, ni_agents: Sequence[str] = ('I', 'J', 'K'),
                  sup: Tuple[str, str] = ('J', 'expectimax_bayesian'), init: str = 'init'
                  ) -> Dict[str, Any]:
    foes = results['foes']
    agents = results['wins']
    out: Dict[str, Any] = {'rates': {}, 'tests': {}, 'descriptive': {}}
    for name, w in agents.items():
        row = {}
        for f in list(dict.fromkeys(foes)) + ['3 heur.']:
            v = [x for x, ff in zip(w, foes) if x is not None and (f == '3 heur.' or ff == f)]
            k, n = int(sum(v)), len(v)
            lo, hi = wilson(k, n)
            row[f] = {'wins': k, 'n': n, 'rate': k / n if n else float('nan'), 'ci95': [lo, hi]}
        row['crashes'] = int(sum(1 for x in w if x is None))
        out['rates'][name] = row
    c = agents[control]
    pv = {}
    for a in ni_agents:
        if a in agents:
            t = paired(agents[a], c, shift=margin)
            out['tests'][f'N_{a}'] = {'kind': f'no inferioridad {a} vs {control} (margen −{100 * margin:.1f} pts)', **t}
            pv[f'N_{a}'] = t['p']
    sa, sf = sup
    if sa and sa in agents:
        idx = [i for i, f in enumerate(foes) if f == sf]
        t = paired([agents[sa][i] for i in idx], [c[i] for i in idx], shift=0.0)
        out['tests'][f'S_{sa}'] = {'kind': f'superioridad {sa} > {control} contra {sf}', **t}
        pv[f'S_{sa}'] = t['p']
    for k, h in holm(pv, alpha).items():
        out['tests'][k].update(h)
    for name in agents:
        if name == control:
            continue
        for f in list(dict.fromkeys(foes)):
            idx = [i for i, ff in enumerate(foes) if ff == f]
            out['descriptive'][f'{name} - {control} | {f}'] = paired(
                [agents[name][i] for i in idx], [c[i] for i in idx])
    if init in agents:
        for name in agents:
            if name != init:
                out['descriptive'][f'{name} - {init} | 3 heur.'] = paired(agents[name], agents[init])
    return out


def merge_results(paths: Sequence[str], rename: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Une resultados de colas distintas con los MISMOS enfrentamientos (mismo bloque y rivales).

    ``rename``: ``{'archivo:agente': 'nuevo'}`` o ``{'agente': 'nuevo'}`` para distinguir agentes homónimos.
    """
    rename = rename or {}
    out: Optional[Dict[str, Any]] = None
    for p in paths:
        r = json.loads(Path(p).read_text(encoding='utf-8'))
        if out is None:
            out = {'foes': r['foes'], 'seeds': r['seeds'], 'wins': {}}
        elif r['foes'] != out['foes'] or r['seeds'] != out['seeds']:
            raise ValueError(f'{p}: enfrentamientos distintos; no se pueden parear')
        for name, w in r['wins'].items():
            new = rename.get(f'{p}:{name}', rename.get(name, name))
            if new in out['wins']:
                raise ValueError(f'agente repetido {new!r}; use --rename')
            out['wins'][new] = w
    assert out is not None
    return out


def format_stats(st: Dict[str, Any], title: str = 'F1 QD 2.0: confirmación contra eval (pre-registro: protocolo §4.11)') -> str:
    lines = [f'=== {title} ===', '']
    foes = [k for k in next(iter(st['rates'].values())) if k not in ('crashes',)]
    lines.append(f"{'agente':10s} " + ' '.join(f'{f:>22s}' for f in foes) + '  caídas')
    for name, row in st['rates'].items():
        cells = [f"{100 * row[f]['rate']:5.1f} % [{100 * row[f]['ci95'][0]:4.1f}–{100 * row[f]['ci95'][1]:4.1f}]"
                 for f in foes]
        lines.append(f'{name:10s} ' + ' '.join(f'{c:>22s}' for c in cells) + f"  {row['crashes']}")
    lines += ['', 'Pruebas (Holm m = %d, α = %.2f):' % (len(st['tests']), ALPHA)]
    for k, t in st['tests'].items():
        lines.append(f"  {k}: {t['kind']}: d = {100 * t['diff']:+.2f} pts (cota inf. 95 % "
                     f"{100 * t['lower95']:+.2f}), n = {t['n']}, p = {t['p']:.2e}, p Holm = "
                     f"{t['p_holm']:.2e} → {'RECHAZA H0' if t['rejected'] else 'no rechaza'}")
    lines += ['', 'Descriptivo (diferencias pareadas):']
    for k, t in st['descriptive'].items():
        lines.append(f"  {k}: {100 * t['diff']:+.2f} pts (± {100 * 1.96 * t['se']:.2f}, n = {t['n']})")
    return '\n'.join(lines)


# --------------------------------------------------------------------------- juego

def _play_all(cfg, states: Dict[str, Dict[str, Any]], env, matches: List[Dict[str, Any]], *,
              chunk: int, tag: str) -> Dict[str, List[Optional[float]]]:
    """Todos los agentes juegan los mismos ``matches`` (a la vez, por bloques, con progreso)."""
    from showdown.rl.qd_ppo.matchups import play_matchups_multi
    from showdown.rl.qd_ppo.train2 import _agent_from

    names = list(states)
    agents = [_agent_from(cfg, states[n]) for n in names]
    out: Dict[str, List[Optional[float]]] = {n: [] for n in names}
    t0 = time.time()
    total = len(matches)
    for s in range(0, total, int(chunk)):
        block = matches[s:s + int(chunk)]
        res = play_matchups_multi(agents, env, [block] * len(agents))
        for n, r in zip(names, res):
            out[n].extend(r.returns)
        done = s + len(block)
        el = time.time() - t0
        eta = el / done * (total - done)
        crashes = sum(1 for n in names for x in out[n] if x is None)
        print(f'[f1] {tag} {100 * done / total:5.1f}% | enfrentamientos {done}/{total} × {len(names)} '
              f'agentes | {done * len(names) / max(el, 1e-9):.1f} partidas/s | ETA {eta / 60:.0f} min | '
              f'caídas {crashes}', flush=True)
    return out


def _setup(cfg_checkpoint: str, *, mock: bool, num_envs: int, envs_per_server: int,
           base_port: Optional[int], agent_team: str, device: str = 'cpu'):
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    from showdown.rl.qd_ppo.train2 import _make_env

    a = PPOAgent(PPOConfig(device=device))
    a._ensure_torch()
    if cfg_checkpoint:
        a.load(cfg_checkpoint)
    a.cfg.device = device
    os.environ['RL_OBS_ABILITIES'] = '1' if a.cfg.obs_abilities else '0'
    env = _make_env(mock=mock, seed=0, start_ps=not mock, num_envs=num_envs,
                    envs_per_server=envs_per_server, base_port=base_port, agent_team=agent_team)
    return a.cfg, env


def _state_of(path: str, cfg) -> Dict[str, Any]:
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    from showdown.rl.qd_ppo.train2 import _state

    a = PPOAgent(PPOConfig(**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__}))
    a._ensure_torch()
    a.load(path)
    return _state(a)


def run_deploy(archives: Dict[str, str], *, cfg_checkpoint: str, out: Path, top: int = 10,
               games: int = 1000, seed_block: int = DEPLOY_SEED_BLOCK, mock: bool = False,
               num_envs: int = 32, envs_per_server: int = 4, base_port: Optional[int] = None,
               agent_team: str = '', chunk: int = 300) -> Dict[str, Any]:
    from showdown.rl.qd_ppo.archive import Archive
    from showdown.rl.qd_ppo.train2 import _agent_from

    out.mkdir(parents=True, exist_ok=True)
    cfg, env = _setup(cfg_checkpoint, mock=mock, num_envs=num_envs, envs_per_server=envs_per_server,
                      base_port=base_port, agent_team=agent_team)
    per_foe = int(math.ceil(int(games) / len(FOES)))
    matches = balanced_matchups(seed_block, per_foe, None if mock else 'val')
    states, meta = {}, {}
    for name, path in archives.items():
        arch = Archive()
        arch.load(path)
        ranked = sorted(arch.cells.items(), key=lambda kv: kv[1].fitness, reverse=True)[:int(top)]
        for k, e in ranked:
            key = f'{name}:{k[0]},{k[1]}'
            states[key] = e.state_dict
            meta[key] = {'agent': name, 'cell': list(k), 'bd': list(e.bd), 'archive_fitness': e.fitness,
                         'n_eval': e.n_eval}
    print(f'[f1] deploy: {len(states)} élites ({", ".join(archives)}), {len(matches)} enfrentamientos val',
          flush=True)
    try:
        rets = _play_all(cfg, states, env, _for_env(matches, mock), chunk=chunk, tag='deploy')
    finally:
        env.close()
    rows = []
    for key, r in rets.items():
        w = [x for x in wins_of(r) if x is not None]
        v = [float(x) for x in r if x is not None]
        rows.append({'name': key, **meta[key], 'games': len(w), 'win_rate': float(np.mean(w)) if w else 0.0,
                     'fitness': float(np.mean(v)) if v else 0.0})
    chosen = {}
    for name in archives:
        cand = [r for r in rows if r['agent'] == name]
        best = max(cand, key=lambda r: (r['win_rate'], r['fitness']))
        chosen[name] = best
        _agent_from(cfg, states[best['name']]).save(out / f'deploy_{name}.pt')
        print(f"[f1] desplegado {name}: {best['name']} wr={best['win_rate']:.3f} ({best['games']} partidas val)",
              flush=True)
    summary = {'games': len(matches), 'top': top, 'seed_block': seed_block, 'rows': rows, 'chosen': chosen}
    (out / 'deploy.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def run_confirm(agents: Dict[str, str], *, cfg_checkpoint: str, out: Path, per_foe: int = 4000,
                seed_block: int = CONFIRM_SEED_BLOCK, mock: bool = False, num_envs: int = 32,
                envs_per_server: int = 4, base_port: Optional[int] = None, agent_team: str = '',
                chunk: int = 600) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    cfg, env = _setup(cfg_checkpoint, mock=mock, num_envs=num_envs, envs_per_server=envs_per_server,
                      base_port=base_port, agent_team=agent_team)
    matches = balanced_matchups(seed_block, per_foe, None if mock else 'eval')
    states = {n: _state_of(p, cfg) for n, p in agents.items()}
    print(f'[f1] confirm: agentes {list(agents)}, {len(matches)} enfrentamientos eval '
          f'({per_foe} por heurístico)', flush=True)
    try:
        rets = _play_all(cfg, states, env, _for_env(matches, mock), chunk=chunk, tag='confirm')
    finally:
        env.close()
    results = {'seed_block': seed_block, 'per_foe': per_foe, 'agents': agents,
               'foes': [m['opponent'] for m in matches], 'seeds': [m['seed'] for m in matches],
               'returns': rets, 'wins': {n: wins_of(r) for n, r in rets.items()}}
    (out / 'results.json').write_text(json.dumps(results), encoding='utf-8')
    return results


def export_elites(archive: str, deploy_json: str, agent: str, top: int, cfg_checkpoint: str,
                  out: Path) -> List[Path]:
    """Las ``top`` élites de ``agent`` con más victorias en el despliegue de F1, como checkpoints PPO."""
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    from showdown.rl.qd_ppo.archive import Archive
    from showdown.rl.qd_ppo.train2 import _agent_from

    base = PPOAgent(PPOConfig(device='cpu'))
    base._ensure_torch()
    base.load(cfg_checkpoint)
    arch = Archive()
    arch.load(archive)
    rows = [r for r in json.loads(Path(deploy_json).read_text(encoding='utf-8'))['rows'] if r['agent'] == agent]
    rows.sort(key=lambda r: (r['win_rate'], r['fitness']), reverse=True)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, r in enumerate(rows[:int(top)]):
        pth = out / f'{agent}_elite{k}_{r["cell"][0]}_{r["cell"][1]}.pt'
        _agent_from(base.cfg, arch.cells[tuple(r['cell'])].state_dict).save(pth)
        paths.append(pth)
    return paths


def _kv(items: Sequence[str]) -> Dict[str, str]:
    out = {}
    for it in items:
        k, v = it.split('=', 1)
        out[k.strip()] = v.strip()
    return out


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description='F1 de la familia QD 2.0 (despliegue y confirmación)')
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('deploy', 'confirm'):
        q = sub.add_parser(name)
        q.add_argument('--cfg-checkpoint', required=True, help='checkpoint PPO del que se toma la config (P5)')
        q.add_argument('--out', required=True)
        q.add_argument('--mock', action='store_true')
        q.add_argument('--num-envs', type=int, default=32)
        q.add_argument('--envs-per-server', type=int, default=4)
        q.add_argument('--base-port', type=int, default=None)
        q.add_argument('--agent-team', default='')
        q.add_argument('--chunk', type=int, default=None)
        from showdown.rl.frozen_preview import add_preview_cli_args
        add_preview_cli_args(q)
    d = sub.choices['deploy']
    d.add_argument('--archive', action='append', default=[], help='NOMBRE=archive.pt')
    d.add_argument('--top', type=int, default=10)
    d.add_argument('--games', type=int, default=1000)
    d.add_argument('--seed-block', type=int, default=DEPLOY_SEED_BLOCK)
    c = sub.choices['confirm']
    c.add_argument('--agent', action='append', default=[], help='NOMBRE=checkpoint.pt')
    c.add_argument('--per-foe', type=int, default=4000)
    c.add_argument('--seed-block', type=int, default=CONFIRM_SEED_BLOCK)
    s = sub.add_parser('stats')
    s.add_argument('--results', action='append', required=True, help='uno o más results.json pareables')
    s.add_argument('--rename', action='append', default=[], help='[archivo:]agente=nuevo')
    s.add_argument('--margin', type=float, default=MARGIN)
    s.add_argument('--control', default='control')
    s.add_argument('--ni', default='I,J,K', help='agentes de las pruebas de no inferioridad')
    s.add_argument('--sup', default='J:expectimax_bayesian', help='agente:rival de la superioridad ("" = ninguna)')
    s.add_argument('--init', default='init')
    s.add_argument('--out', default='', help='prefijo de salida (defecto: confirm_stats junto al primero)')
    s.add_argument('--title', default='F1 QD 2.0: confirmación contra eval (pre-registro: protocolo §4.11)')
    e = sub.add_parser('export', help='guarda élites de un archivo como checkpoints PPO')
    e.add_argument('--archive', required=True)
    e.add_argument('--deploy', required=True, help='deploy.json de F1 (orden por victorias en val)')
    e.add_argument('--agent', required=True)
    e.add_argument('--top', type=int, default=5)
    e.add_argument('--cfg-checkpoint', required=True)
    e.add_argument('--out', required=True)
    a = p.parse_args(argv)

    if a.cmd == 'stats':
        res = merge_results(a.results, dict(x.rsplit('=', 1) for x in a.rename))
        sup = tuple(a.sup.split(':', 1)) if a.sup else ('', '')
        st = compute_stats(res, control=a.control, margin=a.margin,
                           ni_agents=[x for x in a.ni.split(',') if x], sup=sup, init=a.init)
        outp = Path(a.out) if a.out else Path(a.results[0]).with_name('confirm_stats')
        outp.with_suffix('.json').write_text(json.dumps(st, indent=2), encoding='utf-8')
        txt = format_stats(st, a.title)
        outp.with_suffix('.txt').write_text(txt + '\n', encoding='utf-8')
        print(txt)
        return 0
    if a.cmd == 'export':
        paths = export_elites(a.archive, a.deploy, a.agent, a.top, a.cfg_checkpoint, Path(a.out))
        print(','.join(str(x) for x in paths))
        return 0

    from showdown.rl.foe_pool import trainer_defaults
    from showdown.rl.frozen_preview import configure_preview
    with trainer_defaults():
        if not a.mock:
            print('[f1] ' + configure_preview('qd_ppo', checkpoint=a.preview_checkpoint,
                                              required=a.require_preview), flush=True)
        common = dict(cfg_checkpoint=a.cfg_checkpoint, out=Path(a.out), mock=a.mock, num_envs=a.num_envs,
                      envs_per_server=a.envs_per_server, base_port=a.base_port, agent_team=a.agent_team)
        if a.cmd == 'deploy':
            run_deploy(_kv(a.archive), top=a.top, games=a.games, seed_block=a.seed_block,
                       chunk=a.chunk or 300, **common)
        else:
            run_confirm(_kv(a.agent), per_foe=a.per_foe, seed_block=a.seed_block, chunk=a.chunk or 600,
                        **common)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
