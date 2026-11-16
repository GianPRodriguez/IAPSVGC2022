"""Verificación del preview v2 (resultado de 96,7 % frente a 60,5 % del heurístico, 16-11-2026).

Tres comprobaciones, todas pareadas contra el preview heurístico del piloto (greedy_joint):

1. **Equipos rivales no vistos:** la misma evaluación con rivales del grupo ``train`` y del grupo
   ``eval``. Si el preview solo memorizó qué traer contra los equipos de entrenamiento, cae en
   ``eval``.
2. **Formación fija:** la formación más frecuente del preview jugada SIEMPRE, contra el
   heurístico. Si rinde casi igual que el preview, la ventaja es la formación (los 4 elegidos
   de tu equipo), no una adaptación a cada rival.
3. **Replays:** guarda los replays HTML de algunas batallas (preview y heurístico) y la lista de
   formaciones elegidas, para revisarlas a mano.

    python -m showdown.rl.preview_check --checkpoint <dir>/best.pt --own-team showdown/rl/data/ddqn_team.txt \\
        --n 512 --num-workers 28 --start-ps --out-dir logs/preview_check
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def _agent(checkpoint: str):
    from showdown.rl.team_preview_ppo.agent import PreviewPPOAgent
    ag = PreviewPPOAgent()
    ag.load(checkpoint)
    return ag


def _names(team, action: int):
    from showdown.rl.team_preview_action import decode_preview_action
    idx = decode_preview_action(int(action))
    return [team.mons[i].species for i in idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--own-team', default='showdown/rl/data/ddqn_team.txt')
    ap.add_argument('--n', type=int, default=512, help='contextos por evaluación')
    ap.add_argument('--replays-n', type=int, default=12)
    ap.add_argument('--num-workers', type=int, default=28)
    ap.add_argument('--envs-per-server', type=int, default=4)
    ap.add_argument('--start-ps', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', default='logs/preview_check')
    args = ap.parse_args(argv)

    from showdown.inference_sim.team_factory import load_team_file
    from showdown.rl.preview_eval import evaluate_preview_policy
    from showdown.rl.preview_live import make_preview_env

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ['PS_PREVIEW_OWN_TEAM'] = args.own_team
    team = load_team_file(args.own_team)
    agent = _agent(args.checkpoint)
    greedy = lambda o, m: int(agent.act(o, m, greedy=True)[0])
    live = {'backend': 'live', 'num_workers': args.num_workers, 'envs_per_server': args.envs_per_server,
            'base_port': None, 'start_ps': args.start_ps, 'pilot': 'greedy_joint', 'max_turns': 40}
    rep = {'checkpoint': args.checkpoint, 'n': args.n}

    # 1. train frente a eval, y frecuencia de formaciones
    top = None
    for split in ('train', 'eval'):
        os.environ['PS_TEAM_SPLIT'] = split
        env, pool = make_preview_env(args.seed, live)
        try:
            ev = evaluate_preview_policy(env, greedy, n=args.n, baseline='pilot', eval_seed=7_000_000)
            rep[f'split_{split}'] = vars(ev) if not hasattr(ev, '_asdict') else ev._asdict()
            env.reseed(7_000_000)
            obs, infos = env.reset_batch(args.n)
            acts = [greedy(obs[k], infos[k].get('action_mask') if infos[k].get('action_mask') is not None
                           else env.action_masks()) for k in range(args.n)]
            freq = Counter(acts)
            rep[f'formaciones_{split}'] = [{'accion': a, 'traídos': _names(team, a), 'frac': c / args.n}
                                          for a, c in freq.most_common(8)]
            rep[f'formaciones_distintas_{split}'] = len(freq)
            if split == 'train':
                top = freq.most_common(1)[0][0]
                # 2. la formación más frecuente jugada siempre
                fixed = lambda o, m, _a=top: _a if np.asarray(m)[_a] > 0.5 else greedy(o, m)
                ev2 = evaluate_preview_policy(env, fixed, n=args.n, baseline='pilot', eval_seed=7_000_000)
                rep['formacion_fija'] = {'accion': top, 'traídos': _names(team, top),
                                         **(vars(ev2) if not hasattr(ev2, '_asdict') else ev2._asdict())}
        finally:
            if pool is not None:
                pool.close()

    # 3. replays de unas pocas batallas (preview y heurístico, mismos contextos)
    os.environ['PS_TEAM_SPLIT'] = 'train'
    os.environ['PS_PREVIEW_SAVE_REPLAYS'] = str(out / 'replays')
    env, pool = make_preview_env(args.seed, {**live, 'num_workers': min(4, args.num_workers)})
    try:
        env.reseed(9_000_000)
        obs, infos = env.reset_batch(args.replays_n)
        ctx = []
        for k, (b, c) in enumerate(env._batch):
            m = infos[k].get('action_mask') if infos[k].get('action_mask') is not None else env.action_masks()
            a = greedy(obs[k], m)
            ctx.append({'rival': c.get('foe_team'), 'rival_especies': c.get('foe_names'),
                        'preview_trae': _names(team, a)})
        evaluate_preview_policy(env, greedy, n=args.replays_n, baseline='pilot', eval_seed=9_000_000)
        rep['replays'] = {'dir': str(out / 'replays'), 'contextos': ctx}
    finally:
        os.environ.pop('PS_PREVIEW_SAVE_REPLAYS', None)
        if pool is not None:
            pool.close()

    (out / 'preview_check.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str),
                                            encoding='utf-8')
    print(json.dumps({k: v for k, v in rep.items() if k != 'replays'}, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
