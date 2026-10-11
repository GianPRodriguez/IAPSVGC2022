"""CLI: python -m showdown.rl.mcts_puct --seed 0 --sims 16 --smoke"""
from __future__ import annotations

import argparse

from showdown.rl.env import MockBattleBackend, VGCEnv
from showdown.rl.mcts_puct.agent import MCTSPuctAgent, MCTSPuctConfig
from showdown.rl.reward import RewardCandidate


def smoke(seed: int, sims: int) -> int:
    cfg = MCTSPuctConfig(n_sims=int(sims), seed=int(seed), max_turns=40)
    agent = MCTSPuctAgent(cfg)
    env = VGCEnv(
        backend=MockBattleBackend(max_turns=40),
        reward_candidate=RewardCandidate.SPARSE,
        max_episode_turns=40,
    )
    from pathlib import Path

    from showdown.rl.train_log import TrainHeartbeat
    from showdown.rl.train_report import TrainReport, write_train_report

    obs, info = env.reset(seed=seed)
    steps = 0
    hb = TrainHeartbeat(tag='mcts_puct', t_total=80, log_every=20, log_every_sec=5.0)
    ep_return = 0.0
    try:
        while steps < 80:
            assert env._snap is not None
            a1, a2 = agent.act_slots(env._snap)
            obs, r, term, trunc, info = env.step((a1, a2))
            steps += 1
            tel = agent.last_telemetry
            hb.log(
                steps,
                phase='smoke',
                extras={
                    'sims': sims,
                    'nodes': getattr(tel, 'nodes', 0),
                    'fail': getattr(tel, 'failures', 0),
                },
            )
            if term or trunc:
                ep_return = float(info.get('episode_return', 0.0))
                break
    finally:
        env.close()
    tel = agent.last_telemetry
    out = Path('logs/mcts_puct_smoke')
    write_train_report(
        out,
        TrainReport(
            agent='mcts_puct',
            seed=seed,
            smoke=True,
            global_steps=steps,
            t_total=80,
            elapsed_sec=hb.elapsed(),
            steps_per_sec=hb.rate(max(1, steps)),
            config={'n_sims': sims},
            metrics={
                'episode_return': ep_return,
                'tree_nodes': getattr(tel, 'nodes', 0),
                'failures': getattr(tel, 'failures', 0),
            },
            artifacts={'checkpoint_dir': str(out)},
            notes=[
                'Planning agent: no gradient train loop; this is a smoke/eval stub report.',
                'Full offline suite TBD — use inference_sim / robust e2e for large batches.',
            ],
        ),
    )
    hb.log(
        steps,
        phase='SMOKE:DONE',
        extras={
            'sims': sims,
            'nodes': getattr(tel, 'nodes', 0),
            'fail': getattr(tel, 'failures', 0),
            'reporte': str(out / 'reporte.txt'),
        },
        force=True,
    )
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='MCTS+PUCT (Cand. 1, S2, open-loop)')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--sims', type=int, default=200)
    p.add_argument('--smoke', action='store_true', help='Mock BO1 smoke')
    args = p.parse_args(argv)
    if args.smoke:
        return smoke(args.seed, args.sims)
    p.error('specify --smoke (eval suite TBD)')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
