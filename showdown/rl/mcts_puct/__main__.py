"""CLI: python -m showdown.rl.mcts_puct --seed 0 --sims 16 --smoke"""
from __future__ import annotations

import argparse
import time

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
    obs, info = env.reset(seed=seed)
    steps = 0
    t0 = time.perf_counter()
    try:
        while steps < 80:
            assert env._snap is not None
            a1, a2 = agent.act_slots(env._snap)
            obs, r, term, trunc, info = env.step((a1, a2))
            steps += 1
            if term or trunc:
                break
    finally:
        env.close()
    elapsed = time.perf_counter() - t0
    tel = agent.last_telemetry
    print(
        f'[mcts_puct] SMOKE ok steps={steps} sims={sims} '
        f'tree_nodes={getattr(tel, "nodes", 0)} '
        f'failures={getattr(tel, "failures", 0)} elapsed={elapsed:.2f}s'
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
