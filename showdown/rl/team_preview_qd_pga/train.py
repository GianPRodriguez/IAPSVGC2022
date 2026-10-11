"""Train Team Preview QD-PGA (docs/team_preview_qd_pga_spec.md)."""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np

from showdown.rl.team_preview_env import TeamPreviewEnv
from showdown.rl.team_preview_obs import OBS_DIM
from showdown.rl.team_preview_ppo.network import PreviewActorCritic, apply_mask_logits
from showdown.rl.team_preview_qd_pga.archive import (
    Archive,
    actor_state_dict,
    load_actor_into,
)
from showdown.rl.team_preview_qd_pga.critic import SharedPreviewCritic
from showdown.rl.team_preview_qd_pga.evaluate import evaluate_actor
from showdown.rl.team_preview_qd_pga.improve import burn_rollout, pg_improve
from showdown.rl.team_preview_qd_pga.mutate import mutate_actor_state

B_BOOT = 4096
T_EMIT = 128
N_EMIT = 4
N_OUTER = 256
P_MUT = 0.5
SIGMA = 0.02
K_PG = 2
G_FIT = 20
G_MIN = 10


def _make_act_fn(net, device, rng):
    import torch

    def act_fn(obs, mask, greedy=True):
        net.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            m = torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = net(o)
            z = apply_mask_logits(logits, m)
            if greedy:
                return int(torch.argmax(z, dim=-1).item())
            dist = torch.distributions.Categorical(logits=z)
            return int(dist.sample().item())

    return act_fn


def train(
    *,
    seed: int = 0,
    smoke: bool = False,
    device: str = 'cpu',
) -> Archive:
    import torch

    b_boot = 64 if smoke else B_BOOT
    t_emit = 16 if smoke else T_EMIT
    n_emit = 2 if smoke else N_EMIT
    n_outer = 2 if smoke else N_OUTER
    g_fit = 4 if smoke else G_FIT
    g_min = 2 if smoke else G_MIN

    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    dev = torch.device(device)
    env = TeamPreviewEnv(seed=seed, backend='mock')
    net = PreviewActorCritic(OBS_DIM).to(dev)
    critic = SharedPreviewCritic(OBS_DIM).to(dev)
    actor_optim = torch.optim.Adam(
        [p for n, p in net.named_parameters() if not n.startswith('value.')],
        lr=3e-4,
    )
    critic_optim = torch.optim.Adam(critic.parameters(), lr=3e-4)
    archive = Archive(10, 10)
    act_fn = _make_act_fn(net, dev, rng)

    # Bootstrap: PG improves + try insert seed policy
    print(f'bootstrap episodes={b_boot}', flush=True)
    pg_improve(
        env, net, critic, actor_optim, critic_optim, dev,
        n_episodes=b_boot, epochs=K_PG,
    )
    f, bd, g, meta = evaluate_actor(env, act_fn, G=g_fit, g_min=g_min, greedy=True)
    if meta['eligible']:
        archive.try_insert(
            bd1=bd[0], bd2=bd[1], fitness=f,
            state_dict=actor_state_dict(net),
            meta={'phase': 'boot'},
        )
    print(
        f'boot done cov={archive.coverage():.2f} f={f:.3f} bd={bd}',
        flush=True,
    )

    for outer in range(int(n_outer)):
        for _e in range(int(n_emit)):
            filled = archive.filled_keys()
            if filled:
                key = filled[int(rng.integers(0, len(filled)))]
                parent_sd = archive.cells[key].state_dict
                load_actor_into(net, parent_sd)
            else:
                parent_sd = actor_state_dict(net)

            do_mut = bool(rng.random() < P_MUT)
            if do_mut:
                mut_sd = mutate_actor_state(parent_sd, rng, sigma=SIGMA)
                load_actor_into(net, mut_sd)
                burn_rollout(env, net, dev, n_episodes=t_emit, rng=rng)
            else:
                pg_improve(
                    env, net, critic, actor_optim, critic_optim, dev,
                    n_episodes=t_emit, epochs=K_PG,
                )

            f, bd, g, meta = evaluate_actor(
                env, act_fn, G=g_fit, g_min=g_min, greedy=True,
            )
            if meta['eligible']:
                archive.try_insert(
                    bd1=bd[0], bd2=bd[1], fitness=f,
                    state_dict=actor_state_dict(net),
                    meta={'outer': outer, 'mut': do_mut},
                )

        best = archive.best()
        bf = best.fitness if best else float('nan')
        print(
            f'outer={outer} cov={archive.coverage():.2f} '
            f'qd={archive.qd_score():.3f} best_f={bf:.3f}',
            flush=True,
        )

    return archive


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description='Team Preview QD-PGA')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args(argv)
    arch = train(seed=args.seed, smoke=args.smoke, device=args.device)
    print(
        f'DONE team_preview_qd_pga cov={arch.coverage():.3f} '
        f'cells={len(arch.cells)}',
        flush=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
