"""Perfil del paso de aprendizaje de DDQN (checklist DDQN: el aprendiz limita el ritmo).

Llena un replay sintético del tamaño pedido y mide, por paso de gradiente, cuánto toma
cada parte: muestreo (float16 → float32), aumentación por simetría, copia a GPU,
forward/backward/optimizador, sincronizaciones (``.item()``) y las fracciones por rival.

    python -m showdown.rl.ddqn.profile_learner --device cuda --sizes 15000,200000 --steps 100
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np


def _fill(agent, n: int, rng: np.random.Generator) -> None:
    from showdown.rl.constants import JOINT_ACTION_N, OBS_DIM
    rb = agent.replay
    rb.capacity = max(rb.capacity, n)
    rb._alloc()
    while rb._size < n:
        rb._i = rb._size
        rb._alloc()
    obs = (rng.random((n, OBS_DIM)) < 0.15).astype(np.float16)
    rb._s[:n] = obs
    rb._s2[:n] = np.roll(obs, 1, axis=0)
    rb._a[:n] = rng.integers(0, JOINT_ACTION_N, n)
    rb._r[:n] = rng.normal(size=n).astype(np.float32)
    rb._m2[:n] = (rng.random((n, JOINT_ACTION_N)) < 0.05).astype(np.uint8)
    rb._m2[:n, 0] = 1
    rb._opp[:n] = rng.integers(0, 4, n)
    rb._n = n
    rb._i = n % rb.capacity


def profile(agent, steps: int) -> dict:
    """Réplica instrumentada de ``DDQNAgent.learn`` (mismas operaciones, con tiempos)."""
    import torch
    import torch.nn.functional as F
    from showdown.rl import symmetry as SYM
    from showdown.rl.ddqn.replay import ReplayBuffer

    cuda = agent.device.type == 'cuda'
    sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)
    t = defaultdict(float)
    for _ in range(steps):
        t0 = time.perf_counter()
        s, a, r, s2, term, m2, opp = agent.replay.sample(agent.cfg.batch_size, agent._rng)
        t1 = time.perf_counter(); t['muestreo'] += t1 - t0
        g = SYM.sample_elements(s, agent._rng)
        s, a = SYM.augment_obs(s, g), SYM.augment_joint(a, g)
        h = SYM.sample_elements(s2, agent._rng)
        s2, m2 = SYM.augment_obs(s2, h), SYM.augment_joint_mask(m2, h)
        SYM.element_stats(g)
        t2 = time.perf_counter(); t['simetria'] += t2 - t1
        dev = agent.device
        s_t = torch.as_tensor(s, dtype=torch.float32, device=dev)
        a_t = torch.as_tensor(a, dtype=torch.int64, device=dev)
        r_t = torch.as_tensor(r, dtype=torch.float32, device=dev)
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=dev)
        done = torch.as_tensor(term, dtype=torch.float32, device=dev)
        m2_t = torch.as_tensor(m2, dtype=torch.float32, device=dev)
        sync(); t3 = time.perf_counter(); t['copia_gpu'] += t3 - t2
        agent.online.train()
        q_sa = agent.online(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            qn = agent.online(s2_t) + (1.0 - m2_t) * (-1.0e9)
            a_star = qn.argmax(dim=1)
            qt = agent.target(s2_t).gather(1, a_star.view(-1, 1)).squeeze(1)
            y = r_t + (1.0 - done) * agent.cfg.gamma * qt
        loss = F.huber_loss(q_sa, y, delta=agent.cfg.huber_delta)
        agent.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.online.parameters(), agent.cfg.grad_clip)
        agent.optim.step()
        agent._soft_update(agent.online, agent.target, agent.cfg.tau)
        sync(); t4 = time.perf_counter(); t['red_gpu'] += t4 - t3
        _ = (float(loss.item()), float(q_sa.mean().item()), float(y.mean().item()),
             float((q_sa - y).abs().mean().item()))
        t5 = time.perf_counter(); t['item_sync'] += t5 - t4
        ReplayBuffer.batch_origin_fractions(opp)
        agent.replay.origin_fractions()
        t6 = time.perf_counter(); t['fracciones_rival'] += t6 - t5
    return {k: 1000.0 * v / steps for k, v in t.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--sizes', default='15000,200000')
    ap.add_argument('--steps', type=int, default=100)
    args = ap.parse_args(argv)
    from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
    rng = np.random.default_rng(0)
    for n in [int(x) for x in args.sizes.split(',')]:
        agent = DDQNAgent(DDQNConfig(device=args.device, replay_capacity=max(n, 4096)))
        _fill(agent, n, rng)
        profile(agent, 5)  # calentamiento (cuDNN, caché)
        ms = profile(agent, args.steps)
        tot = sum(ms.values())
        print(f'replay={n}: total {tot:.1f} ms/paso de gradiente '
              f'(→ máx. {1000 / tot * agent.cfg.train_freq:.0f} pasos de entorno/s)')
        for k, v in sorted(ms.items(), key=lambda kv: -kv[1]):
            print(f'  {k:18s} {v:7.2f} ms ({100 * v / tot:4.1f} %)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
