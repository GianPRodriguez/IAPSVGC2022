r"""PGA-ME fiel para acciones discretas (agente J 2.0, docs/qd_pga_map_elites_spec.md §J 2.0).

Referencia: Nilsson & Cully (2021), implementación QDax (``pga_me_emitter``, ``qpg_emitter``,
``td3_loss``, ``mutation_operators.isoline_variation``). Adaptación discreta:

* crítico gemelo \(Q_{1,2}(s,\cdot)\) con cabeza conjunta de 1600 acciones (``ddqn.network.QNetwork``);
* objetivo del crítico = SARSA esperado con el actor codicioso objetivo:
  \(y = r + \gamma(1-d)\sum_{a'}\pi'(a'|s')\min_k Q'_k(s',a')\) (análogo del suavizado de TD3);
* rama de gradiente y actor codicioso = gradiente de política esperado (Ciosek & Whiteson, 2018):
  \(\max_\theta \mathbb E_s[\sum_a \pi_\theta(a|s)\,Q_1(s,a)]\), exacto sobre las 40 × 40 acciones legales;
* operador genético Iso+LineDD (Vassiliades & Mouret, 2018).
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np

N_SLOT = 40
N_JOINT = N_SLOT * N_SLOT


# --------------------------------------------------------------------------- replay
class JointReplay:
    """(s, a, r, s', terminado, truncado, máscara(s), máscara(s')) con obs float16 y máscaras en bits."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity, self.obs_dim = int(capacity), int(obs_dim)
        self.n = self.i = 0
        self._alloc = False

    def _allocate(self):
        c, d = self.capacity, self.obs_dim
        self.s = np.zeros((c, d), np.float16)
        self.s2 = np.zeros((c, d), np.float16)
        self.a = np.zeros(c, np.int16)
        self.r = np.zeros(c, np.float32)
        self.term = np.zeros(c, np.float32)
        self.trunc = np.zeros(c, np.float32)
        self.m = np.zeros((c, N_JOINT // 8), np.uint8)
        self.m2 = np.zeros((c, N_JOINT // 8), np.uint8)
        self._alloc = True

    def __len__(self) -> int:
        return int(self.n)

    def add(self, t: Dict[str, Any]) -> None:
        if t.get('mask') is None or np.asarray(t['mask']).shape[-1] != N_JOINT:
            return  # sin máscara conjunta no hay objetivo de SARSA esperado
        if not self._alloc:
            self._allocate()
        k = self.i
        self.s[k] = np.asarray(t['obs'], np.float32)
        self.s2[k] = np.asarray(t['next_obs'], np.float32)
        self.a[k] = int(t['action'])
        self.r[k] = float(t['reward'])
        self.term[k] = 1.0 if t['terminated'] else 0.0
        self.trunc[k] = 1.0 if (t['truncated'] and not t['terminated']) else 0.0
        self.m[k] = np.packbits(np.asarray(t['mask']) > 0.5)
        nm = t.get('next_mask')
        self.m2[k] = np.packbits(np.asarray(nm) > 0.5) if nm is not None else 0
        self.i = (k + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        idx = rng.integers(0, self.n, size=int(batch))
        return {
            's': self.s[idx].astype(np.float32), 's2': self.s2[idx].astype(np.float32),
            'a': self.a[idx].astype(np.int64), 'r': self.r[idx], 'term': self.term[idx],
            'trunc': self.trunc[idx],
            'm': np.unpackbits(self.m[idx], axis=1, count=N_JOINT).astype(np.float32),
            'm2': np.unpackbits(self.m2[idx], axis=1, count=N_JOINT).astype(np.float32),
        }


# --------------------------------------------------------------------------- política conjunta
def joint_log_probs(net, obs, jmask):
    """log π(a1, a2 | s) para las 1600 acciones (−inf en las ilegales). ``net`` = ppo ActorCritic."""
    import torch
    import torch.nn.functional as F

    B = obs.shape[0]
    h = net.features(obs)
    jm = jmask.view(B, N_SLOT, N_SLOT) > 0.5
    m1 = jm.any(dim=-1)
    neg = torch.finfo(h.dtype).min / 4
    z1 = net.policy1(h).masked_fill(~m1, neg)
    lp1 = F.log_softmax(z1, dim=-1)                                       # (B, 40)
    eye = torch.eye(N_SLOT, device=h.device, dtype=h.dtype)
    hh = h.unsqueeze(1).expand(B, N_SLOT, h.shape[-1])
    z2 = net.policy2(torch.cat([hh, eye.unsqueeze(0).expand(B, N_SLOT, N_SLOT)], dim=-1))  # (B, 40, 40)
    z2 = z2.masked_fill(~jm, neg)
    lp2 = F.log_softmax(z2, dim=-1)
    lp = lp1.unsqueeze(-1) + lp2
    return lp.masked_fill(~jm, float('-inf')).view(B, N_JOINT)


def expected_q(net, q_vals, obs, jmask):
    """\\(\\sum_a \\pi(a|s)\\,Q(s,a)\\) sobre las acciones legales."""
    import torch
    p = torch.exp(joint_log_probs(net, obs, jmask))
    p = torch.nan_to_num(p, nan=0.0)
    return (p * torch.where(jmask > 0.5, q_vals, torch.zeros_like(q_vals))).sum(dim=-1)


# --------------------------------------------------------------------------- crítico gemelo
class TwinCritic:
    """Q₁, Q₂ con cabeza conjunta + redes objetivo (τ) — TD3 de PGA-ME con SARSA esperado."""

    def __init__(self, obs_dim: int, device, lr: float = 3e-4, gamma: float = 0.95, tau: float = 0.005):
        import torch
        from showdown.rl.ddqn.network import QNetwork
        self.torch = torch
        self.device = device
        self.gamma, self.tau = float(gamma), float(tau)
        self.q = torch.nn.ModuleList([QNetwork(obs_dim), QNetwork(obs_dim)]).to(device)
        self.q_targ = copy.deepcopy(self.q)
        for p in self.q_targ.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.steps = 0

    def q1(self, s):
        return self.q[0](s)

    def update(self, batch: Dict[str, Any], target_actor) -> float:
        torch = self.torch
        s, s2 = batch['s'], batch['s2']
        with torch.no_grad():
            nq = torch.minimum(self.q_targ[0](s2), self.q_targ[1](s2))
            has_next = batch['m2'].sum(dim=-1) > 0
            ev = torch.zeros_like(batch['r'])
            if bool(has_next.any()):
                ev[has_next] = expected_q(target_actor, nq[has_next], s2[has_next], batch['m2'][has_next])
            y = batch['r'] + self.gamma * (1.0 - batch['term']) * ev
        keep = 1.0 - batch['trunc']   # truncadas (tope de turnos) fuera de la pérdida, como QDax
        a = batch['a'].view(-1, 1)
        loss = sum((((qk(s).gather(1, a).squeeze(1) - y) ** 2) * keep).mean() for qk in self.q)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_targ.parameters()):
                pt.mul_(1.0 - self.tau).add_(self.tau * p)
        self.steps += 1
        return float(loss.item())


def to_torch(batch: Dict[str, np.ndarray], device) -> Dict[str, Any]:
    import torch
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


def pg_steps(net, critic: TwinCritic, replay: JointReplay, rng, *, n_steps: int, lr: float,
             batch: int = 256, optim=None) -> Tuple[float, Any]:
    """``n_steps`` de gradiente de política esperado contra Q₁ (rama de gradiente / actor codicioso).

    ``optim=None`` crea un Adam nuevo (hijo de la rama de gradiente, como QDax); el actor codicioso
    pasa su optimizador persistente.
    """
    import torch
    params = [p for n, p in net.named_parameters() if not n.startswith('value.')]
    opt = optim or torch.optim.Adam(params, lr=lr)
    net.train()
    loss_v = 0.0
    for _ in range(int(n_steps)):
        b = to_torch(replay.sample(batch, rng), critic.device)
        with torch.no_grad():
            q = critic.q1(b['s'])
        loss = -expected_q(net, q, b['s'], b['m']).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        opt.step()
        loss_v = float(loss.item())
    return loss_v, opt


# --------------------------------------------------------------------------- Iso+LineDD
def iso_line(sd1: Dict[str, Any], sd2: Dict[str, Any], rng: np.random.Generator, *,
             iso_sigma: float = 0.005, line_sigma: float = 0.05) -> Dict[str, Any]:
    """θ' = θ1 + σ_iso·ε + σ_line·ξ·(θ2 − θ1); ε por peso, ξ un escalar por par (QDax)."""
    import torch
    xi = float(rng.normal(0.0, 1.0)) * float(line_sigma)
    out = {}
    for k, v1 in sd1.items():
        v2 = sd2.get(k, v1)
        if not torch.is_floating_point(v1):
            out[k] = v1.clone()
            continue
        eps = torch.as_tensor(rng.normal(0.0, float(iso_sigma), size=tuple(v1.shape)), dtype=v1.dtype)
        out[k] = v1 + eps + xi * (v2 - v1)
    return out
