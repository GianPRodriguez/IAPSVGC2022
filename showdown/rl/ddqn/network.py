"""Q-network: codificador estructurado → 512 → 512 → 1600 conjuntas, o factorizada por hueco."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from showdown.rl.constants import JOINT_ACTION_N, OBS_DIM
from showdown.rl.encoders import combat_encoder


class QNetwork(nn.Module):
    """Red Q de DDQN.

    ``q_head='joint'`` (por defecto): 1 600 salidas, una por acción conjunta (a1, a2).

    ``q_head='factored'`` (valores Q por hueco, memorizacion_equipos.md §10; ddqn_spec 1.21):

        Q(s, a1, a2) = V(s) + A1(s, a1) + A2(s, a2) + Σ_k U1(s, a1)_k · U2(s, a2)_k

    con dos cabezas de 40 acciones (una por hueco) y una interacción de rango ``q_rank``
    (0 = puramente aditiva). La salida sigue siendo el vector de 1 600 valores conjuntos
    (índice 40·a1 + a2), así que la máscara conjunta, el objetivo Double DQN, el replay y la
    simetría no cambian. Idea: con ~81 salidas en vez de 1 600, la red no puede guardar una
    respuesta distinta para cada combinación (la factorización por hueco es la diferencia
    estructural con PPO, que generaliza).
    """

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = JOINT_ACTION_N,
                 encoder: Optional[str] = None, layer_norm: bool = False, dropout: float = 0.0,
                 q_head: str = 'joint', q_rank: int = 0):
        super().__init__()
        # Dropout tras cada capa oculta (memorizacion_equipos.md §8, opción a). Se aplica en
        # forward de forma funcional para no cambiar los nombres del state_dict: los
        # checkpoints con y sin dropout son intercambiables. 0 = sin dropout.
        self.dropout = float(dropout)
        self.q_head = str(q_head)
        self.q_rank = int(q_rank)
        if self.q_head not in ('joint', 'factored'):
            raise ValueError(f'q_head desconocido: {q_head!r}')
        self.encoder = combat_encoder(obs_dim, encoder)
        # Punto 7 del checklist: LayerNorm tras el codificador (se concatenan embeddings ReLU
        # con segmentos crudos de otra escala). Apagado por defecto.
        self.layer_norm = bool(layer_norm)
        self.norm = nn.LayerNorm(self.encoder.out_dim) if self.layer_norm else nn.Identity()
        if self.q_head == 'joint':
            self.net = nn.Sequential(
                nn.Linear(self.encoder.out_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions),
            )
        else:
            n_slot = int(round(n_actions ** 0.5))
            assert n_slot * n_slot == n_actions
            self.n_slot = n_slot
            self.net = nn.Sequential(
                nn.Linear(self.encoder.out_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
            )
            self.v = nn.Linear(512, 1)
            self.a1 = nn.Linear(512, n_slot)
            self.a2 = nn.Linear(512, n_slot)
            if self.q_rank > 0:
                self.u1 = nn.Linear(512, n_slot * self.q_rank)
                self.u2 = nn.Linear(512, n_slot * self.q_rank)

    def _trunk(self, h: torch.Tensor) -> torch.Tensor:
        if self.dropout <= 0.0:
            return self.net[3](self.net[2](self.net[1](self.net[0](h))))
        import torch.nn.functional as F
        h = F.dropout(self.net[1](self.net[0](h)), self.dropout, self.training)
        return F.dropout(self.net[3](self.net[2](h)), self.dropout, self.training)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.encoder(obs))
        if self.q_head == 'joint':
            if self.dropout <= 0.0:
                return self.net(h)
            return self.net[4](self._trunk(h))
        z = self._trunk(h)
        q = self.v(z)[:, :, None] + self.a1(z)[:, :, None] + self.a2(z)[:, None, :]
        if self.q_rank > 0:
            b, n, r = z.shape[0], self.n_slot, self.q_rank
            q = q + torch.einsum('bik,bjk->bij', self.u1(z).view(b, n, r), self.u2(z).view(b, n, r))
        return q.reshape(z.shape[0], -1)


@torch.no_grad()
def soft_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    """θ⁻ ← τθ + (1-τ)θ⁻."""
    t = float(tau)
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - t).add_(op.data, alpha=t)


def copy_weights(online: nn.Module, target: nn.Module) -> None:
    target.load_state_dict(online.state_dict())
