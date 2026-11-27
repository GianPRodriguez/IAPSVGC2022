"""Codificadores estructurados de la observación (rl_env_spec v1.20 §3.c).

La obs es una concatenación de bloques del mismo tipo (10 Pokémon de combate,
8 movimientos propios, 8 movimientos rivales; 12 Pokémon en el Team Preview).
Un MLP plano aprende pesos distintos para "el mismo Pokémon en el bloque 1" y
"en el bloque 3". ``StructuredEncoder`` aplica **el mismo** sub-MLP a cada bloque
de un grupo y concatena los embeddings **en orden** (la posición se conserva:
las acciones son posicionales), más los segmentos crudos (matrices, campo).

``RL_OBS_ENCODER=flat`` restaura el MLP plano (ablación para la tesis). Los
checkpoints de una arquitectura no cargan en la otra.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

ENCODER_ENV = 'RL_OBS_ENCODER'
ENCODER_KINDS = ('shared', 'flat')


def encoder_kind(kind: Optional[str] = None) -> str:
    k = str(kind or os.environ.get(ENCODER_ENV, '') or 'shared').strip().lower()
    if k not in ENCODER_KINDS:
        raise ValueError(f'{ENCODER_ENV}={k!r}: use one of {ENCODER_KINDS}')
    return k


@dataclass(frozen=True)
class BlockGroup:
    """``n`` bloques contiguos de ``dim`` features desde ``offset``; cada uno pasa por
    el MLP compartido ``dim → hidden → out`` (ReLU)."""
    name: str
    offset: int
    n: int
    dim: int
    hidden: int
    out: int


class FlatEncoder(nn.Module):
    """Identidad (MLP plano clásico: la primera capa del torso ve la obs entera)."""

    def __init__(self, obs_dim: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.out_dim = int(obs_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs


class StructuredEncoder(nn.Module):
    def __init__(self, obs_dim: int, groups: Sequence[BlockGroup],
                 raw: Sequence[Tuple[int, int]]):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.groups = tuple(groups)
        self.raw = tuple((int(a), int(b)) for a, b in raw)
        covered = []
        for g in self.groups:
            covered.append((g.offset, g.offset + g.n * g.dim))
        covered.extend(self.raw)
        covered.sort()
        pos = 0
        for a, b in covered:
            if a != pos or b < a:
                raise ValueError(f'encoder layout gap/overlap at {pos} (segment {a}:{b})')
            pos = b
        if pos != self.obs_dim:
            raise ValueError(f'encoder layout covers {pos} != obs_dim {self.obs_dim}')
        self.mlps = nn.ModuleDict({
            g.name: nn.Sequential(
                nn.Linear(g.dim, g.hidden), nn.ReLU(),
                nn.Linear(g.hidden, g.out), nn.ReLU(),
            )
            for g in self.groups
        })
        self.out_dim = int(sum(g.n * g.out for g in self.groups)
                           + sum(b - a for a, b in self.raw))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        lead = obs.shape[:-1]
        x = obs.reshape(-1, self.obs_dim)
        bsz = x.shape[0]
        parts = []
        for g in self.groups:
            seg = x[:, g.offset:g.offset + g.n * g.dim].reshape(bsz, g.n, g.dim)
            parts.append(self.mlps[g.name](seg).reshape(bsz, g.n * g.out))
        for a, b in self.raw:
            parts.append(x[:, a:b])
        return torch.cat(parts, dim=-1).reshape(*lead, self.out_dim)


def combat_encoder(obs_dim: Optional[int] = None, kind: Optional[str] = None) -> nn.Module:
    """ObsSpec v2: 10 Pokémon (522) · 8 movimientos propios (81) · 8 rivales (43) · crudo.

    Con memoria de serie (``OBS_DIM_MEMORY``, rl_env_spec v1.35) el bloque de memoria va al final como
    segmento crudo: los sub-MLP compartidos no cambian y solo crece la entrada de la primera capa del torso.
    """
    from showdown.rl.constants import (
        FOE_MOVE_DIM, MON_BLOCK_DIM, MOVE_SLOT_DIM, N_FOE_MONS, N_OUR_MONS, OBS_DIM, OBS_DIM_MEMORY,
    )
    d = int(OBS_DIM if obs_dim is None else obs_dim)
    if encoder_kind(kind) == 'flat' or d not in (OBS_DIM, OBS_DIM_MEMORY):
        return FlatEncoder(d)
    n_mon = N_OUR_MONS + N_FOE_MONS
    o_moves = n_mon * MON_BLOCK_DIM
    o_foe = o_moves + 8 * MOVE_SLOT_DIM
    o_raw = o_foe + 8 * FOE_MOVE_DIM
    return StructuredEncoder(d, [
        BlockGroup('mon', 0, n_mon, MON_BLOCK_DIM, 128, 64),
        BlockGroup('move', o_moves, 8, MOVE_SLOT_DIM, 64, 32),
        BlockGroup('foe_move', o_foe, 8, FOE_MOVE_DIM, 64, 32),
    ], raw=[(o_raw, d)])


def preview_encoder(obs_dim: Optional[int] = None, kind: Optional[str] = None) -> nn.Module:
    """Team Preview v2: 12 Pokémon (OTS 464, propios y rivales) · matrices crudas."""
    from showdown.rl.team_preview_obs import MON_DIM, N_POOL, OBS_DIM
    d = int(OBS_DIM if obs_dim is None else obs_dim)
    if encoder_kind(kind) == 'flat' or d != OBS_DIM:
        return FlatEncoder(d)
    o_raw = 2 * N_POOL * MON_DIM
    return StructuredEncoder(d, [
        BlockGroup('mon', 0, 2 * N_POOL, MON_DIM, 128, 64),
    ], raw=[(o_raw, d)])
