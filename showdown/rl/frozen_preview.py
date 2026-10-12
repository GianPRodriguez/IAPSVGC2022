"""Frozen family Team Preview for combat train/infer (outside combat MDP).

Pairing:
  DDQN / PPO combate     → team_preview_ppo
  MCTS / MCTS–Nash       → team_preview_nash
  QD-PPO / QD-PGA / PPGA → team_preview_qd_pga

Preview stays outside the combat MDP: combat agents never learn /team.
If no checkpoint is available, falls back to Fixed Export Order `/team 1234`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from showdown.rl.team_preview_action import (
    action_to_team_string,
    decode_preview_action,
    legal_mask,
)
from showdown.rl.team_preview_obs import (
    PreviewObsBundle,
    encode_preview_obs,
    mon_feat_from_species,
)

FAMILY_PREVIEW_KIND: Dict[str, str] = {
    'ddqn': 'team_preview_ppo',
    'ppo': 'team_preview_ppo',
    'mcts_puct': 'team_preview_nash',
    'mcts_puct_nash': 'team_preview_nash',
    'qd_ppo': 'team_preview_qd_pga',
    'qd_pga': 'team_preview_qd_pga',
    'qd_ppga': 'team_preview_qd_pga',
}

DEFAULT_PREVIEW_CKPT: Dict[str, str] = {
    'team_preview_ppo': 'logs/team_preview_ppo_checkpoints/final.pt',
    'team_preview_nash': 'logs/team_preview_nash_checkpoints/final.pt',
    'team_preview_qd_pga': 'logs/team_preview_qd_pga_checkpoints/archive.pt',
}

FIXED_EXPORT_ORDER = '/team 1234'
FIXED_EXPORT_TUPLE = (1, 2, 3, 4)


def resolve_preview_kind(combat_or_preview: str) -> str:
    key = str(combat_or_preview or '').strip().lower().replace('-', '_')
    if key in DEFAULT_PREVIEW_CKPT:
        return key
    return FAMILY_PREVIEW_KIND.get(key, key)


def default_checkpoint_for(kind: str) -> Path:
    k = resolve_preview_kind(kind)
    return Path(DEFAULT_PREVIEW_CKPT.get(k, DEFAULT_PREVIEW_CKPT['team_preview_ppo']))


def _species_name(mon: Any) -> str:
    for attr in ('species', 'name', '_species'):
        v = getattr(mon, attr, None)
        if v:
            return str(v).replace(' ', '').replace('-', '').lower()
    return 'unknown'


def obs_bundle_from_pokeenv_battle(battle: Any) -> PreviewObsBundle:
    our_src = list(getattr(battle, 'teampreview_team', None) or battle.team.values())
    foe_src = list(
        getattr(battle, 'teampreview_opponent_team', None)
        or getattr(battle, 'opponent_team', {}).values()
    )
    our = [mon_feat_from_species(_species_name(m)) for m in our_src[:6]]
    while len(our) < 6:
        our.append(mon_feat_from_species('unknown'))
    foe = [mon_feat_from_species(_species_name(m)) for m in foe_src[:6]]
    while len(foe) < 6:
        foe.append(mon_feat_from_species('unknown'))
    return PreviewObsBundle(our=our, foe=foe)


def mark_selected_in_teampreview(battle: Any, order_1based: Sequence[int]) -> None:
    selected = set(int(x) for x in order_1based)
    for i, mon in enumerate(getattr(battle, 'teampreview_team', None) or [], start=1):
        try:
            mon._selected_in_teampreview = i in selected  # type: ignore[attr-defined]
        except Exception:
            pass


@dataclass
class FrozenPreviewPolicy:
    """Callable frozen preview: battle → `/team ####`."""

    kind: str
    checkpoint: str
    device: str = 'cpu'
    greedy: bool = True
    _chooser: Optional[Callable[[np.ndarray, np.ndarray], int]] = None
    _loaded: bool = False
    _load_error: str = ''

    @property
    def available(self) -> bool:
        self._ensure()
        return self._chooser is not None

    def _ensure(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        path = Path(self.checkpoint) if self.checkpoint else None
        if path is None or not path.is_file():
            self._load_error = f'missing_checkpoint:{self.checkpoint}'
            return
        try:
            kind = resolve_preview_kind(self.kind)
            if kind == 'team_preview_ppo':
                from showdown.rl.team_preview_ppo.agent import PreviewPPOAgent

                agent = PreviewPPOAgent()
                agent.cfg.device = self.device
                agent.load(path)

                def choose(obs, mask):
                    a, _lp, _v = agent.act(obs, mask, greedy=self.greedy)
                    return int(a)

                self._chooser = choose
            elif kind == 'team_preview_nash':
                from showdown.rl.team_preview_nash.agent import PreviewNashAgent

                agent = PreviewNashAgent()
                agent.cfg.device = self.device
                agent.load(path)

                def choose(obs, mask):
                    a, _info = agent.act(obs, mask=mask, greedy=self.greedy)
                    return int(a)

                self._chooser = choose
            elif kind == 'team_preview_qd_pga':
                from showdown.rl.team_preview_qd_pga.agent import PreviewQDPGAAgent
                from showdown.rl.team_preview_qd_pga.archive import Archive

                arch = Archive()
                # archive.pt or best_elite.pt (state_dict only)
                if path.name == 'best_elite.pt' or path.suffix == '.pt':
                    import torch
                    try:
                        payload = torch.load(path, map_location='cpu', weights_only=False)
                    except TypeError:
                        payload = torch.load(path, map_location='cpu')
                    if isinstance(payload, dict) and 'cells' in payload:
                        arch.load(path)
                        elite = arch.best()
                        if elite is None:
                            self._load_error = 'empty_archive'
                            return
                        actor_sd = elite.state_dict
                    elif isinstance(payload, dict) and 'state_dict' in payload:
                        actor_sd = payload['state_dict']
                    else:
                        actor_sd = payload
                else:
                    arch.load(path)
                    elite = arch.best()
                    if elite is None:
                        self._load_error = 'empty_archive'
                        return
                    actor_sd = elite.state_dict

                agent = PreviewQDPGAAgent()
                agent.cfg.device = self.device
                agent.load_actor(actor_sd)

                def choose(obs, mask):
                    a, _info = agent.act(obs, mask, greedy=self.greedy)
                    return int(a)

                self._chooser = choose
            else:
                self._load_error = f'unknown_kind:{kind}'
        except Exception as exc:  # pragma: no cover - env/torch specific
            self._load_error = f'load_failed:{exc}'
            self._chooser = None

    def choose_from_species(
        self,
        our_species: Sequence[str],
        foe_species: Sequence[str],
    ) -> Tuple[int, str]:
        """Returns (action_id, '/team ####') from species name lists."""
        self._ensure()
        if self._chooser is None:
            return 0, FIXED_EXPORT_ORDER
        our = [mon_feat_from_species(str(s)) for s in list(our_species)[:6]]
        foe = [mon_feat_from_species(str(s)) for s in list(foe_species)[:6]]
        while len(our) < 6:
            our.append(mon_feat_from_species('unknown'))
        while len(foe) < 6:
            foe.append(mon_feat_from_species('unknown'))
        obs = encode_preview_obs(PreviewObsBundle(our=our, foe=foe))
        mask = np.asarray(legal_mask(6), dtype=np.float32)
        a = int(self._chooser(obs, mask))
        return a, f'/team {action_to_team_string(a)}'

    def choose_action(self, battle: Any) -> Tuple[int, str]:
        """Returns (action_id, '/team ####'). Falls back to fixed export."""
        self._ensure()
        if self._chooser is None:
            return 0, FIXED_EXPORT_ORDER
        bundle = obs_bundle_from_pokeenv_battle(battle)
        obs = encode_preview_obs(bundle)
        mask = np.asarray(legal_mask(6), dtype=np.float32)
        a = int(self._chooser(obs, mask))
        return a, f'/team {action_to_team_string(a)}'

    def action_tuple_from_species(
        self,
        our_species: Sequence[str],
        foe_species: Sequence[str],
    ) -> Tuple[int, ...]:
        a, _cmd = self.choose_from_species(our_species, foe_species)
        if self._chooser is None:
            return FIXED_EXPORT_TUPLE
        return tuple(i + 1 for i in decode_preview_action(a))

    def teampreview(self, battle: Any) -> str:
        a, cmd = self.choose_action(battle)
        order = tuple(i + 1 for i in decode_preview_action(a))
        mark_selected_in_teampreview(battle, order)
        return cmd

    def action_team_preview_tuple(self, battle: Any) -> Tuple[int, ...]:
        """1-based order for IAP websocket bots."""
        a, _cmd = self.choose_action(battle)
        if self._chooser is None:
            return FIXED_EXPORT_TUPLE
        return tuple(i + 1 for i in decode_preview_action(a))


def load_frozen_preview(
    *,
    kind: str,
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    greedy: bool = True,
    allow_missing: bool = True,
) -> FrozenPreviewPolicy:
    k = resolve_preview_kind(kind)
    ckpt = (
        (checkpoint or '').strip()
        or os.environ.get('PREVIEW_CHECKPOINT', '').strip()
        or str(default_checkpoint_for(k))
    )
    pol = FrozenPreviewPolicy(
        kind=k,
        checkpoint=ckpt,
        device=device or os.environ.get('PREVIEW_DEVICE', 'cpu'),
        greedy=greedy,
    )
    if not allow_missing:
        pol._ensure()
        if not pol.available:
            raise FileNotFoundError(
                f'Frozen preview unavailable kind={k} ckpt={ckpt} err={pol._load_error}'
            )
    return pol


def family_preview_for_combat(
    combat_id: str,
    *,
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
) -> FrozenPreviewPolicy:
    kind = FAMILY_PREVIEW_KIND.get(
        str(combat_id).strip().lower().replace('-', '_'),
        'team_preview_ppo',
    )
    return load_frozen_preview(kind=kind, checkpoint=checkpoint, device=device)
