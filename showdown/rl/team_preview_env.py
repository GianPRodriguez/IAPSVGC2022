"""Gymnasium 1-step Team Preview env (docs/team_preview_masked_ppo_spec.md).

reset() builds 512D obs (+ optional calc cache).
step(a) decodes /team, runs frozen combat outcome (mock or live hook), returns Cand.1.

Backends:

- ``mock``: proxy de cobertura (sin batallas). No debe reportarse como live.
- ``live``: cada etiqueta sale de un BO1 real con π_c congelado (greedy_joint),
  jugado por ``showdown.rl.preview_live.LivePreviewPool``. Los contextos son
  equipos reales del pool MetaVGC Reg M-C (validados).

API por lotes (``reset_batch`` / ``step_batch`` / ``pair_label_batch``): los
entrenadores piden N episodios a la vez y, en live, las N batallas se juegan en
paralelo. Con ``mock`` el resultado es el mismo que con la API de a uno.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from showdown.rl.team_preview_action import (
    N_ACTIONS,
    action_to_team_string,
    decode_preview_action,
    legal_mask,
)
from showdown.rl.team_preview_obs import (
    OBS_DIM,
    PreviewMonFeat,
    PreviewObsBundle,
    encode_preview_obs,
    mon_feat_from_species,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None  # type: ignore
    spaces = None  # type: ignore


# Lightweight Champions-like pools for mock episodes
_DEFAULT_POOLS = [
    ['Incineroar', 'Flutter Mane', 'Amoonguss', 'Tornadus', 'Landorus-Therian', 'Rillaboom'],
    ['Kyogre', 'Groudon', 'Incineroar', 'Pikachu', 'Chi-Yu', 'Urshifu'],
    ['Calyrex-Shadow', 'Urshifu-Rapid-Strike', 'Raging Bolt', 'Ogerpon', 'Farigiraf', 'Indeedee-F'],
    ['Zacian', 'Miraidon', 'Koraidon', 'Terapagos', 'Whimsicott', 'Amoonguss'],
]


def _fix_pool_name(name: str) -> str:
    # typo guard from draft pools
    if name == 'Incinerorar':
        return 'Incineroar'
    return name


class TeamPreviewEnv:
    """
    Contextual bandit: one preview action → Cand.1 reward.

    backend='mock' (default): deterministic proxy outcome from coverage scores
    (no PS). Live greedy_joint combat is wired via optional `combat_fn`.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        *,
        seed: int = 0,
        backend: str = 'mock',
        combat_fn=None,
        our_pool: Optional[Sequence[str]] = None,
        foe_pool: Optional[Sequence[str]] = None,
        live_pool=None,
        team_source: Optional[str] = None,
    ):
        if gym is None:
            raise ImportError('gymnasium required for TeamPreviewEnv')
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.backend = str(backend)
        if self.backend == 'live' and live_pool is None and combat_fn is None:
            raise ValueError("backend='live' requiere live_pool (LivePreviewPool) o combat_fn")
        self.combat_fn = combat_fn
        self.live_pool = live_pool
        # metavgc: equipos reales (6 sets con export); default: pools de especies del proxy.
        self.team_source = team_source or ('metavgc' if self.backend == 'live' else 'default')
        self._teams = None
        self._ctx: Optional[Dict[str, Any]] = None
        self._batch: List[Tuple[PreviewObsBundle, Optional[Dict[str, Any]]]] = []
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._our_names = list(our_pool) if our_pool else None
        self._foe_names = list(foe_pool) if foe_pool else None
        self._bundle: Optional[PreviewObsBundle] = None
        self._obs: Optional[np.ndarray] = None
        self._done = True

    def action_masks(self) -> np.ndarray:
        n = len(self._bundle.our) if self._bundle else 6
        return np.asarray(legal_mask(n), dtype=np.float32)

    def _metavgc_teams(self):
        if self._teams is None:
            from showdown.inference_sim.team_factory import active_teams
            self._teams = list(active_teams())
            if len(self._teams) < 2:
                raise RuntimeError('se necesitan >= 2 equipos MetaVGC para preview live')
        return self._teams

    def _sample_metavgc_context(self) -> Dict[str, Any]:
        from showdown.inference_sim.team_factory import to_showdown_export

        teams = self._metavgc_teams()
        i, j = (int(x) for x in self._rng.choice(len(teams), size=2, replace=False))
        ours, foes = teams[i], teams[j]
        return {
            'our_names': [m.species for m in ours.mons][:6],
            'foe_names': [m.species for m in foes.mons][:6],
            'our_export': to_showdown_export(ours),
            'foe_export': to_showdown_export(foes),
            'our_team': ours.name,
            'foe_team': foes.name,
        }

    def _sample_pools(self) -> Tuple[List[str], List[str]]:
        if self._our_names and self._foe_names:
            return (
                [_fix_pool_name(x) for x in self._our_names],
                [_fix_pool_name(x) for x in self._foe_names],
            )
        i = int(self._rng.integers(0, len(_DEFAULT_POOLS)))
        j = int(self._rng.integers(0, len(_DEFAULT_POOLS)))
        while j == i and len(_DEFAULT_POOLS) > 1:
            j = int(self._rng.integers(0, len(_DEFAULT_POOLS)))
        return (
            [_fix_pool_name(x) for x in _DEFAULT_POOLS[i]],
            [_fix_pool_name(x) for x in _DEFAULT_POOLS[j]],
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._ctx = None
        if self.team_source == 'metavgc':
            self._ctx = self._sample_metavgc_context()
            our_n, foe_n = self._ctx['our_names'], self._ctx['foe_names']
        else:
            our_n, foe_n = self._sample_pools()
        our = [mon_feat_from_species(n) for n in our_n[:6]]
        foe = [mon_feat_from_species(n) for n in foe_n[:6]]
        while len(our) < 6:
            our.append(mon_feat_from_species(our_n[0] if our_n else 'Pikachu'))
        while len(foe) < 6:
            foe.append(mon_feat_from_species(foe_n[0] if foe_n else 'Mew'))
        self._bundle = PreviewObsBundle(our=our, foe=foe)
        # Coverage cached here (heuristic; Node calc optional later)
        self._obs = encode_preview_obs(self._bundle)
        self._done = False
        info = {
            'our_species': our_n[:6],
            'foe_species': foe_n[:6],
            'action_mask': self.action_masks(),
        }
        if self._ctx is not None:
            info['our_team'] = self._ctx['our_team']
            info['foe_team'] = self._ctx['foe_team']
        return self._obs.copy(), info

    def _score_action(self, a: int, *, as_attacker: bool) -> float:
        """Coverage proxy for leads+bench of action a (0..359)."""
        assert self._bundle is not None
        from showdown.rl.team_preview_obs import coverage_matrix
        m = coverage_matrix(self._bundle.our, self._bundle.foe)
        i1, i2, i3, i4 = decode_preview_action(int(a))
        if as_attacker:
            return float(m[i1, 0] + m[i1, 1] + m[i2, 0] + m[i2, 1] + 0.25 * (m[i3, 0] + m[i4, 1]))
        # Foe action j: use columns as "our slots threatened" proxy
        return float(m[0, i1] + m[1, i2] + m[0, i2] + m[1, i1] + 0.25 * (m[2, i3] + m[3, i4]))

    def _mock_cand1(self, a: int) -> float:
        """Proxy Cand.1 for our action vs foe greedy leads (legacy step)."""
        assert self._bundle is not None and self._obs is not None
        score = self._score_action(a, as_attacker=True)
        foe_base = self._score_action(0, as_attacker=False)
        if score > foe_base + 0.05:
            return 1.0
        if score < foe_base - 0.05:
            return -1.0
        return float(self._rng.choice([-1.0, 0.0, 1.0], p=[0.35, 0.3, 0.35]))

    def pair_label(self, i: int, j: int) -> float:
        """
        WR-scale label r ∈ {0, 0.5, 1} for (our action i, foe action j).
        Mock backend: coverage differential. Live: combat_fn if provided with both.
        """
        assert self._bundle is not None
        i, j = int(i), int(j)
        if self.live_pool is not None and self._ctx is not None:
            r = float(self._live_rewards([(self._ctx, i, j)])[0])
            return 1.0 if r > 0 else (0.0 if r < 0 else 0.5)
        if self.combat_fn is not None:
            # Optional hook: combat_fn(team_us, team_foe, bundle) → Cand.1
            team_us = action_to_team_string(i)
            team_foe = action_to_team_string(j)
            try:
                reward = float(self.combat_fn(team_us, self._bundle, team_foe=team_foe))
            except TypeError:
                reward = float(self.combat_fn(team_us, self._bundle))
            if reward > 0:
                return 1.0
            if reward < 0:
                return 0.0
            return 0.5
        our_s = self._score_action(i, as_attacker=True)
        foe_s = self._score_action(j, as_attacker=False)
        if our_s > foe_s + 0.05:
            return 1.0
        if our_s < foe_s - 0.05:
            return 0.0
        return 0.5

    def step(self, action: int):
        if self._done or self._bundle is None:
            raise RuntimeError('call reset() before step()')
        a = int(action)
        mask = self.action_masks()
        if mask[a] < 0.5:
            # illegal → treat as loss (should be masked by agent)
            reward = -1.0
            team = '????'
        else:
            team = action_to_team_string(a)
            if self.live_pool is not None and self._ctx is not None:
                reward = float(self._live_rewards([(self._ctx, a, None)])[0])
            elif self.combat_fn is not None:
                reward = float(self.combat_fn(team, self._bundle))
            elif self.backend == 'mock':
                reward = float(self._mock_cand1(a))
            else:
                raise RuntimeError(f'unknown backend {self.backend!r}; pass combat_fn')
            # Clamp to Cand.1 set
            if reward > 0:
                reward = 1.0
            elif reward < 0:
                reward = -1.0
            else:
                reward = 0.0
        self._done = True
        # Terminal obs = zeros (bandit); terminated True
        term_obs = np.zeros(OBS_DIM, dtype=np.float32)
        info = {
            'team': team,
            'preview_action': a,
            'pi_c': self._pi_c_label(),
        }
        return term_obs, reward, True, False, info

    # --- procedencia --------------------------------------------------------

    def _pi_c_label(self) -> str:
        if self.live_pool is not None:
            return str(getattr(self.live_pool, 'pilot', 'greedy_joint'))
        return 'greedy_joint' if self.combat_fn else 'mock_proxy'

    @property
    def provenance(self) -> str:
        """``live`` solo si las etiquetas salen de BO1 reales (spec: no mezclar)."""
        return 'live' if (self.live_pool is not None or self.combat_fn is not None) else 'mock'

    # --- API por lotes --------------------------------------------------------

    def _live_rewards(self, items) -> np.ndarray:
        """items: [(ctx, our_action, foe_action|None)] -> Cand. 1 por BO1 real."""
        from showdown.rl.preview_live import PreviewMatch

        matches = [
            PreviewMatch(
                our_export=ctx['our_export'],
                foe_export=ctx['foe_export'],
                our_order=action_to_team_string(int(a)),
                foe_order=None if b is None else action_to_team_string(int(b)),
            )
            for ctx, a, b in items
        ]
        return np.asarray(self.live_pool.run(matches), dtype=np.float32)

    def reset_batch(self, n: int) -> Tuple[np.ndarray, List[dict]]:
        """N contextos independientes (mismo muestreo que N llamadas a reset)."""
        obs, infos = [], []
        self._batch = []
        for _ in range(int(n)):
            o, info = self.reset()
            obs.append(o)
            infos.append(info)
            self._batch.append((self._bundle, self._ctx))
        self._done = True
        if not obs:
            return np.zeros((0, OBS_DIM), dtype=np.float32), []
        return np.stack(obs).astype(np.float32), infos

    def _check_batch(self, n: int) -> None:
        if n != len(self._batch):
            raise ValueError(f'lote de {n} acciones para {len(self._batch)} contextos; '
                             'llama reset_batch(n) antes')

    def step_batch(self, actions: Sequence[int]) -> np.ndarray:
        """Cand. 1 de cada acción sobre los contextos del último ``reset_batch``."""
        acts = [int(a) for a in actions]
        self._check_batch(len(acts))
        if self.live_pool is not None:
            items = [(ctx, a, None) for (_b, ctx), a in zip(self._batch, acts)]
            return self._live_rewards(items)
        out = np.zeros(len(acts), dtype=np.float32)
        for k, ((bundle, ctx), a) in enumerate(zip(self._batch, acts)):
            self._bundle, self._ctx, self._done = bundle, ctx, False
            _o, r, _t, _tr, _i = self.step(a)
            out[k] = float(r)
        return out

    def pair_label_batch(self, i_actions: Sequence[int], j_actions: Sequence[int]) -> np.ndarray:
        """Etiquetas WR {0, 0.5, 1} para pares (i nuestra, j rival) del último lote."""
        ii = [int(a) for a in i_actions]
        jj = [int(b) for b in j_actions]
        if len(ii) != len(jj):
            raise ValueError('i_actions y j_actions deben tener la misma longitud')
        self._check_batch(len(ii))
        if self.live_pool is not None:
            items = [(ctx, a, b) for (_b, ctx), a, b in zip(self._batch, ii, jj)]
            r = self._live_rewards(items)
            return np.where(r > 0, 1.0, np.where(r < 0, 0.0, 0.5)).astype(np.float32)
        out = np.zeros(len(ii), dtype=np.float32)
        for k, ((bundle, ctx), a, b) in enumerate(zip(self._batch, ii, jj)):
            self._bundle, self._ctx = bundle, ctx
            out[k] = float(self.pair_label(a, b))
        return out


def make_team_preview_env(**kwargs) -> TeamPreviewEnv:
    return TeamPreviewEnv(**kwargs)
