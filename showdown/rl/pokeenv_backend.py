"""Live poke-env BattleBackend for VGCEnv (local PS, Spec v1.1)."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from showdown.rl.env import BattleBackend
from showdown.rl.pe_adapter import (
    joint_order_from_spec_actions,
    pe_autoregressive_masks,
    snapshot_from_battle,
)
from showdown.rl.snapshot import BattleSnapshot

logger = logging.getLogger(__name__)

TeamLike = Union[str, Any]  # showdown export str or TeamMeta


def _team_export(team: Optional[TeamLike]) -> Optional[str]:
    if team is None:
        return None
    if isinstance(team, str):
        return team
    from showdown.inference_sim.team_factory import to_showdown_export
    return to_showdown_export(team)


def _team_meta(team: Optional[TeamLike]):
    """TeamMeta si el equipo vino como objeto (no export str); None si no se conoce."""
    if team is None or isinstance(team, str):
        return None
    return team if hasattr(team, 'mons') else None


def rl_team_split() -> str:
    """Grupo de equipos del entrenamiento RL: ``train`` (200) salvo que PS_TEAM_SPLIT diga otra cosa.

    Lo fija en el entorno del proceso para que el prior de SP y los mundos de expectimax de
    los rivales salgan del mismo grupo (``team_split.prior_split``).
    """
    import os
    os.environ.setdefault('PS_TEAM_SPLIT', 'train')
    return os.environ['PS_TEAM_SPLIT']


def _default_team_metas(seed: Optional[int] = None, split: Optional[str] = None):
    """Par (nuestro, rival) del grupo ``split`` (por defecto el de entrenamiento, ver foe_pool).

    Sin las opciones de ``foe_pool`` el sorteo es idéntico al anterior (mismos números aleatorios).
    """
    from showdown.rl.foe_pool import base_pool, maybe_synthetic_foe, training_split

    split = training_split() if split is None else split
    pool = base_pool(split)
    if len(pool) < 2:
        raise RuntimeError('Need ≥2 teams in MetaVGC pool for live RL backend')
    rng = np.random.default_rng(seed)
    i, j = int(rng.integers(0, len(pool))), int(rng.integers(0, len(pool) - 1))
    if j >= i:
        j += 1
    return pool[i], maybe_synthetic_foe(rng, split, pool[j])


def _default_teams(seed: Optional[int] = None) -> Tuple[str, str]:
    from showdown.inference_sim.team_factory import to_showdown_export

    a, b = _default_team_metas(seed)
    return to_showdown_export(a), to_showdown_export(b)


# Rivales heurísticos con wrapper validated (espejo IAP + OTS + prior leave-one-out),
# igual que en run_robust_e2e: "ganarle a joint" significa lo mismo en toda la tesis.
VALIDATED_FOES = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')

# PPO v2 (memorizacion_equipos.md §17): rivales de entrenamiento nuevos.
# - greedy_joint_arq: greedy_joint en modo arquetipo nivel 2 (Tailwind / Trick Room, Protect de
#   cobertura y sueño), solo en esta instancia (``wrap_archetype_level``).
# - ppo_v1: PPO v1 congelado (checkpoints en PPO_FOE_CHECKPOINTS, separados por comas; uno al azar
#   por partida) jugando con el MISMO equipo que el agente (espejo): v1 es especialista de ese equipo.
FOE_BASE_KIND = {'greedy_joint_arq': 'greedy_joint'}
ARCHETYPE_FOE_LEVEL = {'greedy_joint_arq': 2}
MIRROR_FOES = ('ppo_v1',)


def _base_kind(kind: str) -> str:
    key = str(kind or '').strip().lower().replace('-', '_')
    return FOE_BASE_KIND.get(key, key)


def foe_validated(kind: str) -> bool:
    """True si el rival `kind` va validated (PS_RL_FOE_VALIDATED=0 → rival crudo antiguo)."""
    import os
    if os.environ.get('PS_RL_FOE_VALIDATED', '1').strip().lower() in ('0', 'false', 'no', 'off'):
        return False
    return _base_kind(kind) in VALIDATED_FOES


# Rivales de ENTRENAMIENTO con sorteo entre casi-empates (docs/diversidad_baselines.md):
# misma fuerza medida (costo ~0) pero sin responder siempre igual al mismo estado.
# Solo afecta a los rivales vivos del entorno RL; la evaluación (run_robust_e2e,
# round-robin de la Conclusión) usa las versiones deterministas.
FOE_TIE_MARGINS = {
    'greedy_joint': ('tie_margin', 8.0),             # ~14 % de jugadas distintas, -0,2 pts n.s.
    'expectimax_bayesian': ('phase1_tie_margin', 3.0),  # ~4 %, -0,06 pts
}


def foe_tie_enabled() -> bool:
    """Sorteo en los rivales de entrenamiento (por defecto sí; PS_RL_FOE_TIE=0 lo apaga)."""
    import os
    return os.environ.get('PS_RL_FOE_TIE', '1').strip().lower() not in ('0', 'false', 'no', 'off')


def apply_foe_tie(player, kind: str) -> bool:
    """Activa el sorteo de casi-empates en la instancia ``player`` (no toca variables globales)."""
    key = _base_kind(kind)
    spec = FOE_TIE_MARGINS.get(key)
    if spec is None:
        return False
    field_name, margin = spec
    if key == 'greedy_joint':
        player.tie_margin = float(margin)
        return True
    from dataclasses import replace
    from showdown.battle_bots.expectimax_bayesian.expectimax import load_expectimax_config
    cfg = getattr(player, 'expectimax_config', None) or load_expectimax_config()
    player.expectimax_config = replace(cfg, **{field_name: float(margin)})
    return True


LIVE_OPPONENTS = (
    'random',
    'greedy_solo',
    'greedy_joint',
    'expectimax_bayesian',
    'greedy_joint_arq',
    'ppo_v1',
)


def _make_foe_player(
    kind: str,
    *,
    account_configuration,
    battle_format: str,
    team: str,
    server_configuration,
    log_level: int,
    foe_meta=None,
    agent_meta=None,
    tie: Optional[bool] = None,
):
    """Build poke-env opponent Player for RL live training (con sorteo, ver ``apply_foe_tie``)."""
    player = _build_foe_player(
        kind, account_configuration=account_configuration, battle_format=battle_format,
        team=team, server_configuration=server_configuration, log_level=log_level,
        foe_meta=foe_meta, agent_meta=agent_meta,
    )
    if foe_tie_enabled() if tie is None else bool(tie):
        apply_foe_tie(player, kind)
    level = ARCHETYPE_FOE_LEVEL.get(str(kind or '').strip().lower().replace('-', '_'))
    if level:
        from showdown.battle_bots.greedy_joint.joint import wrap_archetype_level
        wrap_archetype_level(player, level)
    return player


def _build_foe_player(
    kind: str,
    *,
    account_configuration,
    battle_format: str,
    team: str,
    server_configuration,
    log_level: int,
    foe_meta=None,
    agent_meta=None,
):
    """Build poke-env opponent Player for RL live training.

    Con ``foe_meta`` (equipo del rival) y ``agent_meta`` (equipo del agente RL) como
    TeamMeta, los heurísticos van con wrapper validated: misma información que en
    run_robust_e2e (OTS del rival, SP según su modo por defecto, prior leave-one-out).
    Sin TeamMeta (equipos como export str) se usa el rival crudo: stats del rival =
    bases de especie y HP en % (pierde ~2 pts solo por información).
    """
    key = _base_kind(str(kind or 'random'))
    if key == 'ppo_v1':
        return _build_ppo_v1_foe(account_configuration=account_configuration, battle_format=battle_format,
                                 team=team, server_configuration=server_configuration, log_level=log_level)
    if foe_validated(key) and foe_meta is not None and agent_meta is not None:
        from showdown.inference_sim.battle_state_check import StateCheckReport
        from showdown.inference_sim.validated_greedy_player import validated_player_class

        return validated_player_class(key)(
            user_team=foe_meta,
            opp_team=agent_meta,
            state_report=StateCheckReport(),
            soundness_report=None,
            account_configuration=account_configuration,
            battle_format=battle_format,
            server_configuration=server_configuration,
            log_level=log_level,
        )
    kwargs = dict(
        account_configuration=account_configuration,
        battle_format=battle_format,
        team=team,
        server_configuration=server_configuration,
        accept_open_team_sheet=True,
        max_concurrent_battles=1,
        log_level=log_level,
    )
    if key == 'random':
        from poke_env.player import RandomPlayer
        return RandomPlayer(**kwargs)
    if key == 'greedy_solo':
        from showdown.battle_bots.greedy_solo.pokeenv_player import GreedyDamagePlayer
        return GreedyDamagePlayer(**kwargs)
    if key == 'greedy_joint':
        from showdown.battle_bots.greedy_joint.pokeenv_player import GreedyJointPlayer
        return GreedyJointPlayer(**kwargs)
    if key == 'expectimax_bayesian':
        from showdown.battle_bots.expectimax_bayesian.pokeenv_player import (
            ExpectimaxBayesianPlayer,
        )
        return ExpectimaxBayesianPlayer(**kwargs)
    raise ValueError(
        f'oponente live inválido: {kind!r}. Opciones: {", ".join(LIVE_OPPONENTS)}'
    )


def ppo_foe_checkpoints() -> List[str]:
    import os
    raw = os.environ.get('PPO_FOE_CHECKPOINTS', '')
    return [p.strip() for p in raw.split(',') if p.strip()]


def _build_ppo_v1_foe(*, account_configuration, battle_format, team, server_configuration, log_level):
    """Rival PPO congelado (P3): un checkpoint al azar de PPO_FOE_CHECKPOINTS por partida, greedy."""
    import multiprocessing
    import random as _random

    from showdown.rl.ppo.player import PPOPlayer

    cks = ppo_foe_checkpoints()
    if not cks:
        raise ValueError('ppo_v1: falta PPO_FOE_CHECKPOINTS (checkpoints del PPO v1 congelado)')
    try:  # 32 entornos con una red cada uno: un hilo de torch por proceso de entorno
        import torch
        if multiprocessing.current_process().name != 'MainProcess':
            torch.set_num_threads(1)
    except Exception:
        pass

    class PPOv1Foe(PPOPlayer):
        """Cambia de checkpoint (al azar) en cada partida nueva."""

        def __init__(self, *a, **kw):
            import os
            prev = os.environ.get('PPO_DEVICE')
            os.environ['PPO_DEVICE'] = 'cpu'  # el rival nunca usa la GPU del entrenamiento
            try:
                super().__init__(*a, checkpoint=None, greedy=True, **kw)
            finally:
                if prev is None:
                    os.environ.pop('PPO_DEVICE', None)
                else:
                    os.environ['PPO_DEVICE'] = prev
            from showdown.rl.ppo.agent import PPOAgent, PPOConfig
            self._pool = []
            for c in cks:
                ag = PPOAgent(PPOConfig(device='cpu'))
                ag.load(c, map_location='cpu')
                self._pool.append(ag)
            self.agent = self._pool[0]
            self.checkpoint_path = cks[0]
            self._cur_tag = None
            self.foe_checkpoint_counts = {c: 0 for c in cks}

        def choose_move(self, battle):
            tag = getattr(battle, 'battle_tag', None)
            if tag != self._cur_tag:
                self._cur_tag = tag
                i = _random.randrange(len(self._pool))
                self.agent = self._pool[i]
                self.foe_checkpoint_counts[cks[i]] += 1
            return super().choose_move(battle)

    return PPOv1Foe(account_configuration=account_configuration, battle_format=battle_format, team=team,
                    server_configuration=server_configuration, accept_open_team_sheet=True,
                    max_concurrent_battles=1, log_level=log_level)


class PokeEnvBattleBackend(BattleBackend):
    """
    Sync Gym bridge: controlled `_EnvPlayer` vs configurable foe on local Showdown.

    Episode = one BO1 game (finished battle). Requires PS listening
    (`node pokemon-showdown start --no-security`).
    """

    def __init__(
        self,
        *,
        battle_format: Optional[str] = None,
        server_configuration=None,
        our_team: Optional[TeamLike] = None,
        opp_team: Optional[TeamLike] = None,
        opponent: str = 'random',
        challenge_timeout: float = 90.0,
        step_timeout: float = 180.0,
        start_ps: bool = False,
        log_level: int = 30,
        account_prefix: str = 'RlEnv',
        combat_family: str = 'ddqn',
        preview_checkpoint: Optional[str] = None,
        preview_kind: Optional[str] = None,
        port: Optional[int] = None,
        foe_tie: Optional[bool] = None,
        team_split: Optional[str] = None,
    ):
        from poke_env.ps_client import LocalhostServerConfiguration

        from showdown.inference_sim.paths import BATTLE_FORMAT, server_configuration_for_port
        from showdown.rl.frozen_preview import family_preview_for_combat, load_frozen_preview

        self.battle_format = battle_format or BATTLE_FORMAT
        # port: servidor Showdown propio de este entorno (entrenamiento vectorizado).
        self.port = int(port) if port is not None else None
        # Sorteo de casi-empates de los rivales (v1.21): None = PS_RL_FOE_TIE (sí por defecto);
        # False = rivales deterministas (validación de checkpoints, ddqn_spec 1.16).
        self._foe_tie = foe_tie
        # Grupo de equipos de este entorno (None = el de entrenamiento; 'val' en la validación
        # reservada, 'eval' en diagnósticos). Ver showdown/rl/foe_pool.py.
        self._team_split = team_split
        self.last_foe_team: Optional[str] = None
        self.last_our_team: Optional[str] = None  # equipo del agente en la última partida (diagnósticos)
        self._fs_base: Optional[Tuple[int, int, int, int]] = None
        if server_configuration is None and self.port is not None:
            server_configuration = server_configuration_for_port(self.port)
        self.server_configuration = server_configuration or LocalhostServerConfiguration
        self._our_team_arg = our_team
        self._opp_team_arg = opp_team
        self._opponent = str(opponent or 'random').strip().lower().replace('-', '_')
        if self._opponent not in LIVE_OPPONENTS:
            raise ValueError(
                f'oponente live inválido: {opponent!r}. '
                f'Opciones: {", ".join(LIVE_OPPONENTS)}'
            )
        self.challenge_timeout = float(challenge_timeout)
        self.step_timeout = float(step_timeout)
        self.start_ps = bool(start_ps)
        self.log_level = int(log_level)
        self.account_prefix = str(account_prefix)
        self.combat_family = str(combat_family or 'ddqn')
        if preview_kind:
            self._preview_policy = load_frozen_preview(
                kind=preview_kind, checkpoint=preview_checkpoint,
            )
        else:
            self._preview_policy = family_preview_for_combat(
                self.combat_family, checkpoint=preview_checkpoint,
            )

        self._agent = None
        self._foe = None
        self._battle = None
        self._challenge_task = None
        self._uid = 0
        self._closed = False
        self._cached_opp_export: Optional[str] = None

    @property
    def opponent(self) -> str:
        return self._opponent

    def set_opponent(self, kind: str) -> None:
        """Switch foe policy between episodes (recreates poke-env Player B)."""
        key = str(kind or 'random').strip().lower().replace('-', '_')
        if key not in LIVE_OPPONENTS:
            raise ValueError(
                f'oponente live inválido: {kind!r}. '
                f'Opciones: {", ".join(LIVE_OPPONENTS)}'
            )
        if key == self._opponent and self._foe is not None:
            return
        self._abort_open_battle()
        self._stop_player(self._foe)
        self._foe = None
        self._opponent = key
        # Recreate foe on next reset via _ensure_players
        if self._agent is not None:
            self._ensure_players(seed=None, force_new_foe=True)

    # --- BattleBackend -----------------------------------------------------

    def reset(self, *, seed: Optional[int] = None) -> BattleSnapshot:
        self._ensure_ps()
        self._abort_open_battle()
        self._ensure_players(seed=seed)

        assert self._agent is not None and self._foe is not None
        loop = self._agent.ps_client.loop
        self._challenge_task = asyncio.run_coroutine_threadsafe(
            self._agent.battle_against(self._foe, n_battles=1),
            loop,
        )
        try:
            self._battle = self._agent.battle_queue.get(timeout=self.challenge_timeout)
        except Exception as exc:
            self._challenge_task = None
            raise TimeoutError(
                f'RL env challenge timed out ({self.challenge_timeout}s): {exc!r}'
            ) from exc
        self._fs_base = self._foe_state_counts()
        try:  # P5: base de los contadores de la calculadora para este episodio
            from showdown.rl.observation import calc_stats
            self._oc_base = calc_stats()
        except Exception:
            self._oc_base = None
        return snapshot_from_battle(self._battle)

    def step(
        self,
        action: Tuple[int, int],
    ) -> Tuple[BattleSnapshot, Dict[str, Any]]:
        if self._battle is None or self._agent is None:
            raise RuntimeError('PokeEnvBattleBackend.step called before reset')
        if self._battle.finished:
            return snapshot_from_battle(self._battle), {
                'engine': 'pokeenv',
                'used_fallback': False,
                'already_finished': True,
            }

        order, used_fallback = joint_order_from_spec_actions(self._battle, action)
        self._agent.order_queue.put(order)
        try:
            self._battle = self._agent.battle_queue.get(timeout=self.step_timeout)
        except Exception as exc:
            raise TimeoutError(
                f'RL env step timed out ({self.step_timeout}s): {exc!r}'
            ) from exc

        meta: Dict[str, Any] = {
            'engine': 'pokeenv',
            'used_fallback': bool(used_fallback),
            'order': str(order),
        }
        fs = self.foe_state_episode()
        if fs is not None:
            meta['foe_state'] = fs
        return snapshot_from_battle(self._battle), meta

    # --- contraste de estado del rival heurístico (battle_state_check) ---------

    def _foe_state_counts(self) -> Optional[Tuple[int, int, int, int]]:
        rep = getattr(self._foe, 'state_report', None)
        if rep is None:
            return None
        return (int(rep.checks), int(rep._total_mismatches), int(rep.feed_errors), len(rep.mismatches))

    def foe_state_episode(self) -> Optional[Dict[str, Any]]:
        """Contraste de estado del rival en este episodio (acumulado desde el reset).

        Los heurísticos validated comparan cada turno su estado interno (clases del repo) con
        poke-env. Antes el resultado se descartaba; ahora va al registro del entrenamiento.
        El rival procesa su turno en paralelo, así que el último turno de un episodio puede
        contarse en el siguiente (los totales no cambian). ``None`` para rivales sin espejo.
        """
        oc = self._obs_calc_episode()
        cur = self._foe_state_counts()
        if cur is None:
            return {'obs_calc': oc} if oc else None
        base = self._fs_base if self._fs_base is not None else (0, 0, 0, 0)
        if cur[0] < base[0]:  # rival reconstruido: reporte nuevo
            base = (0, 0, 0, 0)
        out: Dict[str, Any] = {'checks': cur[0] - base[0], 'mismatches': cur[1] - base[1],
                               'feed_errors': cur[2] - base[2]}
        if out['mismatches'] > 0:
            rep = self._foe.state_report
            out['samples'] = [f'{m.field}: esperado {m.expected!r}, espejo {m.got!r} (turno {m.turn})'
                              for m in rep.mismatches[base[3]:base[3] + 3]]
        if oc:
            out['obs_calc'] = oc
        return out

    def _obs_calc_episode(self) -> Optional[Dict[str, int]]:
        """P5: observaciones de este episodio con la calculadora real / con respaldo (proceso del entorno)."""
        try:
            from showdown.rl.observation import calc_stats
        except Exception:
            return None
        cur = calc_stats()
        base = getattr(self, '_oc_base', None) or {}
        d = {k: int(cur.get(k, 0)) - int(base.get(k, 0)) for k in cur}
        return d if d.get('obs', 0) > 0 else None

    def action_masks(
        self,
        *,
        slot1_action_id: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._battle is None:
            return None
        return pe_autoregressive_masks(self._battle, slot1_action_id=slot1_action_id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._abort_open_battle()
        self._stop_player(self._agent)
        self._stop_player(self._foe)
        self._agent = None
        self._foe = None
        self._battle = None

    # --- internals ---------------------------------------------------------

    def _ensure_ps(self) -> None:
        from showdown.inference_sim.ps_server import ensure_ps, is_ps_up, ps_start_hint

        if self.port is None:
            from showdown.inference_sim.paths import PS_PORT
            port = int(PS_PORT)
        else:
            port = self.port
        if not ensure_ps(start=self.start_ps, port=port):
            raise RuntimeError(
                f'Pokémon Showdown no responde en localhost:{port}. '
                f'Arranca: {ps_start_hint(port)}  o pasa start_ps=True'
            )
        if not is_ps_up(port=port):
            raise RuntimeError(f'PS still down after ensure; {ps_start_hint(port)}')

    def _ensure_players(
        self,
        *,
        seed: Optional[int],
        force_new_foe: bool = False,
    ) -> None:
        from poke_env.environment.env import _EnvPlayer
        from poke_env.ps_client import AccountConfiguration

        our_meta = _team_meta(self._our_team_arg)
        opp_meta = _team_meta(self._opp_team_arg)
        our = _team_export(self._our_team_arg)
        opp = _team_export(self._opp_team_arg)
        if our is None or opp is None:
            d0, d1 = _default_team_metas(seed=seed, split=self._team_split)
            if our is None:
                our, our_meta = _team_export(d0), d0
            if opp is None:
                opp, opp_meta = _team_export(d1), d1
        if self._opponent in MIRROR_FOES:
            # Espejo (P3): el PPO v1 congelado juega con el equipo del agente.
            opp, opp_meta = our, our_meta
        self._cached_opp_export = opp
        if opp_meta is not None:
            from showdown.inference_sim.team_split import team_key
            self.last_foe_team = team_key(opp_meta)
        if our_meta is not None:
            from showdown.inference_sim.team_split import team_key
            self.last_our_team = team_key(our_meta)
        foe_kw = dict(foe_meta=opp_meta, agent_meta=our_meta)

        self._uid += 1
        stamp = int(time.time()) % 100000
        name_a = f'{self.account_prefix}A{stamp}{self._uid}'
        name_b = f'{self.account_prefix}B{stamp}{self._uid}'

        preview_pol = self._preview_policy

        class _FamilyPreviewEnvPlayer(_EnvPlayer):
            def teampreview(self, battle):
                return preview_pol.teampreview(battle)

        if self._agent is None:
            self._agent = _FamilyPreviewEnvPlayer(
                account_configuration=AccountConfiguration(name_a, None),
                battle_format=self.battle_format,
                team=our,
                server_configuration=self.server_configuration,
                accept_open_team_sheet=True,
                max_concurrent_battles=1,
                log_level=self.log_level,
                choose_on_teampreview=True,
            )
            self._foe = _make_foe_player(
                self._opponent,
                account_configuration=AccountConfiguration(name_b, None),
                battle_format=self.battle_format,
                team=opp,
                server_configuration=self.server_configuration,
                log_level=self.log_level,
                tie=self._foe_tie,
                **foe_kw,
            )
        else:
            self._agent.update_team(our)
            if self._foe is None or force_new_foe or self._foe_needs_rebuild(**foe_kw):
                self._stop_player(self._foe)
                self._foe = _make_foe_player(
                    self._opponent,
                    account_configuration=AccountConfiguration(name_b, None),
                    battle_format=self.battle_format,
                    team=opp,
                    server_configuration=self.server_configuration,
                    log_level=self.log_level,
                    tie=self._foe_tie,
                    **foe_kw,
                )
            else:
                self._update_foe_team(opp, **foe_kw)
            # Drain queues
            while not self._agent.battle_queue.empty():
                try:
                    self._agent.battle_queue.get(timeout=0.1)
                except Exception:
                    break
            while not self._agent.order_queue.empty():
                try:
                    self._agent.order_queue.get(timeout=0.1)
                except Exception:
                    break
            self._agent.battle = None

    def _update_foe_team(self, opp: str, *, foe_meta=None, agent_meta=None) -> None:
        """Nuevo episodio: el rival validated reinicia su espejo con ambos equipos."""
        from showdown.inference_sim.validated_greedy_player import ValidatedBattleMixin

        if isinstance(self._foe, ValidatedBattleMixin):
            self._foe.reset_for_next_battle(foe_meta, agent_meta)
            return
        self._foe.update_team(opp)

    def _foe_needs_rebuild(self, *, foe_meta=None, agent_meta=None) -> bool:
        """El tipo de rival (validated / crudo) debe coincidir con los equipos disponibles."""
        from showdown.inference_sim.validated_greedy_player import ValidatedBattleMixin

        want = foe_validated(self._opponent) and foe_meta is not None and agent_meta is not None
        return isinstance(self._foe, ValidatedBattleMixin) != want

    def _stop_player(self, pl) -> None:
        if pl is None:
            return
        try:
            from showdown.inference_sim.ps_client_shutdown import stop_player_client

            fut = asyncio.run_coroutine_threadsafe(
                stop_player_client(pl),
                pl.ps_client.loop,
            )
            fut.result(timeout=10)
        except Exception:
            pass

    def _abort_open_battle(self) -> None:
        from poke_env.player.battle_order import ForfeitBattleOrder

        if self._agent is None or self._battle is None:
            self._challenge_task = None
            return
        if not getattr(self._battle, 'finished', True):
            try:
                if not self._agent.battle_queue.empty():
                    self._agent.battle_queue.get(timeout=1.0)
                self._agent.order_queue.put(ForfeitBattleOrder())
                self._agent.battle_queue.get(timeout=15.0)
            except Exception as exc:
                logger.debug('forfeit during abort: %r', exc)
        if self._challenge_task is not None:
            try:
                self._challenge_task.result(timeout=30)
            except Exception:
                pass
            self._challenge_task = None
        self._battle = None
