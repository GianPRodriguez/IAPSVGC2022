"""DDQN agent: masked ε-greedy + Double update (docs/ddqn_spec.md)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from showdown.rl.action_space import (
    apply_q_mask,
    decode_joint_action,
    encode_joint_action,
)
from showdown.rl.constants import JOINT_ACTION_N, OBS_DIM, PASS_ID


@dataclass
class DDQNConfig:
    gamma: float = 0.95
    lr: float = 3e-4
    tau: float = 0.005
    batch_size: int = 256
    replay_capacity: int = 200_000
    warmup_steps: int = 5_000
    train_freq: int = 4
    huber_delta: float = 1.0
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 500_000
    device: str = 'cpu'
    seed: int = 0
    # Aumentación por simetría (showdown/rl/symmetry.py): a cada transición
    # muestreada se le aplica un elemento aleatorio del grupo a/b propio, a/b
    # rival y reservas, a (s, a) y otro independiente a (s', m'). Exacto: Q es
    # invariante. Apagada por defecto (RL_SYM_AUG=1 la enciende; ver symmetry.sym_aug_enabled).
    sym_aug: bool = True
    # Dónde vive el replay (ddqn_spec 1.13). 'auto' = en la GPU si el agente entrena en
    # CUDA: muestreo y aumentación sin copiar ~25 MB por lote desde la CPU. Mismos lotes
    # que en CPU. RL_REPLAY_DEVICE=cpu fuerza la versión anterior.
    replay_device: str = 'auto'
    # Sobreestimación de Q (piloto 15-11-2026: Q medio hasta 5,6 con retornos <= 1,8).
    # target_update_every > 0: copia dura de la red objetivo cada N pasos de gradiente en vez
    # de la actualización suave con tau (separa más objetivo y online para Double DQN).
    # target_clip > 0: el objetivo y se acota a [-target_clip, target_clip] (rango real de
    # retornos). 0 = desactivado. Por defecto 2,0 desde ddqn_spec 1.15: los retornos de la
    # recompensa DENSE van de -1,6 a +1,8; pilotos A/B: Q calibrado y 36,7 % frente a 25,3 %.
    target_update_every: int = 0
    target_clip: float = 2.0
    # LayerNorm tras el codificador (punto 7 del checklist; se evalúa en el tuning).
    layer_norm: bool = False
    # Remedios contra la memorización de equipos rivales (memorizacion_equipos.md §8).
    # weight_decay > 0: AdamW (decaimiento desacoplado) en vez de Adam.
    # dropout > 0: dropout tras cada capa oculta de la red Q (solo al aprender).
    # **REG por defecto desde el 17-11-2026 (ddqn_spec 1.20, decisión del usuario):** weight decay
    # 0,01 + dropout 0,1 (decimocuarta cola: 62,3 / 65,3 % contra equipos eval con final.pt
    # frente a 50,0 / 49,0 % sin ellos). weight_decay=0 y dropout=0 restauran la versión anterior.
    # foe_bench_dropout > 0: al aprender, cada uno de los 4 Pokémon rivales no activos de la
    # obs s se borra (bloque y pares a cero = "ausente") con esa probabilidad. Apagado: no mejoró.
    # Desde el 18-11-2026 (ddqn_spec 1.22, decisión del usuario): weight decay 0,03, elegido por la
    # decimoquinta cola con `val` (74,1 % frente a 67,6 % con 0,01; en `eval` +0,9).
    weight_decay: float = 0.03
    dropout: float = 0.1
    foe_bench_dropout: float = 0.0
    # Cabeza de la red Q (memorizacion_equipos.md §10; ddqn_spec 1.21). 'joint' = 1 600 salidas
    # (por defecto); 'factored' = V + A1(a1) + A2(a2) + interacción de rango q_rank.
    q_head: str = 'joint'
    q_rank: int = 0


class DDQNAgent:
    def __init__(self, cfg: Optional[DDQNConfig] = None):
        self.cfg = cfg or DDQNConfig()
        self._torch = None
        self.online = None
        self.target = None
        self.optim = None
        self.replay = None
        self._rng = np.random.default_rng(int(self.cfg.seed))
        self.env_steps = 0
        self.grad_steps = 0
        # Tras reanudar (--resume) el replay está vacío: no aprender hasta volver a juntar
        # warmup_steps transiciones (en una corrida nueva vale 0 y no cambia nada).
        self.min_replay_to_learn = 0
        self._ensure_torch()

    def _ensure_torch(self) -> None:
        if self.online is not None:
            return
        try:
            import torch
            import torch.nn.functional as F
            from showdown.rl.ddqn.network import QNetwork, copy_weights, soft_update
            from showdown.rl.ddqn.replay import ReplayBuffer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'DDQN requires PyTorch. Install: pip install -r requirements-rl.txt'
            ) from exc
        self._torch = torch
        self._F = F
        self._soft_update = soft_update
        self._copy_weights = copy_weights
        device = torch.device(self.cfg.device)
        self.device = device
        kw = dict(layer_norm=bool(self.cfg.layer_norm), dropout=float(self.cfg.dropout),
                  q_head=str(self.cfg.q_head), q_rank=int(self.cfg.q_rank))
        self.online = QNetwork(**kw).to(device)
        self.target = QNetwork(**kw).to(device)
        copy_weights(self.online, self.target)
        self.target.eval()
        if float(self.cfg.weight_decay) > 0.0:
            self.optim = torch.optim.AdamW(self.online.parameters(), lr=self.cfg.lr,
                                           weight_decay=float(self.cfg.weight_decay))
        else:
            self.optim = torch.optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.replay = ReplayBuffer(capacity=self.cfg.replay_capacity, obs_dim=OBS_DIM,
                                   device=self._replay_device(device))

    def _replay_device(self, device) -> str:
        import os
        want = os.environ.get('RL_REPLAY_DEVICE', '').strip() or str(self.cfg.replay_device or 'auto')
        if want == 'auto':
            return str(device) if getattr(device, 'type', str(device)).startswith('cuda') else 'cpu'
        return want

    @property
    def epsilon(self) -> float:
        t = min(1.0, float(self.env_steps) / max(1, self.cfg.eps_decay_steps))
        return float(self.cfg.eps_start + t * (self.cfg.eps_end - self.cfg.eps_start))

    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        *,
        epsilon: Optional[float] = None,
    ) -> int:
        """Return joint action id.

        ``epsilon=None`` → schedule de entrenamiento (incluye warm-up aleatorio).
        ``epsilon`` explícito → se respeta tal cual, sin warm-up: ``0`` es
        siempre greedy (inferencia), aunque ``env_steps`` sea menor que warm-up.
        """
        self._ensure_torch()
        m = np.asarray(mask, dtype=np.float32).reshape(JOINT_ACTION_N)
        legal = np.flatnonzero(m > 0.5)
        if legal.size == 0:
            return encode_joint_action(PASS_ID, PASS_ID)
        if epsilon is None:
            if self.env_steps < self.cfg.warmup_steps or self._rng.random() < self.epsilon:
                return int(self._rng.choice(legal))
            return self.greedy(obs, m)
        eps = float(epsilon)
        if eps > 0.0 and self._rng.random() < eps:
            return int(self._rng.choice(legal))
        return self.greedy(obs, m)

    def act_batch(
        self,
        obs: np.ndarray,
        masks: np.ndarray,
        *,
        epsilon: Optional[float] = None,
    ) -> np.ndarray:
        """ε-greedy en lote para N entornos (una sola pasada de la red).

        Misma política que :meth:`act` aplicada fila a fila: warmup/ε → acción
        legal aleatoria; si no, argmax de Q enmascarado.
        """
        self._ensure_torch()
        torch = self._torch
        obs = np.asarray(obs, dtype=np.float32).reshape(-1, OBS_DIM)
        masks = np.asarray(masks, dtype=np.float32).reshape(-1, JOINT_ACTION_N)
        n = obs.shape[0]
        eps = self.epsilon if epsilon is None else float(epsilon)
        # Warm-up solo con el schedule (epsilon=None); epsilon explícito se respeta.
        warm = epsilon is None and self.env_steps < self.cfg.warmup_steps
        explore = warm or (eps >= 1.0)
        rand = self._rng.random(n) < eps if eps > 0.0 else np.zeros(n, dtype=bool)
        need_q = (not explore) and bool((~rand).any())
        q = None
        if need_q:
            self.online.eval()
            with torch.no_grad():
                x = torch.as_tensor(obs, device=self.device)
                q = self.online(x).float().cpu().numpy()
        out = np.zeros(n, dtype=np.int64)
        for i in range(n):
            legal = np.flatnonzero(masks[i] > 0.5)
            if legal.size == 0:
                out[i] = encode_joint_action(PASS_ID, PASS_ID)
            elif explore or rand[i] or q is None:
                out[i] = int(self._rng.choice(legal))
            else:
                out[i] = int(np.argmax(apply_q_mask(q[i], masks[i])))
        return out

    def greedy(self, obs: np.ndarray, mask: np.ndarray) -> int:
        self._ensure_torch()
        torch = self._torch
        self.online.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
            q = self.online(x).cpu().numpy().reshape(-1)
        q_m = apply_q_mask(q, mask)
        return int(np.argmax(q_m))

    def observe(
        self,
        *,
        s: np.ndarray,
        a: int,
        r: float,
        s_next: np.ndarray,
        terminated: bool,
        m_next: np.ndarray,
        skip_buffer: bool = False,
        truncated: bool = False,
        opponent_code: int = 0,
    ) -> Optional[Dict[str, float]]:
        """Store transition e_t=(s,a,r,s',done,m'); maybe learn."""
        self._ensure_torch()
        self.env_steps += 1
        self.replay.add(
            s=s, a=a, r=r, s_next=s_next, terminated=terminated, m_next=m_next,
            skip_buffer=skip_buffer, truncated=truncated,
            opponent_code=int(opponent_code),
        )
        if self.env_steps < self.cfg.warmup_steps:
            return None
        if len(self.replay) < self.min_replay_to_learn:
            return None
        if self.env_steps % self.cfg.train_freq != 0:
            return None
        if len(self.replay) < self.cfg.batch_size:
            return None
        return self.learn()

    def learn(self) -> Dict[str, float]:
        self._ensure_torch()
        torch = self._torch
        F = self._F
        from showdown.rl.ddqn.replay import ReplayBuffer

        sym_stats: Dict[str, float] = {}
        if self.replay.on_gpu:
            # Replay en GPU: mismo sorteo (índices y elementos del grupo, en ese orden) que
            # la rama NumPy; el muestreo y las permutaciones se hacen en la GPU.
            s_t, a_t, r_t, s2_t, done, m2_t, opp = self.replay.sample_torch(
                self.cfg.batch_size, self._rng)
            if s_t.shape[1] == OBS_DIM:
                from showdown.rl import symmetry as SYM

                if SYM.sym_aug_enabled(self.cfg.sym_aug):
                    g = SYM.sample_elements_t(s_t, self._rng)
                    s_t, a_t = SYM.augment_obs_t(s_t, g), SYM.augment_joint_t(a_t, g)
                    h = SYM.sample_elements_t(s2_t, self._rng)
                    s2_t, m2_t = SYM.augment_obs_t(s2_t, h), SYM.augment_joint_mask_t(m2_t, h)
                    sym_stats = SYM.element_stats(g)
            a_t, r_t, done = a_t.long(), r_t.float(), done.float()
        else:
            s, a, r, s2, term, m2, opp = self.replay.sample(self.cfg.batch_size, self._rng)
            if s.shape[1] == OBS_DIM:
                from showdown.rl import symmetry as SYM

                if SYM.sym_aug_enabled(self.cfg.sym_aug):
                    g = SYM.sample_elements(s, self._rng)
                    s, a = SYM.augment_obs(s, g), SYM.augment_joint(a, g)
                    h = SYM.sample_elements(s2, self._rng)
                    s2, m2 = SYM.augment_obs(s2, h), SYM.augment_joint_mask(m2, h)
                    sym_stats = SYM.element_stats(g)
            s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
            a_t = torch.as_tensor(a, dtype=torch.int64, device=self.device)
            r_t = torch.as_tensor(r, dtype=torch.float32, device=self.device)
            s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
            done = torch.as_tensor(term, dtype=torch.float32, device=self.device)
            m2_t = torch.as_tensor(m2, dtype=torch.float32, device=self.device)

        reg_stats: Dict[str, float] = {}
        if float(self.cfg.foe_bench_dropout) > 0.0 and s_t.shape[1] == OBS_DIM:
            from showdown.rl.ddqn.regularize import foe_bench_dropout_t
            s_t, frac = foe_bench_dropout_t(s_t, float(self.cfg.foe_bench_dropout), self._rng)
            reg_stats['foe_bench_dropped'] = frac

        with torch.no_grad():
            # Selección de a* sin dropout (red online en modo evaluación) y con s' completo.
            self.online.eval()
            q_online_next = self.online(s2_t)
            # Mask illegal before argmax
            q_online_next = q_online_next + (1.0 - m2_t) * (-1.0e9)
            a_star = q_online_next.argmax(dim=1)
            q_target_next = self.target(s2_t).gather(1, a_star.view(-1, 1)).squeeze(1)
            y = r_t + (1.0 - done) * self.cfg.gamma * q_target_next
            if float(self.cfg.target_clip) > 0.0:
                c = float(self.cfg.target_clip)
                y = y.clamp(-c, c)

        self.online.train()
        q_all = self.online(s_t)
        q_sa = q_all.gather(1, a_t.view(-1, 1)).squeeze(1)

        loss = F.huber_loss(q_sa, y, delta=self.cfg.huber_delta)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optim.step()
        if int(self.cfg.target_update_every) > 0:
            if (self.grad_steps + 1) % int(self.cfg.target_update_every) == 0:
                self._copy_weights(self.online, self.target)
        else:
            self._soft_update(self.online, self.target, self.cfg.tau)
        self.grad_steps += 1
        batch_frac = ReplayBuffer.batch_origin_fractions(opp)
        buf_frac = self.replay.origin_fractions()
        out: Dict[str, float] = {
            'loss': float(loss.item()),
            'mean_q': float(q_sa.mean().item()),
            'mean_target': float(y.mean().item()),
            'td_error': float((q_sa - y).abs().mean().item()),
            'eps': self.epsilon,
        }
        out.update(sym_stats)
        out.update(reg_stats)
        for name, frac in batch_frac.items():
            out[f'batch_frac_{name}'] = float(frac)
        for name, frac in buf_frac.items():
            out[f'buf_frac_{name}'] = float(frac)
        return out

    def save(self, path: Path | str) -> Path:
        self._ensure_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'config': asdict(self.cfg),
            'online': self.online.state_dict(),
            'target': self.target.state_dict(),
            'env_steps': self.env_steps,
            'grad_steps': self.grad_steps,
            'optim': self.optim.state_dict() if self.optim is not None else None,
        }
        self._torch.save(payload, path)
        return path

    def load(self, path: Path | str, *, map_location: Optional[str] = None) -> None:
        self._ensure_torch()
        torch = self._torch
        loc = map_location or self.cfg.device
        try:
            payload = torch.load(path, map_location=loc, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=loc)
        if 'config' in payload and isinstance(payload['config'], dict):
            # Checkpoints anteriores a ddqn_spec 1.19 no guardan las opciones de regularización:
            # se entrenaron sin ellas (0), no con los valores por defecto actuales.
            for k in ('weight_decay', 'dropout', 'foe_bench_dropout'):
                setattr(self.cfg, k, 0.0)
            self.cfg.q_head, self.cfg.q_rank = 'joint', 0
            for k, v in payload['config'].items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)
        self.online.dropout = self.target.dropout = float(self.cfg.dropout)
        if (bool(getattr(self.online, 'layer_norm', False)) != bool(self.cfg.layer_norm)
                or getattr(self.online, 'q_head', 'joint') != str(self.cfg.q_head)
                or int(getattr(self.online, 'q_rank', 0)) != int(self.cfg.q_rank)):
            # el checkpoint usa otra arquitectura (con/sin LayerNorm): reconstruir las redes
            self.online = None
            self._ensure_torch()
        self.online.load_state_dict(payload['online'])
        self.target.load_state_dict(payload.get('target', payload['online']))
        self.env_steps = int(payload.get('env_steps', 0))
        self.grad_steps = int(payload.get('grad_steps', 0))
        if payload.get('optim') and self.optim is not None:
            try:
                self.optim.load_state_dict(payload['optim'])
            except (ValueError, KeyError):
                pass  # checkpoint de otra arquitectura: se sigue con el optimizador nuevo

    @staticmethod
    def decode(joint_id: int) -> Tuple[int, int]:
        return decode_joint_action(joint_id)

    @staticmethod
    def encode(a1: int, a2: int) -> int:
        return encode_joint_action(a1, a2)
