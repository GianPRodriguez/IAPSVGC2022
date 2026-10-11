# DDQN Agent Spec — Double Deep Q-Network joint (VGC Gen 9)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo de implementación — DDQN** |
| **Versión** | **1.1** (alineado con `docs/rl_env_spec.md` **v1.10** §6.b) |
| **Estado** | **FROZEN — implementado en `showdown/rl/ddqn/` + `battle_bots/ddqn`** |
| **Algoritmo** | Double DQN (van Hasselt et al., 2016) + máscara de acciones ilegales |
| **Rol** | Agente RL **basado en valor** principal (complemento de Masked PPO) |
| **Code root** | `showdown/rl/ddqn/` |
| **CLI** | `python -m showdown.rl.ddqn.train` |
| **Deps** | `requirements-rl.txt` — `torch >= 2.0.0` (heuristics sin este extra) |

> **Alcance.** Fuente de verdad para **implementar DDQN**. Obs 1024D, Reward Cand. 2,
> `VGCEnv` L1/L2/L3 y ActionSpec viven en [`rl_env_spec.md`](rl_env_spec.md) —
> **no** redefinir fórmulas aquí.
>
> **Contraste con PPO.** DDQN aprende \(Q(s,(a_1,a_2))\) sobre **576** acciones joint.
> PPO factoriza \(\pi(a_1\mid s)\,\pi(a_2\mid s,a_1)\). Ver [`ppo_spec.md`](ppo_spec.md).
> Misma Obs / Reward / preview / foes ⇒ comparación de tesis válida.
>
> **Inmutabilidad.** Bump **1.2+** aquí + `rl_env_spec` §6.b en el mismo cambio.

---

## 0. Por qué existe (lectura no técnica)

Imagina que, para cada posición de batalla, quieres una **tabla de “qué tan buena
es cada jugada conjunta”**. DDQN aproxima esa tabla con una red: mira el estado
(vector de 1024 números) y asigna un valor \(Q\) a cada par de órdenes posibles
de tus dos Pokémon. Luego elige (casi siempre) la jugada legal de mayor \(Q\),
y a veces explora al azar entre las legales (\(\varepsilon\)-greedy).

**Double** significa: una red propone “cuál es la mejor acción siguiente” y
**otra** (la target, más lenta) dice “cuánto vale esa acción”. Así se reduce el
sesgo de siempre mirar el máximo con la misma red.

**Analogía:** un crítico que puntúa cada *acorde* posible (acción izq × der) y
aprende de partidas guardadas en una memoria (replay), no solo de la última.

| Término | Significado aquí |
|---------|------------------|
| \(Q(s,a)\) | Estimación de “cuán bueno es hacer \(a\) en \(s\)” |
| Joint 576 | \(a=24a_1+a_2\); un id por par de slots |
| Máscara | Pone \(Q=-\infty\) en ilegales antes de elegir |
| Replay | Buffer de transiciones pasadas para aprender off-policy |
| Curriculum \(\Omega\) | Conjunto de rivales que crece (random → … → expectimax) |

---

## 1. Entorno heredado (no reimplementar)

| Contrato | Fuente | Uso DDQN |
|----------|--------|----------|
| `VGCEnv` Gymnasium | `rl_env_spec` §1 | Collector `reset` / `step` |
| Obs \(s \in \mathbb{R}^{1024}\), `float32` | §3 | Entrada de la red |
| Reward Candidate 2 + clip \([-1.30,+1.40]\) | §4 | \(r_t\) en replay |
| Slot ids 0–23 | §2 | Factores del id joint |
| L2 `skip_buffer`, L3 `truncated` | §2 | **Nunca** entran al replay |
| No `@smogon/calc` en `step()` | §1 / §8 | Solo heurística local en obs |

**Regla de obs DDQN:** el segmento belief 256D es **ceros / priors planos**
(sin SP Expectimax). OTS sí se acepta en harness por sync; no refina el encoder.

---

## 2. Delimitación MDP

| Ítem | Contrato |
|------|----------|
| Episodio | 1 partida (BO1). Max turns = **40**. Truncate ≠ Loss. |
| Decisión | Un turno Showdown → una acción joint \((a_1,a_2)\). |
| Team Preview | **Fuera del MDP.** `/team 1234` train = eval. WR **condicional** a ese preview. |
| Self-play | **Prohibido** en esta versión. |
| Player | `ValidatedDDQNPlayer` (sync + `accept_open_team_sheet=True`) como player A. |

---

## 3. Espacio joint y máscara

### Codificación

\[
a = 24\cdot a_1 + a_2 \in \{0,\ldots,575\}, \quad a_1,a_2 \in \{0,\ldots,23\}
\]

| Slot id | Significado |
|---------|-------------|
| 0–15 | move \(i\) × target \(t\) |
| 16–19 | hard switch banca |
| 20 | pass |
| 21 | mega (compact; decode `pe_adapter`) |
| 22 | tera (compact; decode `pe_adapter`) |
| 23 | reserved |

**Mega / Tera:** ids 21–22 solo si `valid_orders` los permite en \(s_t\).

### Máscara

\[
M_{\mathrm{joint}}(a_1,a_2) = M_1(a_1)\cdot M_2(a_2\mid a_1) \in \{0,1\}^{576}
\]

| API | Contrato |
|-----|----------|
| `legal_joint_mask(battle) -> (576,)` | Canónico; fuente = **`valid_orders`** |
| `action_masks(slot1_action_id=None)` | \(M_2\) condicionada si hay slot1 |
| Illegal Q | \(Q \leftarrow Q + (1-M)\cdot(-10^{9})\) antes de \(\arg\max\) / \(\varepsilon\)-greedy |
| Sampling | Nunca índice con \(M=0\). Warmup y \(\varepsilon\)-greedy = **uniforme sobre legales** |

Cobertura L1 objetivo: \(>99\%\) de acciones aceptadas sin L2.

---

## 4. Update Double DQN

Online \(\theta\), target \(\theta^-\), legales \(L(s')=\{a:M(s',a)=1\}\):

\[
y =
\begin{cases}
r_t & \text{si }\texttt{terminated} \\
r_t + \gamma\, Q_{\theta^-}\big(s',\, \arg\max_{a \in L(s')} Q_{\theta}(s', a)\big) & \text{si no}
\end{cases}
\]

\[
L(\theta) = \mathrm{Huber}\big(Q_\theta(s_t, a_t),\; y;\; \delta=1.0\big)
\]

| Regla | Contrato |
|-------|----------|
| Double | \(a^*\) de **online**; evalúa con **target**. Prohibido \(\max Q_{\theta^-}\). |
| Soft sync | Cada grad step: \(\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-\), \(\tau=0.005\). |
| Drop | `skip_buffer` o `truncated` → **no** insert, **no** bootstrap. |
| Grad clip | `max_norm = 10` |

---

## 5. Red

| Ítem | Contrato |
|------|----------|
| Input | `float32[1024]` |
| Body | `Linear(1024,512) → ReLU → Linear(512,512) → ReLU → Linear(512,576)` |
| Output | Q de todas las joint (máscara fuera de la net) |
| Off | Dueling, NoisyNets, PER, n-step, Rainbow, C51, QR-DQN |
| Device | **CPU** por defecto; GPU opcional por CLI |

---

## 6. Hiperparámetros congelados

| Parámetro | Valor |
|-----------|--------|
| \(\gamma\) | `0.95` |
| Optimizer | Adam |
| Learning rate | `3e-4` |
| \(\tau\) (soft) | `0.005` / grad step |
| Batch size | `256` |
| Replay capacity | `200_000` |
| Warmup env steps | `5_000` (legal-uniform) |
| Train frequency | 1 grad / 4 env steps |
| Loss | Huber \(\delta=1.0\) |
| Grad clip | `max_norm=10` |
| \(\varepsilon\) | `1.0 → 0.05` linear en primeros `500_000` env steps |
| Eval / gate \(\varepsilon\) | `0` |
| Replay tuple | `(s_t, a_joint, r_t, s', terminated, m_next)` |
| Seed | CLI obligatorio; logueado |

---

## 7. Módulos y CLI

| Módulo | Responsabilidad |
|--------|-----------------|
| `network.py` | Online + target MLP; soft update |
| `replay.py` | Capacidad 200k; rechazo skip/truncated |
| `agent.py` | \(\varepsilon\)-greedy / greedy enmascarado; codec joint |
| `train.py` | Collector, curriculum, checkpoint, logging |
| `player.py` | `DDQNPlayer` + `ValidatedDDQNPlayer` |
| Tests | `tests/test_rl_ddqn.py` |

| Flag | Significado |
|------|-------------|
| `--seed` | Obligatorio |
| `--device` | Default `cpu` |
| `--opponent` | `mock` \| `random` \| `greedy_solo` \| `greedy_joint` \| `expectimax_bayesian` \| `curriculum` |
| `--steps` / `--phases` | Pasos o unlocks `opp:steps,...` |
| `--expand-omega` / `--no-expand-omega` | \(\Omega\) acumulativo (default on) vs fases exclusivas |
| `--opponent-weights` | p.ej. `expectimax_bayesian:0.35` |
| `--no-expectimax` | Omite Expectimax del curriculum |
| `--curriculum-phase-steps` | Default `500000` |
| `--start-ps` | PS local (`--no-security`) |
| `--checkpoint-dir` | Default `logs/ddqn_checkpoints` |

```bat
python -m showdown.rl.ddqn.train --seed 0 --opponent mock --steps 20000
python -m showdown.rl.ddqn.train --seed 0 --opponent curriculum --start-ps
python -m showdown.rl.ddqn.train --seed 0 --start-ps --expand-omega --phases random:100k,greedy_solo:200k,greedy_joint:300k,expectimax_bayesian:50k
```

---

## 8. Curriculum ( \(\Omega\) monótono)

**Default (`--expand-omega`):** el set de rivales crece; el replay **no** se limpia.

| Fase | Unlock | \(\Omega_t\) | Advance |
|------|--------|--------------|---------|
| 0 Warmup | — | legal-uniform | 5k antes de aprender |
| 1 | `random` | `{random}` | `500_000` steps |
| 2 | `greedy_solo` | `{random, greedy_solo}` | igual |
| 3 | `greedy_joint` | `{random, solo, joint}` | igual |
| 4 | `expectimax_bayesian` | \(\Omega_3\cup\{\mathrm{expectimax}\}\) | igual (`--no-expectimax` omite) |

Pesos de muestreo: `random=solo=joint=1.0`, `expectimax_bayesian=0.35`.
**Checkpoint best:** WR vs **`greedy_joint`** (\(\varepsilon=0\)).

---

## 9. Suite offline (WR reportado)

| Rival | Juegos | Modo | \(\varepsilon\) | Preview |
|-------|--------|------|-----------------|---------|
| `random` | 200 | BO1 | 0 | `/team 1234` |
| `greedy_joint` | 200 | BO1 | 0 | `/team 1234` |
| `expectimax_bayesian` | 200 | BO1 | 0 | `/team 1234` |

---

## 10. Telemetría (no entra en \(R_t\))

`illegal_action_fallbacks`, `mean_game_length`, `mean_q` / `mean_target` / `td_error`,
`eps`, `curriculum_phase`, `buf_frac_*` / `batch_frac_*`, `eval_wr_<opponent>`.

---

## 11. Tests de aceptación

1. Double: \(a^*\) solo de online.  
2. Replay: `skip_buffer=True` no aumenta `len(replay)`.  
3. Codec joint idempotente en \(\{0,\ldots,575\}\).  
4. Máscara: con \(\varepsilon=0\), argmax siempre \(M=1\).  
5. Smoke ≥100 steps en `MockBattleBackend` (CPU).

---

## 12. Fortalezas y límites (tesis)

| Fortaleza | Límite |
|-----------|--------|
| Coordina ambos slots en un solo \(Q\) | Espacio 576 grande; sin modelo explícito de \(\pi\) |
| Off-policy + replay eficiente | Sin self-play en v1.1 |
| Fácil de contrastar con PPO | Belief plano; preview fijo |

---

## 13. Non-goals

PER, n-step, Rainbow, C51, QR-DQN, Dueling, NoisyNets; preview en MDP;
entrenar en L2/truncated; \(Q\) como \(\Phi(s)\); self-play; IQL/BDQN como agente principal reportado; BC obligatorio.

---

## 14. Referencias

- Huang & Ontañón (2022). Invalid action masking. *FLAIRS*.  
- Mnih et al. (2015). Deep RL. *Nature*.  
- van Hasselt, Guez & Silver (2016). Double Q-learning. *AAAI*.  
- [`rl_env_spec.md`](rl_env_spec.md); [`ppo_spec.md`](ppo_spec.md); heuristics §6.h.

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.1** | 2026-10-10 | Redacción amena (ES) + glosario/analogía; **sin** cambio de semántica FROZEN. |
| **1.0** | 2026-10-10 | Contrato DDQN + curriculum \(\Omega\) monótono. |

---

*Fin del contrato DDQN v1.1.*
