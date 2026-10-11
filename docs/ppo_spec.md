# Masked PPO Agent Spec — Auto-regresivo Slot1→Slot2 (VGC Gen 9)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo de implementación — Masked PPO** |
| **Versión** | **1.1** (alineado con `docs/rl_env_spec.md` **v1.10** §6.a) |
| **Estado** | **FROZEN — implementado en `showdown/rl/ppo/` + `battle_bots/ppo`** |
| **Algoritmo** | Proximal Policy Optimization (Schulman et al., 2017) + IAM (Huang & Ontañón, 2022) |
| **Rol** | Agente RL de **gradiente de política** principal (complemento de DDQN joint) |
| **Code root** | `showdown/rl/ppo/` |
| **CLI** | `python -m showdown.rl.ppo.train` |
| **Deps** | `requirements-rl.txt` — `torch >= 2.0.0` |

> **Alcance.** Fuente de verdad para **Masked PPO**. Env compartido en
> [`rl_env_spec.md`](rl_env_spec.md). Sibling valor: [`ddqn_spec.md`](ddqn_spec.md).
>
> **Contraste con DDQN.** PPO: \(\pi(a_1\mid s)\,\pi(a_2\mid s,a_1)\) con máscaras
> por slot. DDQN: \(Q\) joint 576. Misma Obs/Reward/preview/foes ⇒ contraste válido.
>
> **Inmutabilidad.** Bump **1.2+** aquí + `rl_env_spec` §6.a juntos.

---

## 0. Por qué existe (lectura no técnica)

En lugar de puntuar *todas* las jugadas conjuntas (como DDQN), PPO aprende una
**política**: una distribución de probabilidad sobre “qué hace el Pokémon de la
izquierda” y, *condicionado a eso*, “qué hace el de la derecha”. Entrena jugando
rollouts recientes (on-policy) y actualiza con un recorte (clip) para no cambiar
la política de golpe — de ahí “proximal”.

La **máscara de acciones inválidas (IAM)** pone a cero la probabilidad de órdenes
ilegales *antes* de muestrear y *también* al recalcular \(\pi\) en las épocas PPO
(no basta con castigar ilegales en la recompensa).

**Analogía:** un director que primero elige la línea del primer músico y luego la
del segundo, ensayando con partituras recientes y sin saltar a un estilo totalmente
distinto de un ensayo a otro.

| Término | Significado aquí |
|---------|------------------|
| Política \(\pi\) | Probabilidad de acciones (no valores \(Q\)) |
| Auto-regresivo | Slot2 depende del Slot1 elegido |
| IAM | Softmax solo sobre legales |
| GAE | Estimador de ventaja temporal |
| BC | Imitación previa de Expectimax/Joint |
| Self-play pool | Rivales = checkpoints propios + best vs Joint |

---

## 1. Entorno heredado

| Contrato | Fuente | Uso PPO |
|----------|--------|---------|
| `VGCEnv` | §1 | Rollout `reset` / `step` |
| Obs 1024D | §3 | Torso actor–crítico |
| Reward Cand. 2 → Cand. 1 | §4 | Curriculum §8 |
| Slot ids 0–23 | §2 | Cabezas factorizadas |
| L2 `skip_buffer` | §2 | **Drop** del buffer |
| `trunc_reason` | §1 / §4 | turn_cap vs crash |
| No `@smogon/calc` en `step()` | §1 / §8 | Heurística local |

**Obs PPO:** belief 256D = ceros / priors planos. OTS solo para sync en harness.

---

## 2. Delimitación MDP

| Ítem | Contrato |
|------|----------|
| Episodio | BO1; max turns **40**; truncate ≠ Loss |
| Decisión | Par ordenado \((a_1,a_2)\) |
| Action space env | `MultiDiscrete([24, 24])` |
| Preview | Fuera del MDP: `/team 1234`; WR condicional |
| Self-play | **En alcance** tras bootstrap heurístico (§8) |
| Player | `ValidatedPPOPlayer` |

---

## 3. Espacio auto-regresivo y IAM

\[
\pi_\theta(a_1, a_2 \mid s)
= \pi_\theta^{(1)}(a_1 \mid s)\,\,
  \pi_\theta^{(2)}(a_2 \mid s, a_1)
\]

| Slot id | Significado |
|---------|-------------|
| 0–15 | move × target |
| 16–19 | hard switch |
| 20 | pass |
| 21–22 | mega / tera (si `valid_orders`) |
| 23 | reserved |

### Invalid Action Masking (IAM)

Fuente: `valid_orders` vía `action_masks(slot1_action_id=...)`.

1. Logits \(\mathbf{z}^{(1)}\), luego \(\mathbf{z}^{(2)}\) condicionados a \(a_1\).
2. Máscaras \(M_1(s),\; M_2(s\mid a_1)\).
3. Softmax legal:

\[
\mathrm{Softmax}\big(\mathbf{z} + (1-M)\cdot(-10^{9})\big)
\]

4. Misma máscara en muestreo interactivo **y** en épocas PPO (Huang & Ontañón).

Helpers: `apply_logit_mask` / `softmax_masked` en `action_space.py`.

| API | Contrato |
|-----|----------|
| `action_masks(...)` | Path de muestreo PPO |
| `legal_joint_mask` | Telemetría / paridad DDQN; **no** path PPO |
| Log-prob | \(\log\pi=\log\pi^{(1)}+\log\pi^{(2)}\) enmascarado |
| L1 | \(>99\%\) sin L2 |

---

## 4. Objetivo PPO (clip) + GAE

### Truncation (`info['trunc_reason']`)

| Evento | `terminated` | `truncated` | Buffer | Bootstrap \(V(s')\) |
|--------|----------------|-------------|---------|----------------------|
| Win/Loss/Tie | True | False | Store | No (\(d_t=1\)) |
| Turn-cap (40) | False | True | Store | **Sí** (\(d_t=0\)) |
| Crash / L3 | False | True | **Drop** | No |
| L2 fallback | — | — | **Drop** | No |

### GAE-\(\lambda\)

\[
\delta_t = r_t + \gamma\, V_\phi(s_{t+1})\,(1-d_t) - V_\phi(s_t),\quad
\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l\, \delta_{t+l}
\]

Normalizar ventajas por rollout. \(\hat{R}_t=\hat{A}_t+V_{\phi_{\mathrm{old}}}(s_t)\).

### Surrogate

\[
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)},\quad
a_t=(a_1,a_2)
\]

\[
L^{\mathrm{CLIP}}=\mathbb{E}_t\Big[\min\big(
r_t\hat{A}_t,\;
\mathrm{clip}(r_t,1-\varepsilon_{\mathrm{clip}},1+\varepsilon_{\mathrm{clip}})\hat{A}_t
\big)\Big]
\]

\[
L^V=\mathbb{E}_t[(V_\phi(s_t)-\hat{R}_t)^2]
\]

**Value clip:** **OFF**. Entropía \(H_t=H[\pi^{(1)}]+H[\pi^{(2)}(\cdot\mid a_{1,t})]\)
(condicionada al \(a_1\) del rollout).

\[
L=-L^{\mathrm{CLIP}}+c_v L^V-c_e\,\mathbb{E}_t[H_t]
\]

On-policy: limpiar buffer tras \(K\) épocas. Ratio con \(\pi\) **enmascarada**. Grad clip `max_norm=0.5`.

---

## 5. Red

| Ítem | Contrato |
|------|----------|
| Input | `float32[1024]` |
| Torso | `1024→512→512` ReLU → \(\mathrm{h}\) |
| Policy Slot1 | `Linear(512,24)` |
| Policy Slot2 | \([\mathrm{h};\,\mathrm{onehot}(a_1)]\rightarrow\) `Linear(536,24)` |
| Value | `Linear(512,1)` → \(V(s)\) (**no** \(Q\)) |
| Off v1.1 | LSTM, attention, dual actors, FiLM |
| Device | CPU default |

---

## 6. Hiperparámetros congelados

| Parámetro | Valor |
|-----------|--------|
| \(\gamma\) | `0.95` |
| \(\lambda\) (GAE) | `0.95` |
| \(\varepsilon_{\mathrm{clip}}\) | `0.2` |
| Optimizer | Adam |
| LR (scratch / post-BC warmup done) | `3e-4` |
| LR **post-BC fine-tune** | \(10^{-4}\) (full; **sin** freeze) |
| Rollout \(T\) | `2048` |
| Num envs | `8` |
| Minibatch | `256` |
| Epochs \(K\) | `4` |
| \(c_v\) | `0.5` |
| \(c_e\) | `0.01` |
| Grad clip | `0.5` |
| Max turns | `40` |
| Value clip | **Off** |
| Seed | CLI obligatorio |

Sin experience replay en el path PPO (on-policy). BC = etapa offline separada.

---

## 7. Módulos y CLI

| Módulo | Rol |
|--------|-----|
| `network.py` | Torso + AR + \(V\) |
| `buffer.py` | Rollout; reject skip / crash |
| `agent.py` | Sample / greedy masked |
| `train.py` | Collect → GAE → PPO → curriculum / self-play |
| `player.py` | `PPOPlayer` + `ValidatedPPOPlayer` |
| `bc.py` | Pretrain BC |
| Tests | `tests/test_rl_ppo.py` |

Flags: `--seed`, `--device`, `--opponent` (`…|curriculum|self`), `--phases`,
`--expand-omega`, `--reward candidate2|candidate1|anneal`, `--bc-checkpoint`,
`--start-ps`, `--checkpoint-dir`.

```bat
python -m showdown.rl.ppo.train --seed 0 --opponent mock --steps 20000
python -m showdown.rl.ppo.train --seed 0 --opponent curriculum --start-ps --expand-omega
python -m showdown.rl.ppo.train --seed 0 --opponent self --reward anneal --bc-checkpoint logs/ppo_bc.pt --start-ps
```

---

## 8. Curriculum

### Stage 0 — Behavioral Cloning (primario)

| Ítem | Contrato |
|------|----------|
| Loss | Cross-entropy enmascarada sobre \((a_1^*,a_2^*)\) expertos |
| Data | Interno `expectimax_bayesian` / `greedy_joint`; sin scrape Smogon ELO |
| Filter | Preferir `sync_ok` |
| Transfer | Full fine-tune; LR inicial on-policy \(10^{-4}\) |

### Stage 1 — Bootstrap heurístico (Cand. 2)

\(\Omega\) monótono (defaults como DDQN): random → solo → joint → expectimax opcional;
pesos `1/1/1/0.35`. Gate cada **10_000** steps: 100 BO1, avanzar si \(WR\ge 0.60\).

### Stage 2 — Self-play pool (anneal Cand. 1)

\[
\mathcal{P}=\{\text{últimos }K=10\text{ checkpoints}\}
\cup\{\texttt{best\_vs\_greedy\_joint}\}
\]

`best_vs_greedy_joint` **nunca** se evicta. Matchmaking v1.1: uniforme \(1/|\mathcal{P}|\).
PFSP completo: **fuera de alcance**. Reward: `--reward anneal`.

---

## 9. Suite offline

200 BO1 cada uno vs `random`, `greedy_joint`, `expectimax_bayesian`; política greedy masked; `/team 1234`. Reportar `illegal_action_fallbacks`.

---

## 10. Telemetría (no en \(R_t\))

`illegal_action_fallbacks`, `approx_kl` / `clip_fraction`, `entropy_slot*`,
`explained_variance`, `curriculum_phase` / `reward_mode`, `eval_wr_*`,
`bc_acc_legal`, `selfplay_pool_size`.

---

## 11. Tests de aceptación

1. Samples siempre legales bajo IAM.  
2. Log-prob = suma de slots; ratio finito.  
3. `skip_buffer` no crece buffer.  
4. Turn-cap aporta \(V(s')\) en GAE.  
5. Crash nunca entra al buffer.  
6. Si \(\theta=\theta_{\mathrm{old}}\), \(L^{\mathrm{CLIP}}=\mathbb{E}[\hat{A}]\) (salvo norm).  
7. Smoke ≥100 steps mock CPU.

---

## 12. Fortalezas y límites (tesis)

| Fortaleza | Límite |
|-----------|--------|
| Factorización natural 2v2 + IAM | On-policy: más sample-hungry que DDQN |
| BC + self-play en alcance | Pool uniforme (no PFSP) |
| Contraste metodológico claro vs joint-Q | Belief plano; preview fijo |

---

## 13. Non-goals

Joint-576 como política **primaria** PPO; PFSP completo; value clip; entrenar en L2/crash;
preview en MDP; BD en \(R_t\); reemplazar DDQN; actor recurrente/attention.

---

## 14. Nota de paridad PPO vs DDQN

La comparación empírica AR-Masked-PPO vs joint-DDQN es válida bajo contrato de env
compartido. Las topologías de acción son inherentes a cada algoritmo y deben
declararse en la tesis; no obligan a forzar cabezas idénticas.

---

## 15. Referencias

- Schulman et al. (2017). PPO. *arXiv:1707.06347*.  
- Schulman et al. (2015). GAE. *arXiv:1506.02438*.  
- Huang & Ontañón (2022). IAM.  
- [`rl_env_spec.md`](rl_env_spec.md); [`ddqn_spec.md`](ddqn_spec.md).

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.1** | 2026-10-10 | Redacción amena (ES); semántica FROZEN intacta. |
| **1.0** | 2026-10-10 | Contrato Masked PPO: AR IAM, GAE, BC, self-play uniforme. |

---

*Fin del contrato Masked PPO v1.1.*
