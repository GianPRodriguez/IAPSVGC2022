# Deep Payoff Nash — Team Preview Exclusive Spec (VGC Gen 9 Regulation M-C)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo — Value Net + Nash solo Team Preview** |
| **Versión** | **1.0** (alineado con `docs/rl_env_spec.md` **v1.12** §6.j) |
| **Estado** | **FROZEN — implementado en `showdown/rl/team_preview_nash/`** |
| **Algoritmo** | Low-rank payoff \(V_\theta\) + Nash LP (HiGHS); FP fallback |
| **Rol** | Bring + leads via equilibrio mixto sobre \(\hat M\in[0,1]^{360\times360}\) |
| **Formato** | Gen 9 VGC **Regulation M-C**, Open Team Sheets |
| **Code root** | `showdown/rl/team_preview_nash/` |
| **Env / Obs / Action** | Reuso estricto: `team_preview_env.py`, `team_preview_obs.py`, `team_preview_action.py` |
| **CLI** | `python -m showdown.rl.team_preview_nash.train --seed N` |
| **Deps** | `torch >= 2.0`; `scipy` (HiGHS); combate frozen `greedy_joint` |

> **Alcance.** Solo Team Preview. Ortogonal a combate (p. ej. Nash-preview + MCTS-PUCT).
>
> **Hermanos:** [`team_preview_masked_ppo_spec.md`](team_preview_masked_ppo_spec.md);
> [`team_preview_qd_pga_spec.md`](team_preview_qd_pga_spec.md).
> Baselines: Random / Greedy / Expectimax preview.
>
> **Inmutabilidad.** Bump **1.1+** aquí + `rl_env_spec` §6.j juntos.

---

## 0. Por qué existe

Estima una matriz de pagos esperados sobre el espacio Flat-360 y resuelve el
equilibrio de Nash en mixtas. No aprende una política por PPO: la política de
preview **es** \(p^\star\) del meta-juego \(360\times360\).

**Analogía:** en vez de memorizar “siempre saco este once”, construye la tabla
de matchups bring/lead y juega el equilibrio (impredecible si hay soporte).

---

## 1. Formalismo

Juego matricial **simultáneo**, info perfecta de plantillas OTS, imperfecta
sobre la **acción** del rival. Un paso.

\[
|A|=P(6,4)=360,\quad a\in\{0,\ldots,359\}
\]

Misma biyección Lehmer que Masked PPO preview (`team_preview_action.py`).
Showdown: `/team t1t2t3t4` (4 dígitos).

### 1.1 Pagos y escala Cand. 1

\[
\hat M_{ij}\approx\mathbb{P}(\text{win}\mid a_1=i,a_2=j)\in[0,1]
\]

\[
A = 2\hat M - 1 \in [-1,1]^{360\times360}
\]

\(A\) es **suma constante** (escala Cand. 1), **no** antisimétrica
(\(A\neq -A^\top\) en general).

Labels de train: \(r\in\{0,\,0.5,\,1\}\) (loss / tie / win) ↔ Cand. 1
\(\{-1,0,+1\}\) vía \(r=(R+1)/2\).

### 1.2 Nash

\[
\max_{p\in\Delta^{359}}\min_{q\in\Delta^{359}} p^\top A q
\]

Valor \(v^\star=p^{\star\top} A q^\star\): \(>0\) ventaja estructural P1;
\(<0\) desventaja; \(\approx 0\) empate teórico de preview.

---

## 2. Reúso inmutable

| Pieza | Contrato |
|-------|----------|
| Acción | `team_preview_action.py` — Lehmer 0..359 |
| Obs | `team_preview_obs.py` — \(512=210_{P1}+210_{P2}+36_M+56_{\mathrm{ctx}}\) |
| Env | `team_preview_env.py` — bandit \(T=1\), \(\pi_c\) via `combat_fn` o mock |
| Prohibido | Encoder/action codec paralelo; `src/agents/` |

Clima/terreno en \(56_{\mathrm{ctx}}\): **0** en preview. Sin Spe-order implícito
fuera del contrato obs.

---

## 3. Value Net (low-rank)

Trunk **MLP** compartido sobre \(x\in\mathbb{R}^{512}\) (sin convoluciones).

\[
e_a\in\mathbb{R}^{d_a},\quad d_a=32,\quad d_{\mathrm{lat}}=64
\]

\[
\Phi_{\mathrm{row}}(h,e_i),\;\Phi_{\mathrm{col}}(h,e_j)\in\mathbb{R}^{64}
\]

\[
\hat M_{ij}=\sigma\big(\langle\Phi_{\mathrm{row}},\Phi_{\mathrm{col}}\rangle\big)
\]

Inferencia: precomputar 360 embeddings fila y 360 columna → un matmul
\(360\times360\) (no 129 600 forwards).

---

## 4. Nash solver

| Prioridad | Método |
|-----------|--------|
| **Primario v1.0** | LP `scipy.optimize.linprog` (`highs`) sobre \(A\) |
| Fallback / ablación | Fictitious Play (`mcts_puct_nash.nash.fictitious_play`) |
| Fallback absoluto | Softmax sobre media de filas \(\bar A_{i,\cdot}\) |

Timeout solver: ~100–200 ms → cascada a FP → softmax. Tests usan tolerancia
\(p_k<10^{-6}\) para filas estrictamente dominadas (no idéntico 0).

---

## 5. Train (value / payoff updates)

No es policy gradient. Actualiza solo \(V_\theta\).

| Ítem | Contrato |
|------|----------|
| \(\pi_c\) | **`greedy_joint`** frozen (mock proxy en CI) |
| Pares \((i,j)\) v1.0 | **Uniforme** independiente |
| On-policy Nash | **Out** de v1.0 (\(i\sim p^\star,\,j\sim q^\star\) → v1.1+) |
| Self-play preview | **Off** |
| Loss | BCE o MSE: \(\hat M_{ij}\) vs \(r\in\{0,0.5,1\}\) |
| \(N_{\mathrm{prod}}\) | **2048** pares / update |
| \(N_{\mathrm{smoke}}\) | **64** |
| Mini-batch \(B\) | **128** |
| Epochs \(K\) | **4** |
| LR | \(3\mathrm{e}{-4}\) |
| Grad clip | `0.5` |
| Device | CPU default |
| Seed | CLI **obligatorio** |

---

## 6. Inferencia E2E

1. `encode` obs 512D (env / ots).
2. \(V_\theta\) → \(\hat M\) → \(A=2\hat M-1\).
3. NashSolver → \(p^\star,q^\star,v^\star\).
4. **In-game:** \(a\sim p^\star\) (\(\tau=1\); **sin** reescalar temperatura).
5. **Eval / tesis:** \(a=\arg\max_i p_i^\star\) (max masa).
6. `/team` vía `action_to_team_string`.

Métricas: \(v^\star\), \(H(p^\star)\), \(|\{i:p_i>10^{-4}\}|\).

---

## 7. Eval

Suite reportada: **200 BO1** vs baselines preview (random / greedy /
expectimax). Smoke: **64** episodios (enrutamiento + solver), no bloqueante
de latencia en CI. Ideal E2E &lt;250 ms (indicador).

---

## 8. Non-goals

- `src/agents/` u obs/action paralelos.
- Salida densa \(360\times360\) como primario.
- Temperatura sobre \(p^\star\) en eval.
- 130k BO1 en vivo por decisión de preview.
- Entrenar \(\pi_c\).
- Equipar uniforme ≡ Nash-on-policy.
- Llamar \(A\) “antisimétrica”.

---

## 9. Referencias

- von Neumann & Morgenstern (1944). *Theory of games and economic behavior*.
- McMahan, Gordon & Blum (2003). Planning in the presence of cost functions
  controlled by an adversary. *ICML*.
- Lanctot et al. (2017). A unified game-theoretic approach to multiagent RL.
  *NeurIPS*.
- [`rl_env_spec.md`](rl_env_spec.md) §4 Cand. 1, §6.j.
- [`team_preview_masked_ppo_spec.md`](team_preview_masked_ppo_spec.md) (hermano).

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.0** | 2026-10-11 | FROZEN: low-rank \(d_a{=}32\), \(d_{\mathrm{lat}}{=}64\), LP HiGHS, uniforme \((i,j)\), \(r\in\{0,0.5,1\}\), \(A{=}2M{-}1\), reuso Flat-360/512D. |

---

*Fin del contrato Team Preview Deep Payoff Nash v1.0.*
