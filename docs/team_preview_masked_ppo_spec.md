# Masked PPO — Team Preview Exclusive Spec (VGC Gen 9 Regulation M-C)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo — Masked PPO solo Team Preview** |
| **Versión** | **1.0** (alineado con `docs/rl_env_spec.md` **v1.11** §6.i) |
| **Estado** | **FROZEN — implementado en `showdown/rl/team_preview_ppo/`** |
| **Algoritmo** | PPO (Schulman et al., 2017) + IAM flat-360 (Huang & Ontañón, 2022) |
| **Rol** | Bring + leads (4 de 6 ordenados); paradigma RL estándar en preview |
| **Formato** | Gen 9 VGC **Regulation M-C**, Open Team Sheets |
| **Code root** | `showdown/rl/team_preview_ppo/` |
| **Env / Obs** | `showdown/rl/team_preview_env.py`, `team_preview_obs.py`, `team_preview_action.py` |
| **CLI** | `python -m showdown.rl.team_preview_ppo.train` |
| **Deps** | `torch >= 2.0`; combate frozen `greedy_joint` |

> **Alcance.** MDP de **solo Team Preview**. El PPO de combate
> ([`ppo_spec.md`](ppo_spec.md)) sigue con `/team 1234` fuera de su MDP.
>
> **Hermanos de capítulo:** Payoff Nash
> ([`team_preview_nash_spec.md`](team_preview_nash_spec.md) v1.0);
> QD-PGA preview ([`team_preview_qd_pga_spec.md`](team_preview_qd_pga_spec.md) v1.0).
> Baselines: Random / Greedy / Expectimax preview.
>
> **Inmutabilidad.** Bump **1.1+** aquí + `rl_env_spec` §6.i juntos.

---

## 0. Por qué existe

Aprende *qué formación sacar* (4 de 6 + leads), no cómo pelear el turno.
Tras elegir `/team …`, un piloto de combate **congelado** (`greedy_joint`) juega
el BO1 y la recompensa es Cand. 1 del resultado.

**Analogía:** el director elige el once inicial; el partido lo juega un entrenador
ya fijado.

---

## 1. MDP (bandit contextual, \(T=1\))

| Ítem | Contrato |
|------|----------|
| Decisión | 1 acción de preview por episodio |
| Episodio | Preview + **1 BO1** con \(\pi_c\) frozen |
| Max turns combate | 40 |
| Self-play preview | **Off** en v1.0 |
| BC warm-start | **Out** de v1.0 |
| \(\pi_c\) default | **`greedy_joint`** (combate \(A_1\times A_2\)) |
| Ablación reporte | `expectimax_bayesian` como \(\pi_c\) (opcional) |
| Prohibido | Actualizar pesos de \(\pi_c\) al entrenar preview |

---

## 2. Acción Flat-360 + IAM

\[
|A| = P(6,4) = 360,\quad a \in \{0,\ldots,359\}
\]

**Gym / logits:** 0-based. **Showdown:** 1-based `/team t1t2t3t4`.

### 2.1 Encode (tupla 0-based \(\to a\))

Sea \(S=\{0,1,2,3,4,5\}\), \((i_1,i_2,i_3,i_4)\in\mathcal{I}_4(S)\):

\[
a = 60\,c_1 + 12\,c_2 + 3\,c_3 + c_4
\]

\[
\begin{aligned}
c_1 &= i_1 \\
c_2 &= i_2 - \mathbf{1}[i_2>i_1] \\
c_3 &= i_3 - \mathbf{1}[i_3>i_1] - \mathbf{1}[i_3>i_2] \\
c_4 &= i_4 - \mathbf{1}[i_4>i_1] - \mathbf{1}[i_4>i_2] - \mathbf{1}[i_4>i_3]
\end{aligned}
\]

Ejemplo: \((0,1,2,3)\Rightarrow a=0\Rightarrow\) `/team 1234`.

### 2.2 Decode (\(a\to\) tupla 0-based)

Inversa del código de Lehmer / sistema factorial sobre el conjunto restante
ordenado (implementación: `decode_preview_action` en `team_preview_action.py`).
Tests de ida y vuelta obligatorios.

### 2.3 Showdown

\(t_k = i_k + 1\), comando `/team t1t2t3t4`.

### 2.4 IAM

\[
\tilde z_i = z_i + (1-M_i)(-10^9),\quad
\pi(a\mid s)=\mathrm{Softmax}(\tilde z)
\]

Con pool completo de 6, \(M=\mathbf{1}_{360}\) (IAM = paridad metodológica +
robustez si pool \(<6\)). Misma máscara en sample y épocas PPO.

**Capacidad:** PPO ve 360 órdenes; Greedy/Expectimax son indiferentes al orden
de banca (\(\approx 90\) brings×leads). Declarar en resultados.

---

## 3. Observación \(S\in\mathbb{R}^{512}\) (`float32`)

\[
\dim = 210_{P1} + 210_{P2} + 36_{M} + 56_{\mathrm{ctx}} = 512
\]

| Bloque | Dim | Contenido |
|--------|-----|-----------|
| P1 / P2 | \(6\times 35\) c/u | Ver §3.1 |
| Cobertura \(M_{ij}\) | 36 | **Única** copia; solo \(P_1\to P_2\) |
| Contexto | 56 | 9D clima/terreno (\(=\mathbf{0}\) en preview) + **47D** padding/flags (**sin** re-incluir \(M\)) |

### 3.1 Por mon (35D)

| Sub | Dim | Contenido |
|-----|-----|-----------|
| Stats | 6 | HP, Atk, Def, SpA, SpD, Spe — OTS o dex; \(/255\) |
| Tipos | 18 | Multi-hot Gen 9 (sin Stellar / sin Tera — Reg M-C) |
| Item/abi flags | 6 | Choice, Focus Sash, boost, immunidad, speed-control, misc |
| Move roles | 5 | Spread, Priority, Setup, Support, Pivot (desde moves OTS si hay) |

### 3.2 Cobertura

\[
M_{ij}=\log\big(d_{\max}(i\!\to\!j)/HP_j + \varepsilon\big),\quad
\varepsilon=10^{-6}
\]

- \(d_{\max}\): max daño nominal sobre moves OTS del atacante vs set OTS defensor
  (sin accuracy/crit). Preferir `@smogon/calc` **cacheado en `reset()`**;
  fallback heurística tipada local si Node no está.
- \(HP_j\): maxHP OTS L50 (no solo BST dex).
- **0 llamadas calc en `step()`.**

---

## 4. Recompensa (Cand. 1 estricto)

\[
R=\begin{cases}+1 & \text{win}\\ -1 & \text{loss}\\ 0 & \text{tie}\end{cases}
\]

Sin \(\Delta\)HP en \(R\) de preview v1.0.

---

## 5. Red

\[
512 \xrightarrow{\mathrm{ReLU}} 256 \xrightarrow{\mathrm{ReLU}} 256
\to \big(\underbrace{\mathbb{R}^{360}}_{\mathrm{logits}},\;
\underbrace{\mathbb{R}}_{\,V(s)}\big)
\]

---

## 6. Hiperparámetros

| Parámetro | Valor |
|-----------|--------|
| \(\gamma\) | `1.0` |
| \(\lambda\) (GAE) | `1.0` |
| \(\varepsilon_{\mathrm{clip}}\) | `0.2` |
| \(c_v\) | `0.5` |
| \(c_e\) | `0.02` |
| LR \(\eta\) | `3\mathrm{e}{-4}` (decay lineal permitido) |
| Grad clip | `0.5` |
| \(N_{\mathrm{prod}}\) | **2048** episodios BO1 / update |
| \(N_{\mathrm{smoke}}\) | **64** |
| Mini-batch \(B\) | **128** |
| Epochs \(K\) | **4** |
| Device | CPU default |
| Seed | CLI obligatorio |

\(c_e=0.02\): exploración en bandit 360-way (no Bo3).

---

## 7. Curriculum / eval

Rivales de **preview**: `random` → `greedy` (mismo preview Solo/Joint) →
`expectimax_bayesian`. Combate: \(\pi_c\) frozen ambos lados (o foe heuristic
preview + mismo \(\pi_c\)).

Suite reportada: **200 BO1** vs cada baseline preview; política eval = argmax masked;
declarar \(\pi_c\).

---

## 8. Non-goals

- Meter preview en Obs 1024D / ActionSpec de combate.
- Entrenar \(\pi_c\) junto al actor preview.
- Flat-90 / AR×4 como primario.
- Cand. 2 / \(\Delta\)HP en \(R\).
- BC (v1.1+).
- Nash / QD preview (specs hermanos).

---

## 9. Referencias

- Schulman et al. (2017). PPO. *arXiv:1707.06347*.
- Huang & Ontañón (2022). Invalid action masking. *FLAIRS*.
- [`rl_env_spec.md`](rl_env_spec.md) §4 Cand. 1, §6.i.
- [`ppo_spec.md`](ppo_spec.md) (combate; no forkar).
- Heuristics preview: [`random_spec`](random_spec.md) / [`greedy_solo_spec`](greedy_solo_spec.md) /
  [`expectimax_bayesian_spec`](expectimax_bayesian_spec.md) §3.

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.0** | 2026-10-11 | FROZEN: Flat-360 IAM, obs 512D, Cand.1, \(\pi_c\)=greedy_joint, Lehmer 0..359, calc solo reset, code under `showdown/rl/`. |

---

*Fin del contrato Team Preview Masked PPO v1.0.*
