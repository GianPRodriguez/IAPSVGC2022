# QD-PPGA Agent Spec — Proximal Policy Gradient Arborescence (VGC Gen 9)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo — QD-PPGA** |
| **Versión** | **1.1** (alineado con `docs/rl_env_spec.md` **v1.10** §6.g) |
| **Estado** | **FROZEN — implementado en `showdown/rl/qd_ppga/`** |
| **Algoritmo** | Grid MAP-Elites + emitters **solo proximales** (Masked PPO) + \(V_\phi\) compartido + selección de padres por **arborescencia** |
| **Rol** | Camino QD de política (PPGA); iluminar BD sin mutación genética |
| **Code root** | `showdown/rl/qd_ppga/` |
| **CLI** | `python -m showdown.rl.qd_ppga.train` |
| **Deps** | `torch>=2.0`; reusa `ppo/`, `qd_ppo/`, `qd_pga/` (crítico/replay/PG) |

> **Contraste.** vs QD-PPO: actor-only + \(V_\phi\) + arborescencia (no full AC uniforme).
> vs QD-PGA: **\(p_{\mathrm{mut}}=0\)** (sin mutación) + padres arborescentes.
> Non-goal: DQD continuo / measure grads / xNES (arborescencia Batra adaptada a grid discreto).
>
> **Inmutabilidad.** Bump **1.2+** + `rl_env_spec` §6.g juntos.

---

## 0. Por qué existe (lectura no técnica)

QD-PGA a veces “tuerce” pesos al azar. QD-PPGA dice: **toda** variación debe ser
un paso proximal de PPO (seguro, con clip), y además elige *a quién clonar* con
criterio de **arborescencia**: la mitad del tiempo un elite al azar; la otra mitad,
el elite más cercano (en BD) a una casilla “frontera” vacía o débil — así se
empuja a llenar el mapa y a mejorar las salas flojas.

**Analogía:** solo se aceptan retoques de taller con reglas (PPO), y el maestro
elige el alumno según qué sala del museo está vacía o mal cubierta.

| Término | Significado aquí |
|---------|------------------|
| Proximal-only | \(p_{\mathrm{mut}}=0\); sin ruido genético |
| Arborescencia \(\alpha\) | Mezcla uniforme vs frontera |
| \(c^\star\) | Celda objetivo (vacía o min \(f\)) |
| Parent \(L_1\) | Elite filled más cercano en BD a \(c^\star\) |

---

## 0b. OPEN → FROZEN

| ID | Decisión |
|----|----------|
| **R1** | Grid \(10\times10\); actor-only; shared \(V_\phi\) |
| **R2–R3** | \(p_{\mathrm{mut}}=0\); variación = PPO-clip + IAM; Cand. 2; mosaic |
| **R4–R5** | Crítico TD(0) Huber; replay 200k; GAE actor; norm por emit |
| **R6–R7** | Arborescencia \(\alpha=0.5\); \(c^\star\) empty else min \(f\); parent min \(L_1\) |
| **R8–R12** | \(T_{\mathrm{emit}}=2048\), \(K=4\), \(n_{\mathrm{emit}}=4\), \(N_{\mathrm{outer}}=109\); fitness/mosaic/bootstrap como hermanos; \(T=10^6\) |

Defaults de paridad: \(c_e=0.01\), \(\varepsilon=0.2\), \(c_v=0.5\); crítico \(G_c=8\times256\);
suite 200 BO1; ≤4 aux PG desde Stage-1; archivo vacío → seed actor.

---

## 1–3. Entorno, archivo, crítico/replay

Igual que QD-PGA: `VGCEnv`, Obs, AR+IAM, Cand. 2/1, BDSpec, `/team 1234`;
payload actor-only; \(V_\phi\) TD(0) Huber + replay 200k (escribe PG/bootstrap; drop L2/crash).

---

## 4. Variación proximal (única)

```
parent ← sample_arborescence(A; α=0.5)   # o seed si vacío
π' ← deepcopy(actor(parent))
π' ← PPO_improve(π'; T_emit=2048, K=4, R=Cand2, mosaic, V_φ, GAE)
(f, b, g) ← evaluate Cand1 (G=20)
if g ≥ 10: try_insert(A, π', f, b)
```

Sin rama de mutación; sin burn-rollout mut.

---

## 5. Selección de padres (arborescencia)

| Caso | Acción |
|------|--------|
| Archivo vacío | Seed actor de bootstrap |
| \(U\sim\mathrm{Unif}<\alpha\) (\(\alpha=0.5\)) | Uniforme sobre filled |
| Else | \(c^\star\): uniforme entre **vacías** si hay; si no, celda con **min** \(f\). Parent = filled que minimiza \(L_1\) (BD o centros de bin) a \(c^\star\) |

---

## 6. Mosaic / fitness / suite

Idénticos a QD-PPO / QD-PGA (25/25/50; \(G=20\); \(g_{\min}=10\); 200 BO1).

---

## 7. Bucle externo y presupuesto

```
bootstrap ≤ B_boot Cand2
seed ≤ 4 aux PG emits
N_outer ← 109
for each emit: arborescence → PG T_emit → critic → PPO-K=4 → eval → insert?
deploy π* = argmax f
```

| Parámetro | Valor |
|-----------|--------|
| \(T_{\mathrm{total}}\) | `1_000_000` |
| \(B_{\mathrm{boot}}\) | `100_000` |
| \(T_{\mathrm{emit}}\) | `2048` |
| \(K_{\mathrm{pg}}\) | `4` |
| \(n_{\mathrm{emit}}\) | `4` |
| \(N_{\mathrm{outer}}\) | `109` |
| \(\alpha\) | `0.5` |
| \(p_{\mathrm{mut}}\) | `0` |
| Workers | `1` |

---

## 8. Módulos y CLI

`arborescence.py`; `improve.py` reexporta `qd_pga.improve.pg_improve`;
crítico/replay/archive reexport QD-PGA; mosaic/eval reexport QD-PPO;
`train.py`, `ValidatedQDPPGAPlayer`; tests `tests/test_rl_qd_ppga.py`.

```bat
python -m showdown.rl.qd_ppga.train --seed 0 --smoke
```

---

## 9. Aceptación / non-goals

\(p_{\mathrm{mut}}=0\) (sin path mutate); \(N_{\mathrm{outer}}=109\); arborescencia con \(\alpha=0\) siempre frontera;
actor-only; IAM; \(g<10\) reject; smoke.

Non-goals: mutación genética (PGA); DQD continuo / Jacobianos / xNES; crítico por elite;
CVT; ME-ES primario; expand-\(\Omega\)/self-play como diversidad primaria; BD en \(R_t\).

---

## 10. Fortalezas y límites (tesis)

| Fortaleza | Límite |
|-----------|--------|
| Variación estable (solo PPO-clip) | Menos exploración “loca” que mutación |
| Arborescencia empuja cobertura | Depende de calidad del BD |
| Paridad de presupuesto con QD-PPO (\(N=109\)) | Sin híbrido mut de PGA |

---

## 11. Referencias

Batra et al. (2024) PPGA *ICLR*; Nilsson & Cully (2021); Schulman et al. (2017);
Huang & Ontañón (2022); [`ppo_spec.md`](ppo_spec.md), [`qd_pga_map_elites_spec.md`](qd_pga_map_elites_spec.md),
[`qd_ppo_spec.md`](qd_ppo_spec.md).

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.1** | 2026-10-10 | Redacción amena (ES); semántica intacta. |
| **1.0** | 2026-10-10 | Frozen: \(p_{\mathrm{mut}}=0\); arborescencia \(\alpha=0.5\); \(N_{\mathrm{outer}}=109\). |

---

*Fin del contrato QD-PPGA v1.1.*
