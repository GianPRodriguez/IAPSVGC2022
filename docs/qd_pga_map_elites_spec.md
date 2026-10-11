# QD-PGA-MAP-Elites Agent Spec — Policy Gradient Assisted MAP-Elites (VGC Gen 9)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo — QD-PGA-MAP-Elites** |
| **Versión** | **1.1** (alineado con `docs/rl_env_spec.md` **v1.10** §6.f) |
| **Estado** | **FROZEN — implementado en `showdown/rl/qd_pga/`** |
| **Algoritmo** | Grid MAP-Elites + variación híbrida: mutación Gaussiana **o** PPO-clip corto (PGA) con crítico **compartido** \(V_\phi\) |
| **Rol** | Camino QD de política (variante PGA); iluminar BD; elites actor-only |
| **Code root** | `showdown/rl/qd_pga/` |
| **CLI** | `python -m showdown.rl.qd_pga.train` |
| **Deps** | `torch>=2.0`; reusa `ppo/`, helpers de `qd_ppo/` |

> **Contraste con QD-PPO.** Elites full AC + solo PPO vs **actor-only** + \(V_\phi\)
> compartido + híbrido mut/PG.
>
> **Contraste con QD-PPGA.** Aquí \(p_{\mathrm{mut}}=0.5\); PPGA fuerza \(p_{\mathrm{mut}}=0\) + arborescencia.
>
> **Inmutabilidad.** Bump **1.2+** + `rl_env_spec` §6.f juntos.

---

## 0. Por qué existe (lectura no técnica)

Igual que QD-PPO, se quiere un **museo de estilos**. La diferencia es *cómo se
proponen nuevos cuadros*:

1. **Mutación:** se tiembla un poco los pesos del actor (ruido Gaussiano) y se
   “quema” un rollout para medir el estilo — barato, exploratorio.
2. **PGA:** un mini-entrenamiento PPO usando un crítico **compartido** por todo
   el archivo (como un profesor de valoración común) — más dirigido a subir
   calidad.

La moneda \(p_{\mathrm{mut}}=0.5\) elige el camino. Los elites guardan **solo el actor**
(menos memoria que un crítico por celda).

**Analogía:** taller con dos herramientas — tijeras aleatorias (mutación) y
cincel guiado por un crítico maestro compartido (PGA).

| Término | Significado aquí |
|---------|------------------|
| Actor-only elite | Se serializa política; no `value.*` por celda |
| \(V_\phi\) compartido | Un crítico para todos los emits |
| Mutación \(\sigma\) | Ruido isótropo en floats del actor |
| Burn-rollout | Pasos Cand. 2 sin escribir replay |
| PGA emit | Collect + PPO-K + updates de crítico |

---

## 0b. OPEN → FROZEN

| ID | Decisión |
|----|----------|
| **P1** | Grid \(10\times 10\) (mismos bins que QD-PPO) |
| **P2** | Actor-only + shared \(V_\phi\) |
| **P3–P5** | \(p_{\mathrm{mut}}=0.5\); \(\sigma=0.02\); mut **no** escribe replay |
| **P6–P9** | \(T_{\mathrm{emit}}=1024\), \(K_{\mathrm{pg}}=2\); replay 200k; crítico TD(0) Huber; GAE actor |
| **P10–P13** | Fitness/BD como QD-PPO; mosaic; bootstrap 100k; \(T=10^6\) env steps |

| Operativo | Contrato |
|-----------|----------|
| \(n_{\mathrm{emit}}\) | `4` |
| \(N_{\mathrm{outer}}\) | \(\lfloor 900000/4096\rfloor=219\) |
| Cada emit | Cuesta \(T_{\mathrm{emit}}\) env steps |
| IAM | Obligatorio en sample + recompute |

---

## 1. Entorno y taxonomía \(R\)/\(U\)

Igual que QD-PPO: Cand. 2 en PG/bootstrap/mut burn; Cand. 1 fitness; BD fuera de \(R_t\)/\(U\); preview `/team 1234`.

---

## 2. Archivo (actor-only)

Payload: keys de actor (excluir `value.*`) + \(f\) + BD. Insert/replace/Coverage/QD-score como QD-PPO. Padres: uniforme sobre filled.

---

## 3. Crítico compartido \(V_\phi\)

| Ítem | Contrato |
|------|----------|
| Net | MLP \(1024\to512\to512\to1\) |
| Target | \(y=r+\gamma V(s')\) stop-grad; \(\gamma=0.95\) |
| Loss | Huber; opcional \(c_v=0.5\) |
| Data | Replay \(200\,000\) |
| LR | \(3\times10^{-4}\) |
| Contador | Gradientes de crítico **no** incrementan \(T\) |

---

## 4. Replay

Capacidad 200k; tupla \((s,r,s',\mathrm{terminated})\); escribe **solo** rama PG + bootstrap;
drop L2/crash; turn-cap store con bootstrap \(V(s')\).

---

## 5. Variación híbrida

Con prob. \(p_{\mathrm{mut}}=0.5\):

1. \(\theta'\leftarrow\theta+\mathcal{N}(0,\sigma^2 I)\), \(\sigma=0.02\).
2. Burn-rollout \(T_{\mathrm{emit}}\) Cand. 2; **sin** replay / **sin** PG.
3. Eval Cand. 1; try insert.

Si no (PG):

1. Clone actor; collect \(T_{\mathrm{emit}}\) Cand. 2; **write** replay.
2. GAE vs \(V_\phi\); normalizar ventajas **por emit**.
3. PPO-clip \(K=2\), \(\varepsilon=0.2\), \(c_e=0.01\); update **solo actor**.
4. Updates crítico desde replay (`8×256` default).
5. Eval Cand. 1; try insert.

---

## 6–8. Mosaic, bucle, fitness

Mosaic 25/25/50. Outer loop: bootstrap → ≤4 aux emits (moneda mut) → \(N_{\mathrm{outer}}=219\) → deploy \(\pi^\star\).

| Parámetro | Valor |
|-----------|--------|
| \(T_{\mathrm{total}}\) | `1_000_000` |
| \(B_{\mathrm{boot}}\) | `100_000` |
| \(T_{\mathrm{emit}}\) | `1024` |
| \(n_{\mathrm{emit}}\) | `4` |
| \(N_{\mathrm{outer}}\) | `219` |
| \(p_{\mathrm{mut}}\) | `0.5` |
| \(\sigma\) | `0.02` |
| \(K_{\mathrm{pg}}\) | `2` |
| Workers | `1` |

Fitness/BD/suite: idénticos a QD-PPO (\(G=20\), \(g_{\min}=10\), 200 BO1).

---

## 9. Módulos y CLI

`archive.py`, `critic.py`, `replay.py`, `mutate.py`, `improve.py`, `evaluate.py`,
`mosaic.py`, `train.py`, `agent.py` / `player.py` (`ValidatedQDPGAPlayer`);
tests `tests/test_rl_qd_pga.py`.

```bat
python -m showdown.rl.qd_pga.train --seed 0 --smoke
```

---

## 10. Aceptación / non-goals

IAM; mut no crece replay; PG sí; elites sin `value.*`; \(N_{\mathrm{outer}}=219\); \(g<10\) reject; smoke.

Non-goals: QD-PPGA; ME-ES primario; CVT; crítico por elite; mut→replay;
expand-\(\Omega\)/self-play como diversidad primaria; BD en \(R_t\); multi-emitter; preview en MDP.

---

## 11. Fortalezas y límites (tesis)

| Fortaleza | Límite |
|-----------|--------|
| Exploración mut + mejora PG | Mutación ciega puede degradar IAM-sensible |
| Crítico compartido eficiente | Menos “personalidad” de valor por elite |
| Más emits que QD-PPO (\(N=219\)) | \(T_{\mathrm{emit}}\) más corto (1024) |

---

## 12. Referencias

Nilsson & Cully (2021) PGA-MAP-Elites; Mouret & Clune (2015); Schulman et al. (2017);
Sutton & Barto (2018); Huang & Ontañón (2022);
[`ppo_spec.md`](ppo_spec.md), [`qd_ppo_spec.md`](qd_ppo_spec.md), [`qd_ppga_spec.md`](qd_ppga_spec.md).

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.1** | 2026-10-10 | Redacción amena (ES); semántica intacta. |
| **1.0** | 2026-10-10 | Frozen: shared \(V_\phi\); \(p_{\mathrm{mut}}=0.5\); \(N_{\mathrm{outer}}=219\). |

---

*Fin del contrato QD-PGA-MAP-Elites v1.1.*
