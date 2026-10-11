# QD-PPO Agent Spec — Quality–Diversity + Masked PPO (VGC Gen 9)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo — QD-PPO** |
| **Versión** | **1.1** (alineado con `docs/rl_env_spec.md` **v1.10** §6.e) |
| **Estado** | **FROZEN — implementado en `showdown/rl/qd_ppo/`** |
| **Algoritmo** | Grid MAP-Elites sobre BDSpec + Masked PPO (IAM) como único operador de variación |
| **Rol** | Camino QD de **política**: iluminar el espacio de comportamientos; varios elites |
| **Code root** | `showdown/rl/qd_ppo/` |
| **CLI** | `python -m showdown.rl.qd_ppo.train` |
| **Deps** | `torch>=2.0`; reusa `showdown/rl/ppo/` |

> **Alcance.** SoT de QD-PPO. Env/BD/Reward en [`rl_env_spec.md`](rl_env_spec.md).
> Matemática PPO en [`ppo_spec.md`](ppo_spec.md) — **reusar; no forkar**.
>
> **Contraste.** Masked PPO = una política (+ pool). QD-PPO = **archivo de elites**.
> Fitness archivo \(U=f\) = Cand. **1**; PPO interno \(R_t\) = Cand. **2**.
> PGA/PPGA: [`qd_pga_map_elites_spec.md`](qd_pga_map_elites_spec.md), [`qd_ppga_spec.md`](qd_ppga_spec.md).
>
> **Inmutabilidad.** Bump **1.2+** + `rl_env_spec` §6.e juntos.

---

## 0. Por qué existe (lectura no técnica)

Un solo agente “campeón” puede ser frágil: gana de una sola forma. QD-PPO mantiene
un **mapa** (rejilla 10×10) donde cada casilla guarda el mejor agente visto con un
*estilo* de juego medido por dos números de comportamiento (BD). Para mejorar un
estilo, se copia un elite y se le aplica un poco de entrenamiento PPO; si el
resultado es mejor en utilidad Cand. 1 y cae en una casilla, entra al archivo.

**Analogía:** un museo de estrategias: no solo el cuadro más caro, sino el mejor
de cada sala (agresivo, pivoteo, …). La “calidad” es ganar (Cand. 1); la “diversidad”
es ocupar salas distintas (BD), **sin** meter la diversidad dentro de la recompensa
de entrenamiento.

| Término | Significado aquí |
|---------|------------------|
| Archive / MAP-Elites | Rejilla de elites por BD |
| BD | Coordenada de comportamiento \([0,1]^2\) |
| Fitness \(f\) | Media Cand. 1 (utilidad de archivo) |
| Improve | PPO corto con Cand. 2 |
| Mosaic | Mezcla fija de rivales 25/25/50 |
| Coverage / QD-score | % celdas llenas / suma de fitness |

---

## 0b. OPEN → FROZEN (v1.0 conservado)

| ID | Decisión |
|----|----------|
| **O1–O2** | Grid 2D regular \(10\times 10\) (100 celdas) |
| **O3** | Full actor–critic `state_dict` por celda; **sin** crítico compartido |
| **O4** | Padres: uniforme sobre celdas **llenas** |
| **O5** | Solo gradientes PPO; **sin** ME-ES noise |
| **O6–O7** | \(f=\) media Cand. 1 en \(G=20\); BD media en elegibles; insert si \(g\ge 10\) |
| **O8** | PPO interno = Cand. 2; Cand. 1 solo utilidad archivo |
| **O9** | Bootstrap \(B_{\mathrm{boot}}=100\,000\) Cand. 2 en mosaic; no seed solo-random |
| **O10–O12** | Fitness foes = mosaic; emitter serial; reportar Coverage, QD-score, WR(\(\pi^\star\)) |

| Ítem operativo | Contrato |
|----------------|----------|
| Improve / Stage 1 | Mosaic i.i.d.; **no** expand-\(\Omega\); **no** self-play |
| Replace | Estricto \(f_{\mathrm{new}}>f_{\mathrm{old}}\) |
| Deploy | \(\pi^\star=\arg\max f\) |
| Budget | \(T_{\mathrm{total}}=10^6\) **incl.** bootstrap; \(N_{\mathrm{outer}}=109\) |
| Eval suite | Overhead; **no** cuenta en \(T_{\mathrm{total}}\) |

---

## 1. Entorno heredado

| Contrato | Uso QD-PPO |
|----------|------------|
| `VGCEnv`, Obs 1024D | Rollouts; BD en `info` |
| AR + **IAM** | Sample + recompute PPO |
| Cand. 2 / Cand. 1 | Improve+bootstrap / fitness archivo |
| BDSpec | Coordenadas de archivo |
| Preview `/team 1234` | Fuera del MDP |
| Belief | Priors planos |

Helper: `BehavioralTracker` (`archive_eligible` iff turns ≥ 4).

### Taxonomía \(R\) vs \(U\)

| Señal | Rol | Cuándo |
|-------|-----|--------|
| Cand. 2 \(R_t^{(2)}\) | Optimización de política | Improve + Stage 1 |
| Cand. 1 \(U=f\) | Fitness archivo | Insert + suite |
| BD \(b\) | Coordenada | Índice de celda; **no** en \(R_t\) ni \(U\) |

---

## 2. Archivo (Grid MAP-Elites)

| Ítem | Contrato |
|------|----------|
| Grid | \(10\times 10\) en \([0,1]^2\) |
| Bin | \(i=\min(9,\lfloor\mathrm{BD}_1\cdot 10\rfloor)\) (igual \(j\)) |
| Payload | Full AC `state_dict` + \(f\) + BD + meta |
| Insert | Vacía → insert; else replace si \(f'>f\); empate → incumbent |

\[
\mathrm{Coverage}=\frac{|\mathrm{filled}|}{100},\quad
\mathrm{QD\text{-}score}=\sum_{c\in\mathrm{filled}}\max(0,f(e_c))
\]

---

## 3. Mosaic foes

\[
P(\texttt{random})=0.25,\ 
P(\texttt{greedy\_solo})=0.25,\ 
P(\texttt{greedy\_joint})=0.50
\]

i.i.d. por episodio.

---

## 4. Bucle externo y presupuesto

```
bootstrap ≤ B_boot Cand2 mosaic
N_outer ← floor((T - used) / (n_emit * 2048))   # 109
for t in 1..N_outer:
    parents ← Uniform(filled, n_emit)
    for parent: deepcopy → PPO_improve 2048 Cand2 → eval Cand1 → insert?
deploy π* = argmax f
```

| Parámetro | Valor |
|-----------|--------|
| \(T_{\mathrm{total}}\) | `1_000_000` |
| \(B_{\mathrm{boot}}\) | `100_000` |
| \(n_{\mathrm{emit}}\) | `4` |
| Steps / emit | `2048` |
| \(N_{\mathrm{outer}}\) | `109` |
| Workers | `1` (serial) |
| ME-ES | Off |

IAM bajo variación: mismo softmax enmascarado en collect y épocas PPO.

---

## 5. Improve PPO (interno)

Importar [`ppo_spec.md`](ppo_spec.md) §§3–6. Reward: **solo Cand. 2**. Foes: mosaic. Sin self-play pool.

---

## 6. Fitness y BD (asimétrico)

| Ítem | Contrato |
|------|----------|
| \(G\) | 20 BO1, Cand. 1, greedy masked, mosaic |
| \(g\) | Conteo `archive_eligible` |
| Discard | \(g<10\) → no escribe archivo |
| BD | Media sobre \(g\) elegibles |
| \(f\) | Media Cand. 1 sobre **todos** \(G=20\) |

Suite offline en \(\pi^\star\): 200 BO1 vs `random` / `greedy_joint` / `expectimax_bayesian`.

---

## 7. Bootstrap (O9)

Stages 0–3: BC opcional → PPO Cand. 2 ≤100k → eval+insert ≤4 emits aux desde Stage-1 (**no** random-init) → outer loop.
**Prohibido:** seedear archivo solo con redes max-entropy / random-init.

---

## 8. Módulos y CLI

`archive.py`, `mosaic.py`, `evaluate.py`, `improve.py`, `train.py`, `agent.py` / `player.py`
(`ValidatedQDPPOPlayer`); reuso `ppo/*`; tests `tests/test_rl_qd_ppo.py`.

```bat
python -m showdown.rl.qd_ppo.train --seed 0 --smoke
```

---

## 9. Telemetría / aceptación / non-goals

Telemetría: `qd_coverage`, `qd_score`, `archive_best_f`, `emit_accept_rate`, salud PPO, ilegales.

Aceptación: IAM; BD formulas; \(g<10\) no inserta; bins extremos; replace estricto;
Cand1 fitness / Cand2 improve; elites independientes; \(N_{\mathrm{outer}}=109\); smoke.

Non-goals: QD-PGA/PPGA; ME-ES; CVT; shared critic; expand-\(\Omega\)/self-play como diversidad
primaria; BD en \(R_t\); seed random-only; multi-emitter; preview en MDP; head joint-576;
train en L2/crash.

---

## 10. Fortalezas y límites (tesis)

| Fortaleza | Límite |
|-----------|--------|
| Diversidad explícita vía archivo | Presupuesto repartido entre muchos elites |
| Reusa PPO+IAM maduro | Sin crítico compartido (más memoria) |
| Separación clara \(R\) vs \(U\) | Mosaic fijo (no curriculum expand) |

---

## 11. Referencias

Mouret & Clune (2015); Schulman et al. (2017); Huang & Ontañón (2022);
Yannakakis & Togelius (2018); [`ppo_spec.md`](ppo_spec.md); [`rl_env_spec.md`](rl_env_spec.md).

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.1** | 2026-10-10 | Redacción amena (ES); semántica intacta. |
| **1.0** | 2026-10-10 | Frozen grid 10×10; PPO+IAM; mosaic; \(N_{\mathrm{outer}}=109\). |

---

*Fin del contrato QD-PPO v1.1.*
