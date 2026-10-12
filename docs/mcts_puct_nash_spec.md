# MCTS–PUCT/Nash Agent Spec — SM-MCTS B1 (VGC Gen 9)

| Campo | Valor |
|---|---|
| **Documento** | Contrato exclusivo — MCTS–PUCT/Nash simultáneo (solo turnos) |
| **Versión** | **1.5** (alineada con [`rl_env_spec.md`](rl_env_spec.md) v1.16) |
| **Estado** | FROZEN; `showdown/rl/mcts_puct_nash/` |
| **Training** | **No**: sin gradientes ni `.pt` de combate |
| **Ajuste** | Presupuesto \(N_{\rm sim}\), Top-K, FP; barrido automatizado de sims |
| **CLI smoke** | `python -m showdown.rl.mcts_puct_nash --seed 0 --sims 8 --smoke` |
| **CLI barrido** | `python -m showdown.inference_sim.run_mcts_sims_sweep --agent mcts_puct_nash ...` |
| **Env presupuesto** | `MCTS_NASH_SIMS` (fallback `MCTS_SIMS`; default config `50`) |
| **Dependencias** | NumPy; reutiliza S2 de `mcts_puct` |

> **Para quien aprende IA.** En VGC ambos eligen a la vez. Este agente no asume
> un rival al azar “después” de ti: mantiene una **tabla** (matriz) de “mi
> jugada × su jugada”, la rellena con simulaciones y resume cada nodo con un
> equilibrio aproximado (Nash / Fictitious Play). Sigue sin entrenar una red.

---

## 1. Por qué existe

El MCTS open-loop ([`mcts_puct_spec.md`](mcts_puct_spec.md)) samplea al rival de
forma uniforme al expandir. Aquí se pregunta: *¿qué mezcla mía resiste una buena
respuesta suya?* Explora **celdas** \((i,j)\) con PUCT y hace backup del valor
de la matriz resuelta, no solo de una muestra open-loop.

**Analogía:** no solo “¿qué pasa si yo hago A y el mundo tira un dado?”, sino
“si yo mezclo A/B y él responde bien, ¿quedo cubierto?”.

Preview familiar en Conclusión: [`team_preview_nash_spec.md`](team_preview_nash_spec.md)
(otro Nash: formación, no turnos).

---

## 2. Glosario

| Término | Significado |
|---|---|
| Celda \((i,j)\) | Acción propia Top-K contra acción rival Top-K |
| \(n_{ij}\) | Visitas de la celda (análogo a \(N\) por celda) |
| \(W_{ij}\) | Suma de retornos en la celda |
| \(\hat A_{ij}=W_{ij}/n_{ij}\) | Media empírica (análogo a \(Q\) por celda) |
| Top-K | Recorte greedy de joints legales por lado |
| Fictitious Play (FP) | Aproxima mezclas \((\sigma,\tau)\) y valor \(v=\sigma^\top\hat A\tau\) |
| PUCT (celdas) | Explora celdas poco visitadas de la matriz |
| \(N_{\rm sim}\) | Sims por decisión de turno (llena celdas) |
| SM-MCTS | Árbol para movimientos simultáneos |

---

## 3. Qué son \(Q\)/\(N\) aquí (celdas de la matriz)

### 3.1 Misma idea que open-loop, otra unidad

En open-loop, \(N\) y \(Q\) viven en **aristas propias**.  
En MCTS–Nash viven en **celdas** \((i,j)\):

| Open-loop | Nash (esta spec) |
|---|---|
| Arista acción propia \(a\) | Celda (mi joint \(i\), su joint \(j\)) |
| \(N(s,a)\) | \(n_{ij}\) |
| \(Q(s,a)=W/N\) | \(\hat A_{ij}=W_{ij}/n_{ij}\) |

**Tampoco se entrenan:** se reinician con cada árbol de turno. No hay checkpoint
neural de combate.

### 3.2 Por qué la matriz necesita más que un \(Q\) propio

Una acción propia con buen promedio contra un rival aleatorio puede ser
explotable. Tras actualizar una celda, el nodo **resuelve** la matriz y sube el
valor Nash aproximado a los ancestros. La decisión de raíz juega
\(\arg\max_i \sigma_i\) (masa de la mezcla propia), no \(\arg\max N\) de una
sola arista open-loop.

### 3.3 Ejemplo intuición \(n\) y \(\hat A\)

Si una celda tiene \(\hat A_{ij}=1\) con \(n_{ij}=1\), es una victoria suelta:
puede deformar la mezcla. Más \(N_{\rm sim}\) reparte visitas y estabiliza
\(\hat A\). Más iteraciones FP estabilizan \((\sigma,\tau)\) **dado** ese
contenido (otro knob, no sustituye sims).

Para el aprendiz: cada casilla de la tabla es un “mini \(Q\)”; \(n_{ij}\) dice
cuántas veces se miró esa casilla; el Nash resume la tabla entera.

### 3.4 Mini-ejemplo de mezcla

Para
\(\hat A=\begin{bmatrix}1&-1\\-1&1\end{bmatrix}\),
jugar siempre una fila permite al rival castigar. \(\sigma\approx(0.5,0.5)\)
hace indiferente al rival (valor ~0). Con pocas muestras, una celda ruidosa
tuerce \(\sigma\); con más sims la tabla se llena.

---

## 4. Algoritmo paso a paso

1. Ordenar joints legales; Top-K (hasta 16) por lado (rival = snapshot volteado).
2. Crear matriz de celdas; prior uniforme \(1/(|I||J|)\).
3. Por cada simulación:
   - Elegir celda no visitada o PUCT sobre celdas.
   - S2 aplica ambas joints a la vez.
   - Terminal → Candidate 1; hoja nueva → rollout greedy joint.
   - Actualizar \(n_{ij},W_{ij}\); FP interno (~200 it.) en ancestros.
4. Tras \(N_{\rm sim}\): FP raíz (~500 it.) → jugar \(\arg\max_i\sigma_i\).

Top-K evita matrices hasta \(576\times576\). Empates de ranking: joint id menor.

### Ciclo de presupuesto

Costo = `step_sim` **más** solves FP. Comparar solo \(N_{\rm sim}\) con open-loop
**no** iguala tiempo de pared. Default config: **50** sims (`8` smoke).

---

## 5. Observación, acción y máscaras

Live: [`rl_env_spec.md`](rl_env_spec.md). Árbol sobre `BattleSnapshot`. Solo
joints legales; Mega ok; Tera 22 off. Hojas Candidate 1.

**No hay Team Preview en este MDP de combate.** Preview familiar congelado:
[`team_preview_nash`](team_preview_nash_spec.md) vía `frozen_preview`
(`MCTSPuctNashPlayer`); fallback `/team 1234` si falta checkpoint.

---

## 6. “Mejorar” sin entrenamiento

| Palanca | Efecto |
|---|---|
| \(N_{\rm sim}\) | Más cobertura de celdas; meseta WR/tiempo |
| Top-K | Más cobertura estratégica; costo cuadrático en K |
| FP it. | Mezcla más estable; CPU por nodo |
| Ranking Top-K | Qué joints entran (cambio de agente) |

### 6.1 Meseta de \(N_{\rm sim}\)

Misma filosofía que open-loop, **rejilla propia** (más cara): default
`25,50,100,200`. Env: `MCTS_NASH_SIMS`.

### 6.2 Barrido automatizado

```bat
python -m showdown.inference_sim.run_mcts_sims_sweep --agent mcts_puct_nash ^
    --player-b greedy_joint --seed 10 --teams 40 ^
    --delta-wr-pp 1.5 --time-ratio 1.8 --metric wr_raw --start-ps
```

Ambos agentes:

```bat
python -m showdown.inference_sim.run_mcts_sims_sweep --agent both ^
    --player-b greedy_joint --seed 10 --teams 40 --start-ps
```

| Parámetro CLI | Default | Propósito | Cómo influye |
|---|---:|---|---|
| `--agent` | `both` / `mcts_puct_nash` | Qué barrer | Nash usa grilla más corta por costo FP |
| `--player-b` | `greedy_joint` | Rival robust e2e | Fijar al reportar meseta |
| `--seed` | `10` | Repro + `MCTS_NASH_SEED` | Cambia curva, no metodología |
| `--teams` | `40` | Tamaño batch | Precisión vs horas |
| `--sims` | `25,50,100,200` | Rejilla | Vacío = default Nash |
| `--delta-wr-pp` | `1.5` | Umbral de mejora WR | Criterio de meseta |
| `--time-ratio` | `1.8` | Factor tiempo | Meseta si el salto es caro y poco ganancia |
| `--metric` | `wr_raw` | Métrica | `pair_wr` opcional |
| `--dry-run` | off | Sin PS | Prueba criterio/tabla |

Por punto: `MCTS_NASH_SIMS=<N>` + robust e2e `--player-a mcts_puct_nash`.

#### Tracking del barrido

Igual que open-loop: solo varía `n_sims`; Top-K / FP / \(c\) se registran como
congelados. Artefactos:

| Artefacto | Rol |
|-----------|-----|
| `..._mcts_puct_nash_history.jsonl` | Eventos `hyperparam_step` + decisión final |
| `..._mcts_puct_nash_progress.json` | Progreso vivo |
| `..._mcts_puct_nash_tracking.txt` | Seguimiento legible prev→nuevo N, ΔWR, time_ratio |
| `..._mcts_puct_nash.{txt,json}` | Resumen + tracking embebido |
| `logs/figures/mcts_puct_nash_sims_sweep_latest.svg` | WR vs paso (figura real al DONE) |

Código: `mcts_sims_sweep.py` + `run_mcts_sims_sweep.py`.  
Tests: `tests/test_mcts_sims_sweep.py`.

Congelar el \(N_{\rm sim}\) recomendado antes de Conclusión.

```bat
python -m showdown.rl.mcts_puct_nash --seed 0 --sims 8 --smoke
```

---

## 7. Hiperparámetros de búsqueda

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| \(N_{\rm sim}\) | `50` (`8` smoke) | Presupuesto por decisión | Más llena la matriz; barrer hasta meseta |
| \(c\) PUCT | `1.5` | Explorar celdas | Mayor dispersa visitas |
| \(K\) | `16` | Acotar matriz | Mayor cubre más; hasta \(K^2\) celdas |
| FP interno | `200` | Valor de nodos | Estabiliza backup |
| FP raíz | `500` | Mezcla jugada | Reduce error de decisión |
| Ventana FP | `50` | Promedio final | Suaviza oscilaciones |
| \(\varepsilon_{\rm opt}\) | `1e-6` | Empates óptimos | Max-entropía en óptimos |
| Prior | uniforme por celda | Inicio neutral | Sin red |
| Samples/visita | `1` | Update online | Más reduce ruido a más costo |
| Max turns | `40` | Acotar sim | Profundidad |
| Descuento | `1` | Backup Cand. 1 | Escala cero-suma |
| RNG | seed+turn+sim | Repro | Azar S2 |
| Acción final | `argmax σ` | Orden jugable | Desempate joint id |

---

## 8. Ilegales, máscaras y fallos

- Top-K tras filtrar legalidad; nunca rehabilita ilegales.
- Tera 22 fuera; Mega según snapshot.
- \(I\) vacío → `(pass, pass)`; matriz vacía → valor 0.
- L1/L2/L3 como en open-loop / `rl_env_spec`.
- Fallo S2 → cero + `failures`.
- FP aproximado ≠ equilibrio exacto.

---

## 9. Training frente a inferencia

Todo es inferencia. “Presupuesto” = sims + solver por turno. El smoke no es
entrenamiento. Conclusión congela K, sims, FP, ranking, seed y rollout.
Protocolo: [`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

---

## 10. Team Preview y Conclusión

Pipeline familiar: **`team_preview_nash` → MCTS–PUCT/Nash combate**
(cableado en train live e inferencia; no solo en el registro de Conclusión).
Mismo apellido “Nash”, problemas distintos (formación vs matrices locales).

---

## 11. Mapa de código

| Archivo | Responsabilidad |
|---|---|
| `showdown/rl/mcts_puct_nash/agent.py` | config / acción |
| `showdown/rl/mcts_puct_nash/search.py` | SM-MCTS + backup |
| `showdown/rl/mcts_puct_nash/tree.py` | matrices, celdas, PUCT |
| `showdown/rl/mcts_puct_nash/nash.py` | Fictitious Play |
| `showdown/rl/mcts_puct_nash/topk.py` | ranking Top-K |
| `showdown/rl/mcts_puct/simulator.py` | S2 compartido |
| `showdown/rl/mcts_puct_nash/player.py` | poke-env + `MCTS_NASH_SIMS` |
| `showdown/inference_sim/mcts_sims_sweep.py` | meseta |
| `showdown/inference_sim/run_mcts_sims_sweep.py` | CLI barrido |

---

## 12. Limitaciones

- Top-K puede excluir respuestas con score greedy bajo.
- Muchos solves FP limitan sims asequibles.
- FP aproximado propaga error en cada backup.
- Sesgo S2 + rollout greedy.
- No es belief search de información imperfecta estilo póker.
- Tera off.

---

## 13. Lectura de telemetría

Además de sims/nodos/fallos: entropía de \(\sigma\), fracción de celdas raíz
visitadas, y conteos \(n_{ij}\). Valor Nash ≠ probabilidad calibrada de victoria.
Comparar configs congelando K y midiendo `step_sim` + tiempo FP. Conservar seed
y turno.

---

## Changelog

| Versión | Fecha | Notas |
|---|---|---|
| **1.5** | 2026-10-11 | Inferencia bajo Preview Nash congelado (`frozen_preview`). |
| **1.4** | 2026-10-11 | §Q/N por celdas; meseta \(N_{\rm sim}\); CLI sweep; rival/seed/criterio; **tracking** de cambios de N_sim. |
| **1.3** | 2026-10-11 | Matemática matricial, presupuesto, fallos, code map. |
| **1.2** | 2026-10-11 | Dual-audience; Nash-preview. |
| **1.1** | 2026-10-10 | Lectura pedagógica. |
| **1.0** | 2026-10-10 | Contrato SM-MCTS B1. |
