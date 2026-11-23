# MCTS–PUCT/Nash Agent Spec — SM-MCTS B1 (VGC Gen 9)

| Campo | Valor |
|---|---|
| **Documento** | Contrato exclusivo — MCTS–PUCT/Nash simultáneo (solo turnos) |
| **Versión** | **2.0** (H 2.0: S3 de G open-loop + preview por búsqueda + SP meta, §4b; confirmada, H de la Conclusión) |
| **Estado** | FROZEN; `showdown/rl/mcts_puct_nash/` |
| **Training** | **No**: sin gradientes ni `.pt` de combate |
| **Ajuste** | Presupuesto \(N_{\rm sim}\), Top-K, FP; barrido automatizado de sims |
| **CLI smoke** | `python -m showdown.rl.mcts_puct_nash --seed 0 --sims 8 --smoke` |
| **CLI barrido** | `python -m showdown.inference_sim.run_mcts_sims_sweep --agent mcts_puct_nash ...` |
| **Env presupuesto** | `MCTS_NASH_SIMS` (fallback `MCTS_SIMS`; 200 en v2, 50 en v1) |
| **Env versión** | `MCTS_NASH_VERSION` (2 por defecto en el player; 1 = SM-MCTS B1) |
| **Dependencias** | NumPy; v2 reutiliza S3, la poda y el preview por búsqueda de `mcts_puct` (G); v1 reutiliza S2 |

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

Top-K evita matrices hasta \(1600\times1600\). Empates de ranking: joint id menor.

### Ciclo de presupuesto

Costo = `step_sim` **más** solves FP. Comparar solo \(N_{\rm sim}\) con open-loop
**no** iguala tiempo de pared. Default config: **50** sims (`8` smoke).

---

## 4b. Versión 2.0 (agente H 2.0, default del player desde el 22-11-2026, confirmada el 23-11-2026)

`MCTS_NASH_VERSION=2` (default del player; `1` = SM-MCTS B1 de §4). Código:
`showdown/rl/mcts_puct_nash/openloop_nash.py`. Se conserva lo que define a H (cada nodo es un juego
simultáneo de suma cero y sube el **valor de Nash** \(v^*(\hat A)\), no la media; se juega
\(\arg\max\sigma^*\)) y se porta lo que G demostró que funciona (diagnóstico de G §2–§9):

| Aspecto | v1.7 (SM-MCTS B1) | v2.0 | Por qué |
|---|---|---|---|
| Modelo de transición | S2 (`step_sim`) | **S3 fase 3c** de G (daño real de la calculadora, velocidad/prioridad, Protect, Fake Out, redirección, Tailwind, Trick Room, estados, boosts, Intimidate, clima, pivotes) | S3 llevó a G de 56 % a 65,5 % contra joint |
| Árbol | Cerrado: el hijo de una celda se crea una vez (una tirada de azar fija) | **Open-loop**: los nodos guardan solo estadísticas; cada sim vuelve a muestrear el azar desde la raíz | Igual que G v2 |
| Filas \(I\) | Top-16 por `score_joint` (S2) de todos los joints | Raíz: joints legales reales (`valid_orders`), Top-12 por S3; dentro: joints podados, Top-8 | Sin pases ni golpes al aliado |
| Columnas \(J\) | Top-16 (S2) | Top-8 podados por S3 desde el lado rival | — |
| Prior | Uniforme por celda | \(P_{ij} = p_i\,q_j\), softmax del daño estimado (+ ε) | — |
| Selección | Primero las celdas no visitadas **en orden de id**; luego \(\max Q+U\) sobre celdas | **Desacoplada contra la mezcla actual**: fila \(\arg\max_i \sum_j \tau_j \hat A_{ij} + c\,p_i\sqrt N/(1+n_i)\); columna \(\arg\min_j \sum_i \sigma_i \hat A_{ij} - c\,q_j\sqrt N/(1+n_j)\) | v1 con 50 sims solo veía las primeras filas por id; \(\max Q+U\) sobre celdas supone un rival que colabora |
| \(\hat A\) no visitadas | 0 | Media del nodo (FPU) | 0 sesgaba \(v^*\) hacia el empate |
| Hoja | Rollout greedy S2 hasta el turno 40 | 2 turnos greedy S3 + material en [−1, 1] | Igual que G |
| \(N_{\rm sim}\) / profundidad | 50 / sin tope | **200** (como G) / 4 plies | Mismo presupuesto de sims que G (~1,7× su tiempo por decisión, por el FP) |
| FP interno / raíz | 200 / 500 | 100 / 500 | Costo |
| Preview | `team_preview_nash` (nunca entrenado) → `/team 1234` | **Preview por búsqueda de G** (`mean`, 90 s, fase 3c) | +4,1 pts en G (protocolo §5.5) |
| SP del rival | Inferencia | **Prior meta** (`default_sp_info_mode = 'meta'`) | La inferencia estrecha 0 stats en Reg M-C (diagnóstico de G §9.1) |
| Anomalías | Silenciosas | `MCTS_STATS_FILE` (excepción, fallback, calculadora, preview), como G | Auditoría de la cola |

Medición pre-registrada: protocolo §4.6 (cola `scripts/server/mcts_h_v2.sh`). Tests:
`tests/test_mcts_puct_nash_v2.py` (7).

**Resultado (protocolo §5.7, 23-11-2026):** confirmada contra los tres baselines en C13/C14 (34/34
semillas cada una): 61,57 % contra greedy_joint, 61,73 % contra expectimax v1.9, 63,33 % contra
greedy_solo. Frente a G 5.2 (mismas series, contra joint): **−7,25 pts** (1087 frente a 1815, p Holm =
6·10⁻⁴²; C13 −7,37, C14 −7,13), con 1,7× su tiempo (378 frente a 219 s por semilla). H 1.7 en la
exploración: 10,9 %. H 2.0 es el H de la Conclusión; con este presupuesto el backup de Nash no mejora al
open-loop con media de G. Causas posibles, sin probar: matriz de 96 celdas dispersa para 200 sims, y el
valor de Nash supone un rival del peor caso frente a baselines explotables.

**Paso 0 de H 2.1: diagnóstico de H 2.0 (23-11-2026, commit `47c0331`, cola
`scripts/server/mcts_h_diag.sh`).** H 2.0 contra greedy_joint en 9101–9102 × 300 con
`MCTS_NASH_DIAG_FILE` (`showdown/rl/mcts_puct_nash/diag.py`): 7628 decisiones no triviales; en cada una
se registra la matriz de la raíz y el joint que elige G 5.2 en el mismo estado (misma raíz, creencias y
semilla). El registro no cambia la partida: 189 y 187 victorias, idénticas a N2 en esas semillas.
Descriptivo, sin hipótesis.

| Métrica (raíz) | Mediana (p25–p75) |
|---|---|
| Matriz \(R \times C\) | 12 × 8 |
| Celdas sin visitar (\(\hat A\) desde el FPU) | **42 %** (21–53 %) |
| Visitas por celda visitada | **2** (1–3) |
| \(\max \sigma^*\) / soporte > 5 % | 0,57 (0,38–1,00) / 3 filas |
| Distancia TV entre \(\tau^*\) y el prior del rival \(q\) | 0,63 (0,35–0,82) |

| Regla sobre la misma \(\hat A\) | Coincide con H (argmax \(\sigma^*\)) | Coincide con G 5.2 |
|---|---|---|
| argmax \(\sigma^*\) (H 2.0) | — | **45,9 %** (turnos 1–2: 41,9 %) |
| Fila más visitada | 60,1 % | **60,2 %** |
| Mejor respuesta a \(q\) | 53,5 % | 40,2 % |
| Mezcla \(\tfrac12(\hat A\tau^* + \hat A q)\) | 57,8 % | 40,5 % |

El joint de G está entre las filas de H en el 96,4 % de los estados, pero H le da \(\sigma^* = 0\) en el
22,9 % (mediana 0,11) aunque recibe el 28 % de las visitas. Tiempo por decisión: H 1,81 s, G 0,61 s.

Lectura: (1) la matriz está muy submuestreada (2 visitas por celda, 42 % de huecos), así que \(\sigma^*\)
se calcula sobre una \(\hat A\) ruidosa; (2) la regla de decisión final es la mayor divergencia: la
fila más visitada de H coincide con G 14 pts más que argmax \(\sigma^*\); (3) responder al prior del
rival \(q\), solo o mezclado, aleja a H de G, así que la causa "rival del peor caso" no parece la
principal. Candidatos para H 2.1, en este orden: matriz más chica o RM+ (más visitas por celda en el
mismo tiempo) y decisión por visitas (o \(\sigma^*\) solo con la matriz bien muestreada); la mezcla con
\(q\) queda como brazo secundario. Que H coincida con G no implica que gane más; lo decide una medición
pre-registrada en semillas 10601+.

---

## 4c. Versión 2.1 (candidatas tras el paso 0, pre-registradas el 23-11-2026)

Variantes de H 2.0 que atacan lo que mostró el diagnóstico del paso 0 (§4b): la matriz de la raíz está
submuestreada (42 % de celdas sin visitar, 2 visitas por celda) y la regla final argmax \(\sigma^*\)
sobre esa \(\hat A\) ruidosa es la mayor divergencia con G. Se activan por entorno; **sin variables el
player es H 2.0 exactamente** (test de regresión contra el código congelado `c70f0f2` en
`tests/test_mcts_puct_nash_v21.py`).

| Brazo | Entorno | Qué cambia | Por qué |
|---|---|---|---|
| H1 matriz chica | `MCTS_NASH_MATRIX=8,6,4` | Raíz 8 filas, interior 6, rival 4 (antes 12/8/8): 32 celdas en la raíz en vez de 96 | ~6 visitas por celda con las mismas 200 sims |
| H2 decisión por visitas | `MCTS_NASH_DECISION=visits` | Se juega la fila más visitada (desempate por \(\sigma^*\), menor id); el árbol, la selección y el backup de Nash no cambian | La fila más visitada coincide con G 14 pts más que argmax \(\sigma^*\) |
| H3 = H1 + H2 | ambas | — | Las dos causas a la vez |
| H4 RM+ | `MCTS_NASH_SOLVER=rm+`, `MCTS_NASH_SIMS=350` | Cada nodo resuelve su matriz con **regret matching+** con arranque en caliente (regrets y promedios lineales guardados en el nodo; 10 iteraciones por backup, 200 en la raíz) en lugar de un FP de 100 iteraciones; el valor de Nash sigue subiendo al padre. El ahorro (−41 % por decisión con 200 sims, benchmark local) se gasta en sims: 350 ≈ tiempo de H 2.0 | Más visitas por celda con el mismo tiempo; RM+ es el resolvedor estándar del SM-MCTS (Lanctot et al., 2013; CFR+) |

Código: `NashOLConfig.decision`, `.solver`, `.t_rm_internal`, `.t_rm_root` y `k_self_root/k_self/k_opp`
(`openloop_nash.py`); `regret_matching_plus` (`nash.py`); `_h21_knobs` (player). Medición pre-registrada:
protocolo §4.7 (cola `scripts/server/mcts_h_v21.sh`). Si un brazo se confirma, pasa a ser el default del
player como H 2.1.

---

## 5. Observación, acción y máscaras

Live: [`rl_env_spec.md`](rl_env_spec.md). Árbol sobre `BattleSnapshot`. Solo
joints legales; Mega ok; Tera 22 off. Hojas Candidate 1.

**No hay Team Preview en este MDP de combate.** Preview familiar congelado:
[`team_preview_nash`](team_preview_nash_spec.md) vía `frozen_preview`
(`MCTSPuctNashPlayer`); fallback `/team 1234` si falta checkpoint. **Desde v2.0 el preview por
defecto es el de búsqueda de G** (`MCTS_PREVIEW=nash` vuelve al de familia).

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

**Desde v2.0 (Conclusión): preview por búsqueda de G → H 2.0 combate.** Pipeline familiar anterior
(`MCTS_PREVIEW=nash`): **`team_preview_nash` → MCTS–PUCT/Nash combate**
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
| **2.1** (candidatas) | 2026-11-23 | Variantes por entorno tras el paso 0 (§4c): matriz chica 8/6/4, decisión por visitas, ambas, y RM+ con arranque en caliente y 350 sims. Defaults = H 2.0 exacto. Diagnóstico opcional `MCTS_NASH_DIAG_FILE` (§4b). Medición pre-registrada: protocolo §4.7. |
| **2.0** (confirmada 23-11: 61,6 % joint, 61,7 % expectimax, 63,3 % solo; −7,25 pts frente a G 5.2, protocolo §5.7) | 2026-11-22 | H 2.0 (§4b): SM-MCTS open-loop con S3 fase 3c de G, Top-K podado por S3, prior \(p_i q_j\), selección desacoplada contra la mezcla de Nash, FPU en \(\hat A\), hoja de G, 200 sims; preview por búsqueda de G; SP por prior meta (sin inferencia); anomalías en `MCTS_STATS_FILE`. v1.7 sigue con `MCTS_NASH_VERSION=1`. Medición pre-registrada: protocolo §4.6. |
| **1.7** | 2026-11-15 | Objetivo aliado del slot 0 corregido en el adaptador de órdenes (−2; antes −1 = sí mismo, rechazado por Showdown). Sin cambios de obs. |
| **1.6** | 2026-11-14 | ActionSpec v2 (40 ids por slot, mega como flag de cada movimiento; joint 1600) y máscaras desde `valid_orders`. |
| **1.5** | 2026-10-11 | Inferencia bajo Preview Nash congelado (`frozen_preview`). |
| **1.4** | 2026-10-11 | §Q/N por celdas; meseta \(N_{\rm sim}\); CLI sweep; rival/seed/criterio; **tracking** de cambios de N_sim. |
| **1.3** | 2026-10-11 | Matemática matricial, presupuesto, fallos, code map. |
| **1.2** | 2026-10-11 | Dual-audience; Nash-preview. |
| **1.1** | 2026-10-10 | Lectura pedagógica. |
| **1.0** | 2026-10-10 | Contrato SM-MCTS B1. |
