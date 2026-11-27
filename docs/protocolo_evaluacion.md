# Protocolo de evaluación: cómo se afirma que un método es mejor que otro

Este documento fija cómo la tesis afirma "A > B" entre agentes de combate. Se aplica a la
Fase 3 de expectimax y a la escalera de heurísticos.

La herramienta de los análisis es `python -m showdown.inference_sim.confirm_stats`.

## 1. Exploración frente a confirmación

| Etapa | Semillas | Uso | Cómo se reporta |
|---|---|---|---|
| Exploración | 8101–8117, 9101–9117 | Diagnóstico, ajuste de parámetros, elección de variantes | Como **exploratoria**. No respalda afirmaciones de superioridad |
| **Confirmación** | **Bloques nuevos** (C1 = 9201–9217, C2 = 9301–9317) que no se usaron para ajustar nada | **Una** medición por hipótesis, con la versión congelada (commit fijo, configuración por defecto) | **Este es el resultado de la tesis** |

Elegir la mejor variante y medirla con las mismas semillas infla el resultado por
selección. Medir con semillas nuevas da una estimación de la ventaja sin ese sesgo.

## 2. Condiciones fijas

- `run_robust_e2e --teams 300 --procs 28 --restart-ps`: 300 BO3 por semilla, cruzadas.
  Cada par de equipos se juega con los equipos intercambiados y con pools A/B disjuntos.
  17 semillas por bloque = 5100 BO3.
- **Información simétrica:** B validado (por defecto), prior leave-one-out (por defecto) y
  el modo de SP por defecto de cada agente (A en solo, joint y expectimax).
- **Preview:** el propio de cada agente. Desde expectimax v1.7, solo, joint y expectimax
  comparten el preview de tipado + BST, así que las diferencias entre ellos son de combate.
- **Trazabilidad:** commit, comando, semillas y logs por semilla archivados. El reporte de
  `run_robust_e2e` registra `sp_info`, `b_validated` y `preview`.
- **`inf_ok=NO` con la opción C es esperado.** El chequeo de inferencia del reporte comprueba
  que la creencia del lado A contenga los SP reales. Con `meta` la creencia es un valor
  puntual (el reparto más frecuente del prior), que no acierta en ~26 % de los Pokémon, así
  que el chequeo falla por diseño (p. ej. H2: `fail_bo3_inf=6304`, `NEED_FIX=YES`). No
  afecta al juego ni a la validez de la corrida.

## 2b. Separación de equipos: 200 para entrenar, 100 para evaluar

Decisión del 15-11-2026 (checklist DDQN, punto 11). Si un agente RL entrena con los mismos
equipos con que se evalúa, la evaluación mide en parte memorización: en 2M pasos ve cada
equipo rival ~800 veces, puede aprender respuestas específicas por equipo y aprende de
hecho sus SP reales, información que los heurísticos no tienen (prior leave-one-out).

| Grupo | Equipos | Uso |
|---|---|---|
| `train` | 200 (109 combinaciones de especies) | Entrenamiento de todos los agentes RL y previews; base del "meta conocido" (prior de SP y mundos de expectimax) |
| `eval` | 100 (58 combinaciones de especies; 79 únicos por especie + ítem) | Solo la evaluación final (Conclusión). Ningún agente los ve antes |

Cómo se hizo la separación (`showdown/inference_sim/team_split.py`, archivo congelado
`team_split.json`):

- **Por grupos de clones.** El pool repite arquetipos: 300 equipos, pero solo 167
  combinaciones distintas de 6 especies (la más común aparece 17 veces). Todos los equipos
  con las mismas 6 especies van al mismo grupo, así que **ningún equipo de evaluación
  comparte sus 6 especies con uno de entrenamiento**. (Un primer sorteo equipo por equipo
  dejaba ~18 equipos de evaluación con un clon en entrenamiento.)
- **Estratificada por especies.** Entre 500 semillas se eligió la de menor desbalance en la
  frecuencia de las 15 especies más usadas (semilla 237). El criterio solo mira la
  composición de los equipos, no resultados de juego. Diferencias: ≤ 5 puntos en casi
  todas; las mayores, Sneasler (55 % frente a 45 %) e Incineroar (40 % frente a 31 %).

Qué usa cada parte (`PS_TEAM_SPLIT=all|train|eval`):

| Componente | Grupo |
|---|---|
| Entrenamiento RL (`PokeEnvBattleBackend`) y previews | `train` por defecto |
| Prior de SP y mundos de expectimax | `train` siempre que haya separación (`train` o `eval`); con `all`, los 300 + leave-one-out |
| Conclusión (`run_conclusion_e2e`) | `eval`: snapshot de 78 equipos (los únicos, en número par) |
| `run_robust_e2e` | `all` por defecto (como se midieron los baselines); `--team-split train|eval` |

Consecuencias:

- **La escalera de heurísticos ya confirmada (§5.1) se midió con los 300 equipos.** Sigue
  siendo válida: los heurísticos no aprenden, así que no hay memorización. En la Conclusión
  todos los agentes, heurísticos incluidos, se miden sobre `eval` con el prior de `train`.
- **Medida de memorización.** La diferencia entre la tasa de un agente RL en `train` y en
  `eval` (mismas condiciones) estima cuánto de su rendimiento depende de equipos vistos.
- **Validación reservada (16-11-2026).** Con `RL_HOLDOUT_VAL=1`, `train` se divide en `fit`
  (170, entrenamiento) y `val` (30, elección del checkpoint), con el mismo método (grupos de
  clones, estratificación). Motivo: el primer mini kick-off de DDQN ganó 84 % con equipos
  `train` y 50 % con `eval`, y elegir el checkpoint con equipos de entrenamiento premia esa
  memorización. `eval` no cambia. Ver [`memorizacion_equipos.md`](memorizacion_equipos.md).
- El vocabulario de identidad del preview (especies, ítems, movimientos) se construyó con
  los 300 equipos. Solo define índices, no estadísticas; una especie que solo aparece en
  `eval` tiene un embedding sin entrenar.

## 3. Pruebas estadísticas

| Pregunta | Diseño | Prueba | Herramienta |
|---|---|---|---|
| ¿A > B en enfrentamiento directo? | A contra B | **Binomial exacta bilateral** contra 50 %, con **IC 95 % por *cluster bootstrap***. El clúster es (semilla, par de equipos), porque las dos BO3 cruzadas de un par no son independientes | `confirm_stats binom` |
| ¿La versión A' mejora a la A? (ambas contra el mismo rival) | Pareado: mismas semillas y series | **McNemar exacta** sobre las series en que una gana y la otra pierde | `confirm_stats mcnemar` |
| Varias afirmaciones a la vez | Familia de hipótesis | **Holm-Bonferroni**, α familiar = 0,05 | `confirm_stats holm` |

**Regla de decisión para afirmar "A > B"**:

1. el p ajustado por Holm es ≤ 0,05;
2. el IC 95 % por *cluster bootstrap* excluye el 50 % y queda del lado de A;
3. la dirección se repite en cada bloque de confirmación medido (consistencia).

Si falla cualquiera de las tres, se reporta "sin diferencia demostrada", aunque la tasa
observada esté por encima de 50 %.

**Potencia.** Con 5100 BO3, el margen de error a ciertas tasas (cerca de 50 %) es de
±1,37 puntos al 95 %. Para efectos de alrededor de 1,5 puntos conviene usar dos bloques
(10 200 BO3). En el diseño pareado (McNemar) se detectan efectos menores, de ~0,5 puntos,
porque la mayoría de las series salen iguales.

## 3b. Tamaño del efecto: dos niveles de afirmación (todos los agentes)

Que una diferencia sea significativa no implica que sea relevante. Toda afirmación lleva su
tamaño de efecto y se clasifica en uno de dos niveles, **declarados antes de medir**:

| Nivel | Criterio | Frase en la tesis |
|---|---|---|
| **Superioridad estadística** | Regla de la §3: Holm ≤ 0,05, IC 95 % que excluye el 50 % y consistencia entre bloques | "A supera a B de forma significativa, con ventaja pequeña (+x puntos)" |
| **Superioridad práctica** | Lo anterior **y** límite inferior del IC 95 % **≥ 55 %** de victorias en BO3 (≈ +35 Elo) | "A es claramente superior a B" |

El umbral de superioridad práctica es **provisional (55 %)**. Puede subirse (57–60 %) a
medida que avance la tesis, pero siempre **antes** de medir la comparación que se quiere
afirmar. Cualquier cambio se anota en el registro de cambios del final, con su fecha.

Equivalencia aproximada entre la tasa de victorias en BO3 y la diferencia de Elo
(400 · log₁₀(p/(1−p))): 52,5 % ≈ +17, 55 % ≈ +35, 60 % ≈ +70, 65 % ≈ +108. Una BO3 amplifica
la ventaja por partida: un 52 % por juego se convierte en ~53 % por serie.

Con este criterio, la escalera de heurísticos medida en la exploración (+2,2 y +2,8 puntos)
queda en **superioridad estadística con ventaja pequeña**.

**No inferioridad / equivalencia (TOST).** Para afirmar "X no es peor que B por más de δ"
(p. ej. δ = 2 puntos), se usan dos pruebas unilaterales: el IC 90 % debe quedar por encima
de 50 % − δ (no inferioridad), o entre 50 % ± δ (equivalencia). Es útil cuando un agente RL
empata con un heurístico.

## 3c. Agentes con aprendizaje (DDQN, PPO, MCTS, QD)

Los agentes entrenados tienen fuentes de variación que los heurísticos no tienen:

1. **Varias semillas de entrenamiento** (3–5 entrenamientos independientes). La unidad de
   la afirmación es el **método**, no un agente entrenado: se reporta la media entre
   semillas de entrenamiento con su IC entre semillas, además del IC por partidas.
2. **Checkpoint: el último como resultado principal** (decisión del usuario, 17-11-2026). Se
   reporta el checkpoint final de cada semilla (`final.pt`); el mejor según la validación
   reservada (`best.pt`, subgrupo `val` de `train`, no usado para entrenar, §2b) se reporta como
   resultado secundario. Ninguno se elige con los bloques de confirmación (C1/C2) ni con los
   equipos de `eval`. Motivo: con 30 equipos y 50 partidas por rival, `val` no distingue
   checkpoints cercanos (± 7 puntos por celda); en DDQN con regularización eligió el checkpoint
   de 750k, ~8 puntos peor contra `eval` que el final (decimocuarta cola). Usar el final además
   evita que la selección dependa de una medida ruidosa.
3. **Evaluación fuera de distribución.** Además del rival con el que entrenó, se evalúa
   contra rivales y equipos no vistos, para descartar que solo explote a su rival de
   entrenamiento. Los equipos no vistos son el grupo `eval` (§2b); la diferencia entre
   `train` y `eval` mide la memorización.
4. **Preview frente a combate.** Los pipelines RL usan su propio preview (p. ej. E = preview
   PPO + DDQN). La ventaja se descompone con el diseño factorial de `--preview-a/b`, como se
   hizo con expectimax.
5. **Información simétrica.** Los rivales heurísticos del entrenamiento y de la evaluación
   van validados (`rl_env_spec` v1.19 y `run_robust_e2e` por defecto).
6. **Rivales variados en el entrenamiento, deterministas en la evaluación.** En el
   entrenamiento, joint y expectimax sortean entre casi-empates (`rl_env_spec` v1.21, costo
   medido ~0); la evaluación usa las versiones deterministas congeladas
   (`docs/diversidad_baselines.md`).

## 4. Hipótesis pre-registradas (fijadas el 15-11-2026, antes de medir)

**Versiones congeladas (15-11-2026):** greedy_solo y greedy_joint v1.8 (opción C por defecto) y expectimax v1.9 (cambios voluntarios al 25 %, mundos bayesianos, modo `none`).

**Versión congelada de expectimax para la Fase 3:** v1.8 + los cambios que pasen la
exploración (cambios voluntarios al 25 % y lo que salga de los bloques 1–3). Se congela como
**v1.9 por defecto** en un commit antes de la confirmación.

**Familia principal** (escalera; Holm con m = 3):

| Id | Hipótesis | Comparación | Bloques |
|---|---|---|---|
| H1 | expectimax v1.9 > greedy_joint | A = expectimax, B = greedy_joint | C1 + C2 |
| H2 | greedy_joint > greedy_solo | A = greedy_joint, B = greedy_solo | C1 |
| H3 | expectimax v1.9 > greedy_solo | A = expectimax, B = greedy_solo | C1 |

**Hipótesis secundarias** (no entran en la familia; se reportan con su p sin ajustar y se
etiquetan como secundarias):

| Id | Hipótesis | Prueba |
|---|---|---|
| S1 | v1.9 mejora a v1.8 contra joint (efecto de la Fase 2) | McNemar en C1 |
| S2 | el oráculo de SP no mejora a v1.9 (valor de la información) | McNemar en C1 |
| S3 | v1.9 > random (control de cordura) | binomial en C1 |

### 4.1 Hipótesis de MCTS+PUCT v3 (agente G), fijadas el 20-11-2026 antes de medir

**Motivo.** En exploración (semillas 9101–9114, 4200 BO3), G v3.0 (modelo S3, fase 3a)
ganó 57,2 % contra greedy_joint, frente a 37,1 % de v2 en las mismas semillas
(`mcts_puct_diagnostico.md` §6). Decisión del usuario: confirmar antes de seguir con 3b.

**Versión congelada:** G v3.1 por defecto (código del commit `bb9ef00`: combate idéntico a
v3.0 / `6f85aa9`, más el preview greedy por defecto; 200 sims, `MCTS_PUCT_VERSION=3`,
`MCTS_SEED=0`). Rivales: solo los **baselines** (decisión del usuario: ningún agente RL),
random, greedy_solo y greedy_joint v1.8 y expectimax v1.9, los congelados de §4. La cola
aborta si el código de `mcts_puct`, `greedy_joint`, `greedy_solo` o `expectimax_bayesian`
cambió desde `bb9ef00`.

**Bloques nuevos:** C3 = 9401–9417 y C4 = 9501–9517 (no usados antes por ningún agente).

**Preview:** G juega con el preview greedy (tipado + BST), el mismo que solo, joint y
expectimax, para que la comparación sea de combate como en H1–H3. Desde v3.1 es el preview
por defecto de G (decisión del usuario, 20-11-2026: `team_preview_nash` nunca se entrenó;
no se usa el preview PPO para no mezclar componentes RL).

**Familia de G** (Holm con m = 3; binomial exacta + IC por *cluster bootstrap*; regla de
la §3, consistencia en C3 y C4):

| Id | Hipótesis | Comparación | Bloques |
|---|---|---|---|
| G1 | G v3.1 > greedy_joint | A = mcts_puct, B = greedy_joint | C3 + C4 |
| G2 | G v3.1 > expectimax v1.9 | A = mcts_puct, B = expectimax_bayesian | C3 + C4 |
| G3 | G v3.1 > greedy_solo | A = mcts_puct, B = greedy_solo | C3 + C4 |

Nivel práctico (§3b): además, límite inferior del IC 95 % ≥ 55 %.

**Control de cordura (fuera de Holm):** G4 = G v3.1 > random, binomial en C3 (como S3).

~~**Secundaria GS1**~~ (pipeline G con preview Nash contra joint en C3): **eliminada el
20-11-2026, antes de medir nada en C3/C4.** No existe un checkpoint entrenado de
`team_preview_nash` (solo el de un smoke de tests: 1 actualización, etiquetas *mock*), y el
clon de medición ni siquiera lo tiene, así que G caía al orden fijo `/team 1234`; GS1 habría
medido ese orden, no un preview Nash.

### 4.2 Hipótesis de MCTS+PUCT v4 (agente G, fase 3b), fijadas el 20-11-2026 antes de medir

**Motivo.** En exploración (9101–9117, 5100 BO3) v4 ganó 60,3 % contra joint frente a 57,5 % de
v3.1 en las mismas series (McNemar 745 frente a 606, p = 0,0002; `mcts_puct_diagnostico.md`
§6.3). Decisión del usuario: confirmar antes de pasar v4 a ser el valor por defecto.

**Versión congelada:** código del commit `0df75cb` (v4 = `MCTS_PUCT_VERSION=4`; v3.1 =
`MCTS_PUCT_VERSION=3`; 200 sims, `MCTS_SEED=0`). Solo **baselines** como rivales: random,
greedy_solo, greedy_joint v1.8, expectimax v1.9. La cola aborta si cambió el código de
`mcts_puct`, `team_preview_nash`, `frozen_preview` o de los baselines desde `0df75cb`.

**Preview (igual para v4 y v3.1):** lo decide de forma automática la regla del diagnóstico §7
sobre el resultado de la cola del preview Nash (`nash_decision`): si se adopta una semilla del
preview Nash, su `final.pt`; si no, el preview greedy. Se registra cuál se usó.

**Bloques nuevos:** C5 = 9601–9617 y C6 = 9701–9717 (no usados antes).

**Familia de v4** (Holm con m = 4; regla de la §3, consistencia en C5 y C6):

| Id | Hipótesis | Prueba | Bloques |
|---|---|---|---|
| V1 | v4 > greedy_joint | binomial + IC cluster | C5 + C6 |
| V2 | v4 > expectimax v1.9 | binomial + IC cluster | C5 + C6 |
| V3 | v4 > greedy_solo | binomial + IC cluster | C5 + C6 |
| V4 | v4 > v3.1 (ambos contra greedy_joint, mismas series) | McNemar exacta | C5 + C6 |

Nivel práctico (§3b) para V1–V3. **Control (fuera de Holm):** V5 = v4 > random en C5.
**Cambio de valor por defecto:** v4 pasa a ser el default de G solo si V4 se confirma (y
V1–V3 se mantienen).

### 4.3 Hipótesis de MCTS+PUCT v5 (agente G, fase 3c), fijadas el 21-11-2026 antes de medir

**Motivo.** En exploración (9101–9117, 5100 BO3) v5 ganó 64,2 % contra joint frente a 60,3 % de
v4 en las mismas series (McNemar 737 frente a 533, p = 10⁻⁸; `mcts_puct_diagnostico.md` §6.5).
v6 (terreno) no mejoró a v5 (§6.6), así que se confirma v5. Orden acordado con el usuario.

**Versión congelada:** código del commit `cce1356` (v5 = `MCTS_PUCT_VERSION=5`; v4 =
`MCTS_PUCT_VERSION=4`; 200 sims, `MCTS_SEED=0`; preview greedy). Solo **baselines** como rivales:
random, greedy_solo, greedy_joint v1.8, expectimax v1.9. La cola aborta si cambió el código de
`mcts_puct` o de los baselines desde `cce1356`.

**Bloques nuevos:** C7 = 9801–9817 y C8 = 9901–9917 (no usados antes).

**Familia de v5** (Holm con m = 4; regla de la §3, consistencia en C7 y C8):

| Id | Hipótesis | Prueba | Bloques |
|---|---|---|---|
| W1 | v5 > greedy_joint | binomial + IC cluster | C7 + C8 |
| W2 | v5 > expectimax v1.9 | binomial + IC cluster | C7 + C8 |
| W3 | v5 > greedy_solo | binomial + IC cluster | C7 + C8 |
| W4 | v5 > v4 (ambos contra greedy_joint, mismas series) | McNemar exacta | C7 + C8 |

Nivel práctico (§3b) para W1–W3. **Control (fuera de Holm):** W5 = v5 > random en C7.
**Cambio de valor por defecto:** v5 pasa a ser el default de G solo si W4 se confirma (y
W1–W3 se mantienen).

### 4.4 Preview por búsqueda para G v5, fijado el 21-11-2026 antes de medir

**Motivo.** v5 quedó confirmada (§5.4). El preview greedy no elige leads; el preview por
búsqueda (`MCTS_PREVIEW=search`, `mcts_puct_spec.md` §4.6) simula los primeros 3 turnos de
cada formación. Detalle también en `mcts_puct_diagnostico.md` §8.

**Versión congelada:** commit `3269711` (G v5, 200 sims, `MCTS_SEED=0`; preview por búsqueda
con memo por proceso, `MCTS_PREVIEW_DEADLINE=90` bajo carga de 28 procesos). Solo **baselines**
como rivales. La cola aborta si cambia el código de `mcts_puct` o de los baselines.

**Exploración** (9101–9117 × 300 BO3, contra greedy_joint, mismas series):

| Config | Preview |
|---|---|
| S0 | greedy (`--preview-a greedy`, el actual) |
| S1 | búsqueda, decisión `mean` |
| S2 | búsqueda, decisión `maximin` |

Una variante es **elegible** si tiene 0 errores de serie y fallbacks del preview (deadline o
fallo → greedy) en ≤ 5 % de las series. Se elige la elegible con mayor diferencia McNemar
frente a S0 y pasa a confirmación **solo si** la diferencia es > 0 y p < 0,05. Si no, G se
queda con el preview greedy, se documenta y la cola termina.

**Confirmación** (bloques nuevos C9 = 10001–10017 y C10 = 10101–10117; variante elegida):

| Id | Hipótesis | Prueba |
|---|---|---|
| X1 | preview por búsqueda > greedy, ambos con G v5 contra greedy_joint, mismas series | McNemar exacta, C9 + C10 |
| X2 | preview por búsqueda > greedy, ambos con G v5 contra expectimax v1.9, mismas series | McNemar exacta, C9 + C10 |

Holm con m = 2 y consistencia de signo en C9 y C10. **Cambio de valor por defecto:** el
preview por búsqueda pasa a ser el de G solo si X1 se confirma y X2 no es significativamente
negativa (p Holm < 0,05 con diferencia < 0). Se reportan además las tasas binomiales de cada
brazo y los fallbacks del preview. Cola: `scripts/server/mcts_g_search_preview.sh`.

### 4.5 Fase 3e (objetos) para G, fijado el 22-11-2026 antes de medir

Candidato: **G v7** = S3 fase 3e (3c + objetos con efecto dinámico; diagnóstico §10,
`MCTS_PUCT_VERSION=7`), frente al default **v5.2** (combate 3c, preview por búsqueda `mean`, SP
del rival por prior meta). Ambos con el preview y el `sp_info` por defecto del código congelado;
la fase de S3 del preview sigue a la versión (3c en v5.2, 3e en v7). Cola
`scripts/server/mcts_g_v7.sh`; código congelado en el commit del pre-registro.

- **Exploración** (9101–9117 × 300, contra greedy_joint, mismas series): E0 = v5.2, E1 = v7. Pasa a
  confirmación si E1 tiene 0 errores de serie, fallbacks del preview ≤ 5 % de las series y McNemar
  E1 frente a E0 con diferencia > 0 y p < 0,05. Si no, G se queda en v5.2.
- **Confirmación** (bloques nuevos C11 = 10201–10217 y C12 = 10301–10317, × 300 BO3):
  Z1 v7 > greedy_joint, Z2 v7 > expectimax v1.9, Z3 v7 > greedy_solo (binomial, IC bootstrap por
  semilla), Z4 v7 > v5.2 (McNemar, ambos contra joint, mismas series). Holm m = 4.
- **Consecuencia:** si Z4 se confirma (p Holm < 0,05) y Z1–Z3 mantienen superioridad, v7 pasa a
  ser el default de G. Se reportan por bloque (C11 / C12), el tiempo por semilla y las anomalías.

### 4.6 Agente H 2.0, fijado el 22-11-2026 antes de medir

Candidato: **H 2.0** (spec de H v2.0 §4b): SM-MCTS/Nash open-loop con el modelo S3 fase 3c de G,
preview por búsqueda de G y SP por prior meta, 200 sims. Base: **H 1.7** (SM-MCTS B1 con S2, 50 sims)
con el mismo preview y SP. Solo baselines; G 5.2 entra como referencia contra el mismo rival (H no
juega contra G). Cola `scripts/server/mcts_h_v2.sh`; código congelado en el commit del pre-registro.

- **Exploración** (9101–9117 × 300, contra greedy_joint, mismas series): N1 = H 1.7, N2 = H 2.0. Pasa a
  confirmación si N2 tiene 0 errores de serie, fallbacks del preview ≤ 5 % y McNemar N2 frente a N1
  con diferencia > 0 y p < 0,05. Descriptivo: N2 frente a G 5.2 (E0 de la cola 3e, mismas series).
- **Confirmación** (bloques nuevos C13 = 10401–10417 y C14 = 10501–10517, × 300 BO3): W1 H 2.0 >
  greedy_joint, W2 H 2.0 > expectimax v1.9, W3 H 2.0 > greedy_solo (binomial, IC bootstrap por
  semilla); W4 H 2.0 frente a G 5.2 (McNemar **bilateral**, ambos contra joint, mismas series; G 5.2 se
  vuelve a medir en C13/C14). Holm m = 4.
- **Consecuencia:** si W1–W3 se confirman, H 2.0 es el H de la Conclusión. W4 responde la pregunta de
  la tesis "¿el backup de Nash mejora al open-loop con media?" en cualquier dirección; si no se
  rechaza, se reporta como "sin diferencia detectable" con su IC. Se reportan por bloque, el tiempo
  por semilla y las anomalías.
- **Desviación (22-11-2026, 20:45, decidida por el autor):** N1 (H 1.7) se cortó tras 4 de 17 semillas
  (130 / 1200 = 10,8 %, ~860 s/semilla y subiendo, 5431 fallbacks del adaptador por órdenes ilegales de
  la búsqueda v1) frente a N2 (H 2.0) 62,1 % en las 17. La regla de paso se aplica sobre las series
  emparejadas de esas 4 semillas (9101–9104). No afecta a la confirmación (W1–W4), que se mide completa.

### 4.7 Agente H 2.1, fijado el 23-11-2026 antes de medir

Candidatos (spec de H §4c, tras el diagnóstico del paso 0 en §4b): H1 matriz chica (8/6/4), H2 decisión
por la fila más visitada, H3 = H1 + H2, H4 RM+ con arranque en caliente y 350 sims (≈ tiempo de H 2.0).
Base: **H 2.0** (confirmado en §5.7). Solo baselines; G 5.2 como referencia descriptiva contra el mismo
rival. Cola `scripts/server/mcts_h_v21.sh`; código congelado en el commit del pre-registro.

- **Exploración** (9101–9117 × 300, contra greedy_joint, mismas series): H0 = H 2.0 se toma de N2 de
  §4.6 (mismo código por defecto, verificado con test de regresión); H1–H4 se miden. **Regla:** entre los
  brazos con 0 errores de serie, fallbacks del preview ≤ 5 % y McNemar frente a H0 con diferencia > 0 y
  p < 0,05, gana el de mayor diferencia (empate: menor tiempo por semilla). Si ninguno pasa, H se queda en
  2.0 y no hay confirmación. Exploratorio: sin corrección por los cuatro brazos; la confirmación lo cubre.
- **Confirmación** (bloques nuevos C15 = 10601–10617 y C16 = 10701–10717, × 300 BO3) del brazo ganador
  (H 2.1): V1 H 2.1 > greedy_joint, V2 H 2.1 > expectimax v1.9, V3 H 2.1 > greedy_solo (binomial, IC
  bootstrap por semilla); V4 H 2.1 > H 2.0 (McNemar, contra joint, mismas series; H 2.0 se vuelve a medir
  en C15/C16; exige diferencia > 0). Holm m = 4. Descriptivo: H 2.1 frente a G 5.2 (McNemar, contra joint).
- **Consecuencia:** si V1–V4 se confirman, H 2.1 pasa a ser el default del player y el H de la
  Conclusión. Si V4 no se confirma, H se queda en 2.0 (como G en §5.6). Se reportan por bloque, el tiempo
  por semilla y las anomalías.

### 4.8 Agente H 2.2, exploración fijada el 24-11-2026 antes de medir

Candidatos (spec de H §4d): B1 matriz 6/4/3, B2 H3 + RM+ con 200 sims, B3 backup mixto
(½·valor de Nash + ½·media del nodo), todos sobre **H3** (matriz 8/6/4 + decisión por visitas). Se fija
mientras la confirmación de H 2.1 (§4.7) aún mide, con la autorización del usuario; si H3 no se
confirmara, la exploración queda solo como descriptiva. Solo baselines. Cola
`scripts/server/mcts_h_v22.sh` (espera al paso A; reglas en `scripts/server/h22_rules.py`); código
congelado en el commit del pre-registro.

- **Exploración** (9101–9117 × 300, contra greedy_joint, mismas series): R = H3 se toma de la exploración
  de §4.7 (mismo código, verificado con test de regresión); B1–B3 se miden.
  - **Superioridad (B1, B3):** 0 errores de serie, fallbacks del preview ≤ 5 % y McNemar frente a R con
    diferencia > 0 y p < 0,05; gana la mayor diferencia (empate: menor tiempo por semilla).
  - **No inferioridad (B2):** 0 errores, fallbacks ≤ 5 %, cota inferior unilateral 95 % de la diferencia
    pareada frente a R > −1,5 pts (aproximación normal sobre las series emparejadas) y tiempo por semilla
    ≤ 85 % del de R.
  - Si nada pasa, H 2.2 = H 2.1 y no hay confirmación. Exploratorio: sin corrección por los tres brazos.
- **Confirmación:** no se encola con esta exploración. Si algo pasa, se pre-registra aparte (bloques nuevos
  desde 10801, contra joint, expectimax y solo, más la comparación pareada frente a H 2.1) y se encola con
  la aprobación del usuario.

### 4.9 Confirmación de H 2.2 = B2, fijada el 25-11-2026 antes de medir

Candidato (spec de H §4e): **B2 = H 2.1 + RM+ con arranque en caliente y 200 sims**. En la exploración
(§4.8) pasó la no inferioridad frente a H3: −0,04 pts, cota inferior 95 % −1,09, con el 57 % del tiempo.
Base: **H 2.1** (confirmada en §5.8). Solo baselines; G 5.2 como referencia descriptiva. Cola
`scripts/server/mcts_h_v22c.sh`, estadística en `scripts/server/h22c_stats.py`; código congelado en el
commit del pre-registro.

- **Bloques nuevos:** C17 = 10801–10817 y C18 = 10901–10917, × 300 BO3. Brazos: B2 contra joint,
  expectimax y solo; H 2.1 contra joint (pareja de U4; se mide en la misma cola, no se reutiliza); G 5.2
  contra joint (descriptivo, mismo coste que B2).
- **U1–U3:** B2 > greedy_joint, > expectimax v1.9, > greedy_solo (binomial, IC bootstrap por clúster).
- **U4 (no inferioridad):** B2 frente a H 2.1 contra joint, McNemar sobre las mismas series. H0: la
  diferencia es ≤ −1,5 pts; p unilateral por aproximación normal de la diferencia pareada. Se reporta la
  cota inferior unilateral 95 %, que debe quedar por encima de −1,5 pts.
- **Holm m = 4** sobre U1–U4. **Condición de tiempo:** s/semilla de B2 contra joint ≤ 85 % del de
  H 2.1 en la misma cola.
- **Descriptivo:** B2 frente a G 5.2 contra joint (McNemar), la comparación a igual coste.
- **Consecuencia:** si U1–U4 se rechazan y se cumple el tiempo, B2 pasa a ser el default del player como
  **H 2.2** y el H de la Conclusión. Si no, H se queda en 2.1. El margen −1,5 pts es el de la
  exploración, fijado antes de ver sus datos; equivale a un tercio de la ganancia de H 2.1 sobre H 2.0.

### 4.10 Hipótesis a priori de la familia QD 2.0 (I, J, K), fijada el 26-11-2026 antes de medir

Se fija antes de que existan resultados: la semilla de PPO v2 todavía se está entrenando y las colas
`qd_i2` y `qd_jk2` aún no arrancan. Es una **expectativa del investigador**, no una hipótesis
confirmatoria. No cambia ninguna prueba. Sirve para que la Conclusión diga honestamente si los resultados
coincidieron con lo esperado o sorprendieron. Las confirmaciones de I, J y K se pre-registran aparte,
después de sus exploraciones.

**Referencias.** PPO v2 de arranque es la semilla elegida por `val`. El **control** es PPO v2 + 1M pasos
con el mismo preview, mosaico y equipo; es la comparación justa a igual presupuesto. Se mide la tasa de
victorias contra las 3 heurísticas (greedy_solo, greedy_joint, expectimax) en el split eval.

| Agente | vs PPO v2 de arranque | vs control | P(supera al control), subjetiva |
|---|---|---|---|
| I 2.0 (QD-PPO) | mejor, por el entrenamiento extra | empate (±2 pts) | ~30 % |
| J 2.0 (PGA-ME) | parecido o algo peor | peor | ~10 % |
| K 2.0 (PPGA) | parecido o algo mejor | algo peor | ~15 % |

**Ranking esperado en rendimiento contra baselines: I > K > J.** En calidad-diversidad (QD-score,
cobertura) se espera **K ≥ I > J**.

**Razones:**
1. **Objetivo distinto.** QD reparte el presupuesto entre celdas. El control lo dedica entero a ganar
   contra el mismo mosaico. A igual presupuesto, la literatura QD rara vez supera a un optimizador puro en
   rendimiento; gana en QD-score y cobertura.
2. **I.** Cada hijo recibe 2 actualizaciones de PPO desde una élite, lo que lo acerca a un
   *population-based training*. La selección pareada y la reevaluación del top-10 en `val` evitan quedarse
   con un agente que solo tuvo suerte en pocas partidas, lo que podría dar una ventaja pequeña. Pero el
   presupuesto se reparte entre hijos y evaluaciones.
3. **J.** Depende de un crítico Q sobre 1600 acciones conjuntas, en un juego estocástico con información
   parcial, entrenado con ~1M de transiciones. Un crítico mal calibrado desvía el gradiente de política, y
   las mutaciones Iso+LineDD sobre redes grandes suelen dañarlas. Se espera que su mejor élite quede cerca
   del arranque.
4. **K.** Es el método QD más fuerte de los tres en la literatura (rendimiento cercano a PPO en Brax). Pero
   aquí corre con N1 = N2 = 3 en lugar de 10, y su avance optimiza una recompensa combinada con los BD
   según xNES, no solo ganar. Se espera una mejora pequeña sobre el arranque y quedar por debajo del
   control. Es el mejor candidato a QD-score porque su archivo suave y xNES están diseñados para cubrir el
   mapa de comportamientos.
5. **Ruido.** Con 200 partidas por rival en la exploración solo se distinguen diferencias de ~3–4 pts.
   Una diferencia de 1–2 pts frente al control necesita la confirmación pre-registrada.

**Escenario a favor de QD.** Si PPO v2 se estanca en un estilo único y algún rival, en particular
expectimax, lo castiga, una élite con otro estilo podría rendir más contra ese rival. Por eso los
resultados se leen **por rival**, no solo como el promedio de las 3 heurísticas.

**Resultado de la exploración (26-11-2026, qd_ppo_spec §I 2.0):** contra eval, control 81,0 %, I 80,8 %, K 78,3 % y J 81,8 %. I ≈ control y K algo peor coinciden con lo esperado; J por encima del control no, aunque todo cae dentro del ruido (± 4,5 pts por diferencia). En QD-score salió K > J > I (se esperaba K ≥ I > J).

### 4.11 F1 de la familia QD 2.0: despliegue y confirmación, fijada el 26-11-2026 antes de medir

Decisión del usuario: primera fase de la propuesta FODA, medir bien antes de cualquier mejora. Parte de la
exploración (§4.10, qd_ppo_spec §I 2.0): ningún QD se distinguió del control, y el despliegue por `val`
con 300 partidas inflaba al elegido. Cola `scripts/server/qd_f1.sh`, código en
`showdown/rl/qd_ppo/confirm_f1.py`, congelado en el commit de este pre-registro.

- **1a. Despliegue (selección, no prueba).** Las 10 élites con mejor estimación de cada archivo (I, J, K)
  juegan los mismos **1000 enfrentamientos `val`** (334 por heurístico, bloque de semillas 30001). Se
  despliega la de más victorias (desempate: retorno medio). Esto reemplaza la reevaluación de 300
  partidas. `eval` no interviene.
- **1b. Partidas de la confirmación.** I, J y K desplegados, el **control** (PPO v2 s1 + 1M,
  `qd_i2/control/final.pt`) y **PPO v2 s1** (arranque, solo descriptivo) juegan los mismos **4000
  enfrentamientos `eval` por heurístico** (greedy_solo, greedy_joint, expectimax; 12 000 por agente;
  bloque de semillas 40001, nuevo). Un enfrentamiento fija el rival y los equipos de los dos lados; el azar
  del simulador es propio de cada partida. Las partidas caídas se excluyen del par. El instrumento es el
  entorno vectorizado de los entrenadores QD (`play_matchups`), con el preview v2.1b de PPO v2 y el equipo
  fijo, en modo determinista. No es `policy_diag`, así que las cifras absolutas pueden diferir algo de la
  exploración; las pruebas son internas a esta cola.
- **N_I, N_J, N_K (no inferioridad):** agente frente al control sobre los 12 000 pares, diferencia
  pareada de victorias. H0: d ≤ −1,5 pts; p unilateral por aproximación normal (equivale a McNemar con n
  grande). Se reporta la cota inferior unilateral 95 %. El margen es el de H 2.2 (decisión del usuario).
- **S_J (superioridad):** J > control solo contra expectimax (4000 pares), H0: d ≤ 0. Prueba el indicio
  de la exploración (+5,5 pts, 77,5 frente a 72,0).
- **Holm m = 4** sobre N_I, N_J, N_K y S_J, α = 0,05.
- **Potencia (planificación):** con ~30 % de pares discordantes, el error estándar de d es ≈ 0,5 pts con
  12 000 pares. Si la diferencia real es 0, la probabilidad de rechazar N con el umbral más estricto de
  Holm (0,0125) es ≈ 0,8.
- **Descriptivo:** tasas por rival con IC de Wilson, diferencias pareadas por rival frente al control y
  frente al arranque, caídas. La diversidad (cobertura y QD-score de §4.10) se reporta como resultado
  descriptivo de los archivos, sin prueba (opción (c) de la propuesta).
- **Consecuencia:** cada N rechazada permite afirmar que ese QD "rinde al menos como PPO con el mismo
  presupuesto, con un margen de 1,5 pts, y aporta diversidad". Si S_J se rechaza, J castiga mejor a
  expectimax que PPO puro, que es lo que la Fase 2 (selector de élite por rival) quiere explotar. Una N no
  rechazada se reporta tal cual, sin reformular la prueba.

### 4.12 F2 de la familia QD 2.0: selector de élite por rival (exploración), fijada el 26-11-2026 antes de medir

Decisión del usuario: implementar F2–F4 y encolarlas detrás de F1, en una cadena: `qd_f2.sh` → `qd_f3.sh`
→ `qd_f4.sh`, un método a la vez. Código en `showdown/rl/qd_ppo/selector_f2.py` y `team_roles.py`,
congelado en el commit de este pre-registro. Las tres fases son **exploraciones**: no tienen pruebas
confirmatorias. Si alguna da un efecto prometedor, su confirmación se pre-registra aparte.

- **Idea.** El repertorio de Cully et al. (2015) aplicado a combate: el preview muestra el equipo rival,
  y entre élites de estilos distintos se elige la que mejor le va. El selector solo ve el equipo rival
  (especies y conteos de roles a partir de los sets); no ve el algoritmo del rival.
- **Repertorio.** Las 10 élites con más victorias en el despliegue de F1 (`deploy.json`, 1000 partidas
  `val`), de cualquiera de los tres archivos.
- **Datos.** Cada élite juega los mismos 1000 enfrentamientos `fit` por heurístico (bloque 50001), así
  que cualquier elección tiene su resultado observado: es un bandit con información completa.
- **Modelo.** Por élite, una regresión logística con L2. λ se elige por validación cruzada agrupada por
  equipo rival (5 pliegues en `fit`), según las victorias de la élite elegida.
- **Prueba.** En `eval` (bloque 60001, 1000 enfrentamientos por heurístico) juegan las 10 élites, el
  control y PPO v2 s1. El resultado del selector en cada enfrentamiento es el de la élite que eligió. No
  hay sesgo de selección, porque la elección usa solo el equipo rival.
- **Comparación primaria (descriptiva):** selector − control, pareada, con IC 95 % y p unilateral.
  **Secundarias:** selector − mejor élite única (elegida en `fit`), por rival, y el oráculo por equipo,
  que es optimista porque elige con los resultados de `eval`. También se reporta qué élites se eligieron.
- **Expectativa a priori:** ganancia pequeña o nula (0 a +2 pts). Los equipos de `eval` son distintos de
  los de `fit` (separación por clones), y con ~200 equipos de entrenamiento generalizar por especie es
  difícil. Un oráculo muy por encima del selector indicaría que la diversidad sí sirve, pero que el
  contexto no alcanza para elegir.

### 4.13 F3 de la familia QD 2.0: réplica y K con N = 10, fijada el 26-11-2026 antes de medir

- **3a. Réplica desde otra semilla.** PPO v2 **s4** es la segunda mejor en `val` (88,3 %); la regla se
  fija ahora, sin mirar `eval`. Se entrenan desde s4 el control, J 2.0 y K 2.0, con la misma receta y 1M
  pasos. I no se replica: es el más caro y en la exploración quedó igual al control.
- **3b. K 2.0 con N1 = N2 = 10** (las iteraciones del paper) desde s1, con el mismo presupuesto de 1M
  pasos.
- **Despliegue** como en F1: top-10 por archivo, 1000 partidas `val`, bloque 30002.
- **Partidas.** J4, K4, control4, PPO v2 s4, K10, K1 (el desplegado de F1) y el control de s1 juegan los
  mismos 2000 enfrentamientos `eval` por heurístico (bloque 70001).
- **Lectura.** F3a usa las mismas pruebas de F1 sobre s4 (N_J4 y N_K4 frente a control4, S_J4 contra
  expectimax, Holm m = 3), pero como réplica: confirma la dirección y no suma afirmaciones nuevas a la
  Conclusión. F3b compara K10 y K1 frente al control de s1 (no inferioridad, descriptiva), y también
  cobertura y QD-score.
- **Expectativa:** el patrón de F1 se repite (QD ≈ control). K10 queda igual o algo mejor que K1.

### 4.14 F4 de la familia QD 2.0: preview por roles y archivo como pool de rivales, fijada el 26-11-2026 antes de medir

- **4a. Preview QD-PGA v1.7.** Usa descriptores por roles para el equipo fijo (team_preview_qd_pga_spec
  1.7): b1 = leads de soporte / 2 y b2 = físicos entre los 4 que se llevan / 4, calculados con los sets
  reales. Piloto = el control; 64 iteraciones externas; despliegue por reevaluación del top-10. El
  control juega con este preview los mismos enfrentamientos `eval` de F1 (bloque 40001, 4000 por
  heurístico) y se parea con el control de F1, que usa el preview v2.1b.
  Comparación descriptiva: preview por roles − v2.1b, con el mismo combate.
- **4b. El archivo como pool de rivales.** PPO v2 s1 + 1M pasos con el mosaico `pool`: random 10 %,
  greedy_solo 20 %, greedy_joint 25 %, expectimax 25 % y 20 % contra las 5 mejores élites de K en espejo
  (`ppo_v1` con `PPO_FOE_CHECKPOINTS`). Juega los mismos enfrentamientos `eval` de F1 y se parea con el
  control. Comparación descriptiva: pool − control.
- **Expectativa:** 4a ≈ v2.1b, porque el preview pesa poco con el equipo fijo y el v2.1b ya está
  entrenado. 4b ≈ control contra las heurísticas, aunque podría ganar robustez frente a estilos variados.

### 4.15 F2b: selector entrenado con equipos no vistos, fijada el 26-11-2026 antes de medir

Decisión del usuario tras F2 (§5.11). Es una exploración y es la corrección de la falla de diseño de F2:
los datos del selector venían de `fit`, equipos memorizados en los que todas las élites ganaban 95–97 %.

- **Datos del selector:** el mismo repertorio de F2 (10 élites) juega 2000 enfrentamientos `val` por
  heurístico (bloque 50002, 30 equipos rivales). Ningún agente se entrenó con esos equipos. El
  repertorio se eligió en `val` (despliegue de F1), pero `eval` sigue intacto.
- **Prueba:** la parte `eval` de F2 (mismas 3000 partidas, mismo control, mismo arranque). Así la
  diferencia entre F2 y F2b se debe solo a los datos del selector.
- **Modelo y comparaciones:** como en §4.12. Primaria: selector − control. Secundarias: selector − mejor
  élite única y selector F2b − selector F2.
- **Expectativa:** mejor que F2, pero difícilmente por encima del control. Con 30 equipos rivales la
  generalización es limitada.
- **Cola:** `scripts/server/qd_f2b.sh`, después de F4 (un método a la vez).

### 4.16 PPO v3 / DDQN v2: memoria de serie BO3 y máscara del Protect consecutivo, fijada el 27-11-2026 antes de medir

**Origen.** Las series de la web contra personas (`docs/online_web_partidas.md`): PPO v2 perdió 0–2 contra adojs
(lluvia + Trick Room) y contra Sam smithy (sol + Tailwind). Patrones: preview casi fijo, Protect consecutivo con
`protect_counter` ya en la observación y ninguna adaptación entre partidas. Idea del usuario: en un BO3 los dos
repiten equipo, así que lo que el rival hizo en la partida 1 sirve para la 2 y la 3, y el agente puede recordarlo
exacto. Por juego limpio, **PPO y DDQN reciben los mismos cambios**: PPO v3 (ppo_spec 1.18) y DDQN v2 (ddqn_spec
1.26), rl_env_spec v1.35. El código está implementado y probado; **nada está encolado**.

**M1 (descriptiva, ya medida, `scripts/analysis/preview_distribution.py`).** Preview v2.1b (`preview_v21b_s0`),
sorteo T = 1, hoja abierta del rival:

| Grupo | H / ln 90 | Formaciones efectivas | I(rival; formación) / ln 90 | Formaciones argmax distintas | Traer Raichu / Arcanine-H |
|---|---|---|---|---|---|
| `train` (200) | 0,07 | 1,5 | 0,46 | 16 | 0,43 / 0,92 |
| `val` (30) | 0,17 | 2,4 | 0,32 | 9 | 0,39 / 0,93 |
| `eval` (100) | 0,16 | 2,6 | 0,40 | 16 | 0,50 / 0,95 |

Por rival el preview está casi colapsado (el sorteo T = 1 apenas añade variedad), pero sí reacciona al equipo rival
(información mutua 32–46 % del máximo). Hay un sesgo fuerte hacia Arcanine-Hisui (92–95 %). Con el rival
condicionado a 4 Pokémon (los otros 2 como desconocidos), los dos equipos humanos convergen a la misma formación
(Gholdengo, Staraptor, Rillaboom, Arcanine): una entrada nunca vista empuja al preview a una formación por defecto.

**P1. Máscara del Protect consecutivo (sin reentrenar; exploración con regla).** PPO v2 (s0–s4) y DDQN v1
(s0–s4) con y sin la máscara (`RL_MASK_PROTECT_STREAK`), `policy_diag` sobre `eval`, 200 partidas por heurístico
(joint, solo, expectimax; random de control), rivales deterministas, mismas semillas en los dos brazos (pareado).
Prueba: no inferioridad con margen −1,5 pts (McNemar), por familia. Se reporta además cuántas decisiones cambió la
máscara. **Expectativa:** diferencia ≈ 0 (contra los heurísticos la situación es rara; la ganancia esperada es
contra personas). **Regla:** si es no inferior, la máscara se activa en la web para las versiones actuales y entra
en PPO v3 / DDQN v2; si no, se revisa antes de entrenar.

**D1/D2. Diagnósticos por arquetipo (descriptivos, añadidos el 27-11-2026 antes de medir, decisión del usuario).**
Corrección previa: la observación sí tiene el orden de velocidad efectivo y los arquetipos de las personas (lluvia,
sol, Trick Room, Tailwind, Defiant/Competitive) están entre los equipos de entrenamiento (`train`: TR 24 %, Tailwind
68 %, lluvia 17 %, sol 12 %, Defiant/Competitive 38 %). Hipótesis: los heurísticos casi nunca ejecutan esos planes y
por eso el agente no aprende a responder. Desde las evaluaciones sin máscara de P1 (con `--context`),
`scripts/analysis/archetype_diag.py` informa: D1a victorias por arquetipo; D1b fracción de partidas en que el rival
activó Trick Room / Tailwind cuando su equipo lo tiene, y victorias según lo activara o no; D1c uso propio de Tailwind
cuando trae a Staraptor; D2 preview real por arquetipo (traer cada Pokémon, abrir con Intimidate contra
Defiant/Competitive, Arcanine-Hisui contra ≥ 2 atacantes de Tierra/Agua). **Guía de decisión (para proponer al
usuario, no automática):** si los heurísticos activan su Trick Room o Tailwind en menos del 25 % de las partidas en que
lo tienen, se propone un rival de entrenamiento que sí lo ejecute (distinto del `greedy_joint_arq` descartado en la
cola 24); si el preview trae a Arcanine-Hisui en más del 80 % de las partidas contra ≥ 2 atacantes de Tierra/Agua y
con menos victorias que sin él, se propone reentrenar el preview. Muestra de prueba local (12 partidas, no cuenta):
Trick Room activado 0/3, Tailwind 0/9, Tailwind propio 0/7.

**S1. Preview condicionado a la serie (sin reentrenar; exploración con regla).** PPO v2 y DDQN v1 (s0–s2, decisión del usuario del 27-11-2026; antes s0–s4) en series BO3
(`policy_diag --series`, `eval`, 100 series por heurístico, rivales deterministas), con y sin `--preview-series`.
La partida 1 es idéntica en los dos brazos; la comparación es la tasa de victorias en las partidas 2–3. **Regla:**
se adopta solo con +1,5 pts o más y p < 0,05 unilateral. **Expectativa a priori (por M1):** ≤ 0.

**V. Memoria de serie (confirmación).** Dos brazos por familia, mismos rivales, programa y presupuesto:

- **C (control):** versión anterior + entrenamiento adicional en series BO3, con máscara, **sin memoria**.
- **V (v3 / v2):** igual, **con memoria** (arranque en caliente con columnas nuevas en cero).

Así la diferencia V − C se debe solo a la memoria (el entrenamiento adicional y la máscara son comunes). Semillas
de arranque: PPO v2 s0–s2 (`checkpoints/ppo_v2/`), DDQN v1 s0–s2 (`~/iapsvgc_runs/kickoff_ddqn_v2/ddqn_s*`).
Rivales de entrenamiento con sorteo de casi-empates (§3c.6), para que las series no sean copias exactas.

- **Evaluación:** `final.pt` (§3c.2), `policy_diag --series` sobre `eval`, 100 series por heurístico (joint, solo,
  expectimax; random de control), rivales deterministas, bloque de semillas `--seed 40000` (no usado antes).
- **Primaria (V1 PPO, V2 DDQN):** tasa de victorias en las partidas 2–3, V − C, a nivel de método (media de las 3
  semillas, IC entre semillas y por partidas, §3c.1). Superioridad unilateral p < 0,05, Holm m = 2.
- **Secundarias:** series ganadas; no inferioridad en la partida 1 (margen −1,5 pts: la memoria no debe dañar la
  partida 1, donde su bloque es casi cero); variante con `--foe-tie` (descriptiva); V frente a la versión anterior
  sin entrenamiento adicional (descriptiva).
- **Expectativa a priori:** PPO +1 a +3 pts en las partidas 2–3 (probabilidad de confirmar ~35 %); DDQN ≈ 0.
  Límite reconocido: los heurísticos no se adaptan entre partidas, así que la memoria puede aprender "el rival
  repite"; contra personas habría que verlo en la web.

**Decisiones del usuario (27-11-2026, antes de medir):** +1M pasos por semilla; 3 semillas por brazo; diseño
con control (brazos C y V, así V − C se atribuye solo a la memoria); orden P1 → S1 → PPO → DDQN. Mezcla de rivales
del entrenamiento: la del kick-off (`random:150k, greedy_solo:250k, greedy_joint:300k, expectimax_bayesian:300k`)
con el sorteo de casi-empates por defecto. En DDQN el arranque en caliente empieza con ε = 0,05 (el final del
programa) en los dos brazos, para no destruir la política aprendida. P1 usa PPO v2 s0–s4 y DDQN v1 s0–s4; S1,
s0–s2 y sin máscara, para aislar el preview. Las evaluaciones corren en paralelo (`JOBS`, 6 por defecto, un puerto
cada una); los entrenamientos, de uno en uno. Tiempo estimado: ~1,5–2 días (en serie habrían sido 4–5,5). Cola: `scripts/server/v3_queue.sh` (reanudable; informe con
`scripts/analysis/v3_report.py`), **sin encolar hasta el visto bueno del usuario**.

## 5. Plantilla de reporte

| Hipótesis | Victorias / N | Tasa | IC 95 % (cluster) | p | p Holm | Semillas > 50 % | Veredicto |
|---|---|---|---|---|---|---|---|
| H1 | … | … | … | … | … | …/34 | … |

Cada afirmación lleva su alcance: "en Reg M-C, con el pool MetaVGC de 300 equipos e
información simétrica".

### 5.1 Resultados de la confirmación (15-11-2026, commit a6d706c)

| Hipótesis | Victorias / N | Tasa | IC 95 % (cluster) | p | p Holm | Semillas > 50 % | Veredicto |
|---|---|---|---|---|---|---|---|
| H1 expectimax v1.9 > joint v1.8 (C1+C2) | 5119 / 10103 | 50,67 % | [49,82; 51,51] | 0,18 | 0,18 | 23/34 | **Sin diferencia demostrada** (C1 49,71 %, C2 51,62 %: tampoco hay consistencia) |
| H2 joint v1.8 > solo v1.8 (C1) | 2609 / 5054 | 51,62 % | [50,53; 52,74] | 0,022 | 0,044 | 12/17 | **Superioridad estadística, ventaja pequeña (+1,6 pts)**; no práctica (límite inferior < 55 %) |
| H3 expectimax v1.9 > solo v1.8 (C1) | 2677 / 5049 | 53,02 % | [51,83; 54,18] | 1,9·10⁻⁵ | 5,6·10⁻⁵ | 16/17 | **Superioridad estadística, ventaja pequeña (+3,0 pts)**; no práctica |

Secundarias (p sin ajustar, C1):

| Id | Resultado | Lectura |
|---|---|---|
| S1 v1.9 frente a v1.8, ambos contra joint | McNemar +0,04 pts (84 frente a 82 discordantes), p = 0,94 | La ganancia de la Fase 2c (+0,82 pts en exploración) **no se replica** |
| S2 oráculo frente a v1.9, ambos contra joint | McNemar +1,17 pts (384 frente a 326), p = 0,032 | Contra lo esperado, **el oráculo de SP sí aporta algo** (secundaria, sin ajustar) |
| S3 v1.9 frente a random | 99,78 %, 17/17 semillas | Control de cordura correcto |

**Diagnóstico exploratorio (agregado tras ver H1 C1, no confirmatorio):** expectimax v1.9
contra joint con la regla anterior (`--sp-info-b none`) en C1 = **52,22 %** [51,12; 53,36],
15/17 semillas. McNemar frente a H1 C1: +2,33 pts (595 frente a 479), p = 0,0004.

**Réplica en C2** (misma corrida, semillas 9301–9317): contra joint con la regla anterior,
**51,90 %** [50,72; 53,07], 14/17 semillas; McNemar frente a H1 C2: **+0,32 pts** (517 frente
a 501), p = 0,64. El efecto grande de C1 **no se replica**.

| Bloque | expectimax contra joint `none` | contra joint `meta` (H1) | Efecto de C en joint (McNemar) |
|---|---|---|---|
| C1 | 52,22 % | 49,71 % | +2,33 pts (p = 0,0004) |
| C2 | 51,90 % | 51,62 % | +0,32 pts (p = 0,64) |
| C1+C2 | ≈ 52,1 % | 50,67 % | **+1,33 pts** (1112 frente a 980; p = 0,004) |

Lectura (exploratoria): contra el joint con la regla anterior, expectimax gana ~52 % de forma
estable en los dos bloques. La opción C fortalece a joint contra expectimax en ~1,3 pts en
promedio, con mucha variación entre bloques (C1 probablemente sobrestima). En espejo, la
exploración daba +0,48 pts. **C no empeora a joint y lo mejora un poco, más contra
expectimax que contra sí mismo.** Eso, sumado al sesgo de selección (53,5 % en exploración
frente a ~52 % en confirmación), explica que H1 quede sin diferencia demostrada.

**Escalera confirmada:** expectimax > solo y joint > solo (estadística, ventajas pequeñas).
**No confirmada:** expectimax > joint. Ninguna comparación alcanza el umbral práctico.

### 5.2 Resultados de la confirmación de G (20-11-2026, código congelado `bb9ef00`)

Pre-registro: §4.1 (G1–G3 + control G4, solo baselines, preview greedy). Cola
`scripts/server/mcts_g_confirm.sh` (commit de ejecución `65f271e`: solo cambia la caché de
la calculadora, por corrida; ningún valor de daño cambia). 0 anomalías del player. Las
corridas que fallaron por el disco lleno (11:36–11:43, sin reporte) se repitieron completas.

| Hipótesis | Victorias / N | Tasa | IC 95 % (cluster) | p | p Holm | Semillas > 50 % | Veredicto |
|---|---|---|---|---|---|---|---|
| G1 G v3.1 > greedy_joint (C3+C4) | 5710 / 10146 | 56,28 % | [55,44; 57,11] | 1,1·10⁻³⁶ | 2,1·10⁻³⁶ | 34/34 | **Superioridad práctica** (+6,3 pts; límite inferior ≥ 55 %, por 0,44) |
| G2 G v3.1 > expectimax v1.9 (C3+C4) | 5702 / 10137 | 56,25 % | [55,42; 57,10] | 2,4·10⁻³⁶ | 2,4·10⁻³⁶ | 34/34 | **Superioridad práctica** (+6,3 pts; límite inferior ≥ 55 %, por 0,42) |
| G3 G v3.1 > greedy_solo (C3+C4) | 6047 / 10146 | 59,60 % | [58,81; 60,41] | 9,5·10⁻⁸⁴ | 2,8·10⁻⁸³ | 34/34 | **Superioridad práctica** (+9,6 pts) |

Consistencia por bloque (regla §3.3):

| Hipótesis | C3 | C4 |
|---|---|---|
| G1 | 56,57 % [55,43; 57,81], 17/17 | 55,99 % [54,81; 57,15], 17/17 |
| G2 | 56,81 % [55,61; 58,02], 17/17 | 55,69 % [54,59; 56,89], 17/17 |
| G3 | 59,46 % [58,33; 60,57], 17/17 | 59,74 % [58,58; 60,91], 17/17 |

Control G4 (C3): G contra random 5073 / 5080 = 99,86 % [99,76; 99,94], 17/17. Correcto.

Lectura: G v3.1 supera a todos los baselines con ventaja grande y en las 34 semillas de
los dos bloques. Para G1 y G2 el nivel práctico se cumple con el IC combinado (C3+C4),
pero por poco (0,4 puntos); en C4 solo, el límite inferior queda en 54,8 % y 54,6 %. La
frase en la tesis: "G es claramente superior a greedy_joint y a expectimax (+6,3 pts,
IC [55,4; 57,1]), aunque el margen sobre el umbral práctico es estrecho". Alcance: Reg
M-C, pool MetaVGC de 300 equipos (`all`), información simétrica, preview greedy común.

### 5.3 Resultados de la confirmación de G v4 (21-11-2026, código congelado `0df75cb`)

Pre-registro: §4.2. Cola `scripts/server/mcts_g_confirm_v4.sh` (commit de ejecución
`debcda9`; preview greedy por la regla del diagnóstico §7, porque el preview Nash empeoraba
a G). 0 anomalías del player; código congelado verificado al arrancar.

| Hipótesis | Victorias / N | Tasa | IC 95 % (cluster) | p | p Holm | Semillas > 50 % | Veredicto |
|---|---|---|---|---|---|---|---|
| V1 v4 > greedy_joint (C5+C6) | 6360 / 10158 | 62,61 % | [61,80; 63,41] | 6·10⁻¹⁴⁴ | 2·10⁻¹⁴³ | 34/34 | **Superioridad práctica** (+12,6 pts) |
| V2 v4 > expectimax v1.9 (C5+C6) | 6215 / 10158 | 61,18 % | [60,34; 62,11] | 2·10⁻¹¹³ | 5·10⁻¹¹³ | 34/34 | **Superioridad práctica** (+11,2 pts) |
| V3 v4 > greedy_solo (C5+C6) | 6534 / 10162 | 64,30 % | [63,47; 65,08] | 4·10⁻¹⁸⁵ | 1·10⁻¹⁸⁴ | 34/34 | **Superioridad práctica** (+14,3 pts) |
| V4 v4 > v3.1 (McNemar, ambos contra joint) | 1582 frente a 1066 discordantes (10 128 emparejadas) | +5,09 pts | — | 1·10⁻²³ | 1·10⁻²³ | — | **Confirmada** |

Consistencia por bloque: V1 62,98 % / 62,24 %; V2 61,17 % / 61,19 %; V3 63,20 % / 65,40 %;
V4 +5,10 / +5,09 pts (C5 / C6), todos con 17/17 semillas > 50 %. Referencia: v3.1 contra joint
en las mismas series 57,53 % [56,73; 58,39]. Control V5: v4 contra random 99,84 %, 17/17.

**Consecuencia (regla del pre-registro):** v4 pasa a ser el valor por defecto de G
(`MCTS_PUCT_VERSION=4`). Ahora todos los niveles prácticos se cumplen con holgura (límites
inferiores ≥ 60,3 %), a diferencia de v3.1, que superaba el 55 % por solo 0,4 puntos.

### 5.4 Resultados de la confirmación de G v5 (21-11-2026, código congelado `cce1356`)

Pre-registro: §4.3. Cola `scripts/server/mcts_g_confirm_v5.sh` (commit de ejecución
`79b0cb1`; preview greedy; 07:45–14:18). Tests 38/38, smoke sin errores, **0 anomalías del
player** en toda la cola; código congelado verificado al arrancar.

| Hipótesis | Victorias / N | Tasa | IC 95 % (cluster) | p | p Holm | Semillas > 50 % | Veredicto |
|---|---|---|---|---|---|---|---|
| W1 v5 > greedy_joint (C7+C8) | 6665 / 10172 | 65,52 % | [64,68; 66,34] | 1·10⁻²¹⁸ | 4·10⁻²¹⁸ | 34/34 | **Superioridad práctica** (+15,5 pts) |
| W2 v5 > expectimax v1.9 (C7+C8) | 6468 / 10170 | 63,60 % | [62,72; 64,44] | 1·10⁻¹⁶⁷ | 3·10⁻¹⁶⁷ | 34/34 | **Superioridad práctica** (+13,6 pts) |
| W3 v5 > greedy_solo (C7+C8) | 6795 / 10172 | 66,80 % | [65,96; 67,57] | 2·10⁻²⁵⁶ | 6·10⁻²⁵⁶ | 34/34 | **Superioridad práctica** (+16,8 pts) |
| W4 v5 > v4 (McNemar, ambos contra joint) | 1535 frente a 1080 discordantes (10 155 emparejadas) | +4,48 pts | — | 6·10⁻¹⁹ | 6·10⁻¹⁹ | — | **Confirmada** |

Consistencia por bloque (C7 / C8): W1 65,32 % / 65,72 %; W2 64,00 % / 63,19 %; W3 66,14 % /
67,47 %; W4 +3,46 pts (p = 1·10⁻⁶) / +5,50 pts (p = 8·10⁻¹⁵); todas con 17/17 semillas > 50 %.
Referencia: v4 contra joint en las mismas series 61,05 % [60,21; 61,91]. Control W5: v5 contra
random 99,82 % [99,71; 99,92], 17/17.

**Consecuencia (regla del pre-registro):** W4 se confirma y W1–W3 se mantienen, así que v5
pasa a ser el valor por defecto de G (`MCTS_PUCT_VERSION=5`). Frente a v4 confirmada (§5.3),
v5 sube +2,9 pts contra joint, +2,4 contra expectimax y +2,5 contra solo, con un costo de
~25 % más de tiempo por decisión.

### 5.5 Resultados de la confirmación del preview por búsqueda de G (22-11-2026, código congelado `3269711`)

Pre-registro: §4.4 (diagnóstico §8). Cola `scripts/server/mcts_g_search_preview.sh` (commit de
ejecución `cf73765`, `MCTS_PREVIEW_DEADLINE=90`; exploración 17:51–20:07, confirmación
20:07–04:16). Variante confirmada: `mean` (selección de la exploración, diagnóstico §8.1).
**0 errores de serie, 0 fallbacks del preview y 0 anomalías** en toda la cola. Bloques C9
(10001–10017) y C10 (10101–10117), × 300 BO3, en pareado (mismas series, solo cambia el preview).

| Hipótesis | Comparación | Discordantes (A mejor / B mejor) | Diferencia | p | p Holm | Por bloque (C9 / C10) | Veredicto |
|---|---|---|---|---|---|---|---|
| X1 búsqueda > greedy, contra greedy_joint | 10 112 series emparejadas | 1979 / 1563 | **+4,11 pts** | 3·10⁻¹² | 6·10⁻¹² | +4,76 (p = 2·10⁻⁸) / +3,47 (p = 2·10⁻⁵) | **Confirmada** |
| X2 búsqueda > greedy, contra expectimax v1.9 | 10 104 series emparejadas | 2011 / 1601 | **+4,06 pts** | 1·10⁻¹¹ | 1·10⁻¹¹ | +4,33 (p = 3·10⁻⁷) / +3,78 (p = 7·10⁻⁶) | **Confirmada** (no solo no empeora: mejora) |

| Brazo (C9+C10) | Victorias / series | Tasa | IC 95 % (cluster) | Semillas > 50 % |
|---|---|---|---|---|
| G v5 + búsqueda contra joint (X1s) | 7007 / 10 152 | **69,02 %** | [68,19; 69,89] | 34/34 |
| G v5 + greedy contra joint (X1g) | 6600 / 10 166 | 64,92 % | [64,11; 65,75] | 34/34 |
| G v5 + búsqueda contra expectimax (X2s) | 6873 / 10 152 | **67,70 %** | [66,86; 68,55] | 34/34 |
| G v5 + greedy contra expectimax (X2g) | 6462 / 10 161 | 63,60 % | [62,75; 64,45] | 34/34 |

**Consecuencia (regla del pre-registro):** X1 se confirma (Holm m = 2, mismo signo en ambos
bloques) y X2 no es negativa (es significativamente positiva), así que el **preview por búsqueda
(`mean`) pasa a ser el preview por defecto de G** (`MCTS_PREVIEW=search`; `MCTS_PREVIEW=greedy`
vuelve al anterior). Frente a v5 con greedy (§5.4), G sube ~+4 pts contra los dos rivales; coste:
~7 s por preview en el servidor (una vez por BO3 gracias al memo) y ~+60 % de tiempo por semilla
en la cola (≈ 240 s frente a ≈ 150 s, con 28 procesos).

### 5.6 Resultados de la confirmación de la fase 3e de G (22-11-2026, código congelado `8efb619`)

Pre-registro §4.5; exploración en diagnóstico §10.1 (+1,90 pts, p = 0,0086, pasó a confirmación).
C11 = 10201–10217 y C12 = 10301–10317, × 300 BO3, terminada 17:33.

| Hipótesis | Victorias / series | Tasa / diferencia | IC 95 % (cluster) | C11 | C12 | p Holm | Resultado |
|---|---|---|---|---|---|---|---|
| Z1 v7 > greedy_joint | 7198 / 10 146 | 70,94 % | [70,08; 71,87] | 71,74 % | 70,15 % | ≈ 0 | confirmada (34/34 semillas) |
| Z2 v7 > expectimax v1.9 | 7046 / 10 146 | 69,45 % | [68,57; 70,27] | 69,29 % | 69,60 % | ≈ 0 | confirmada (34/34) |
| Z3 v7 > greedy_solo | 7426 / 10 148 | 73,18 % | [72,36; 74,03] | 73,07 % | 73,28 % | ≈ 0 | confirmada (34/34) |
| **Z4 v7 > v5.2** (McNemar, contra joint) | 1319 frente a 1243 (10 054 emparejadas) | **+0,76 pts** | — | +1,55 (p = 0,031) | **−0,04 (p = 0,98)** | **0,138** | **no confirmada** |

Referencia: v5.2 contra joint (Zj5) 7116 / 10 148 = 70,12 % [69,26; 70,98]. Tiempo por semilla:
v7 229 s frente a v5.2 219 s contra joint (+4 %). 0 errores de serie y 0 fallbacks del preview en
las 136 corridas.

**Decisión (regla pre-registrada): Z4 no se confirma (p Holm = 0,138) → G se queda en v5.2.** El
efecto de la exploración se replica en C11 pero desaparece en C12; lo más probable es que la ganancia
real de los objetos dinámicos sea menor que 1 pt, por debajo de lo que este diseño puede separar.
v7 queda disponible con `MCTS_PUCT_VERSION=7`. Z1–Z3 confirman de paso que G con objetos sigue
por encima de los tres baselines.

### 5.7 Resultados de la confirmación del agente H 2.0 (23-11-2026, código congelado `c70f0f2`)

Pre-registro §4.6; exploración: H 2.0 62,06 % contra joint, +51,7 pts sobre H 1.7 (p = 3·10⁻¹⁵⁸),
pasó a confirmación. C13 = 10401–10417 y C14 = 10501–10517, × 300 BO3, terminada el 23-11 a las 10:02.

| Hipótesis | Victorias / series | Tasa / diferencia | IC 95 % (cluster) | C13 | C14 | p Holm | Resultado |
|---|---|---|---|---|---|---|---|
| W1 H 2.0 > greedy_joint | 6254 / 10 158 | 61,57 % | [60,71; 62,51] | 61,25 % | 61,88 % | 6·10⁻¹²¹ | confirmada (34/34 semillas) |
| W2 H 2.0 > expectimax v1.9 | 6274 / 10 164 | 61,73 % | [60,79; 62,60] | 61,07 % | 62,38 % | 4·10⁻¹²⁴ | confirmada (34/34) |
| W3 H 2.0 > greedy_solo | 6431 / 10 155 | 63,33 % | [62,44; 64,20] | 63,17 % | 63,49 % | 4·10⁻¹⁶⁰ | confirmada (34/34) |
| **W4 H 2.0 frente a G 5.2** (McNemar bilateral, contra joint) | 1087 frente a 1815 (10 038 emparejadas) | **−7,25 pts** | — | −7,37 (p = 2·10⁻²²) | −7,13 (p = 5·10⁻²¹) | 6·10⁻⁴² | **rechaza H0: G 5.2 es mejor** |

Referencia: G 5.2 contra joint (Gj) 6978 / 10 124 = 68,93 % [68,09; 69,81]. Tiempo por semilla:
H 2.0 378 s (joint), 421 s (expectimax), 368 s (solo); G 5.2 219 s contra joint (H cuesta 1,7×).
0 errores de serie, 0 fallbacks del preview y 0 anomalías en las 136 corridas.

**Decisión (regla pre-registrada): W1–W3 se confirman → H 2.0 es el H de la Conclusión.** W4 responde
la pregunta de la tesis en contra del backup de Nash: con el mismo modelo S3, el mismo preview, el mismo
SP y las mismas 200 sims, el open-loop con media de G gana 7,25 pts más contra joint que el SM-MCTS con
valor de Nash, de forma igual en los dos bloques y como en la exploración (−7,18). Hipótesis no probadas
(spec de H §4b): la matriz de 12 × 8 celdas es demasiado dispersa para 200 sims, y el valor de Nash
supone un rival del peor caso mientras que los baselines son explotables. Una H 2.1 necesitaría un
pre-registro nuevo en semillas 10601+.

### 5.8 Resultados de la confirmación del agente H 2.1 (24-11-2026, código congelado `4db8267`)

Pre-registro §4.7; exploración: gana H3 (matriz 8/6/4 + decisión por visitas), +3,24 pts sobre H 2.0
(p = 1,6·10⁻⁵). C15 = 10601–10617 y C16 = 10701–10717, × 300 BO3, terminada el 24-11 a las 11:28.

| Hipótesis | Victorias / series | Tasa / diferencia | IC 95 % (cluster) | C15 | C16 | p Holm | Resultado |
|---|---|---|---|---|---|---|---|
| V1 H 2.1 > greedy_joint | 6703 / 10 160 | 65,97 % | [65,11; 66,81] | 65,73 % | 66,22 % | 7·10⁻²³¹ | confirmada (34/34 semillas) |
| V2 H 2.1 > expectimax v1.9 | 6664 / 10 151 | 65,65 % | [64,76; 66,51] | 64,85 % | 66,44 % | 2·10⁻²²¹ | confirmada (34/34) |
| V3 H 2.1 > greedy_solo | 6851 / 10 155 | 67,46 % | [66,60; 68,34] | 67,49 % | 67,44 % | 2·10⁻²⁷⁶ | confirmada (34/34) |
| **V4 H 2.1 > H 2.0** (McNemar, contra joint) | 1665 frente a 1213 (10 127 emparejadas) | **+4,46 pts** | — | +4,17 (p = 3·10⁻⁸) | +4,75 (p = 2·10⁻¹⁰) | 4·10⁻¹⁷ | **confirmada** |

Referencias contra joint en las mismas series: H 2.0 (Hj) 6246 / 10 159 = 61,48 %; G 5.2 (Gj)
6995 / 10 126 = 69,08 %. Descriptivo H 2.1 frente a G 5.2: −3,17 pts (1011 frente a 1330, p = 5·10⁻¹¹;
con H 2.0 era −7,25). Tiempo por semilla: H 2.1 373 s (joint), 419 s (expectimax), 361 s (solo); H 2.0
383 s; G 5.2 221 s. 0 errores de serie, 0 fallbacks del preview y 0 anomalías en las 170 corridas.

**Decisión (regla pre-registrada): V1–V4 se confirman → H 2.1 (H3) es el default del player y el H de
la Conclusión.** Las dos correcciones del paso 0 (menos celdas y decidir por visitas) explican más de la
mitad de la distancia con G; el resto sigue a favor del open-loop con media.

**Paso A (descriptivo, 24-11, mismas series C15/C16; spec de H §4c):** G 5.2 contra expectimax 68,57 %
y contra solo 71,51 %. H 2.1 frente a G 5.2: −3,17 pts contra joint, −3,00 contra expectimax
(p = 4·10⁻⁹) y −4,09 contra solo (p = 10⁻¹⁸). La desventaja de H frente a G es de 3–4 pts contra los tres
baselines.

### 5.9 Resultados de la confirmación de H 2.2 = B2 (25-11-2026, código congelado `0f56856`)

Pre-registro §4.9; exploración: B2 (H 2.1 + RM+) no inferior a H3 (−0,04 pts, cota −1,09) en el 57 % del
tiempo. C17 = 10801–10817 y C18 = 10901–10917, × 300 BO3, terminada el 25-11 a las 19:19.

| Hipótesis | Victorias / series | Tasa / diferencia | IC 95 % (cluster) | C17 | C18 | p Holm | Resultado |
|---|---|---|---|---|---|---|---|
| U1 B2 > greedy_joint | 6689 / 10 124 | 66,07 % | [65,15; 66,98] | 66,26 % | 65,88 % | 5·10⁻²³³ | confirmada (34/34 semillas) |
| U2 B2 > expectimax v1.9 | 6616 / 10 149 | 65,19 % | [64,26; 66,11] | 64,75 % | 65,62 % | 2·10⁻²⁰⁸ | confirmada (34/34) |
| U3 B2 > greedy_solo | 6872 / 10 133 | 67,82 % | [66,94; 68,66] | 67,58 % | 68,06 % | 3·10⁻²⁸⁷ | confirmada (34/34) |
| **U4 B2 no inferior a H 2.1** (McNemar, margen −1,5 pts) | 1092 frente a 1092 (10 068 emparejadas) | **+0,00 pts**, cota inferior 95 % −0,76 | — | +0,22 (cota −0,87) | −0,22 (cota −1,29) | 6·10⁻⁴ | **confirmada** |

Condición de tiempo: B2 218 s/semilla frente a H 2.1 385 s contra joint = **57 %** (tope 85 %), cumple.
Referencias contra joint en las mismas series: H 2.1 (Hj) 6701 / 10 153 = 66,00 %; G 5.2 (Gj)
6939 / 10 142 = 68,42 % (231 s/semilla). Descriptivo **B2 frente a G 5.2 al mismo coste: −2,37 pts**
(1144 frente a 1382, p = 2·10⁻⁶; C17 −2,28, C18 −2,46). B2 contra expectimax tardó 288 s y contra
solo 218 s. 0 errores de serie, 0 fallbacks del preview y 0 anomalías en las 171 corridas.

**Decisión (regla pre-registrada): U1–U4 se confirman y se cumple el tiempo → H 2.2 (B2) es el default
del player y el H de la Conclusión.** H 2.2 juega exactamente como H 2.1 (empate 1092–1092 en las
series discordantes) con el 57 % del tiempo, es decir, al coste de G. Con el mismo modelo, preview, SP,
200 sims y el mismo tiempo, el open-loop con media (G 5.2) gana 2,4 pts más contra joint que el
SM-MCTS con valor de Nash (H 2.2): la diferencia entre los dos métodos ya no se explica por tiempo ni
por el resolvedor.

### 5.10 Resultados de F1 de la familia QD 2.0 (26-11-2026, código congelado `bc7f716`)

**Despliegue (1a, 1000 partidas `val`).** Se eligieron I 8,3 (91,4 %), J 7,6 (91,5 %) y K 6,3 (88,5 %).
I y K coinciden con la elección de 300 partidas; J cambia de 7,4 a 7,6.

**Partidas (1b).** 4000 enfrentamientos `eval` por heurístico, 12 000 por agente, 0 caídas. Tasa de
victorias con IC 95 % de Wilson:

| Agente | greedy_solo | greedy_joint | expectimax | **3 heur.** |
|---|---|---|---|---|
| I 2.0 | 83,0 [81,9–84,2] | 84,2 [83,0–85,3] | 71,4 [70,0–72,8] | **79,5 %** [78,8–80,3] |
| J 2.0 | 82,2 [81,0–83,4] | 83,5 [82,3–84,6] | **78,9** [77,6–80,1] | **81,5 %** [80,8–82,2] |
| K 2.0 | 82,4 [81,2–83,6] | 82,6 [81,4–83,8] | 72,1 [70,7–73,4] | **79,0 %** [78,3–79,8] |
| **Control** (PPO v2 s1 + 1M) | **85,8** [84,7–86,9] | **84,9** [83,7–85,9] | 75,9 [74,5–77,2] | **82,2 %** [81,5–82,9] |
| PPO v2 s1 (arranque) | 82,7 [81,5–83,8] | 83,9 [82,7–85,0] | 74,5 [73,1–75,8] | 80,3 % [79,6–81,1] |

**Pruebas (Holm m = 4):**

| Prueba | d (pts) | Cota inf. 95 % | p | p Holm | Resultado |
|---|---|---|---|---|---|
| **S_J**: J > control contra expectimax | **+3,00** | +1,72 | 5,5·10⁻⁵ | 2,2·10⁻⁴ | **Confirmada** |
| N_J: J no inferior al control | −0,64 | −1,31 | 0,017 | 0,051 | No confirmada (por poco: la cota pasa el margen −1,5, pero p = 0,017 > α/3 = 0,0167 en Holm) |
| N_I: I no inferior al control | −2,64 | −3,31 | 0,998 | 1 | No confirmada |
| N_K: K no inferior al control | −3,14 | −3,81 | 1,000 | 1 | No confirmada |

**Lectura:**
1. **El control es el mejor agente en el promedio** (82,2 %). Supera a I (−2,6 pts) y a K (−3,1 pts) más
   allá del margen, así que esos dos rinden menos que PPO con el mismo presupuesto. La exploración de
   200 partidas (I ≈ K ≈ control) no tenía resolución para verlo.
2. **J es el único QD que se acerca al control** (−0,6 pts), y **castiga mejor a expectimax: +3,0 pts,
   confirmado.** Es el primer resultado confirmatorio a favor de un QD. A cambio, pierde contra greedy_solo
   (−3,6) y greedy_joint (−1,3): su estilo rinde distinto según el rival. Eso es justo lo que el selector
   de F2 intenta aprovechar.
3. **El control mejora al arranque** en +1,8 pts (± 0,8). El 1M de pasos extra sí sirve, y por eso el
   control era necesario para no atribuirle esa mejora a QD.
4. **Los QD no mejoran al arranque.** J: +1,2 pts; I y K: −0,8 y −1,3 pts. Su presupuesto rinde menos
   que el de PPO puro.
5. **Frente a la hipótesis a priori (§4.10):** I ≈ control no se cumple (I queda por debajo). K algo peor
   se cumple. J por encima del control no se cumple en el promedio, pero sí contra expectimax.
6. **Consecuencia para la Conclusión.** Se puede afirmar: "J 2.0 (PGA-ME) supera a PPO con el mismo
   presupuesto contra expectimax, por +3,0 pts". No se puede afirmar que algún QD rinda en promedio igual
   que PPO. I y K quedan por debajo del control.

### 5.11 Resultados de F2: selector de élite por rival (26-11-2026, código `463c736`, exploración)

Repertorio: 10 élites (5 de I y 5 de J; ninguna de K entró en el top-10 de `val`). Datos: 3000
enfrentamientos `fit` por élite, de 89 equipos rivales. Prueba: 3000 enfrentamientos `eval` de 58
equipos rivales, 0 caídas. λ = 10 por validación cruzada agrupada.

| Brazo (`eval`, 1000 por rival) | solo | joint | expectimax | **3 heur.** |
|---|---|---|---|---|
| Control | 85,6 | 87,6 | 75,2 | **82,8 %** |
| **Selector** | 84,5 | 82,8 | 75,2 | **80,8 %** |
| Mejor élite única (J 6,5, elegida en `fit`) | 82,5 | 84,1 | 74,8 | 80,5 % |
| PPO v2 s1 | 83,2 | 84,5 | 73,6 | 80,4 % |
| Oráculo por equipo (optimista) | 91,3 | 93,1 | 87,8 | 90,7 % |

- **Primaria:** selector − control = **−1,97 pts** (± 1,53). El selector rinde menos que el control.
- Selector − mejor élite única = +0,37 (± 1,40): elegir por rival no aporta nada medible.
- Por rival frente al control: solo −1,1; joint −4,8; expectimax 0,0.
- El selector eligió sobre todo J 7,6 (47 %), J 6,5 (16 %) e I 9,3 (16 %).

**Lectura:**
1. **El selector no funciona con este diseño, y la causa está en los datos de entrenamiento.** En `fit`
   todas las élites ganan entre 95 y 97 %: son los equipos con los que se entrenaron y los tienen
   memorizados (la brecha de memorización de §19). Sin diferencias entre élites en `fit`, el modelo no
   tiene nada que aprender sobre qué élite le va mejor a cada rival, y en `eval`, con equipos nuevos,
   elige casi al azar entre élites parecidas.
2. **El oráculo (90,7 %) no prueba que haya margen.** Elige la mejor de 10 élites por equipo con ~52
   partidas por equipo en la misma muestra. El error estándar por equipo es de ~5,5 pts y el máximo de 10
   lo infla en ~8 pts, así que casi toda la distancia es ruido de selección.
3. **Expectativa a priori (§4.12):** se esperaba una ganancia de 0 a +2 pts. Salió −2,0 frente al control
   y +0,4 frente a la mejor élite única, en la zona baja de lo esperado.
4. **Mejora posible (no encolada):** entrenar el selector con equipos que los agentes no vieron
   (`val`, o una parte de `eval` separada de la prueba), donde sí hay diferencias entre élites. El costo
   es menos equipos para el selector.

### 5.12 Resultados de F3: réplica desde s4 y K con N = 10 (27-11-2026, código `463c736`)

Despliegue con 1000 partidas `val`: J4 4,3 (87,0 %), K4 6,3 (85,2 %), K10 7,4 (91,7 %). Partidas: 2000
enfrentamientos `eval` por heurístico, 6000 por agente, 0 caídas.

| Agente | solo | joint | expectimax | **3 heur.** | Archivo |
|---|---|---|---|---|---|
| control4 (PPO v2 s4 + 1M) | 88,8 | 88,9 | 79,7 | **85,8 %** | — |
| PPO v2 s4 (arranque) | 86,8 | 89,6 | 81,3 | **85,9 %** | — |
| J4 | 86,2 | 86,7 | 78,6 | 83,8 % | 31 celdas, QD-score 61 |
| K4 | 84,4 | 85,5 | 76,8 | 82,2 % | 83 celdas, QD-score 153 |
| control1 (de F1) | 85,5 | 85,2 | 76,1 | 82,2 % | — |
| K10 (N = 10, desde s1) | 82,3 | 83,0 | 74,0 | 79,8 % | 54 celdas, QD-score 85, 23 iteraciones |
| K1 (K de F1) | 83,2 | 83,9 | 72,2 | 79,8 % | 87 celdas, QD-score 162 |

**F3a (réplica s4, Holm m = 3):**
- N_J4: −1,95 pts (cota −2,85), no rechaza.
- N_K4: −3,60 pts (cota −4,51), no rechaza.
- **S_J4: −1,00 pts contra expectimax (p = 0,83), no rechaza: la ventaja de J contra expectimax de F1 no se
  replica desde s4.**

**F3b (K10 frente a control1):**
- N_K10: −2,47 pts, igual que K1 (−2,48). Las iteraciones del paper no mejoran el rendimiento.
- K10 cubre menos mapa (54 frente a 87 celdas), porque con N = 10 caben menos iteraciones en el mismo
  presupuesto.

**Lectura:**
1. **El patrón de F1 se replica:** los QD rinden menos que el control. J queda más cerca (−0,6 en s1 y
   −2,0 en s4) y K más lejos (−3,1 y −3,6).
2. **La ventaja de J contra expectimax depende de la semilla.** En s1 fue +3,0 (confirmada); en s4 es
   −1,0. La afirmación de F1 se mantiene para el agente desplegado desde s1, pero no se puede generalizar
   a "PGA-ME castiga mejor a expectimax".
3. **Desde s4, el control no mejora al arranque** (−0,1 ± 1,0). s4 ya estaba en su techo con esta receta
   (85,9 %). Ningún QD lo supera: J4 −2,1 y K4 −3,7 frente al arranque.
4. **La reducción N = 3 de K no explica su rendimiento:** con N = 10 rinde igual y cubre menos.
5. **Frente a la expectativa (§4.13):** el patrón QD ≈ control se cumple en la forma "QD ≤ control". La
   expectativa de que K10 rindiera igual o algo mejor que K1 se cumple: rinde igual.

### 5.13 Resultados de F4b: PPO con el archivo de K como pool de rivales (27-11-2026, código `463c736`)

PPO v2 s1 + 1M pasos con el mosaico `pool`: el 20 % de las partidas contra las 5 mejores élites de K en
espejo. Entrenó en 1 h 40 min con 22 reinicios de entorno (el control tuvo 0: los rivales PPO en espejo
son menos estables). Partidas: los mismos 12 000 enfrentamientos `eval` de F1 (bloque 40001), pareados
con el control. 0 caídas.

| Agente | solo | joint | expectimax | **3 heur.** |
|---|---|---|---|---|
| Control (mosaico de PPO v2) | 85,8 | 84,9 | 75,9 | **82,2 %** |
| **Pool** (20 % élites de K) | 84,4 | 82,2 | 77,4 | **81,3 %** |
| PPO v2 s1 (arranque) | 82,7 | 83,9 | 74,5 | 80,3 % |

- **N_pool:** −0,87 pts (cota inferior 95 % −1,56). No rechaza por muy poco: la cota apenas cruza el
  margen −1,5 (p = 0,067).
- **Por rival frente al control:** expectimax +1,52 (± 1,60), solo −1,45 (± 1,26), joint −2,67 (± 1,41).
- **Frente al arranque:** +0,97 (± 0,83). El control mejora +1,83.

**Lectura:**
1. **Entrenar contra la diversidad de K no mejora a PPO contra las heurísticas.** Queda casi igual al
   control y un poco por debajo, porque el 20 % de las partidas contra élites le resta práctica contra
   las heurísticas, que son la métrica.
2. **Hay un desplazamiento como el de J:** mejora contra expectimax (+1,5, no significativo) y pierde
   contra greedy_joint. Lo que aporta la diversidad parece cambiar contra quién rinde, no cuánto rinde.
3. **Frente a la expectativa (§4.14):** se esperaba pool ≈ control. Se cumple (−0,9 pts). La posible
   ganancia en robustez frente a estilos variados no se mide contra las heurísticas; habría que medirla
   contra personas o contra un pool aparte.

## 6. Limitaciones

- Un solo formato (Reg M-C) y un solo pool de equipos.
- La relación entre agentes puede no ser transitiva, por eso cada par se mide directamente.
- Expectimax trabaja con un tope de tiempo de pared por decisión. Los resultados dependen
  de la carga del servidor; se usan siempre `--procs 28` y 2 workers.

- **Hoja abierta en ~5 % de los equipos (corregido el 15-11-2026, db6d4de).** 14 equipos del
  pool exportaban nombres de especie no oficiales (`Garchomp Mega Z`, `Kommo-O`, `Mr-Mime`,
  `Rotom Wash`); poke-env no los emparejaba al leer `|showteam|` y la hoja abierta de esas
  batallas quedaba sin procesar. Afecta por igual a ambos lados en las corridas anteriores
  (confirmación, techos, diversidad); no se espera que cambie las conclusiones.

### Limitaciones de los agentes con aprendizaje (18-11-2026)

- **Decisiones de configuración mirando `eval`** (máscara del aliado, simetría en PPO, REG): la
  cifra final contra `eval` puede estar algo inflada; desde el 17-11-2026 las elecciones se hacen con
  `val`, y la Conclusión formal usa semillas de partida nuevas (`memorizacion_equipos.md` §9).
- **Especialización en el equipo fijo:** con un equipo propio desconocido DDQN gana ~19 % y PPO
  ~31 % contra los heurísticos (§12.1).
- **Los heurísticos casi no ejecutan arquetipos de velocidad** (Trick Room 0–1 %, Tailwind 0–4 % de
  las partidas): se agrega un rival de arquetipos para medirlo (§14.2).

## Registro de cambios del protocolo

| Fecha | Cambio |
|---|---|
| 2026-11-15 | Versión inicial: exploración/confirmación, pruebas, regla de decisión, hipótesis H1–H3 y S1–S3. |
| 2026-11-15 | **Separación de equipos 200/100 (§2b):** por grupos de clones (mismas 6 especies) y estratificada por especies; RL y previews entrenan en `train`, la Conclusión se mide en `eval`, y el prior de SP y los mundos de expectimax salen de `train`. |
| 2026-11-15 | Resultados de la confirmación (§5.1): H2 y H3 confirmadas con ventaja pequeña; H1 sin diferencia demostrada. Diagnóstico exploratorio: la opción C fortalece a joint contra expectimax (+2,3 pts en C1, +0,3 en C2; +1,3 en conjunto). |
| 2026-11-15 | Se agregan los dos niveles de afirmación (umbral práctico provisional: límite inferior del IC ≥ 55 % en BO3), TOST y la sección de agentes con aprendizaje. |
| 2026-11-16 | **Validación reservada (§2b):** subgrupo `val` (30 equipos de `train`, grupos de clones) para elegir checkpoints; `fit` (170) para entrenar. Motivo: memorización medida (84 % con equipos `train` frente a 50 % con `eval`). Ver `memorizacion_equipos.md`. |
| 2026-11-17 | **Checkpoint principal = el último (§3c.2)**; `best.pt` elegido con `val`, resultado secundario. Motivo: `val` eligió en DDQN con regularización un checkpoint ~8 puntos peor contra `eval` que el final. |
| 2026-11-20 | **Hipótesis G1–G2 de MCTS+PUCT v3 (§4.1)**, fijadas antes de medir: G v3.0 contra joint y contra expectimax en bloques nuevos C3 (9401–9417) y C4 (9501–9517), preview greedy, Holm con m = 2; secundaria GS1 (pipeline G con preview Nash) en C3. |
| 2026-11-20 | Enmienda §4.1 antes de medir: se elimina la secundaria GS1 porque no existe un preview Nash entrenado (G caía al orden fijo `/team 1234`). G1–G2 sin cambios. |
| 2026-11-20 | Segunda enmienda §4.1 antes de medir (usuario): se agregan G3 (G > greedy_solo, C3 + C4; Holm pasa a m = 3) y el control G4 (G > random, C3); solo baselines, ningún agente RL. Código congelado `bb9ef00` (v3.1: combate de v3.0 + preview greedy por defecto). |
| 2026-11-20 | Resultados de la confirmación de G (§5.2): G1, G2 y G3 con superioridad práctica (G1 y G2 por poco: límite inferior 55,44 y 55,42 %); control G4 correcto. |
| 2026-11-20 | Hipótesis V1–V4 de MCTS+PUCT v4 (§4.2), fijadas antes de medir: v4 contra joint, expectimax y solo, y v4 frente a v3.1 en pareado, bloques C5 (9601–9617) y C6 (9701–9717); preview según la regla del diagnóstico §7; control V5 contra random. |
| 2026-11-21 | Resultados de la confirmación de G v4 (§5.3): V1–V4 confirmadas con superioridad práctica (62,6 % contra joint, 61,2 % contra expectimax, 64,3 % contra solo) y +5,1 pts sobre v3.1; v4 pasa a ser el valor por defecto de G. |
| 2026-11-21 | Hipótesis W1–W4 de MCTS+PUCT v5 (§4.3), fijadas antes de medir: v5 contra joint, expectimax y solo, y v5 frente a v4 en pareado, bloques C7 (9801–9817) y C8 (9901–9917); control W5 contra random. |
| 2026-11-21 | Resultados de la confirmación de G v5 (§5.4): W1–W4 confirmadas con superioridad práctica (65,5 % contra joint, 63,6 % contra expectimax, 66,8 % contra solo) y +4,5 pts sobre v4 (McNemar p = 6·10⁻¹⁹); v5 pasa a ser el valor por defecto de G. |
| 2026-11-21 | Preview por búsqueda para G v5 (§4.4), fijado antes de medir: exploración S0 greedy / S1 mean / S2 maximin en 9101–9117 con regla de selección; si pasa, X1 (contra joint) y X2 (contra expectimax) en pareado frente al greedy, bloques C9 (10001–10017) y C10 (10101–10117), Holm m = 2. |
| 2026-11-21 | Exploración pre-registrada del motor de inferencia de SP en G v5 (diagnóstico §9): inference frente a oracle_spe y meta en 9101–9117; si meta no es peor (|Δ| ≤ 1 pt o mejor), el default de G pasa a meta. |
| 2026-11-22 | Resultados de la confirmación del preview por búsqueda de G (§5.5): X1 +4,11 pts contra joint (p = 3·10⁻¹²) y X2 +4,06 pts contra expectimax (p = 1·10⁻¹¹), ambas confirmadas en C9 y C10 (Holm m = 2); G v5 + búsqueda: 69,0 % contra joint, 67,7 % contra expectimax. El preview por búsqueda (`mean`) pasa a ser el default de G. |
| 2026-11-22 | Resultado del motor de inferencia de SP en G (diagnóstico §9.1): 0 de 14 016 stats estrechados; meta frente a inference +0,02 pts (p = 1), oracle_spe +0,08 (p = 0,85); por la regla pre-registrada el default de G pasa a `meta` (−14 % de tiempo). |
| 2026-11-22 | Fase 3e (objetos) para G pre-registrada (§4.5, diagnóstico §10): exploración v7 frente a v5.2 en 9101–9117; si mejora, confirmación Z1–Z4 en C11 (10201–10217) y C12 (10301–10317), Holm m = 4. |
| 2026-11-22 | Exploración de la fase 3e (diagnóstico §10.1): v7 71,18 % frente a v5.2 69,21 % contra joint en 9101–9117, McNemar +1,90 pts (p = 0,0086); pasa a confirmación Z1–Z4 en C11/C12. |
| 2026-11-22 | Resultado de la confirmación de la fase 3e de G (§5.6): Z1–Z3 confirmadas (v7 70,9 % contra joint, 69,4 % contra expectimax, 73,2 % contra solo); Z4 v7 frente a v5.2 +0,76 pts, p Holm = 0,138 (C11 +1,55, C12 −0,04): no se confirma, G se queda en v5.2. |
| 2026-11-22 | Agente H 2.0 pre-registrado (§4.6): exploración H 2.0 frente a H 1.7 en 9101–9117; si mejora, confirmación W1–W3 contra baselines y W4 frente a G 5.2 (bilateral) en C13 (10401–10417) y C14 (10501–10517), Holm m = 4. H pasa a SP meta. |
| 2026-11-22 | Desviación en §4.6: N1 (H 1.7) cortado en 4 semillas (10,8 % frente a 62,1 % de H 2.0, ~860 s/semilla); la regla de paso se evalúa en 9101–9104 emparejadas. |
| 2026-11-22 | Exploración de H 2.0 (§4.6): N2 H 2.0 62,06 % contra joint (17/17 semillas > 50 %, 376 s/semilla, 0 errores, 0 anomalías); N1 H 1.7 10,91 % (4 semillas, 5431 fallbacks del adaptador); N2 frente a N1 +51,7 pts (p = 3·10⁻¹⁵⁸) → pasa a confirmación. Descriptivo: N2 frente a G 5.2 −7,18 pts (532 frente a 893, p = 9·10⁻²²). |
| 2026-11-23 | Resultado de la confirmación de H 2.0 (§5.7): W1–W3 confirmadas (61,6 % contra joint, 61,7 % contra expectimax, 63,3 % contra solo, 34/34 semillas cada una); W4 H 2.0 frente a G 5.2 −7,25 pts (p Holm = 6·10⁻⁴², C13 −7,37, C14 −7,13): G 5.2 es mejor. H 2.0 es el H de la Conclusión. |
| 2026-11-23 | Agente H 2.1 pre-registrado (§4.7): exploración de H1 matriz chica, H2 decisión por visitas, H3 ambas y H4 RM+ (350 sims) frente a H 2.0 (N2) en 9101–9117; el mejor que pase McNemar p < 0,05 va a confirmación V1–V4 en C15 (10601–10617) y C16 (10701–10717), Holm m = 4. |
| 2026-11-23 | Exploración de H 2.1 (§4.7, spec de H §4c): H1 +1,52 pts (p = 0,046), H2 +2,43 (p = 0,001), **H3 +3,24 (p = 1,6·10⁻⁵; 65,30 % contra joint)**, H4 RM+ −0,55 (p = 0,49) frente a H 2.0; gana H3 (matriz 8/6/4 + decisión por visitas) y pasa a confirmación V1–V4 en C15/C16. |
| 2026-11-24 | Exploración de H 2.2 pre-registrada (§4.8, spec de H §4d): B1 matriz 6/4/3 y B3 backup mixto (superioridad frente a H3, McNemar p < 0,05), B2 H3 + RM+ con 200 sims (no inferioridad: cota inferior 95 % > −1,5 pts y ≤ 85 % del tiempo de H3); 9101–9117 contra joint, espera al paso A. Sin confirmación encolada. |
| 2026-11-24 | Resultado de la confirmación de H 2.1 (§5.8): V1–V4 confirmadas (H3: 66,0 % contra joint, 65,7 % contra expectimax, 67,5 % contra solo; +4,46 pts sobre H 2.0, p Holm = 4·10⁻¹⁷, los dos bloques). H 2.1 pasa a ser el default del player y el H de la Conclusión. Paso A: H 2.1 frente a G 5.2 −3,17 (joint), −3,00 (expectimax), −4,09 (solo). |
| 2026-11-24 | Exploración de H 2.2 (§4.8, spec de H §4d): B1 matriz 6/4/3 +0,51 pts (p = 0,43) y B3 backup mixto +0,57 (p = 0,33) no pasan la superioridad; **B2 H3 + RM+ pasa la no inferioridad** (−0,04 pts, cota inferior 95 % −1,09 > −1,5; 219 s/semilla = 57 % del tiempo de H3, ≈ G 5.2). Sin brazo ganador; B2 queda como candidato de velocidad, sin confirmación encolada. |
| 2026-11-25 | Confirmación de H 2.2 = B2 pre-registrada (§4.9, spec de H §4e): U1–U3 B2 > joint, expectimax, solo; U4 B2 no inferior a H 2.1 (margen −1,5 pts, McNemar con p unilateral); Holm m = 4; tiempo ≤ 85 % del de H 2.1; descriptivo frente a G 5.2 a igual coste. C17 = 10801–10817, C18 = 10901–10917. |
| 2026-11-25 | Resultado de la confirmación de H 2.2 (§5.9): U1–U4 confirmadas (B2: 66,1 % contra joint, 65,2 % contra expectimax, 67,8 % contra solo; frente a H 2.1 +0,00 pts, cota inferior −0,76 > −1,5, p Holm = 6·10⁻⁴) y tiempo 57 % del de H 2.1. H 2.2 = H 2.1 + RM+ pasa a ser el default del player y el H de la Conclusión. Frente a G 5.2 al mismo coste: −2,37 pts. |
| 2026-11-26 | Hipótesis a priori de la familia QD 2.0 registrada (§4.10), antes de cualquier resultado: frente al control (PPO v2 + 1M) I empata (~30 % de superarlo), K algo peor (~15 %), J peor (~10 %); ranking esperado en rendimiento I > K > J y en QD-score K ≥ I > J. Es descriptiva y no cambia ninguna prueba. |
| 2026-11-26 | Exploración de la familia QD 2.0 frente a la hipótesis a priori (§4.10): I ≈ control y K algo peor coinciden; J (81,8 %) por encima del control (81,0 %) no se esperaba, pero está dentro del ruido. QD-score K > J > I. Confirmación pendiente de decidir. |
| 2026-11-26 | F1 de la familia QD 2.0 pre-registrada (§4.11): despliegue con 1000 partidas `val` (top-10 de cada archivo) y confirmación con 4000 enfrentamientos `eval` por heurístico; N_I/N_J/N_K no inferioridad frente al control (margen −1,5 pts) y S_J J > control contra expectimax, Holm m = 4. Cola `scripts/server/qd_f1.sh`. |
| 2026-11-26 | F2–F4 de la familia QD 2.0 pre-registradas como exploraciones (§4.12–§4.14), encadenadas detrás de F1: F2 selector de élite por rival (logística sobre el equipo rival, bandit con información completa); F3 réplica desde PPO v2 s4 (control, J, K) y K con N1 = N2 = 10; F4a preview QD-PGA por roles; F4b PPO con el archivo de K como pool de rivales. |
| 2026-11-26 | Resultado de F1 de la familia QD 2.0 (§5.10): **S_J confirmada** (J > control contra expectimax, +3,00 pts, p Holm = 2,2·10⁻⁴). N_J no confirmada por poco (−0,64 pts, cota −1,31, p = 0,017 > 0,0167). N_I (−2,64) y N_K (−3,14) no confirmadas: rinden menos que el control. Control 82,2 % frente a I 79,5, J 81,5 y K 79,0 % (12 000 partidas cada uno). |
| 2026-11-26 | Resultado de F2 (§5.11, exploración): el selector rinde 80,8 % frente a 82,8 % del control (−1,97 ± 1,53 pts) y +0,37 frente a la mejor élite única. Causa: en `fit` (equipos de entrenamiento, memorizados) todas las élites ganan 95–97 % y no hay señal para aprender. El oráculo (90,7 %) está inflado por ruido de selección. Mejora posible: entrenar el selector con equipos no vistos. |
| 2026-11-26 | F2b pre-registrada (§4.15, exploración): selector entrenado con 2000 enfrentamientos `val` por heurístico (equipos no vistos) y probado con la parte `eval` de F2. Cola `qd_f2b.sh`, después de F4. |
| 2026-11-27 | **PPO v3 / DDQN v2 pre-registrados (§4.16)**, código implementado y sin encolar: M1 (distribución del preview, ya medida: casi colapsado por rival, pero reacciona al rival; sesgo a Arcanine-Hisui), P1 máscara del Protect consecutivo sin reentrenar (no inferioridad −1,5 pts), S1 preview condicionado a la serie sin reentrenar (regla +1,5 pts), V1/V2 memoria de serie (brazos C y V, partidas 2–3, Holm m = 2). Por juego limpio PPO y DDQN reciben los mismos cambios (decisión del usuario). Presupuesto, semillas y orden pendientes del usuario. |
| 2026-11-27 | Resultado de F3 (§5.12): desde s4, control4 85,8 %, J4 83,8 % (−1,95), K4 82,2 % (−3,60); **la ventaja de J contra expectimax no se replica** (−1,0, p = 0,83), así que la de F1 es propia de s1. K con N = 10 rinde igual que con N = 3 (−2,47 frente a −2,48) y cubre menos mapa (54 frente a 87 celdas). |
| 2026-11-27 | §4.16 cerrada con las decisiones del usuario (+1M pasos, 3 semillas por brazo, control C frente a V, orden P1 → S1 → PPO → DDQN; ε = 0,05 en el arranque en caliente de DDQN). Cola `scripts/server/v3_queue.sh` e informe `scripts/analysis/v3_report.py` preparados, sin encolar. |
| 2026-11-27 | **F4a cortada** por decisión del usuario en la iteración 42 de 64, sin resultado: la calibración salió estrecha ([0,475; 0,525] en los dos descriptores, porque el actor inicial elige casi siempre lo mismo), la rejilla funcionaba con pocos niveles (cobertura 0,13) y aporta poco a las mejoras de PPO y DDQN. F4b y F2b siguen. |
| 2026-11-27 | §4.16: diagnósticos D1/D2 por arquetipo añadidos antes de medir (decisión del usuario), desde las evaluaciones sin máscara de P1 con `--context`; guía de decisión para proponer un rival que ejecute Trick Room / Tailwind o reentrenar el preview. Corrección: la observación ya tenía el orden de velocidad efectivo. |
| 2026-11-27 | §4.16: S1 con 3 semillas por familia (s0–s2) y evaluaciones de la cola en paralelo (`JOBS`, un puerto por evaluación; entrenamientos de uno en uno), decisión del usuario para acortar la cola de ~4–5,5 a ~1,5–2 días. Diseño y reglas sin cambios. |
| 2026-11-27 | Objetivo de equidad de cómputo entre agentes (decisión del usuario): horas de cola totales similares para E–K. Registro y desequilibrio actual en `docs/presupuesto_computo.md` (F 101 h, E 88 h, H 79 h, G 61 h, I/J/K 19 h cada uno). |
| 2026-11-27 | **PPO v3 / DDQN v2 encolados** (§4.16, visto bueno del usuario): `v3_queue.sh` con JOBS = 6 en la copia del repo `IAPSVGC2022_v3` (commit `72a42bc`, con los checkpoints de PPO v2 en `checkpoints/ppo_v2/`). Un envoltorio espera a `QD_F2B_DONE` (un método a la vez) y corre los tests de v3 antes de lanzar la cola. Salida `~/iapsvgc_runs/v3`, marca `V3_DONE`. |
| 2026-11-27 | Resultado de F4b (§5.13): el PPO entrenado con un 20 % de élites de K como rivales rinde 81,3 % frente a 82,2 % del control (−0,87 pts, cota −1,56; no inferioridad no confirmada por poco). Expectimax +1,5, joint −2,7: la diversidad cambia contra quién rinde, no cuánto. |
