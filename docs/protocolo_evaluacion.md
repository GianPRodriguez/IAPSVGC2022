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
