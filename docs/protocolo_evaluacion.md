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
2. **Elección del checkpoint con validación aparte.** El checkpoint se elige con semillas o
   equipos de validación, nunca con los bloques de confirmación (C1/C2).
3. **Evaluación fuera de distribución.** Además del rival con el que entrenó, se evalúa
   contra rivales y equipos no vistos, para descartar que solo explote a su rival de
   entrenamiento.
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

## Registro de cambios del protocolo

| Fecha | Cambio |
|---|---|
| 2026-11-15 | Versión inicial: exploración/confirmación, pruebas, regla de decisión, hipótesis H1–H3 y S1–S3. |
| 2026-11-15 | Resultados de la confirmación (§5.1): H2 y H3 confirmadas con ventaja pequeña; H1 sin diferencia demostrada. Diagnóstico exploratorio: la opción C fortalece a joint contra expectimax (+2,3 pts en C1, +0,3 en C2; +1,3 en conjunto). |
| 2026-11-15 | Se agregan los dos niveles de afirmación (umbral práctico provisional: límite inferior del IC ≥ 55 % en BO3), TOST y la sección de agentes con aprendizaje. |
