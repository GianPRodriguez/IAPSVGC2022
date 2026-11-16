# Memorización de los equipos rivales y cómo reducirla

Estado: **octava cola terminada (16-11-2026). Los sintéticos no cumplen el criterio de adopción; la cantidad de equipos reales no cambia la generalización.** PPO (novena cola, §6) generaliza mucho mejor que DDQN. Decisión del usuario pendiente.
Relacionado: [`protocolo_evaluacion.md`](protocolo_evaluacion.md) §2b (separación de equipos),
[`ddqn_kickoff_checklist.md`](ddqn_kickoff_checklist.md), [`equipos_sinteticos_analisis.md`](equipos_sinteticos_analisis.md).

## 1. El problema

El mini kick-off de DDQN (1M pasos, currículo completo) llegó a 85 % de victorias en la
validación periódica. Esa validación usaba equipos rivales del grupo `train`, los mismos
con que entrenó. Con equipos que nunca vio (grupo `eval`) el resultado es otro:

| Rival | Equipos `train` | Equipos `eval` | Diferencia |
|---|---|---|---|
| random | 91 % | 90 % | −1 |
| greedy_solo | 87 % | 38 % | −49 |
| greedy_joint | 87 % | 36 % | −51 |
| expectimax_bayesian | 72 % | 36 % | −36 |
| **total** | **84,2 %** | **50,0 %** | **−34,2** |

100 partidas por rival, sin exploración, rivales deterministas, equipo fijo del agente.

El preview v2 muestra lo mismo: +33,6 puntos sobre el preview heurístico con equipos
`train` y +10,5 con equipos `eval`. Además, +10,5 es lo mismo que se obtiene jugando
siempre la formación más frecuente (+10,4), así que, con greedy_joint en el combate, no hay
evidencia de que adapte la elección a rivales que no conoce. **Con expectimax en el combate
(séptima cola) sí la hay:** +27,7 con equipos `train`, +17,2 con `eval` y solo +1,2 con la
formación fija. El preview memoriza en parte (pierde 10 puntos con equipos no vistos), pero
también aprendió a adaptar la elección al rival.

**Por qué esto es memorización y no ruido.** La caída contra solo y joint es de unos 50
puntos, con unos ±7 puntos de error por celda. Contra random no cambia, porque ganarle no
requiere conocer al rival.

**Lo llamativo.** Los equipos `eval` no son raros. La separación se hizo por grupos de
clones (equipos con las mismas 6 especies van juntos), así que un equipo `eval` comparte
casi siempre 4 o 5 especies con algún equipo `train`:

| Especies en común con el equipo `train` más parecido | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| Equipos `eval` | 5 % | 14 % | 29 % | 52 % | 0 % |

(Referencia: los 170 equipos de `fit`, sección 3.) El agente pierde rendimiento contra
equipos que solo cambian 1 o 2 piezas. Aprendió a reconocer combinaciones concretas, no a
jugar contra especies y sets.

## 2. Por qué no basta con conseguir más equipos reales

El metajuego real de torneo tiene poca variación, y los 300 equipos lo reflejan bien. Más
equipos reales del mismo metajuego aportarían poca diversidad nueva. Con 1M pasos y 200
equipos, además, cada equipo rival aparece miles de veces; pasar a 300 o 400 no cambia mucho
esa proporción. Antes de invertir en conseguir más equipos hay que medir si la cantidad
importa (curva de escala, sección 4).

## 3. Qué se implementó

Tres opciones del entrenamiento, todas **apagadas por defecto**: sin ellas el sorteo de
equipos es idéntico al anterior, con los mismos números aleatorios (test). Ninguna toca la
evaluación final (Conclusión, grupo `eval`). Código: `showdown/rl/foe_pool.py`.

### 3.1 Validación reservada (`RL_HOLDOUT_VAL=1`)

El mejor checkpoint (`best.pt`) se elegía con equipos de entrenamiento, así que la
selección premiaba al checkpoint que más memorizaba. Ahora:

| Grupo | Equipos | Uso |
|---|---|---|
| `fit` | 170 | Entrenamiento (rivales) |
| `val` | 30 | Validación periódica y elección de `best.pt` (nunca se entrena con ellos) |
| `eval` | 100 | Solo evaluación final (sin cambios) |

`val` sale de `train` con el mismo método que la separación original: grupos de clones
completos y estratificación por especies (la mejor de 500 semillas). `train` y `eval` no
cambiaron. Uso de las especies más comunes:

| Especie | `fit` | `val` |
|---|---|---|
| Rillaboom | 64,7 % | 63,3 % |
| Sneasler | 56,5 % | 46,7 % |
| Salamence | 45,9 % | 46,7 % |
| Incineroar | 40,0 % | 40,0 % |
| Basculegion | 34,7 % | 43,3 % |
| Floette-Eternal | 32,9 % | 30,0 % |
| Kingambit | 24,7 % | 40,0 % |

Con solo 30 equipos hay diferencias de hasta ~15 puntos en algunas especies; es una
validación para elegir checkpoints, no la medida final.

**Límite:** el preview v2 que usan estas corridas se entrenó con los 200 de `train`, así
que ya vio los equipos de `val`. Esto afecta por igual a todos los checkpoints de una corrida
(no cambia cuál se elige), pero la cifra de `val` queda algo inflada por el preview.

### 3.2 Subconjuntos de equipos (`RL_TRAIN_SUBSET=N`)

Entrena solo con N equipos de `fit`, en un orden fijo por hash. Los subconjuntos son
anidados (los 50 están dentro de los 100), así que la curva de escala compara pools que
solo difieren en tamaño.

### 3.3 Equipos sintéticos por recombinación (`RL_FOE_SYNTH=p`)

Con probabilidad p el rival juega un equipo sintético en vez de uno real. Generador:
`showdown/rl/synth_teams.py`; archivo: `showdown/rl/data/synth_teams.json` (2 000 equipos,
semilla 20261116).

**Diseño.** Como el metajuego real varía poco, los sintéticos no son equipos al azar sino
**variantes de equipos reales**:

1. Se toma un equipo real de `fit`.
2. Se reemplazan k de sus 6 Pokémon por Pokémon de otros equipos de `fit`, **cada uno con su
   set real completo** (objeto, habilidad, naturaleza, movimientos, SP). Pesos: k = 1 (50 %),
   2 (35 %), 3 (15 %). El núcleo del equipo (al menos 3 de 6) queda intacto.
3. Los donantes se sortean entre todas las apariciones de Pokémon en `fit`, así que se
   respeta la frecuencia de uso del metajuego.
4. Reglas: sin especies repetidas (por número de Pokédex: Arcanine y Arcanine-Hisui cuentan
   como la misma) ni objetos repetidos. Después, el validador de Showdown de Reg M-C revisa
   cada equipo.
5. Se descartan las combinaciones de especies que ya existen en `fit`, `val` o `eval`: un
   sintético nunca reproduce un equipo de validación ni de evaluación.

**Por qué variantes y no equipos al azar.** Un equipo armado al azar con piezas del
metajuego suele ser incoherente (por ejemplo, sin control de velocidad). El agente
aprendería contra equipos que no aparecen en torneos y que probablemente son más débiles.
Las variantes mantienen el realismo y cubren justo la novedad que el agente encuentra en
la evaluación: casi todas las piezas conocidas en otra combinación.

**Análisis** (detalle en [`equipos_sinteticos_analisis.md`](equipos_sinteticos_analisis.md)):

- 2 000 válidos de 2 244 intentos; el validador de Showdown no rechazó ninguno (los Pokémon
  ya eran sets legales y las cláusulas se aplican al generar). Se comprobó aparte que el
  validador sí rechaza equipos con especie repetida o movimientos ilegales.
- Reemplazos: k = 1: 918, k = 2: 761, k = 3: 321.
- **Parecido con el equipo real más cercano** (especies en común con `fit`):

| Grupo | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| Sintéticos | 0 % | 7 % | 40 % | 53 % |
| `eval` (real) | 5 % | 14 % | 29 % | 52 % |
| `val` (real) | 0 % | 10 % | 33 % | 57 % |

  Los sintéticos se separan de los equipos de entrenamiento casi exactamente como los
  equipos de evaluación reales. Ese es el objetivo.
- **Uso de especies:** se mantiene dentro de ±3,5 puntos en las 15 más usadas, salvo
  Rillaboom (64,7 % real frente a 54,1 % sintético). Las especies más comunes se reemplazan
  más seguido de lo que entran. Es un sesgo menor y conocido.
- **Límite:** no crea especies ni sets que no estén en el pool (81 especies en ambos). No
  cubre equipos realmente fuera del metajuego, solo combinaciones nuevas de piezas
  conocidas. Para equipos muy raros haría falta otra fuente (por ejemplo, sets generados).

### 3.4 Registro para el análisis

- `policy_diag` guarda el equipo rival y el resultado de cada partida. Con eso se mide si el
  agente gana más contra equipos `eval` parecidos a los de entrenamiento.
- `train_metrics.jsonl` registra las opciones del pool de cada corrida (`foe_pool`).

### 3.5 Verificación

- Tests: `tests/test_foe_pool_and_synth.py` (sorteo idéntico sin opciones, grupos fit/val,
  subconjuntos anidados, sintéticos solo en entrenamiento, cláusulas, sets completos,
  determinismo) y `tests/test_kickoff_summary8.py`. Todos pasan en el servidor.
- Prueba en vivo: 4 000 pasos con 100 % de rivales sintéticos y validación con `val`: sin
  reinicios ni errores, y la validación usó solo equipos de `val`.
- **Espejo de los rivales con equipos sintéticos:** los heurísticos comparan cada turno su
  estado interno con poke-env (`battle_state_check`, registrado desde rl_env_spec v1.27). En
  dos pruebas cortas (DDQN y PPO, greedy_joint y expectimax, 100 % sintéticos): unas 430
  partidas por prueba, 1 800 revisiones cada una, **0 discrepancias y 0 errores**. La octava
  cola lo registra en todas sus corridas. Diagnóstico: con
  sintéticos activos, 12 de 12 rivales fueron sintéticos; con `val` y `eval`, 12 de 12 del
  grupo correcto.

## 4. Experimento (octava cola)

Configuración del mini kick-off en todas las corridas (1M pasos, currículo completo, lr 1e-4,
lote 256, preview v2, equipo fijo), con validación reservada. Cada corrida se diagnostica con
100 partidas por rival contra sus propios equipos de entrenamiento y contra `eval`.

| Corrida | Equipos | Sintéticos | Semillas |
|---|---|---|---|
| D0 | mini kick-off anterior (200 de `train`) | no | (ya entrenado) |
| F170 | 170 (`fit`) | no | 0 y 1 |
| SY50 | 170 (`fit`) | 50 % | 0 y 1 |
| S50 | 50 | no | 0 |
| S100 | 100 | no | 0 |

Script: `scripts/server/kickoff_queue8.sh`; resumen: `showdown/rl/ddqn/kickoff_summary8.py`.

**Preguntas y cómo se leerán** (fijado antes de ver resultados):

1. **¿Los sintéticos reducen la memorización?** Comparar SY50 con F170 en victorias contra
   `eval` (3 heurísticos, 300 partidas por corrida), con 2 semillas. Si SY50 mejora en `eval`
   en ambas semillas y por más de ~8 puntos en promedio, se adoptan para el kick-off.
2. **¿Más equipos reales ayudan?** Curva S50 → S100 → F170 de la brecha propios/`eval`. Si la
   brecha baja claramente con el tamaño, vale la pena ampliar el pool real; si no cambia, la
   cantidad no es el problema.
3. **¿Gana más contra equipos parecidos?** Victorias contra `eval` según especies en común
   con el equipo de entrenamiento más parecido. Si suben con el parecido, confirma que
   memoriza combinaciones.
4. **¿La validación reservada elige mejor?** Comparar el `best.pt` elegido con `val` con el
   último checkpoint en `eval`.

**Alcance.** Es una exploración para decidir la configuración del kick-off, no una afirmación
de la tesis (protocolo §3c: los métodos se comparan con 3–5 semillas). La curva de escala
tiene una semilla por punto; diferencias de menos de ~8 puntos son ruido.

## 5. Resultados (octava cola, commit 9727278, 16-11-2026)

Resumen completo en el servidor: `~/iapsvgc_runs/kickoff8/kickoff_summary8.md`. Los
diagnósticos usan el `best.pt` de cada corrida (elegido con `val`), 100 partidas por rival y
grupo, sin exploración. "Propios" = los equipos con que entrenó esa corrida.

### 5.1 Brecha entre equipos propios y `eval` (3 heurísticos, 300 partidas por grupo)

| Corrida | Propios | `eval` | Brecha | `best.pt` (paso) |
|---|---|---|---|---|
| D0: mini kick-off anterior (200 de `train`) | 83,0 % | 35,3 % | −47,7 | — |
| S50 (50 equipos), semilla 0 | 83,7 % | 36,3 % | −47,3 | 750k |
| S100 (100 equipos), semilla 0 | 86,3 % | 33,3 % | −53,0 | 1M |
| F170 (170 equipos), semilla 0 | 80,0 % | 38,7 % | −41,3 | 750k |
| F170, semilla 1 | 84,3 % | 34,0 % | −50,3 | 1M |
| SY50 (170 + 50 % sintéticos), semilla 0 | 71,0 % | 39,0 % | −32,0 | 1M |
| SY50, semilla 1 | 72,0 % | 34,0 % | −38,0 | 750k |

Error estándar de la brecha: ± 3,4–3,8 puntos; de cada cifra de `eval`: ± 2,8.

### 5.2 Respuestas a las preguntas fijadas de antemano

1. **¿Los sintéticos reducen la memorización? No en lo que importa.** Contra `eval`, SY50
   frente a F170: +0,3 (semilla 0) y 0,0 (semilla 1). El criterio era más de ~8 puntos en
   ambas semillas: **no se cumple**. La brecha baja (−32/−38 frente a −41/−50) solo porque el
   agente gana menos contra los equipos reales con que entrenó (71–72 % frente a 80–84 %): los
   sintéticos diluyen la exposición a los equipos reales sin mejorar contra los no vistos. En
   `val` el mejor punto de SY50 es algo mayor (60,5 / 60,5 % frente a 56,5 / 58,0 %; incluye
   random), una diferencia del orden del ruido con 30 equipos.
2. **¿Más equipos reales ayudan? No.** Contra `eval`: 50 equipos 36,3 %, 100 equipos 33,3 %,
   170 equipos 38,7 / 34,0 %. No hay tendencia y las diferencias son del tamaño de la
   variación entre semillas (F170: 4,7 puntos entre semilla 0 y 1). Ampliar el pool real no
   resolvería el problema, como anticipaba el usuario (el metajuego varía poco).
3. **¿Gana más contra equipos `eval` parecidos? Apenas.** Victorias por especies en común con
   el equipo de entrenamiento más parecido (3 heurísticos):

| Corrida | ≤ 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| F170, semilla 0 | 33 % (n=12) | 34 % (n=41) | 39 % (n=90) | 40 % (n=157) |
| F170, semilla 1 | 25 % (n=12) | 27 % (n=41) | 32 % (n=90) | 38 % (n=157) |
| S50, semilla 0 | 41 % (n=39) | 35 % (n=40) | 36 % (n=85) | 35 % (n=136) |
| S100, semilla 0 | 37 % (n=30) | 14 % (n=43) | 32 % (n=74) | 39 % (n=153) |

   F170 sube algo con el parecido, pero incluso contra equipos que comparten 5 de 6 especies
   gana ~40 %, frente a 80–84 % contra los propios. **Lo que memoriza no son combinaciones de
   especies sino los equipos concretos** (sets, objetos y, probablemente, la elección del
   preview, que también se entrenó con los equipos de `train`).
4. **¿La validación reservada elige mejor?** No se puede responder del todo: la cola
   diagnosticó solo el `best.pt` (no el último checkpoint). Sí se ve que `val` sigue la
   realidad: sus cifras contra los heurísticos (p. ej. F170 semilla 0 a 1M: 38 / 52 / 30 %,
   media 40 %) coinciden con `eval` (38,7 %), mientras la validación anterior con equipos de
   entrenamiento marcaba ~85 %. En F170 semilla 0 `val` tocó techo en 750k (56,5 %) y bajó a
   52,0 % en 1M; en las otras corridas el máximo cae entre 750k y 1M.

### 5.3 Otras observaciones

- **Espejo de los rivales:** unas 333 000 revisiones por corrida, **0 discrepancias y 0
  errores**, también con 50 % de rivales sintéticos.
- **Tiempo:** los sintéticos no hacen más lento el entrenamiento (SY50: 43–47 min por 1M
  pasos; el resto 46–48 min).

### 5.4 Conclusión para el kick-off

- Sintéticos: **no adoptarlos** con este diseño (criterio no cumplido); queda como resultado
  negativo documentado.
- Pool real: **no ampliarlo**; la cantidad no es el cuello de botella.
- Validación reservada: **recomendada**, porque es la única que mide lo que importa y no
  cuesta rendimiento.
- La memorización sigue abierta. Hipótesis para después del kick-off: el preview v2
  (entrenado con los equipos de `train`) explicaría parte de la brecha (séptima cola: +27,7
  con `train` frente a +17,2 con `eval`); reentrenarlo con `fit` (opción D) lo aclararía.
  Otras vías (no implementadas): regularización (dropout, menos capacidad), entrenar contra
  rivales que juegan con equipos aleatorios del pool de sets, o mezclar sets por especie.

Decisiones del usuario pendientes: sintéticos y validación reservada para el kick-off.

## 6. PPO frente a DDQN (novena cola, commit 7c7a11e, 16-11-2026)

Mismo diagnóstico con los dos pilotos de PPO (400k pasos: random 150k y greedy_solo 250k;
preview v2; equipo fijo; entrenados con equipos de `train` y sin la máscara del aliado, que el
diagnóstico también apaga). 100 partidas por rival y grupo, sin exploración. Resumen en el
servidor: `~/iapsvgc_runs/kickoff9/kickoff_summary9.md`.

| Agente | Rival | Equipos `train` | Equipos `eval` | Diferencia |
|---|---|---|---|---|
| PPO con simetría | greedy_solo | 97 % | 90 % | −7 |
| | greedy_joint | 89 % | 76 % | −13 |
| | expectimax | 81 % | 74 % | −7 |
| | **3 heurísticos** | **89,0 %** | **80,0 %** | **−9,0 (± 2,9)** |
| PPO sin simetría | greedy_solo | 95 % | 77 % | −18 |
| | greedy_joint | 96 % | 72 % | −24 |
| | expectimax | 80 % | 73 % | −7 |
| | **3 heurísticos** | **90,3 %** | **74,0 %** | **−16,3 (± 3,1)** |
| DDQN, mini kick-off 1M (D0, §5) | **3 heurísticos** | **83,0 %** | **35,3 %** | **−47,7 (± 3,5)** |

Contra random ambos PPO ganan 99–100 % con cualquier grupo.

**Lectura:**

1. **PPO generaliza mucho mejor que DDQN.** Contra equipos no vistos gana 74–80 % frente a
   ~35 % de DDQN (unos 40 puntos, ~10 errores estándar), con menos pasos (400k frente a 1M) y
   sin haber entrenado nunca contra greedy_joint ni expectimax (74 % contra expectimax con
   equipos `eval`). La memorización de §5 no es del montaje común (preview, equipos,
   observación): PPO usa el mismo preview v2, los mismos equipos y la misma observación.
   Apunta a DDQN: aprende un valor Q por cada una de las ~1 600 acciones conjuntas, lo que le
   permite ajustar respuestas a equipos concretos; PPO aprende una política por hueco.
   Factores no separados: PPO entrenó menos pasos y sin las fases joint/expectimax (menos
   tiempo para memorizar), y una semilla por piloto.
2. **La simetría podría ayudar a generalizar en PPO.** Con equipos `train` no cambia (89,0
   frente a 90,3 %), pero con `eval` da +6 puntos (80,0 frente a 74,0 %; z ≈ 1,4) y la brecha
   baja a la mitad (−9 frente a −16). Una semilla, no concluyente. **Importante:** la decisión
   de apagar la simetría por defecto (rl_env_spec v1.28) se tomó con validaciones sobre equipos
   de entrenamiento, que miden memorización. Merece revisarse con equipos `eval` o `val`.
3. **Para el benchmark:** la diferencia DDQN–PPO en equipos no vistos es exactamente lo que
   la evaluación final (Conclusión, grupo `eval`) debe capturar. La validación con equipos de
   entrenamiento la ocultaba (DDQN 85 % frente a PPO 92 %).
