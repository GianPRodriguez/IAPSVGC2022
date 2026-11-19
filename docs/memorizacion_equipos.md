# Memorización de los equipos rivales y cómo reducirla

Estado: **colas 20–22 terminadas (18-11-2026): piloto con equipos al azar (§13.2), rival de arquetipos y entropía (§14.4), preview con su propio combate peor que v2 (§15.1).** Decimocuarta cola (17-11-2026): la regularización (REG) cumple el criterio y sube a DDQN a 62–65 % contra `eval` (§8).** Octava cola (16-11-2026): Los sintéticos no cumplen el criterio de adopción; la cantidad de equipos reales no cambia la generalización.** PPO (novena cola, §6) generaliza mucho mejor que DDQN. Decisión del usuario: sin sintéticos, validación reservada por defecto.
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

**Decisiones del usuario (16-11-2026):** sin sintéticos; validación reservada activa por defecto en DDQN y PPO (rl_env_spec v1.29); ensayo general (B) y corrida de 2M (C) en la undécima cola.

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

## 7. Por qué DDQN pierde con equipos no vistos (análisis local, 16-11-2026)

Datos: las partidas de los diagnósticos ya hechos (rival, equipo y resultado de cada una, y el
desglose de decisiones por rival), sin usar el servidor. DDQN actual = B y C de la undécima cola
(6 diagnósticos, 1 800 partidas con `eval` contra los heurísticos); DDQN anterior = las 6
corridas de la octava cola; PPO = los dos pilotos de la novena. Script:
`scripts/analysis/analisis_derrotas_eval.py`.

### 7.1 Las derrotas se concentran en equipos concretos, y son estables

- La tasa de victorias por equipo `eval` varía mucho más de lo que daría el azar (chi² de
  dispersión 499 con 96 grados de libertad; z ≈ 29). 20 de 97 equipos le ganan a DDQN en el 80 %
  o más de las partidas; contra 13 gana el 80 % o más.
- Es estable: dividiendo los checkpoints en dos mitades, la tasa por equipo se correlaciona
  r = 0,69 entre ambas. La configuración anterior falla contra los mismos equipos (r = 0,62).
- **No son malos emparejamientos de nuestro equipo fijo:** los equipos difíciles para DDQN casi
  no lo son para PPO (r = 0,28). Varios equipos contra los que DDQN gana 0 % (n = 12–24) PPO los
  gana 75–100 %. Solo unos pocos son difíciles para ambos: núcleos del metajuego
  (Kingambit + Basculegion + Rillaboom + Floette + Salamence + Sneasler: DDQN 4–14 %, PPO 0–19 %)
  y Trick Room (Indeedee + Hatterene + Torkoal: DDQN 0 %, PPO 25 %). Muestras pequeñas por
  equipo.

### 7.2 No pierde por piezas desconocidas sino por combinaciones

| Sets del equipo `eval` idénticos a uno de `fit` | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Victorias de DDQN | 51 % | 58 % | 37 % | 62 % | 44 % | 44 % | 60 % |
| Partidas | 78 | 66 | 192 | 162 | 162 | 804 | 336 |

Sin tendencia: incluso contra equipos `eval` cuyos 6 Pokémon tienen sets idénticos a sets de
entrenamiento gana 60 %, frente a 88 % con sus propios equipos. Tampoco importa si hay especies
nunca vistas (0: 49 %, 1: 51 %, 2: 39 % con n = 78). Por especie, las peores tasas con `eval`
son de especies que sí conoce y contra las que gana 80–100 % con sus equipos (Golisopod-Mega,
Delphox, Grimmsnarl, Baxcalibur, Froslass: 17–27 % con `eval`). **DDQN responde a la combinación
completa del equipo rival, no a sus piezas.**

### 7.3 DDQN aprende una política compleja que no transfiere; PPO, una simple que sí

Por cada 100 decisiones (3 heurísticos):

| Agente | Grupo | Decisiones por partida | Ataque | Área | Protect | Cambio voluntario | Pass obligatorio |
|---|---|---|---|---|---|---|---|
| DDQN actual | propios | 12,9 | 47,1 | 11,3 | 11,2 | 7,4 | 10,2 |
| DDQN actual | `eval` | 17,0 | 39,5 | 7,9 | 13,6 | 5,3 | 18,2 |
| PPO pilotos | propios | 11,0 | 56,2 | 16,6 | 2,3 | 0,0 | 12,1 |
| PPO pilotos | `eval` | 12,8 | 52,5 | 14,6 | 2,2 | 0,0 | 16,5 |

DDQN usa Protect (11 %) y cambios voluntarios (7 %), decisiones que dependen de anticipar al
rival. Con equipos nuevos sus partidas se alargan (17 frente a 13 decisiones), ataca menos y
pasa más turnos sin Pokémon disponibles (18 % frente a 10 %). PPO juega casi igual con ambos
grupos: una política agresiva y simple (ataca el 70 %, Protect 2 %, nunca cambia por voluntad
propia) que no depende de conocer al rival. Su debilidad potencial es la contraria: contra
rivales que castiguen esa simpleza (Trick Room, núcleos fuertes) no tiene recursos.

### 7.4 Implicación para el punto 3 (remedio del lado del algoritmo)

Las derrotas no son generales sino puntos ciegos estables ante combinaciones de equipo que no
vio, con piezas conocidas. Eso apunta a que la red Q usa la identidad del equipo rival como
clave (una tabla de respuestas) en lugar de razonar con lo que tiene delante. Remedios que
atacan eso, de menor a mayor costo: (a) regularización (weight decay, dropout); (b) ruido o
dropout sobre la parte de la observación que describe el equipo rival no activo; (c) valores Q
por hueco (arquitectura ramificada), la factorización que usa PPO. **Decisión del usuario:
implementar (a) y (b) y probarlas (§8); (c) solo si no alcanzan.**

## 8. Remedios del lado del algoritmo (decimocuarta cola, 16-11-2026)

Implementados en DDQN (ddqn_spec 1.19), todos apagados por defecto:

| Opción | Qué hace | Por qué |
|---|---|---|
| (a) `--weight-decay`, `--dropout` | AdamW con decaimiento de pesos desacoplado; dropout tras cada capa oculta de la red Q (solo al aprender) | Regularización genérica: dificulta que la red guarde una tabla de respuestas por equipo |
| (b) `--foe-bench-dropout p` | Al aprender, cada uno de los 4 Pokémon rivales no activos de la obs (su bloque y sus entradas de daño y velocidad en `pairs`) se pone a cero, la codificación de "ausente", con probabilidad p, por muestra | Ataca lo encontrado en §7: sin la hoja completa del rival no puede usarla como clave y tiene que decidir con lo que está en juego. La inferencia ve la obs completa |

Detalles: el dropout se aplica de forma funcional, así que los checkpoints con y sin dropout
tienen el mismo `state_dict`. La acción a* del objetivo Double DQN se elige con la red online en
modo evaluación (sin dropout) y con s' completo. Tests: `tests/test_ddqn_regularization.py`
(columnas borradas = solo los rivales no activos, versiones NumPy y torch idénticas, valores por
defecto sin cambios, guardar y cargar).

**Experimento (decimocuarta cola).** Configuración del ensayo general B (1M, máscara del aliado,
sin simetría, validación reservada), semillas 0 y 1:

| Corrida | Opciones |
|---|---|
| REG | `--weight-decay 0.01 --dropout 0.1` |
| FBD | `--foe-bench-dropout 0.5` |
| B (referencia, undécima cola) | ninguna: 49,3 / 47,0 % contra `eval` |

**Criterio fijado de antemano:** un remedio se adopta si mejora a B contra `eval` en ambas
semillas y por más de ~5 puntos de media. Si ninguno lo cumple, la opción (c) (valores Q por
hueco) queda como siguiente paso, o la memorización se documenta como característica de DDQN.
Script: `scripts/server/kickoff_queue14.sh`.

### 8.1 Resultados (commit 5936dc2, 17-11-2026)

3 heurísticos, 100 partidas por rival, sin exploración. Resumen en el servidor:
`~/iapsvgc_runs/kickoff14/kickoff_summary14.md`.

| Corrida | Propios | `eval` (best.pt) | `eval` (final.pt) | `eval` solo / joint / expectimax (final.pt) | best.pt (paso, `val`) |
|---|---|---|---|---|---|
| B, semilla 0 | 87,3 % | 49,3 % | 50,0 % | 47 / 53 / 50 | 1M, 61,0 % |
| B, semilla 1 | 86,7 % | 47,0 % | 49,0 % | 47 / 55 / 45 | 1M, 68,0 % |
| **REG, semilla 0** | 93,0 % | 53,3 % | **62,3 %** | 65 / 65 / 57 | 750k, 74,0 % |
| **REG, semilla 1** | 94,0 % | 59,0 % | **65,3 %** | 67 / 67 / 62 | 750k, 78,5 % |
| FBD, semilla 0 | 88,7 % | 49,7 % | 53,3 % | 52 / 60 / 48 | 1M, 72,0 % |
| FBD, semilla 1 | 87,0 % | 46,3 % | 47,7 % | — | 1M, 69,5 % |

| Remedio frente a B | Semilla 0 | Semilla 1 | Media | Criterio |
|---|---|---|---|---|
| REG (best.pt) | +4,0 (± 4,1) | +12,0 (± 4,0) | +8,0 | **cumple** |
| REG (final.pt) | +12,3 | +16,3 | +14,3 | — |
| FBD (best.pt) | +0,3 (± 4,1) | −0,7 (± 4,1) | −0,2 | no cumple |

**Lectura:**

1. **La regularización funciona.** Con REG, DDQN gana 62–65 % contra equipos no vistos con el
   último checkpoint (57–62 % contra expectimax), frente a 49–50 % sin ella. Es la primera vez que
   supera claramente a los heurísticos con equipos nuevos. También gana más con sus propios
   equipos (93–94 % frente a 87 %): la regularización no le quita capacidad de aprender, solo
   reduce el ajuste a equipos concretos. La brecha sigue grande (−35 a −40 con best.pt), así que
   la memorización baja pero no desaparece.
2. **Borrar el banco rival no sirve.** FBD queda igual que B. Que la red no vea parte de la hoja
   del rival al aprender no basta: la memorización no pasa (solo) por esa parte de la
   observación.
3. **`val` eligió mal en REG.** Marcó 750k como mejor (74,0 / 78,5 %), pero el checkpoint de 1M
   rinde ~8 puntos más contra `eval`. Con 30 equipos y 50 partidas por rival (± 7 puntos por
   celda), `val` no distingue checkpoints cercanos. Opciones: más partidas en la validación,
   o usar el último checkpoint.
4. Espejo de los rivales: 0 discrepancias en las 4 corridas (~1,4 M revisiones).

**Decisiones del usuario (17-11-2026):** REG por defecto en DDQN (ddqn_spec 1.20) y
checkpoint principal final.pt (protocolo §3c.2); sus hiperparámetros se ajustan en la
decimoquinta cola. La opción (c), valores Q por hueco, queda en reserva.

## 9. Limitación: decisiones tomadas mirando `eval`

Durante esta exploración, varias decisiones de configuración se tomaron comparando resultados
contra los equipos `eval`: la máscara del aliado (colas 11 y 13), la simetría en PPO (cola 12) y
REG (cola 14). Esos 100 equipos son también los de la evaluación final, así que la cifra final
contra `eval` puede estar algo inflada: se eligieron las opciones que mejor les fue a esos
equipos. Factores que lo limitan:

- Son pocas decisiones, entre 2 y 4 opciones cada una, con diferencias grandes (de +6 a +17
  puntos) y repetidas en 2 semillas; el sesgo de selección esperable es mucho menor que esos
  efectos.
- Las opciones elegidas tienen un motivo independiente del resultado (no atacar al aliado,
  invariancia exacta del juego, regularización estándar contra el sobreajuste).

**Medidas desde el 17-11-2026:** el ajuste de REG (decimoquinta cola) se elige solo con `val`, y
la Conclusión formal (run_robust_e2e, BO3, bloques de semillas C1/C2) usa semillas de partida
nuevas. Para la tesis se reporta esta limitación junto con los resultados.

## 10. Valores Q por hueco (opción C; decimoctava cola, 17-11-2026)

Aprobada por el usuario para antes del kick-off. Motivo: con REG, DDQN gana 62–65 % contra
`eval`, pero sigue ~30 puntos por debajo de su rendimiento con los equipos propios y ~20 por
debajo de PPO. §7 mostró que responde a la combinación completa del equipo rival. La diferencia
estructural con PPO es que PPO decide por hueco (dos distribuciones de 40 acciones) mientras la red
Q de DDQN tiene una salida por cada una de las 1 600 acciones conjuntas: espacio para guardar una
respuesta distinta por combinación.

**Diseño** (ddqn_spec 1.21, `--q-head factored --q-rank r`):

    Q(s, a1, a2) = V(s) + A1(s, a1) + A2(s, a2) + Σ_k U1(s, a1)_k · U2(s, a2)_k

- Dos cabezas de 40 valores (una por hueco) más un valor de estado; la interacción de rango r
  permite coordinar los huecos (p. ej. concentrar ataques en un objetivo) sin volver a 1 600
  salidas libres. r = 0 es puramente aditiva.
- La salida se arma como el mismo vector de 1 600 valores (índice 40·a1 + a2), así que la máscara
  conjunta (que ya prohíbe, por ejemplo, que ambos huecos cambien al mismo Pokémon), el objetivo
  Double DQN, el replay, la simetría y la inferencia no cambian.
- Salidas: 81 + 2·40·r (rango 8: 721) frente a 1 600; la capa de salida tiene ~5 % (rango 0) y
  ~45 % (rango 8) de los parámetros de la conjunta.

Tests: `tests/test_ddqn_factored_head.py` (fórmula e índice conjunto, menos parámetros, aprendizaje
con simetría, acción greedy legal, guardar y cargar con reconstrucción de la red, checkpoints
anteriores como conjuntos).

**Experimento:** QF0 (rango 0) y QF8 (rango 8), 1M, semillas 0 y 1, configuración por defecto +
REG elegida en la decimoquinta cola. Elección con `val` (200 partidas por rival, final.pt): se
adopta la mejor variante si supera a la cabeza conjunta por más de 4 puntos. `eval` y propios
como registro. Script: `scripts/server/kickoff_queue18.sh`; resumen:
`showdown/rl/kickoff_summary18.py`.

**Resultados (commit 442d741, 17/18-11-2026): no supera el umbral; se mantiene la cabeza conjunta.**

| Variante | `val` s0 / s1 | `val` media | Diferencia con la conjunta (74,1 %) | `eval` s0 / s1 (registro) | `eval` media | Propios |
|---|---|---|---|---|---|---|
| Rango 0 (aditiva) | 76,3 / 78,3 % | 77,3 % | +3,2 (± 1,7) | 66,7 / 56,0 % | 61,3 % | 94,2 % |
| Rango 8 | 77,2 / 73,8 % | 75,5 % | +1,4 (± 1,8) | 58,0 / 63,0 % | 60,5 % | 94,3 % |
| Conjunta (REG 0,03 + 0,1, cola 15) | 74,7 / 73,5 % | 74,1 % | — | 63,3 / 66,0 % | 64,7 % | — |

La cabeza por hueco aprende algo más rápido (rango 0: 76 % de `val` con random a 250k) y sube un
poco en `val`, pero no en `eval` (−3,4 puntos de media; la semilla 0 dio +3,4 y la 1 −10). La
memorización no baja (propios ~94 % en ambas). **Conclusión:** la factorización por hueco no
explica por sí sola la ventaja de PPO en generalización; la diferencia debe estar en otra parte
(on-policy frente a replay, la política estocástica, el objetivo de PPO).

### 8.2 Ajuste de REG (decimoquinta cola, 17-11-2026)

Tres variantes más fuertes, 1M × 2 semillas, elegidas con `val` (200 partidas por rival, final.pt;
regla: más de 4 puntos sobre REG actual). `val` media / `eval` media (registro, 100 por rival):
REG actual 67,6 / 63,8 %; **weight decay 0,03 + dropout 0,1: 74,1 / 64,7 % (elegida)**; 0,01 + 0,2:
66,9 / 61,5 %; 0,03 + 0,2: 72,8 / 62,0 %. Más weight decay ayuda en `val`; más dropout no. La mejora
en `eval` es pequeña (+0,9), así que la distancia con `val` sugiere que parte de la subida es ruido
o propia de los 30 equipos de `val`.

## 11. ¿Los agentes detectan la estrategia (arquetipo) del equipo rival? (17-11-2026)

Pregunta del usuario: hay equipos cuya estrategia gira en torno a pocos movimientos (Trick Room,
Tailwind, Fake Out y posicionamiento, redirección, clima). ¿DDQN y PPO lo detectan?

**La información está en la observación.** Gracias a la hoja abierta, cada Pokémon rival lleva
indicadores de utilidad de sus movimientos (`fakeout`, `tailwind`, `trickroom`, `redirect`,
`protect`, `wideguard`, `helpinghand`…). El bloque de campo indica si Trick Room está activo y
cuántos turnos le quedan, el Tailwind de cada lado y sus turnos, y cuántos Fake Out, Protect,
cambios y ataques en área usó cada lado el turno anterior. Cada Pokémon rival activo marca si
todavía puede usar Fake Out (primer turno en el campo).

**Rendimiento por arquetipo** (diagnósticos ya hechos: DDQN con REG, colas 14–15; PPO con
simetría, cola 12; etiquetas a partir de la hoja del rival; `scripts/analysis/analisis_arquetipos.py`):

#### Rendimiento por arquetipo del equipo rival (3 heurísticos)

| Arquetipo | Equipos `eval` | Equipos `fit` | DDQN (REG) `eval` | PPO (simetría) `eval` | DDQN (REG) propios | PPO (simetría) propios |
|---|---|---|---|---|---|---|
| (todos) | 100 | 170 | 62 % (n=1800, ± 2) | 83 % (n=1200, ± 2) | 94 % (n=600, ± 2) | 96 % (n=600, ± 2) |
| Trick Room | 22 | 42 | 63 % (n=408, ± 5) | 85 % (n=272, ± 4) | 96 % (n=132, ± 3) | 98 % (n=132, ± 3) |
| Tailwind | 70 | 115 | 61 % (n=1164, ± 3) | 81 % (n=776, ± 3) | 94 % (n=436, ± 2) | 98 % (n=436, ± 1) |
| Sin control de velocidad (TR/TW) | 14 | 38 | 65 % (n=288, ± 6) | 91 % (n=192, ± 4) | 90 % (n=116, ± 6) | 89 % (n=116, ± 6) |
| Fake Out ×2+ | 58 | 102 | 61 % (n=1110, ± 3) | 81 % (n=740, ± 3) | 94 % (n=360, ± 2) | 94 % (n=360, ± 2) |
| Redirección | 21 | 35 | 58 % (n=420, ± 5) | 87 % (n=280, ± 4) | 93 % (n=106, ± 5) | 98 % (n=106, ± 3) |
| Clima | 38 | 55 | 60 % (n=690, ± 4) | 90 % (n=460, ± 3) | 92 % (n=186, ± 4) | 98 % (n=186, ± 2) |

#### Solo contra expectimax (el rival que mejor ejecuta la estrategia del equipo), equipos `eval`

| Arquetipo | DDQN (REG) | PPO (simetría) |
|---|---|---|
| (todos) | 61 % (n=600, ± 4) | 81 % (n=400, ± 4) |
| Trick Room | 55 % (n=150, ± 8) | 76 % (n=100, ± 8) |
| Tailwind | 66 % (n=372, ± 5) | 80 % (n=248, ± 5) |
| Sin control de velocidad (TR/TW) | 58 % (n=108, ± 9) | 94 % (n=72, ± 5) |
| Fake Out ×2+ | 60 % (n=324, ± 5) | 77 % (n=216, ± 6) |
| Redirección | 56 % (n=162, ± 8) | 83 % (n=108, ± 7) |
| Clima | 58 % (n=246, ± 6) | 91 % (n=164, ± 4) |

**Lectura:**

1. **No hay un arquetipo que hunda a ninguno de los dos.** Las tasas por arquetipo quedan dentro
   del margen de error de la media (DDQN ~62 %, PPO ~83 % con `eval`). Contra expectimax, Trick
   Room es el arquetipo más difícil para ambos (DDQN 55 %, PPO 76 %), pero dentro del ruido (± 8).
   La debilidad de §7 (equipos concretos) no se explica por el arquetipo: hay equipos Trick Room
   fáciles y difíciles.
2. **Rendir igual no prueba que adapten el juego.** La tasa de victorias no dice si el agente
   reconoce el plan del rival (por ejemplo, usar Protect o cambiar mientras dura el Trick Room
   rival, atacar primero al que lo activa, o protegerse del Fake Out en el primer turno). Para
   saberlo hace falta medir las decisiones según el contexto, algo que `policy_diag` hoy agrega
   sin distinguir.
3. **Los rivales heurísticos solo ejecutan el arquetipo en parte.** greedy_joint y greedy_solo
   valoran Tailwind y Trick Room solo cuando su cálculo de velocidad lo pide, y expectimax los
   incluye en su búsqueda. Ninguno juega el plan completo como un jugador humano (por ejemplo,
   preparar Trick Room con redirección y Fake Out). Que un equipo tenga Trick Room no garantiza
   que el rival lo active en la partida.

## 12. Decisiones según el plan del rival y uso real de los arquetipos (decimonovena cola)

Aprobado por el usuario tras §11, para antes del kick-off. Dos mediciones nuevas en
`policy_diag --context` (`showdown/rl/context_diag.py`):

1. **Decisiones según el contexto.** Cada decisión libre contra los heurísticos se clasifica
   según lo que hace el rival en ese momento: Trick Room rival activo (quién lo puso se lee del
   protocolo), Tailwind rival activo, rival con Fake Out disponible (primer turno en el campo),
   rival con redirección en el campo, rival que puede activar Trick Room y todavía no lo hizo, y
   primer turno. Para cada contexto: % de Protect, cambios voluntarios, ataques, ataques en área y
   movimientos de estado; y, para los ataques a un solo rival, cuántos van al que puede activar
   Trick Room o al redirector, frente a lo esperado si eligiera el objetivo al azar.
2. **Uso real de los movimientos clave.** Por partida, cuántas veces cada lado usó Trick Room,
   Tailwind, Fake Out y redirección, y cuántas veces se activaron Trick Room y Tailwind (protocolo
   crudo de Showdown). Responde si los rivales heurísticos juegan su arquetipo.

**Equipo del agente al azar en cada partida** (decisión del usuario, solo para este análisis): el
agente juega con equipos del grupo `eval` que nunca usó, sorteados en cada partida, contra rivales
también de `eval`. Así el contexto no depende del arquetipo del equipo fijo. Se registra el equipo
propio de cada partida. Límite: los agentes se entrenaron con el equipo fijo, así que las victorias
con equipos al azar miden también la generalización a equipos propios nuevos.

Agentes: DDQN con la REG elegida (decimoquinta cola) y PPO con simetría (duodécima cola), final.pt,
2 semillas, 200 partidas por rival. Tests: `tests/test_context_diag.py`. Script:
`scripts/server/kickoff_queue19.sh`; resumen: `showdown/rl/kickoff_summary19.py`.

### 12.1 Resultados (commit 97bbad2, 18-11-2026)

**1. Con un equipo propio que no conocen, ambos agentes se desploman.**

| Agente | random | greedy_solo | greedy_joint | expectimax |
|---|---|---|---|---|
| DDQN, equipo propio al azar | 88 % | 22 % | 17 % | 18 % |
| PPO, equipo propio al azar | 93 % | 32 % | 34 % | 28 % |
| Referencia con el equipo fijo (`eval`) | — | DDQN ~63 %, PPO ~83 % (3 heurísticos) | | |

Los dos están especializados en el equipo con que entrenaron (el preview v2 también). PPO
generaliza mejor que DDQN a rivales nuevos, pero no a equipos propios nuevos. Para la tesis, los
agentes son pilotos de un equipo concreto, no jugadores generales.

**2. Los rivales heurísticos casi nunca juegan el arquetipo de su equipo** (% de partidas con al
menos un uso; 22 % de los equipos `eval` tienen Trick Room y 70 % Tailwind):

| Rival | Activa Trick Room | Activa Tailwind | Usa Fake Out | Usa redirección |
|---|---|---|---|---|
| greedy_solo | 0 % | 0 % | 0 % | 4 % |
| greedy_joint | 1 % | 2–3 % | 26 % | 0–2 % |
| expectimax | 0 % | 3–4 % | 28 % | 2–3 % |
| random (referencia) | 4–6 % | 15–20 % | 26–32 % | 4–6 % |

Ni siquiera el rival al azar activa tan poco Tailwind como los heurísticos: sus reglas de
velocidad casi nunca lo justifican. **La evaluación actual no pone a prueba el juego contra Trick
Room ni Tailwind**; sí contra Fake Out (joint y expectimax lo usan en ~27 % de las partidas).

**3. Los agentes no adaptan su juego al plan del rival:**

| Situación | DDQN | PPO | Lectura |
|---|---|---|---|
| Ataques al que puede activar Trick Room (antes de que lo haga) | 54 % (azar 58 %) | 47 % (azar 59 %) | No lo priorizan; PPO incluso menos que al azar |
| Ataques al redirector | 46 % (azar 50 %) | 51 % (azar 50 %) | Indiferentes |
| Protect con Fake Out rival disponible | 11,9 % (sin contexto 15,0 %) | 13,3 % (17,7 %) | No se protegen más del Fake Out |
| Protect con Trick Room rival activo | 27,8 % (n = 18) | 20,0 % (n = 15) | Algo más, pero muestras mínimas |

Límite: con equipos propios al azar los agentes juegan fuera de su distribución de entrenamiento;
aun así, ningún indicador muestra que reconozcan el plan del rival. Es coherente con el punto 2:
sus rivales de entrenamiento tampoco ejecutan esos planes, así que no tuvieron de qué aprender.

**Implicaciones:** (a) para que los agentes aprendan a jugar contra arquetipos hacen falta rivales
de entrenamiento y evaluación que los ejecuten (trabajo futuro, p. ej. heurísticos con plan de
Trick Room / Tailwind o self-play); (b) la generalización a equipos propios nuevos requeriría
entrenar con equipos propios variados (hoy se usa el equipo fijo por decisión del usuario).

## 13. Preview y combate por separado; piloto con equipos propios al azar (vigésima cola, 18-11-2026)

Aprobado por el usuario. El agente es una tupla (preview, combate) que decide por separado en
cada etapa, y el resultado final exige que ambas jueguen bien. Hasta ahora casi todo el análisis
fue del combate; el preview v2 (PPO de preview) se entrenó con un combate heurístico
(greedy_joint) como piloto, no con DDQN ni PPO.

**1. Contribución del preview.** Mismo combate (final.pt de la configuración final, 2 semillas
por agente), tres previews, tu equipo fijo, rivales `eval`, 200 partidas por rival:

| Preview | Qué aísla |
|---|---|
| v2 (la red) | La tupla completa |
| Formación fija: la que más elige v2 con ese agente, jugada siempre | v2 − fija = cuánto aporta adaptar la formación al rival |
| `/team 1234` (los cuatro primeros) | fija − 1234 = cuánto aporta elegir bien los cuatro |

Implementación: `PREVIEW_FORCE_ORDER` en `showdown/rl/frozen_preview.py` (fuerza una formación
sin consultar la red) y registro de la formación de cada partida en `policy_diag`. Límite: el
combate se entrenó con las formaciones de v2; con otras formaciones juega algo fuera de su
distribución, así que las diferencias mezclan la calidad del preview con la adaptación del
combate a él.

**2. Combate según el plan del rival con tu equipo** (`--context` en la corrida con v2): repite el
análisis de §12 en la distribución de entrenamiento.

**3. Piloto con equipo propio al azar:** DDQN y PPO, 1M, semillas 0 y 1, configuración final,
`--agent-team pool` (equipo propio sorteado del grupo `fit` en cada partida). Evaluación contra
rivales `eval` con equipo propio al azar y con tu equipo. Pregunta: ¿aprenden a jugar con
cualquier equipo y cuánto pierden con el tuyo? Límite: el preview v2 se entrenó para tu equipo y
aquí trabaja fuera de su distribución.

Tests: `tests/test_preview_analysis.py`. Script: `scripts/server/kickoff_queue20.sh`; resumen:
`showdown/rl/kickoff_summary20.py`.

### 13.1 Resultados de la contribución del preview (commit 86b212f, 18-11-2026)

3 heurísticos, 2 semillas agrupadas (1 200 partidas por celda):

| Agente | Preview v2 | Formación fija (3546, la más frecuente) | /team 1234 | v2 − fija (adaptación) | fija − 1234 (elección) |
|---|---|---|---|---|---|
| DDQN | 64,4 % | 60,3 % | 37,3 % | +4,1 (± 2,0) | +23,0 (± 2,0) |
| PPO | 85,3 % | 82,5 % | 71,3 % | +2,8 (± 1,5) | +11,2 (± 1,7) |

Por semilla, v2 − fija: DDQN +9,0 y −0,8; PPO +0,9 y +4,8.

1. **Elegir bien los cuatro importa mucho; adaptar la formación al rival, poco.** Jugar siempre
   la formación 3546 conserva casi todo el rendimiento; traer los cuatro primeros lo hunde, sobre
   todo a DDQN (−23 puntos).
2. **PPO depende menos de la formación que DDQN** (con /team 1234 pierde 11 puntos frente a 23):
   su combate es más robusto a qué Pokémon tiene en el campo.
3. **El preview elige igual para los dos agentes** (mismo preview, mismo equipo): 17 formaciones
   distintas; 3546 en el 19 % de las partidas y las tres más frecuentes en el 47 %. Es decir, no
   tiene en cuenta cómo juega cada combate.
4. **Por qué /team 1234 hunde tanto el rendimiento.** Con tu equipo (1 Raichu, 2 Rillaboom,
   3 Gholdengo, 4 Arcanine-Hisui, 5 Staraptor, 6 Primarina), 1234 abre con Raichu + Rillaboom y deja
   fuera a Staraptor y Primarina. El preview v2 **nunca** eligió 1234 ni ese par de leads (0 de 2 400
   partidas), y Rillaboom nunca sale de lead (0 %), aunque se trae en el 46 % de las partidas.
   Traídos: Arcanine-Hisui 92 %, Primarina 79 %, Staraptor 72 %, Gholdengo 63 %, Raichu 48 %,
   Rillaboom 46 %. Pares de leads más usados: Raichu + Primarina 32 %, Gholdengo + Staraptor 30 %,
   Raichu + Arcanine-Hisui 13 %. Como el combate se entrenó con este preview, **nunca practicó
   abrir con Raichu + Rillaboom**: con 1234 juega una situación que no vio, además de con una
   formación probablemente peor. DDQN, que memoriza más, sufre más (−23 frente a −11 de PPO). La
   caída mezcla así dos efectos: peor formación y combate fuera de su distribución.
5. **Corrección sobre Protect en PPO.** La "política sin Protect" de §7.3 (2 %) era del piloto de
   400k. El PPO final (1M, simetría, máscara del aliado) usa Protect en el 11,2 % de sus decisiones
   libres con tu equipo, igual que DDQN (11,7 %). Lo que sigue en 0 % son sus cambios voluntarios
   (DDQN 5,3 %). Ninguno ajusta mucho el Protect al contexto: PPO 14,1 % con Fake Out rival
   disponible, 13,6 % en el primer turno y 2,4 % con Tailwind rival activo (n = 42), frente a
   11,0 % sin contexto.
6. Conclusión: el preview aporta sobre todo una buena formación base; su adaptación al rival es
   pequeña (+3–4 puntos) y no está coordinada con el combate de cada agente. Motiva entrenar el
   preview con su propio combate (§15).

### 13.2 Resultados del piloto con equipo propio al azar (18-11-2026)

Entrenado con un equipo propio sorteado en cada partida entre los 170 de `fit` (~107 000 partidas
en 1M pasos: ~630 por equipo propio, frente a ~107 000 con el equipo fijo). Contra los 3
heurísticos, rivales `eval`, 2 semillas, 200 partidas por rival:

| Agente | Entrenado con | Equipo propio al azar (`eval`) | Tu equipo |
|---|---|---|---|
| DDQN | tu equipo | 18,9 % | 64,7 % |
| DDQN | **equipo al azar** | **19,5 %** (+0,6 ± 1,6) | **29,8 %** (−34,8 ± 2,4) |
| PPO | tu equipo | 31,5 % | 83,3 % |
| PPO | **equipo al azar** | **45,5 %** (+14,0 ± 2,0) | **72,8 %** (−10,6 ± 2,0) |

Por semilla: DDQN 19,8 / 19,2 % y 32,0 / 27,7 %; PPO 42,7 / 48,3 % y 73,8 / 71,7 %.

1. **DDQN no aprende nada general** al entrenar con equipos variados y pierde la mitad de su
   rendimiento con tu equipo: diluir la práctica (~630 partidas por equipo) no le deja dominar
   ninguno.
2. **PPO sí generaliza en parte:** +14 puntos con equipos propios desconocidos (casi 50 % contra
   greedy_solo) a cambio de −10,6 con tu equipo, con el que sigue ganando a los tres heurísticos.
3. Es el mismo patrón visto con rivales, formaciones y equipos: **PPO generaliza, DDQN memoriza**.
   Con 1M pasos, especializarse en un equipo sigue siendo lo mejor para ambos; un agente general
   necesitaría más entrenamiento y un preview entrenado con equipos propios variados (el v2 es el de
   tu equipo). Trabajo futuro.

## 14. Rendimiento por baseline, rival de arquetipos y entropía de PPO (18-11-2026)

### 14.1 Rendimiento contra cada baseline (configuración final, 1M, entrenados con tu equipo)

Victorias sin exploración; final.pt salvo que se indique. "Conocidos" = equipos de
entrenamiento; `eval` y `val` = equipos nunca entrenados; "propio al azar" = el agente juega con un
equipo propio que no conoce (cola 19).

| Agente | Rivales | random | greedy_solo | greedy_joint | expectimax | 3 heurísticos |
|---|---|---|---|---|---|---|
| DDQN | conocidos ¹ | 98,5–99,5 % | 93,0–97,5 % | 94,5–96,5 % | 86,5–88,5 % | 92,0–93,5 % |
| DDQN | **`eval`** | 99,5 % | 63,0 % | 66,5 % | **64,5 %** | **64,7 %** |
| DDQN | `val` (200 por rival) | 97,5 % | 84,5 % | 74,0 % | 63,8 % | 74,1 % |
| DDQN | propio al azar (`eval`) | 88,2 % | 21,5 % | 17,2 % | 18,0 % | 18,9 % |
| PPO | conocidos (best.pt) | 100 % | 98,5 % | 97,5 % | 92,0 % | 96,0 % |
| PPO | **`eval`** | 99,0 % | 85,0 % | 85,0 % | **80,0 %** | **83,3 %** |
| PPO | `val` (200 por rival) | 100 % | 94,2 % | 91,2 % | 78,8 % | 88,1 % |
| PPO | propio al azar (`eval`) | 93,0 % | 32,5 % | 33,5 % | 28,5 % | 31,5 % |

¹ DDQN con REG 0,03 a 1M no se midió con equipos conocidos; rango de corridas equivalentes (REG
0,01 best.pt y REG 0,03 con simetría). 2 semillas; 200 partidas por rival y grupo (400 en `val` y
con propio al azar).

**Lectura:** contra rivales desconocidos los dos superan a los tres heurísticos, incluido
expectimax (DDQN ~64 %, PPO ~80 %), y contra conocidos casi los dominan. Contra expectimax la
brecha conocidos − `eval` es ~−23 en DDQN y −12 en PPO. Con un equipo propio desconocido ambos
pierden contra todos los heurísticos: su fuerza depende de pilotar tu equipo. En `val`, greedy_solo
y joint resultan más fáciles que en `eval` (30 equipos, distinta composición).

### 14.2 Rival de arquetipos (vigesimoprimera cola)

La cola 19 mostró que los heurísticos casi nunca activan Trick Room (0–1 % de las partidas) ni
Tailwind (0–4 %), aunque 22 % y 70 % de los equipos los traen: la evaluación no ponía a prueba esos
planes. **Modo arquetipo de greedy_joint** (`PS_JOINT_ARCHETYPE=1`, `joint.py`): Tailwind recibe
un bono dominante (250) siempre que no esté puesto, y Trick Room cuando no está puesto y su lado es
más lento, sin exigir que el compañero haga daño ese turno (el compañero cubre con Fake Out,
redirección, Protect o un ataque). Apagado por defecto: greedy_joint no cambia (test). Uso:
evaluación de los agentes finales (2 semillas, tu equipo, `eval`, 200 partidas, con `--context`)
y, en el futuro, entrenamiento. Mide también si el rival efectivamente activa los planes.
Prueba en vivo (PPO con simetría, semilla 0, 24 partidas contra joint): el rival activó Tailwind en
0/24 partidas en modo normal y 6/24 en modo arquetipo, y Trick Room en 0/24 y 2/24 (solo cuando trae
a quien lo pone y su lado es más lento); PPO ganó 23/24 frente a 19/24. Muestra mínima: la cifra
confiable sale de la cola.

### 14.3 Entropía de PPO (vigesimoprimera cola)

Para igualar el esfuerzo de ajuste con DDQN (que tuvo el de REG): `--ent-coef` 0,003 y 0,03 frente
a 0,01, 1M × 2 semillas, configuración final; elección con `val` (más de 4 puntos). Más entropía
podría además darle a PPO el repertorio que le falta (nunca cambia por voluntad propia); el resumen
registra el % de cambios voluntarios.

Tests: `tests/test_joint_archetype.py`. Script: `scripts/server/kickoff_queue21.sh`; resumen:
`showdown/rl/kickoff_summary21.py`.

### 14.4 Resultados (commit c9a5a3f, 18-11-2026)

**Rival de arquetipos** (tu equipo, `eval`, 2 semillas, 400 partidas contra joint por agente):

| Agente | Contra joint normal | Contra joint arquetipo | Diferencia | Rival activa Trick Room | Rival activa Tailwind |
|---|---|---|---|---|---|
| DDQN (REG 0,03) | 62,3 % | 53,8 % | −8,5 | 0 % → 12 % | 1 % → 29 % |
| PPO (simetría) | 85,8 % | 78,8 % | −7,0 | 0 % → 10 % | 1 % → 28 % |

- El modo arquetipo sí ejecuta los planes (Tailwind en ~29 % de las partidas, Trick Room en
  ~11 %, frente a ~0–1 %), y **ambos agentes pierden 7–9 puntos**: la evaluación con heurísticos
  normales los sobreestimaba frente a planes de velocidad. Siguen ganando (DDQN 54 %, PPO 79 %).
- **PPO sí adapta algo su Protect:** 24,2 % con Tailwind rival activo y 18,1 % con Trick Room rival,
  frente a 10,6 % sin contexto (esperar a que se agote el efecto). DDQN casi no (14,8 % y 9,9 %
  frente a 12,5 %), aunque cambia algo más con Trick Room rival (7,2 % frente a 4,7 %).
- Ninguno ataca primero al que puede activar Trick Room (DDQN 48 %, PPO 41 %, frente a ~55 % al
  azar) ni al redirector.

**Entropía de PPO** (1M, 2 semillas; elección con `val`, umbral +4):

| Entropía | `val` | Diferencia con 0,01 (88,1 %) | `eval` (registro) | Propios | Cambios voluntarios |
|---|---|---|---|---|---|
| 0,003 | 86,0 % | −2,1 (± 1,4) | 84,5 % | 95,3 % | 0,0 % |
| 0,01 (actual) | 88,1 % | — | 83,4 % | ~96 % | 0,0 % |
| 0,03 | 87,6 % | −0,5 (± 1,3) | 84,5 % | 96,0 % | **3,0 %** |

Ninguna supera el umbral: **se mantiene 0,01**. Hallazgo: con entropía 0,03 PPO empieza a cambiar
de Pokémon por voluntad propia (3 %) sin perder rendimiento. Confirma que su 0 % de cambios era
falta de exploración y no una imposibilidad, pero contra estos rivales cambiar no le da ventaja.

## 15. Preview entrenado con su propio combate (vigesimosegunda cola, 18-11-2026)

Aprobado por el usuario ("aplicar recomendación") tras §13.1. El preview v2 se entrenó con
greedy_joint como combate (ambos lados), así que aprendió qué formaciones le sirven a greedy_joint,
no a DDQN ni a PPO; elige lo mismo para los dos agentes y su adaptación al rival aporta solo +3–4
puntos.

**Implementación:** `preview_live` acepta ahora pilotos RL (`--pilot ddqn|ppo`): nuestro lado juega
con el combate congelado del agente (`PREVIEW_PILOT_CHECKPOINT`, jugadores de inferencia
`DDQNPlayer` / `PPOPlayer`) y el rival con un heurístico (`PREVIEW_FOE_PILOT`, por defecto
greedy_joint, como en el entrenamiento original). La formación de nuestro lado la sigue eligiendo el
preview que se entrena. Tests: `tests/test_preview_rl_pilot.py`. Prueba en vivo (4 workers, 2
actualizaciones, piloto DDQN): 5 120 partidas sin fallos ni reinicios, 28,6 partidas/s (~7 por
worker, igual que con greedy_joint), así que un preview completo toma ~45–50 min con 28 workers.

**Experimento:** un preview por agente, con su combate (final.pt semilla 0) y solo con los 170
equipos de `fit` (así `val` queda desconocido también para el preview); misma receta que v2 (200
actualizaciones, entropía 0,005, tu equipo). Después, mismo combate (semillas 0 y 1) con v2 y con
el preview nuevo, contra `val` (elige: más de 4 puntos) y `eval` (registro), 200 partidas por rival.
Script: `scripts/server/kickoff_queue22.sh`; resumen: `showdown/rl/kickoff_summary22.py`.

### 15.1 Resultados (18-11-2026): **el preview nuevo es peor; se mantiene v2**

| Agente | `val` v2 | `val` nuevo | Diferencia | `eval` v2 | `eval` nuevo | Diferencia |
|---|---|---|---|---|---|---|
| DDQN (REG 0,03) | 74,2 % | 68,4 % | −5,8 (± 1,8) | 64,4 % | 61,2 % | −3,2 (± 2,0) |
| PPO (simetría) | 88,2 % | 86,8 % | −1,4 (± 1,4) | 85,3 % | 81,7 % | −3,7 (± 1,5) |

Mismo combate (final.pt, semillas 0 y 1), tu equipo, 200 partidas por rival.

**Por qué salió peor (hipótesis principal: señal saturada).** Durante el entrenamiento del preview,
nuestro lado ganó el 92 % (DDQN) y el 97 % (PPO) de las ~415 000 partidas contra greedy_joint. Si
casi cualquier formación gana, la recompensa casi no distingue entre formaciones y el preview
aprende poco o preferencias poco útiles. Con v2 (greedy_joint contra greedy_joint) también ganaba
~90 %, pero su combate era más débil y dependía más de la formación. Además, entrenar con un solo
combate (semilla 0) puede ajustar el preview a sus manías, y `fit` tiene 30 equipos menos. Nota: la
cifra MEAN_WR del entrenamiento (0,97 / 0,99) compara con el preview propio del piloto, que en los
workers cae a /team 1234 (no tiene checkpoint de preview configurado), así que no sirve como
referencia.

**Para un intento futuro:** un rival más fuerte para el preview (expectimax, o el rival de
arquetipos) para que la recompensa vuelva a discriminar, y entrenar con varios combates (varias
semillas).


## 16. Arquetipos del pool de equipos (18-11-2026)

Pregunta del usuario tras la prueba online (ddqn_kickoff_checklist.md, "Prueba online"): ¿qué
arquetipos hay en nuestros 300 equipos? Script: `scripts/analysis/arquetipos_pool.py` (etiquetas a
partir de las hojas: movimientos, habilidades y objetos; un equipo puede tener varias).

**Arquetipo primario** (precedencia: Trick Room > clima > Tailwind > Campo Psíquico > equilibrado):

| Arquetipo | `train` | `eval` | Total | % |
|---|---|---|---|---|
| Tailwind | 72 | 45 | 117 | 39 % |
| Trick Room | 48 | 22 | 70 | 23 % |
| Equilibrado / ofensivo (sin plan de campo) | 36 | 5 | 41 | 14 % |
| Sol | 18 | 9 | 27 | 9 % |
| Arena | 7 | 9 | 16 | 5 % |
| Lluvia | 9 | 7 | 16 | 5 % |
| Nieve | 10 | 2 | 12 | 4 % |
| Campo Psíquico | 0 | 1 | 1 | 0 % |

**Etiquetas** (no excluyentes; "Equipo fijo" = tu equipo, `ddqn_team.txt`):

| Etiqueta | Total (de 300) | Equipo fijo |
|---|---|---|
| Setup (Swords Dance, Nasty Plot, Dragon Dance…) | 207 | sí |
| Tailwind | 207 | sí (Staraptor) |
| Campo de Hierba (casi siempre Rillaboom) | 195 | sí |
| Fake Out en 2+ Pokémon | 176 | sí |
| Trick Room | 70 | |
| Redirección (Follow Me / Rage Powder) | 60 | |
| Wide Guard | 55 | |
| Lluvia (fuente) | 51 | |
| Disrupción (Encore, Taunt, Trick, Imprison) | 49 | |
| Pantallas | 39 | |
| Sol (fuente + abusador o ataque de fuego/Weather Ball) | 39 | |
| Campo Psíquico | 34 | |
| Sueño (Spore, Sleep Powder, Yawn…) | 32 | |
| Icy Wind / Electroweb / Thunder Wave | 19 | |
| Arena | 18 | |
| Nieve | 13 | |
| Perish Song | 11 | |

Combinaciones de control de velocidad más comunes: solo Tailwind 117, ninguna 42,
Trick Room + Tailwind + Lluvia 20, Tailwind + Sol 19, solo Trick Room 17, Trick Room + Tailwind 12.

**Lectura:**
1. **El meta del pool es Tailwind + Trick Room:** 207 equipos llevan Tailwind y 70 Trick Room, pero
   los heurísticos casi nunca los usan (~0–1 % de las partidas, §14). Es decir, en el entrenamiento
   **el plan principal de ~3 de cada 4 equipos rivales nunca se ejecuta**. Es el hueco más grande, y
   el que el rival de arquetipos cubre solo en parte (§14.4).
2. **El clima sí se vio en el entrenamiento:** viene de habilidades (Drought, Drizzle, Sand Stream,
   Snow Warning, Charizard-Mega-Y) y se activa solo, sin decisión del heurístico. Sol en 39 equipos
   (23 de `train`). **Corrección a la prueba online:** el agente sí había enfrentado sol; lo nuevo
   contra velascoAlvarado fue cómo se jugó (doble Protect el turno 1 mientras evoluciona el Mega,
   Sleep Powder, Parting Shot), no el clima en sí.
3. **Sueño** (32 equipos) y **disrupción** (49) existen en el pool, pero un heurístico codicioso
   rara vez elige movimientos de estado frente a un ataque.
4. **Tu equipo** es un Tailwind con Campo de Hierba, Fake Out doble y setup: el arquetipo más común
   del pool, pero que el propio agente tampoco aprendió a usar como plan (Staraptor con Tailwind).

**Implicaciones para P2** (rival de arquetipos en el entrenamiento): priorizar que el rival ejecute
Tailwind y Trick Room (~77 % del pool) por sobre ampliar a clima (que ya ocurre solo); como añadido,
el patrón "Protect + preparación" y el uso de sueño.

## 17. PPO v2: candidatos de mejora (vigesimocuarta cola, 18-11-2026)

**Decisión del usuario:** crear una versión mejorada, **PPO v2**, solo para PPO, y **congelar la v1**.
- **v1 congelado:** kick-off oficial (1M × 5, equipo fijo, preview v2; 81,2 % contra `eval`). Código:
  commit f671e65, tag git `ppo-v1`. Checkpoints, diagnósticos, resumen y preview en
  `~/iapsvgc_runs/ppo_v1`, en solo lectura y con `MANIFEST.md5` (131 archivos). La cola verifica el
  manifiesto antes de empezar.
- **v2** = v1 + los candidatos que cumplan su regla, entrenado después con 1M × 5 semillas y evaluado
  igual que v1 (requiere la aprobación del usuario).

Origen: la prueba online (0-2 contra velascoAlvarado; checklist, "Prueba online") y los arquetipos
del pool (§16).

| # | Candidato | Qué cambia | Debilidad que ataca | Regla de adopción |
|---|---|---|---|---|
| P1 | Sorteo al jugar | Sin reentrenar: la acción se sortea de la política (T = 1,0 y 0,5) en vez de tomar la más probable. Mismo algoritmo y mismos pesos (ppo_spec 1.15) | Juega siempre igual: en un BO3 el rival lo lee | Δ`val` ≥ −2 (5 semillas de v1); si ambas T cumplen, T = 1 |
| P2 | Rival de arquetipos en el entrenamiento | La mitad de la fase greedy_joint (150k) contra `greedy_joint_arq`: arquetipo **nivel 2** = nivel 1 (Tailwind siempre; Trick Room si es más lento) + Protect del compañero mientras el otro lo activa + sueño | Planes de velocidad (~75 % del pool, §16) que los heurísticos nunca ejecutan | Δ`val` ≥ −2 y Δ(arquetipo nivel 2, `val`) > +4 |
| P4 | P2 + entropía 0,03 | Igual que P2 con `--ent-coef 0.03` | 0 % de cambios voluntarios (con 0,03 aparecían, §14.4) | La misma que P2 |
| P3 | Rival PPO v1 congelado | La mitad de la fase expectimax (150k) contra `ppo_v1`: los 5 checkpoints de v1 (uno al azar por partida) **con tu mismo equipo (espejo)** | Nunca enfrentó a un rival que castigue sus errores | Δ`val` ≥ −2 y > 54 % contra v1 |

Cualquier candidato con Δ`val` > +4 también se adopta (regla general). `eval` es solo registro.

**Decisiones de diseño:**
1. **P3 en espejo:** v1 es especialista de tu equipo (~31 % con equipos al azar, §13.2). Con
   equipos del pool sería un rival débil; con tu equipo es el rival más fuerte disponible. Contra
   v1, v1 gana el 50 % por construcción.
2. **Con `--expand-omega` los rivales se acumulan:** en la última fase de P2 el rival de arquetipos
   sale en ~23 % de las partidas (peso 1,0 frente a 0,35 de expectimax), y en P3 el PPO v1 también
   en ~23 %.
3. **El nivel 2 no reemplaza al nivel 1:** la comparación con el rival de arquetipos del kick-off de
   v1 (nivel 1, −5,7) se mantiene.
4. **Referencia de P2, P4 y P3:** semillas 0 y 1 de v1, con los mismos diagnósticos.

**Diagnósticos de cada candidato** (semillas 0 y 1, 200 partidas por rival): `val`, `eval`,
arquetipo nivel 1 (`eval`), arquetipo nivel 2 (`val` y `eval`) y contra PPO v1 (espejo). El resumen
también da la tasa de victorias por arquetipo del rival (§16).

Código: `showdown/rl/pokeenv_backend.py` (rivales `greedy_joint_arq` y `ppo_v1`),
`showdown/battle_bots/greedy_joint/joint.py` (`_archetype_level`, `wrap_archetype_level`),
`showdown/rl/ppo/agent.py` (temperatura), `showdown/rl/ddqn/policy_diag.py`
(`--sample`, `--temperature`, rivales extra), `showdown/rl/team_archetypes.py`. Tests:
`tests/test_ppo_v2.py`. Script: `scripts/server/kickoff_queue24.sh`; resumen:
`showdown/rl/kickoff_summary24.py`. Duración estimada ~8,2 h. Resultados: pendientes.

**Calibración del enfrentamiento en espejo (19-11-2026).** Con resultados parciales, P2 y P4
ganaron solo el 28,2 % y el 21,8 % contra PPO v1, y la regla de P3 suponía que v1 contra sí mismo
da 50 % sin haberlo medido. Decisión del usuario: medirlo al terminar la cola 24 (v1 semillas 0 y 1
contra el mismo rival, 200 partidas; `scripts/server/h2h_v1_calib.sh`). La regla de P3 pasa a
"más de 4 puntos por encima de la calibración" (antes: > 54 %, que es lo mismo si la calibración da
50 %). La cola 25 espera a la calibración.

## 18. P5: corrección de habilidades en la observación (vigesimoquinta cola, 18-11-2026)

**Hallazgo del usuario en la prueba online:** Primarina atacaba con Hyper Voice teniendo
Charizard-Mega-Y el sol activo. Diagnóstico:
- El agente sí sabe que Charizard con Charizardite Y trae sol: la observación calcula la forma Mega
  con su habilidad (Drought) a partir de la hoja abierta del rival.
- Lo que no sabía es que **su propio Hyper Voice es de tipo Agua**: la estimación de daño no
  aplicaba Liquid Voice y lo trataba como Normal.

| Hyper Voice de Primarina (con sol) | La obs decía (Normal) | Real (Agua, STAB, sol) |
|---|---|---|
| Venusaur (Planta/Veneno) | ×1 | ×0,375 |
| Charizard-Mega-Y (Fuego/Volador) | ×1 | ×1,5 (sin sol, ×3) |
| Garchomp (Dragón/Tierra) | ×1 | ×0,375 |

Revisión de las habilidades del pool que afectan el daño:
- **Sin modelar:** Liquid Voice (10 equipos y el nuestro) y Parental Bond (3).
- **Solo como marca en el perfil:** las habilidades de potencia (Tough Claws 41 equipos, Fairy Aura
  69, Sharpness, Iron Fist, Technician…) aparecían como marca, pero el cálculo de daño no las
  aplicaba.
- **Ya modeladas:** -ate, Adaptability, Huge Power, inmunidades, Multiscale y Filter.
- El estudio de arquetipos (P2) no lo habría corregido: cambia cómo juega el rival, no lo que el
  agente sabe de sus propios movimientos.

**Corrección (P5, rl_env_spec v1.32):**
- Liquid Voice, las habilidades de potencia y algunas defensivas entran en la estimación de daño, el
  código de efectividad, el aviso de redirección y el tipo de los movimientos rivales.
- El perfil compartido con el preview no cambia: el preview v2 sigue igual.

**Decisión del usuario: P5 se activa en PPO y DDQN.**
- Por defecto en los entrenamientos nuevos.
- Cada checkpoint guarda con qué observación se entrenó (`obs_abilities`). PPO v1 y los DDQN
  anteriores cargan con la observación vieja, así que v1 congelado no cambia.
- Entra en la corrida final de PPO v2 y en el kick-off de DDQN.
- La vigesimoquinta cola **mide** su efecto; no decide. Configuración final de cada agente, 1M,
  semillas 0 y 1:
  - PPO frente a v1, más arquetipo nivel 1 y contra v1 en espejo.
  - DDQN frente a REG 0,03.
- Si alguno pierde más de 2 puntos en `val`, se revisa antes de seguir.

Los candidatos de la vigesimocuarta cola (§17) se entrenaron sin P5; sus comparaciones siguen siendo
válidas porque referencia y candidatos usan la misma observación. Tests:
`tests/test_obs_abilities.py`. Script: `scripts/server/kickoff_queue25.sh` (espera a la 24); resumen:
`showdown/rl/kickoff_summary25.py`. Duración estimada ~4,3 h. Resultados: pendientes.

### 18.1 P5 completo: calculadora real y preview v2.1 (decisiones del usuario, 19-11-2026)

1. **v1 congelado no cambia** (observación y preview v2 originales).
2. **Desde PPO v2 y el kick-off de DDQN, la corrección es completa.** Todas las habilidades,
   movimientos y objetos con efectos parecidos, y los efectos de campo como en la realidad (Fairy
   Aura afecta a todo el campo).
3. **Enfoque elegido: la calculadora real.** El daño de la observación de DDQN y PPO se toma de la
   misma calculadora de Showdown (`@smogon/calc`, mecánicas Champions) que usan los heurísticos,
   a través de `damage_calc_bridge`. Hasta ahora DDQN y PPO usaban una aproximación propia en
   Python (`move_damage_frac`).
4. **Preview v2.1:** preview nuevo con la corrección (tipo y potencia efectivos de cada movimiento
   según la calculadora). Lo usarán PPO v2 y el kick-off de DDQN; **el preview v2 queda congelado
   con v1**.
5. **La cola 25 (P5 parcial) se canceló:** quedaba superada (`~/iapsvgc_runs/p5_cancelada_19nov`).

**Auditoría** (`scripts/analysis/audit_damage_effects.js` → `API/JSONs/damage_effects_audit.json`,
manejadores del simulador de Showdown, dex Champions):

| | Legales con efecto sobre daño, tipo o efectividad | Usados en el pool |
|---|---|---|
| Habilidades | 128 | 54 |
| Movimientos | 173 | 84 |
| Objetos | 57 | 29 |

Huecos graves de la aproximación anterior, además de Liquid Voice:
- **Weather Ball** (91 usos en el pool): con sol es Fuego de 100 de potencia; la obs la trataba como
  Normal de 50.
- **Last Respects** (106 usos): la potencia sube con los aliados caídos.
- **Grassy Glide** (191 usos): prioridad con Campo de Hierba.
- **Fairy Aura** en todo el campo (69 equipos).

**Primer paso: medir la velocidad** (`scripts/analysis/bench_calc_obs.py`). La observación pediría
~80 cálculos por paso: ~24 000/s para entrenar a la velocidad actual. Según el resultado se decide
cómo integrarla (lote por observación, caché, cálculo solo de lo necesario).

**Avance (19-11-2026, sin esperar la medición de velocidad):**
- **Efectos que no son de daño**, en Python y solo con la observación nueva (`obs_abilities`):
  - prioridad: Grassy Glide con Campo de Hierba y Gale Wings;
  - velocidad: Iron Ball y Surge Surfer;
  - inmunidades: Bulletproof, Soundproof, Overcoat, Magnet Rise;
  - Sturdy anula el KO de un golpe con HP lleno;
  - Telepathy: el área del aliado no le pega.

  Quick Claw y Quick Draw dan prioridad al azar: no se modelan.
- **`showdown/rl/obs_calc.py`:** arma los pedidos a la calculadora desde el estado de la partida.
  - Forma Mega efectiva y stats según el modo (medio, máximo, mínimo).
  - Aliados caídos (Last Respects).
  - Pantallas del bando defensor, Friend Guard, Battery, Power Spot y Steely Spirit.
  - **Fairy / Dark Aura y Aura Break si cualquiera de los cuatro en campo las tiene**, y las cuatro
    habilidades "de la ruina".
  - Convierte (mínimo, máximo) en fracción del HP con la misma semántica que la aproximación
    anterior.
  - `calc_core.js` debe aceptar `isFairyAura`, `isDarkAura` e `isAuraBreak`. Los heurísticos no
    mandan esos campos, así que no cambian.
- **Propuesta de cambio al contrato** (rl_env_spec, al integrar): quitar de §1 y §8 la prohibición de
  usar `@smogon/calc` dentro de `step()` cuando `obs_abilities` está activo, fijar el commit de
  `damage-calc` y registrar la velocidad medida.

Tests: `tests/test_obs_abilities.py` (efectos que no son de daño) y `tests/test_obs_calc.py`
(incluye el caso de la prueba online con la calculadora real: el sol reduce a la mitad el Hyper Voice
de Primarina porque es de Agua).

### 17.1 Resultados de la vigesimocuarta cola y de la calibración (19-11-2026)

Commit c8bc5ec (calibración y resumen rehecho: 9fc0842). Todo rc=0. 200 partidas por rival, equipo
fijo, preview v2; `val` decide y `eval` es solo registro.

**P1, sorteo al jugar** (5 semillas de v1):

| Modo | `val` | Δ`val` | `eval` | Arquetipo n1 | Regla |
|---|---|---|---|---|---|
| Determinista (v1) | 90,4 % | — | 81,2 % | 77,1 % | — |
| T = 1,0 | 88,0 % | −2,4 (± 1,6) | 80,1 % | 76,7 % | no cumple |
| T = 0,5 | 90,3 % | −0,1 (± 1,5) | 81,2 % | 77,9 % | **cumple** |

**P2, P4 y P3** (2 semillas, frente a v1 semillas 0 y 1):

| | `val` | Δ`val` | Arquetipo n2 `val` | Δ | Contra v1 (espejo) | `eval` | Δ`eval` | Arquetipo n1 `eval` | Regla |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 90,7 % | — | 84,5 % | — | **59,8 %** (calibración) | 81,3 % | — | 76,8 % | — |
| P2 | 89,3 % | −1,3 | 86,2 % | +1,8 | 28,2 % | 81,5 % | +0,2 | 79,5 % | no |
| P4 | 90,2 % | −0,5 | 87,8 % | +3,2 | 21,8 % | **85,7 %** | **+4,3** | 81,0 % | no |
| P3 | 90,5 % | −0,2 | 88,0 % | +3,5 | **99,5 %** | 83,2 % | +1,8 | 77,2 % | sí por la regla (> 64 %) |

**Lectura:**
1. **La calibración no da 50 %: v1 contra v1 en espejo da 59,8 %.** La medición favorece unos 10
   puntos al lado del agente (el jugador del entorno frente al `PPOPlayer` rival: camino de
   observación y orden de mensajes distintos). Suponer 50 % habría sido un error. P2 y P4 quedan
   31 y 38 puntos por debajo de esa referencia: v1 los explota.
2. **P3 cumple la regla, pero la evidencia está contaminada.** P3 se entrenó contra esos mismos
   checkpoints de v1, y el 99,5 % mide que aprendió a explotarlos, no que juegue mejor en general.
   En `val` queda igual (−0,2) y mejora algo contra arquetipos (+3,5, dentro del IC). La regla fijada
   no previó que el rival de prueba fuera el mismo del entrenamiento; queda como limitación y
   decide el usuario.
3. **P4 es el mejor en `eval`** (+4,3, en el límite del IC) y contra arquetipos (Trick Room
   87,4 % frente a 81,0 %; Tailwind 83,8 % frente a 77,6 %). Pero no cumple la regla en `val`
   (+3,2 contra el umbral de +4) y es el más explotable por v1.
4. **Ningún candidato empeora en `val`** (todos dentro de ±2 puntos).

**Medición de velocidad de la calculadora real** (`~/iapsvgc_runs/bench_calc`, servidor libre,
60 s por variante, 0 errores):

| Variante | Cálculos/s | Pasos/s equivalentes (80 por paso) | Latencia del lote p50 / p95 |
|---|---|---|---|
| 32 procesos, lotes de 80 (realista, con caché) | 123 037 | 1 538 | 6,6 / 61,0 ms |
| 32 procesos, lotes de 80, sin caché (peor caso) | 38 653 | 483 | 7,7 / 296,9 ms |
| 8 procesos, lotes de 80 | 164 232 | 2 053 | 2,7 / 4,4 ms |
| 32 procesos, lotes de 240 | 378 476 | 1 577 | 13,6 / 19,5 ms |

El entrenamiento corre a ~300 pasos/s: **la calculadora alcanza incluso en el peor caso** (483
pasos/s sin caché). Lotes más grandes rinden más por cálculo. El riesgo son los picos de latencia
sin caché (p95 297 ms), y que en el entrenamiento los heurísticos usan el mismo servidor a la vez.
Integración: un lote por observación con caché; se mide la velocidad real en el piloto.

**Avance (19-11-2026, partes 2a–2c):**
- **2a** (c2a0f6b): la observación toma el daño de la calculadora real en un lote por paso (~230
  pedidos: bloques de movimientos y matriz de pares), con respaldo en Python y cortacircuito. El uso
  se registra por episodio.
- **2b** (971ddc6): el perfil de cada movimiento, que comparten preview y combate, usa tipo y
  potencia efectivos (Liquid Voice, Weather Ball con clima propio, habilidades de potencia). La
  caché va por versión. El preview guarda su versión: v2 sin corrección, v2.1 con ella.
- **2c:** rl_env_spec v1.33 (la prohibición de la calculadora en `step()` queda solo para la
  observación v1; damage-calc fijado en `bacbbfc`) y team_preview_masked_ppo_spec 1.7 (preview
  v2.1).

Tests en el servidor: 125 pasan. Pendiente: el piloto de velocidad dentro del entrenamiento y
entrenar el preview v2.1.


**Decisiones del usuario (19-11-2026):**
- **P3 no se adopta.** Su único argumento era ganarle al mismo rival con que se entrenó.
- **P4 no se implementa.**
- **El preview v2.1 usa la misma receta que el v2:** greedy_joint como combate, tu equipo, 200
  actualizaciones, entropía 0,005, formaciones canónicas, `best.pt`.
- **Medir antes de las corridas oficiales.**

Vigesimosexta cola (`scripts/server/kickoff_queue26.sh`, espera al piloto de velocidad):
1. Reinicia el servidor de la calculadora.
2. Entrena el preview v2.1.
3. PPO con P5 + preview v2.1 (2 semillas): `val`, `eval`, arquetipo y contra v1 en espejo. El v1
   juega con su preview v2 (`PPO_FOE_PREVIEW`).
4. DDQN con P5 + preview v2.1 (2 semillas): `val` y `eval`, frente a REG 0,03.

Resumen: `showdown/rl/kickoff_summary26.py` (incluye el uso de la calculadora y la velocidad de cada
entrenamiento). PPO v2 = v1 + P5 completo + preview v2.1 + sorteo T = 0,5 al jugar contra personas.

### 17.2 ¿Cuánto se aparta el sorteo del modo determinista? (P1, 19-11-2026)

Pedido del usuario. Las 5 semillas de v1 contra `val`, contando solo las decisiones con más de una
opción legal (`scripts/server/p1_divergencia.sh`, `scripts/analysis/divergencia_p1.py`; commit
6299332, rc=0):

| Temperatura | Jugada distinta | Hueco 1 distinto | Turno 1: jugada distinta | Partidas con alguna decisión distinta | Decisiones | Victorias |
|---|---|---|---|---|---|---|
| **0,5** | **22,7 %** | 13,7 % | **12,0 %** | **71,3 %** | 15 429 | 90,2 % |
| 1,0 | 38,3 % | 23,9 % | 23,3 % | 89,2 % | 7 835 | 87,9 % |

1. **T = 0,5 sí aporta variedad sin costo:** cambia la jugada en ~1 de cada 4–5 decisiones, y 7 de
   cada 10 partidas tienen al menos una decisión distinta del modo determinista. Victorias 90,2 %
   (determinista: 90,4 %).
2. **La apertura cambia poco:** en el turno 1 solo difiere el 12 %. Es justo lo que leyó
   velascoAlvarado (las dos partidas empezaron igual). El sorteo no basta para variar la
   apertura; también cuenta el preview, que es determinista (mismo líder en las dos partidas).
3. T = 1,0 varía más (38 %, turno 1: 23 %), pero pierde victorias (87,9 %).
4. Control: con las mismas semillas que P1 se esperaba 90,3 % y salió 90,2 %. La pequeña diferencia
   sugiere algo de no determinismo del simulador entre corridas; no cambia la conclusión.

**Conclusión:** se mantiene T = 0,5 para jugar contra personas. Una mejora futura para el BO3 es
variar también la apertura: sorteo en el preview o en el turno 1.

**Piloto de velocidad dentro del entrenamiento** (19-11-2026, `~/iapsvgc_runs/p5_speed`, commit
d45fbf6; PPO 30k pasos, 32 entornos, observación nueva):

| Daño de la observación | Pasos/s | Observaciones con la calculadora | Respaldo | Pedidos sin respuesta | Pedidos por observación |
|---|---|---|---|---|---|
| Calculadora real | **270** | 29 616 (100 %) | 0 % | 0 % | ~87 |
| Respaldo en Python (control) | 346 | — | — | — | — |

La calculadora cuesta **−22 % de velocidad** (1M pasos: ~62 min frente a ~48) y funcionó en el 100 %
de las observaciones, sin errores. Los pedidos reales por observación (~87) son menos que los
estimados (~230): no se repiten, y los Pokémon caídos o excluidos no generan pedidos. **Se integra
tal cual.** La vigesimosexta cola pasó el resguardo (270 ≥ 0,4 × 346) y arrancó a las 08:33.

P1b (preview no determinista, `scripts/server/p1b_preview.sh`) quedó encadenado a la cola 26:
- **Modos:** preview sorteado con T = 1,0 y 0,5 (combate determinista), y preview + combate con
  T = 0,5 (el modo contra personas).
- **Combate y preview:** PPO con P5 + preview v2.1, semillas 0 y 1, contra `val`.
- **Sorteo:** solo entre las 90 formaciones canónicas con que se entrenó; el modo determinista no
  cambia.
- **Regla:** como P1 (Δ`val` ≥ −2).

### 18.2 Resultados de la vigesimosexta cola y de P1b (19-11-2026)

Cola 26: commit b7732d8, todo rc=0. P1b: commit 0b08719. 200 partidas por rival, semillas 0 y 1,
`final.pt`.

**P5 completo + preview v2.1 frente a la referencia** (PPO v1; DDQN REG 0,03; ambas con preview v2 y
la observación anterior):

| Agente | `val` antes → con P5 | Δ`val` | `eval` antes → con P5 | Δ`eval` | Otros |
|---|---|---|---|---|---|
| PPO | 90,7 → 88,8 % | −1,9 (± 2,4) | 81,3 → 79,5 % | −1,8 (± 3,2) | Arquetipo n1: 72,0 % (v1 77,1 %); contra v1 en espejo 30,8 % (calibración 59,8 %) |
| DDQN | 74,1 → 75,2 % | +1,2 (± 3,5) | 65,3 → 56,5 % | **−8,8 (± 3,9)** | — |

Calculadora real en los 4 entrenamientos: ~1M observaciones cada uno, **0 % de respaldo y 0 %
de pedidos sin respuesta**. Velocidad: PPO 260 pasos/s (1h04m), DDQN 290 pasos/s (57m). Preview
v2.1: MEAN_WR 0,951.

**Lectura:**
1. **La condición acordada se cumple** (Δ`val` ≥ −2 en ambos y la calculadora sin fallos), pero
   hay señales de empeoramiento:
   - PPO queda en el límite (−1,9) y cae en todo lo demás: `eval` −1,8, arquetipos −5,1, y contra
     v1 queda 29 puntos por debajo de la calibración.
   - DDQN mejora en `val` (+1,2) pero **cae 8,8 puntos en `eval`**, fuera del IC.
2. **Dos cambios a la vez:** la observación nueva (P5) y el preview v2.1. Con estos datos no se
   puede saber cuál de los dos causa la caída.
3. **Patrón sin explicar contra v1:** todos los modelos nuevos pierden mucho contra v1 en espejo
   (P2 28 %, P4 22 %, P5 31 %; v1 contra v1 59,8 %), incluso los que no empeoran contra los
   heurísticos.

**P1b, preview no determinista** (combate PPO con P5 + preview v2.1, contra `val`):

| Modo | Victorias | Δ | Otra formación | Otros líderes | Formaciones distintas |
|---|---|---|---|---|---|
| Determinista | 88,8 % | — | — | — | — |
| Preview T = 1,0 | 89,0 % | +0,3 (± 2,5) | 10,1 % | 2,2 % | 18 |
| Preview T = 0,5 | 87,8 % | −1,0 (± 2,6) | 5,6 % | 0,7 % | 12 |
| Preview T = 0,5 + combate T = 0,5 | 87,7 % | −1,1 (± 2,6) | 4,9 % | 0,4 % | 11 |

El preview está muy concentrado: incluso con T = 1,0 cambia la formación en solo 1 de cada 10
partidas, y casi nunca los líderes (2,2 %). Ambas temperaturas cumplen la regla (Δ ≥ −2). Por la
misma regla de P1 (si ambas cumplen, T = 1) correspondería T = 1,0, que no cuesta nada pero aporta
poca variedad.
