# DDQN: checklist antes del entrenamiento oficial

Plan surgido de la revisión del agente E (14–15 de noviembre de 2026). Cada punto se
implementa en el laptop, se prueba en el servidor y se registra aquí con su commit. El
entrenamiento oficial (pipeline E: preview PPO + DDQN) empieza cuando todo lo obligatorio
está cerrado.

## A. Errores y riesgos

| # | Punto | Estado | Commit / evidencia |
|---|---|---|---|
| 1 | El backend mock congelaba la observación dentro del episodio (la caché de la obs compara por identidad y el mock mutaba el mismo snapshot). Solo afectaba a smoke tests y corridas `--mock` | **hecho**: el mock devuelve un snapshot nuevo por paso | 1627cc0; `tests/test_rl_prekickoff.py` |
| 6 | Orden inestable del resto del rival: revelar un Pokémon (orden "vistos primero") o descartar uno movía los demás bloques a mitad del combate | **hecho**: orden fijo por especie; descartados en cero en su lugar. Cambia la semántica de la obs (rl_env_spec v1.22) | 1627cc0 |
| 3 | `DDQNPlayer.choose_move` se tragaba las excepciones y jugaba la orden por defecto sin avisar | **hecho**: `decision_stats` (decisiones, excepciones por tipo, fallbacks de orden), log de las primeras 5, `PS_RL_STRICT=1` propaga | 1627cc0 |
| 2 | Paridad entre entrenamiento e inferencia: el entorno arma el snapshot sin `stat_beliefs` y el jugador con ellas | **hecho**: obs idéntica en 902 de 902 pasos reales y máscara de entrenamiento = máscara de inferencia. Con `OBS_STAT_INFERENCE` apagado (por defecto) las creencias no entran en la obs; si se enciende, hay que hacerlo en entrenamiento e inferencia a la vez | live_validation, 9b29322 |

## B. Validación en vivo

Corrida corta contra Showdown real, con `PS_RL_STRICT=1`, que mida: tasa de fallback,
historial frente al protocolo, alineación movimiento i ↔ acción i, máscara nunca vacía,
paridad del punto 2, simetrías con batallas reales y costo en ms de obs y codificador.

Herramienta: `python -m showdown.rl.live_validation` (mismo camino que el entrenamiento:
`VGCEnv` + `PokeEnvBattleBackend` + `_joint_masks`, acciones legales al azar contra random,
solo, joint y expectimax) + `run_robust_e2e --player-a ddqn` en modo estricto (camino de
inferencia validado).

**Estado: hecho (15-11-2026).**

| Chequeo | 1.ª corrida (5e414e0: 48 episodios, 403 pasos) | 2.ª corrida, con el arreglo (9b29322: 100 episodios, 902 pasos) |
|---|---|---|
| Máscara vacía | 0 | 0 |
| Máscara de entrenamiento ≠ inferencia | 0 | 0 |
| Paridad de la obs (punto 2) | 0 diferencias | 0 diferencias |
| Alineación movimiento k ↔ acción k | 1995/1995 | 4483/4483 |
| Historial frente al protocolo | 960/960 | 2268/2268 |
| Simetrías con combates reales (7 elementos) | 2821/2821 | 6314/6314 |
| **Fallbacks de orden** | **11/403 (2,7 %)** | **0/902** |
| Caídas (truncado por excepción) | 0 | 0 |
| Costo por paso (mediana) | snapshot + obs 1,7 ms; forward 0,4 ms (GPU) | 1,7 ms; 0,5 ms |
| Inferencia estricta: DDQN validado vs joint y vs expectimax (120 BO3) | 0 excepciones; sync e inferencia OK | 0 excepciones; sync e inferencia OK |

**Error encontrado y corregido (rl_env_spec v1.23):** los 11 fallbacks eran cambios forzados
de los dos huecos con un solo reemplazo. La máscara permitía `(pass, pass)` (Showdown exige
que entre el reemplazo) y el adaptador convertía un `pass` legal en un cambio al mismo
Pokémon que el otro hueco; en ambos casos se jugaba una orden al azar y el replay guardaba
una acción no ejecutada. Ahora la máscara sigue la regla de Showdown (`sim/side.ts`,
`forcedPasses = forzados − reemplazos`) y el `pass` legal se respeta. Un test antiguo
(`test_forced_switch_orders`) esperaba que un hueco forzado con reemplazo pudiera pasar; se
corrigió a la regla real (4ab4609).

Suite completa en el servidor tras el arreglo: 663 pasan, 1 omitido, 6 xfail, más el test
corregido.

## C. Aprendizaje

| # | Punto | Estado |
|---|---|---|
| 7 | LayerNorm después del codificador (ablación corta) | pendiente |
| 9 | Re-tunear hiperparámetros, incluido `sym_coef` ∈ {0; 0,05; 0,1; 0,3} (los `*_best_hp.json` actuales son corridas de prueba) | pendiente |
| — | Calibrar el paralelismo según el servidor (phantom: 32 hilos, 3 × RTX A5500 de 24 GB, 251 GB de RAM) | **hecho** (ver "Calibración del paralelismo" abajo). Recomendación: `--num-envs 32 --envs-per-server 4`. Hallazgo: el cuello de botella es el aprendiz, no las batallas |
| — | El aprendiz limitaba el entrenamiento a ~130–155 pasos/s frente a ~450 de las batallas | **hecho** (5ff2507, ddqn_spec 1.13): replay y aumentación por simetría en GPU (mismos lotes, verificado bit a bit) y fracciones por rival incrementales. Paso de gradiente 34,2 → 4,7 ms. Prueba A/B en entrenamiento real (32 × 4, 4 rivales): replay en CPU 153 pasos/s, **en GPU 410 pasos/s** (2,7×). Un currículo de 2M pasos baja de ~4 h a **~1,4 h** |

### Calibración del paralelismo (15-11-2026, commit db6d4de)

DDQN real con `--device cuda`, Ω con los 4 rivales (random, solo, joint, expectimax con
sorteo; pesos 1/1/1/0,35), 8k–18k pasos por configuración. "Sin aprender" = pasos 1k–5k
(antes del warm-up de 5000 pasos: solo batallas y llenado del replay); "con aprendizaje" =
desde el paso 6500 (un paso de gradiente cada 4 pasos de entorno, lote 256).

| `--num-envs` × `--envs-per-server` | Solo batallas (pasos/s) | Con aprendizaje (pasos/s) | CPU media | Reinicios |
|---|---|---|---|---|
| 8 × 4 | 193 | 99 | 21 % | 0 |
| 16 × 4 | 331 | 122 | 27 % | 0 |
| 24 × 4 | 425 | 124 | 29 % | 0 |
| 28 × 4 | 457 | 138 | 33 % | 0 |
| **32 × 4** | **462** | 127 | 31 % | 0 |
| 40 × 4 | 454 | 154 | 36 % | 0 |
| 48 × 4 | 452 | 155 | 37 % | 0 |
| 32 × 2 | 434 | 101 | 67 % | 0 |
| 32 × 6 | 249 | 100 | 96 % | 0 |

Lectura:

- **Las batallas escalan hasta ~28–32 entornos** (~450–460 pasos/s) y luego se
  estabilizan. Más de 4 entornos por servidor Showdown satura la CPU (32 × 6: 96 %) y rinde
  menos; 2 por servidor también rinde algo menos.
- **Con aprendizaje, el rendimiento queda en ~100–155 pasos/s sin importar los entornos:**
  el aprendiz (un paso de gradiente cada 4 pasos, ~30–40 por segundo) es el cuello de
  botella. La CPU total queda baja (≤ 37 %), así que el límite es el bucle principal, no el
  servidor.
- Duración estimada de un currículo de 2M pasos: ~3,6–4,5 h con el aprendiz actual; ~1,2 h
  si el aprendiz alcanzara el ritmo de las batallas.
- 0 reinicios y 0 errores de `|showteam|` en las 9 configuraciones (con el arreglo de
  nombres de especie, db6d4de). Un primer intento con el código anterior (53889fa) tenía
  7 errores `|showteam|` en 1 minuto.

### Equipo fijo del agente

**Hecho (15-11-2026).** Indicación del usuario: DDQN juega siempre con el mismo equipo
(`showdown/rl/data/ddqn_team.txt`) y los rivales cambian de equipo en cada reinicio (grupo
`train`). Se mantiene **hasta que el usuario indique pasar a un equipo distinto por
reinicio** (`--agent-team pool`). Verificado: 66 SP por Pokémon, legal en Reg M-C, el equipo
llega como `TeamMeta` (rivales validados), el rival cambia casi en cada reinicio.

Verificación en el servidor (15-11-2026, commits 66002cf–0730087):

| Comprobación | Resultado |
|---|---|
| Suite completa | 679 pasan, 1 omitido, 6 xfail (0 fallos) |
| Tests del equipo fijo, pre-kickoff, separación y SP | 55 de 55 |
| Validación en vivo con el equipo fijo (32 episodios, 231 pasos, modo estricto) | el agente juega con el equipo en 32/32 episodios; rivales validados; 0 fallbacks de orden; paridad, alineación, historial y simetrías sin diferencias |
| Equipos distintos del rival en 8 reinicios | solo, joint y expectimax: 8 de 8 (cambia en cada reinicio); random: 1 (sin OTS, no se identifica) |
| Entrenamiento corto (6 000 pasos, 32 × 4, modo estricto) | 0 excepciones, 0 reinicios de workers, 0 fallbacks en 769 episodios; 209 pasos/s con el servidor cargado; el replay respeta los pesos del currículo (random 39 %, solo 25 %, joint 24 %, expectimax 11 %) |

Línea base del agente sin entrenar (ε ≈ 0,99, es decir, jugadas al azar) con el equipo fijo:
46 % contra random, 1,7 % contra solo, 0,5 % contra joint, 1 % contra expectimax; ~6,5
turnos por partida contra los heurísticos. Es la referencia para el piloto de aprendizaje.
Una primera versión del contador de especies marcaba 80 desajustes falsos
("Arcanine-Hisui" frente a "arcaninehisui"); corregido en 0730087.

Pendiente de decidir: cómo evalúa la Conclusión al pipeline E con un equipo fijo (mismos
pares de equipos y lados intercambiados para todos los demás).

### Piloto de aprendizaje (15-11-2026, commit 0476a4c)

Configuración oficial (equipo fijo, 32 × 4, replay en GPU, rivales del grupo `train`,
modo estricto): 150k pasos contra random y 250k con random + solo (Ω creciente), 22,7 min
(~295 pasos/s de media con el servidor compartido). ε baja de 1 a 0,24 (decaimiento en 500k).

**Evaluación final sin exploración** (100 partidas BO1 por rival) frente a la línea base sin
entrenar:

| Rival | Sin entrenar | Tras 400k pasos | ¿Entrenó contra él? |
|---|---|---|---|
| random | 46 % | **60 %** | sí |
| greedy_joint | 0,5 % | **7 %** | no |
| expectimax | 1 % | **9 %** | no |

**Curva con exploración** (victorias por tramo): contra random 0,455 → 0,604; contra solo
0,012 → 0,067. **El agente aprende**, y mejora también contra rivales que no vio (joint y
expectimax), aunque sigue muy lejos de ellos con el 20 % del presupuesto previsto.

**Recompensa:** separa perfecto victoria de derrota (umbral 0,14 clasifica el 100 % de los
episodios; victorias +1,35..+1,8, derrotas −1,6..−0,96). Sin fallbacks, descartes ni truncados.

**Problema: sobreestimación de Q.** El retorno real de un estado no puede pasar de ~1,8,
pero el Q medio de las acciones tomadas sube hasta 5,6 (paso 280k) y termina en 3,4; el
error TD pasa de 0,26 a 0,54–0,65 y la pérdida de 0,06 a 0,25–0,33. Al entrar solo (retornos
negativos) Q **subió** en vez de bajar. No diverge (baja al final, cuando la política mejora),
pero los valores no siguen la realidad. Causa probable: el máximo sobre cientos de pares
legales toma el ruido como valor, y con actualización suave del objetivo (τ = 0,005) la red
objetivo es casi igual a la online, así que Double DQN apenas corrige el sesgo. Se descartó
que el fin de episodio no se marque en el replay (se marca bien).

### Pilotos A/B contra la sobreestimación de Q (16-11-2026, commit ff313a7)

Misma configuración y presupuesto que el piloto A (400k pasos, semilla 0). Opciones nuevas
en `DDQNConfig`, apagadas por defecto: `target_update_every` (copia dura del objetivo cada N
pasos de gradiente) y `target_clip` (objetivo acotado a [−c, c]).

| | A: original (τ = 0,005) | B: copia dura cada 2 000 | **C: objetivo acotado a [−2, 2]** |
|---|---|---|---|
| Q medio en 280k / final | 5,60 / 3,38 | 1,34 / 1,01 | 1,26 / 0,88 |
| Error TD final | 0,54 | 0,18 | 0,15 |
| Pérdida final | 0,26 | 0,03 | 0,03 |
| Eval. sin exploración: random | 60 % | 57 % | **72 %** |
| Eval.: greedy_joint | 7 % | 13 % | **19 %** |
| Eval.: expectimax | 9 % | 11 % | **19 %** |
| Eval.: promedio (300 partidas) | 25,3 % | 27,0 % | **36,7 %** |
| Con exploración, último tramo: random / solo | 0,604 / 0,067 | 0,541 / 0,090 | 0,644 / 0,114 |
| Pasos/s (servidor compartido) | ~295 | 262 | 214 |

Lectura:

- **B y C eliminan la sobreestimación:** Q queda dentro del rango real de retornos (≤ 1,8)
  y el error TD baja a un tercio.
- **C rinde claramente mejor.** En las 300 partidas de evaluación gana 110 frente a 76 de A
  (z ≈ 3); contra joint, 19 frente a 7. B calibra igual pero rinde parecido a A.
- **Alcance:** una sola semilla de entrenamiento y 100 partidas por rival. Es una elección
  exploratoria de configuración antes del kick-off, no una afirmación de la tesis (protocolo
  §3c: los métodos se comparan con 3–5 semillas).
- La diferencia de velocidad es carga del servidor, no del cambio (el recorte es una
  operación trivial en la GPU).
- El límite 2,0 sale del diseño de la recompensa: los retornos observados van de −1,6 a +1,8.
  Si la recompensa cambia, el límite debe revisarse.

**Decisión (16-11-2026):** C por defecto (`target_clip=2.0`, ddqn_spec 1.15). B queda como
opción apagada; la combinación B+C puede probarse en el tuning (punto 9).

### Mini kick-off de 1M pasos (tercera cola, 16-11-2026, commit 5b43e8f)

Currículo completo (random → solo → joint → expectimax), lr 1e-4, lote 256, preview v2
(`preview_v2_ent005`), equipo fijo, semilla 0. Validación sin exploración cada 250k pasos,
50 partidas por rival, **con equipos rivales del grupo `train`**.

| Paso | Media | random | solo | joint | expectimax |
|---|---|---|---|---|---|
| 250k | 35,0 % | 66 % | 24 % | 26 % | 24 % |
| 500k | 61,5 % | 76 % | 64 % | 54 % | 52 % |
| 750k | 79,5 % | 84 % | 82 % | 86 % | 66 % |
| 1M | 85,0 % | 88 % | 88 % | 88 % | 76 % |

Q medio final 1,05; error TD 0,120. Antes (400k pasos, preview v1) lo mejor era 38 %.

### Generalización a equipos rivales no vistos (cuarta cola, commit f61f16d)

`best.pt` del mini kick-off (paso 1M), 100 partidas por rival, rivales con equipos de cada grupo:

| Rival | Equipos `train` | Equipos `eval` | Diferencia |
|---|---|---|---|
| random | 91 % | 90 % | −1 |
| greedy_solo | 87 % | 38 % | −49 |
| greedy_joint | 87 % | 36 % | −51 |
| expectimax_bayesian | 72 % | 36 % | −36 |
| **total** | **84,2 %** | **50,0 %** | **−34,2** |

Lectura:

- **El agente memoriza los equipos rivales de entrenamiento.** Contra los heurísticos con
  equipos que nunca vio, gana alrededor de 37 % (por debajo de 50 %). Contra random no cambia:
  ahí no hace falta conocer al rival.
- Es justo lo que la separación 200/100 (punto 11) debía detectar. Sin ella, el 85 % se habría
  leído como dominio sobre los heurísticos.
- **La validación periódica también usa equipos `train`.** Por eso `best.pt` se elige con una
  medida que premia la memorización. Opciones (pendiente de decisión): validar con un subgrupo
  reservado de `train` que no se use para entrenar, o más diversidad de rivales.
- Alcance: una semilla y un checkpoint. La caída es tan grande (unos 50 puntos contra solo y
  joint, con ±7 de error por celda) que no se explica por ruido.

### Verificación del preview v2 (quinta cola, commit 98599da)

`preview_v2_ent005/best.pt` frente al preview heurístico del piloto; combate greedy_joint en
ambos lados; 512 contextos pareados por fila.

| Rivales | Preview v2 | Heurístico | Diferencia (± EE) |
|---|---|---|---|
| equipos `train` | 95,3 % | 61,7 % | +33,6 (± 2,3) |
| equipos `eval` | 67,6 % | 57,0 % | +10,5 (± 2,7) |
| formación fija (la más frecuente, siempre), `train` | 74,2 % | 63,9 % | +10,4 (± 2,9) |

Formaciones distintas elegidas: 18 (`train`) y 17 (`eval`). La más frecuente (28 % en
`train`) es Gholdengo, Staraptor, Arcanine-Hisui y Primarina.

Lectura:

- **El 96,7 % era sobre todo memorización.** Con equipos no vistos la ventaja baja de +34 a
  +10,5 puntos. Sigue siendo positiva y clara (unos 4 errores estándar).
- **La ventaja en equipos no vistos equivale a jugar siempre una buena formación.** Fija, la
  formación más frecuente da +10,4 contra rivales `train`, igual que el preview adaptativo
  contra `eval`. Por ahora no hay evidencia de que adapte la elección a rivales que no conoce.
- La cola 7 (D4) repite la prueba con expectimax en el combate.

### Simetría a 1M pasos (quinta cola, commit 98599da)

Misma configuración que el mini kick-off (DDQN) y que el piloto de PPO de la cola 2; solo
cambia `RL_SYM_AUG`. Una semilla por variante. **Rivales con equipos `train`** (la medida
que premia memorizar; ver la sección anterior).

| | Con simetría | Sin simetría | Diferencia |
|---|---|---|---|
| DDQN 1M, validación (200 partidas) | 85,0 % | 81,0 % (random 92, solo 78, joint 88, expectimax 66) | +4,0 (EE ≈ 3,8) |
| PPO 400k, suite (300 partidas: random, joint, expectimax) | 92,0 % | 90,7 % | +1,3 (EE ≈ 2,3) |
| Pasos/s PPO | 394 | 384 | |

Lectura:

- **La simetría no muestra efecto demostrable** en ninguno de los dos métodos, igual que en la
  ablación de 400k pasos. Las curvas de DDQN son casi iguales en todos los puntos (250k: 35
  frente a 32; 500k: 61,5 frente a 62,5; 750k: 79,5 frente a 76).
- Decisión pendiente del usuario: mantenerla (no cuesta velocidad apreciable) o apagarla.
- **PPO a 400k pasos (92 %) supera claramente a DDQN a 400k (38 %)**, aun sin haber
  entrenado contra joint ni expectimax (fases random y solo). Falta comprobar con equipos
  `eval` cuánto de eso es memorización, como se hizo con DDQN.

### Análisis previos al kick-off de PPO (sexta cola, commit 5933ebb)

**Validación en vivo con máscaras por hueco** (100 episodios, 715 pasos): 0 máscaras vacías,
0 pares ilegales, 0 diferencias entre la máscara de entrenamiento y la de inferencia, 0
respaldos de orden, 0 diferencias de paridad, alineación, historial o simetrías, y el equipo
fijo en los 100 episodios.

**Jugadas del piloto de PPO** (400k pasos, sin exploración, 50 partidas por rival, equipos
`train`), frente al DDQN de lr 1e-4 de la tercera cola:

| Categoría | PPO | DDQN |
|---|---|---|
| Ataque a un rival | 59,4 % | 35,2 % |
| Ataque en área a los rivales | 17,9 % | 7,2 % |
| **Ataque de daño al aliado** | **0,0 %** | 7,1 % |
| **Cambio voluntario** | **0,0 %** | 7,3 % |
| Protect | 2,5 % | 11,6 % |
| Mega | 8,9 % | 3,9 % |
| Victorias: random / solo / joint / expectimax | 50 / 50 / 47 / 37 de 50 | 35 / 22 / 17 / 12 de 50 |

PPO no ataca a su aliado ni cambia voluntariamente; DDQN sí (~7 % cada uno). Juega casi
siempre ataques directos o en área, con poco Protect. Es una política simple que gana mucho
contra los heurísticos con equipos `train`; falta medirla con equipos `eval`.

**Validación periódica y reanudación de PPO en vivo:** funcionan (validación cada 6 000
pasos, reanudación desde `final.pt` salta la fase ya hecha). **Bug encontrado:** tras
reanudar, la validación olvidaba el mejor previo y marcó 0,375 como nuevo mejor cuando el
mejor era 0,400, así que reemplazó `best.pt` por uno peor. Afectaba a DDQN y PPO. Corregido en
9727278 (`_restore_best`: recupera `best.json` del directorio de la corrida o del de origen);
tests en `tests/test_resume_keeps_best.py`. Verificado en vivo (PPO): mejor previo 0,333 en
el paso 3 000; tras reanudar, 0,292 en el paso 9 000 ya no reemplaza a `best.pt`.

### Ataques al aliado, cambios voluntarios y preview (séptima cola, commit 5786d82)

Diagnósticos sin exploración, rivales deterministas, equipos `train`, 25 partidas por rival
(50 en D3). "Ataque al aliado" = movimiento de daño dirigido al propio aliado.

**D1. ¿Bajan con el entrenamiento?** Sí, pero no desaparecen.

| Checkpoint | Ataque al aliado | Cambio voluntario | Protect | Victorias |
|---|---|---|---|---|
| lr 1e-4, 400k (preview v1) | 6,4 % | 6,6 % | 11,5 % | 49 % |
| mini kick-off 250k (preview v2) | 3,0 % | 5,2 % | 9,6 % | 76 % |
| mini kick-off 500k | 2,9 % | 5,8 % | 10,6 % | 78 % |
| mini kick-off 750k | 2,4 % | 5,6 % | 9,7 % | 83 % |
| mini kick-off 1M | 2,3 % | 4,5 % | 9,3 % | 82 % |

**D2. ¿Influye la simetría?** (ablación de 400k pasos, 3 semillas por variante.) En los
ataques al aliado, apenas: 7,3 / 7,8 / 8,3 % con simetría frente a 7,1 / 7,1 / 7,5 % sin ella.
En victorias, en este diagnóstico, **sin simetría gana más en las 3 semillas**: 37 / 39 / 38 %
frente a 30 / 28 / 25 % (unos +10 puntos; 100 partidas por checkpoint). A 1M pasos (quinta
cola) la diferencia fue al revés y pequeña (85 % con frente a 81 % sin). Conclusión: la
simetría no da una ventaja consistente.

**D3. ¿Es el preview?** Con el mismo DDQN, preview v1: 7,7 % de ataques al aliado y 41 % de
victorias; preview v2: 6,2 % y 42 %. El preview no explica los ataques al aliado.

**D4. Preview v2 con expectimax en el combate** (pareado con el preview heurístico):

| Rivales | Preview v2 | Heurístico | Diferencia (± EE) |
|---|---|---|---|
| equipos `train` | 78,9 % | 51,2 % | +27,7 (± 4,0) |
| equipos `eval` | 65,2 % | 48,0 % | +17,2 (± 4,0) |
| formación fija | 56,6 % | 55,5 % | +1,2 (± 4,4) |

Con expectimax en el combate, el preview v2 **sí** generaliza a equipos no vistos (+17,2) y
la formación fija no sirve (+1,2). Con greedy_joint pasaba al revés (formación fija +10,4,
igual que el preview en `eval`). El preview aprendió a adaptar la elección al rival, y esa
adaptación rinde con un combate más fuerte.

**P5 / P6. Pilotos A/B** (configuración de T0_base: lr 3e-4, preview v1, 400k pasos, una
semilla):

| Corrida | Ataque al aliado | Cambio voluntario | Victorias (diagnóstico, 100) | Validación final (200) |
|---|---|---|---|---|
| T0_base (referencia) | 6,6 % | 6,7 % | 28 % | 30,0 % (random 72, solo 16, joint 10, expectimax 22) |
| P5: recompensa simétrica (`RL_REWARD_HP_SELF=1`) | 7,4 % | 6,3 % | 35 % | 31,0 % (70, 18, 16, 20) |
| P6: máscara de daño al aliado (`RL_MASK_ALLY_DAMAGE=1`) | **0,0 %** | 7,9 % | **47 %** | **44,0 %** (88, 40, 24, 24) |

Lectura:

- **P5 no sirve:** penalizar la pérdida de HP propia no reduce los ataques al aliado y apenas
  cambia el rendimiento.
- **P6 rinde claramente mejor:** +14 puntos en la validación (z ≈ 3) y +19 en el diagnóstico.
  Sin ataques al aliado por construcción, el agente no desperdicia exploración ni turnos en
  ellos. Los cambios voluntarios no cambian.
- **Alcance:** una semilla, exploratorio (protocolo §3c).
- **Límite de la máscara:** permite los movimientos de estado (Helping Hand y similares),
  Pollen Puff y los movimientos que el aliado absorbe por habilidad (Lightning Rod, Water
  Absorb, Justified, etc.). Bloquea otras jugadas de daño al aliado que a veces son legítimas,
  como activar la Weakness Policy del aliado. Los ataques en área que golpean al aliado no se
  bloquean.
- **Decisión del usuario (16-11-2026):** máscara activa por defecto en DDQN y PPO
  (rl_env_spec v1.28); simetría apagada por defecto.

### Respuesta a la memorización (octava cola, 16-11-2026)

Implementado (apagado por defecto): validación reservada (`val`, 30 equipos de `train` no
entrenados), subconjuntos de equipos para una curva de escala y equipos rivales sintéticos
por recombinación (2 000 variantes de equipos reales, validadas). La octava cola compara con
y sin sintéticos (2 semillas) y 50 / 100 / 170 equipos. Diseño, análisis de los sintéticos y
criterios de lectura fijados de antemano: [`memorizacion_equipos.md`](memorizacion_equipos.md).

**Resultados (16-11-2026, §5 de ese documento):** contra equipos `eval` (3 heurísticos),
F170 38,7 / 34,0 %, SY50 39,0 / 34,0 % (semillas 0 / 1), S50 36,3 %, S100 33,3 %.
Los sintéticos no cumplen el criterio de adopción (+0,15 puntos de media; se pedían más de 8)
y la cantidad de equipos reales no cambia la generalización. La validación con `val`
coincide con `eval` (~40 % contra heurísticos), así que sí sirve para elegir checkpoints.
Incluso contra equipos `eval` que comparten 5 de 6 especies con uno de entrenamiento gana
~40 %, frente a 80–84 % con los propios: memoriza equipos concretos, no combinaciones de
especies. Espejo de los rivales: 0 discrepancias en ~2,3 M revisiones. Pendiente: decisión
del usuario sobre sintéticos y validación reservada.

### PPO con equipos `train` frente a `eval` y suite completa (novena cola, commit 7c7a11e)

- Pilotos de PPO (400k) contra los 3 heurísticos: con simetría 89,0 % (`train`) frente a
  **80,0 %** (`eval`); sin simetría 90,3 % frente a **74,0 %**. DDQN (mini kick-off 1M):
  83,0 % frente a 35,3 %. **PPO generaliza mucho mejor**; la memorización es propia de DDQN, no
  del preview ni de los equipos. Con `eval`, la simetría da +6 puntos en PPO (una semilla): la
  decisión de apagarla se tomó con validaciones sobre equipos de entrenamiento y merece
  revisarse. Detalle: [`memorizacion_equipos.md`](memorizacion_equipos.md) §6.
- Suite completa: **744 passed, 1 skipped, 6 xfailed** (3 min, `OMP_NUM_THREADS=4`).

### Validación en vivo con los valores por defecto nuevos (décima cola, 16-11-2026)

Control antes del kick-off: la máscara de daño al aliado nunca se validó en vivo como valor
por defecto. Se repite la validación en vivo (punto B) con máscaras conjuntas (DDQN) y por
hueco (PPO), 25 episodios por rival, sin variables de entorno. `live_validation` cuenta además
en cuántos pasos la máscara del aliado quita acciones y comprueba que nunca agrega ninguna.
Script: `scripts/server/kickoff_queue10.sh`; corre después de la novena cola. Resultados:
pendientes.

## D. Metodología

| # | Punto | Estado |
|---|---|---|
| 11 | ¿Equipos de entrenamiento = equipos de evaluación? Si sí, se mediría memorización; reservar equipos solo para evaluar | **hecho** (decisión del usuario: 200 para entrenar, 100 para evaluar; rl_env_spec v1.24). Verificado: el entrenamiento usaba los mismos 300 equipos que la evaluación. Separación por grupos de clones (mismas 6 especies) y estratificada; RL y previews en `train`, Conclusión en `eval` (78 equipos), prior y mundos desde `train`. Ver `protocolo_evaluacion.md` §2b |
| 12 | Ablación codificador × simetría (≥ 3 semillas, evaluación pareada) | pendiente |
| — | Suite completa en el servidor con los commits desde dbb188b | **hecho**: 651 pasan, 1 falla, 1 omitido, 6 xfail en 1627cc0. La falla era un test con el orden anterior del rival; actualizado al contrato v1.22 en 925b47a y verificado en el servidor |
| — | Suite completa (cuarta cola, f61f16d) | 712 pasan, 1 falla: `test_vectorized_training_smoke_mock[ddqn]`. La validación periódica (nueva por defecto) intentaba levantar Showdown en una corrida solo con `mock`; PPO tenía el mismo problema. Corregido en 31babfe y 8b8517d: la validación se desactiva cuando el calendario solo usa rivales offline. Los 77 tests relacionados pasan en el servidor. Falta repetir la suite completa cuando terminen las colas |

## Prerrequisitos del pipeline E

- Preview: puntos 4 (90 formaciones en vez de 360) y 5 (aumentación por permutación),
  `run_preview_value_probe` y entrenamiento del preview PPO. Sin checkpoint de preview,
  DDQN usa `/team 1234`.

## Ya resuelto en otras etapas

- Evaluación con ε = 0 y `DDQN_REQUIRED` (791e5b2, ddqn_spec 1.8).
- Rivales de entrenamiento validados (rl_env_spec v1.19) y con sorteo entre casi-empates
  (v1.21; evaluación determinista). Ver `docs/diversidad_baselines.md`.
