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
especies. Espejo de los rivales: 0 discrepancias en ~2,3 M revisiones. **Decisión del usuario:** sin
sintéticos; validación reservada activa por defecto en DDQN y PPO (rl_env_spec v1.29).

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
Script: `scripts/server/kickoff_queue10.sh`.

**Resultados (commit 05f5883, 16-11-2026): todo en 0.** DDQN (conjuntas): 794 pasos, 100
episodios; PPO (por hueco): 813 pasos, 100 episodios. 0 caídas, 0 máscaras vacías (conjuntas ni
por hueco), entrenamiento = inferencia en ambas, 0 fallbacks, 0 diferencias de paridad de la obs,
alineación, historial y simetrías, equipo fijo en 100 de 100. La máscara del aliado actúa en el
80 % de los pasos (633 de 794; ~52 acciones conjuntas quitadas por paso) y nunca agrega acciones.
Costo: snapshot + obs 1,8 ms, forward 0,4–0,5 ms (medianas).

### Ensayo general y duración del kick-off (undécima cola, 16-11-2026)

Aprobado por el usuario. B: 1M pasos con la configuración final (máscara del aliado activa,
simetría apagada, validación reservada, sin sintéticos), semillas 0 y 1. C: 2M pasos (fases
×2), validación cada 250k. Cada corrida se diagnostica con `best.pt` (propios y `eval`) y con
`final.pt` (`eval`), para ver si `val` elige mejor que el último checkpoint. Script:
`scripts/server/kickoff_queue11.sh`.

**Resultados (commit b264a68, 16-11-2026).** Diagnósticos: 100 partidas por rival, sin
exploración; cifras = 3 heurísticos (300 partidas).

| Corrida | Checkpoint | Propios | `eval` | solo / joint / expectimax (`eval`) | `val` del checkpoint |
|---|---|---|---|---|---|
| B, semilla 0 (1M) | mejor = 1M | 87,3 % | **49,3 %** | 45 / 55 / 48 | 61,0 % |
| B, semilla 0 | último (1M) | — | 50,0 % | 47 / 53 / 50 | — |
| B, semilla 1 (1M) | mejor = 1M | 86,7 % | **47,0 %** | 49 / 49 / 43 | 68,0 % |
| B, semilla 1 | último (1M) | — | 49,0 % | 47 / 55 / 45 | — |
| C, semilla 0 (2M) | mejor = 1,75M | 90,7 % | **48,7 %** | 52 / 48 / 46 | 73,0 % |
| C, semilla 0 | último (2M) | — | 48,0 % | 44 / 49 / 51 | 64,5 % |
| F170 (cola 8, configuración anterior) | mejor | 80,0 / 84,3 % | 38,7 / 34,0 % | — | 52,0 / 58,0 % |

1. **La configuración final mejora la generalización de DDQN:** +10,6 y +13,0 puntos contra
   `eval` frente a F170 con las mismas semillas (media +11,8; ± ~4 cada una). Cambia la máscara
   del aliado (activa) y la simetría (apagada); no se puede separar el aporte de cada una.
2. **2M pasos no mejoran contra `eval`** (48,7 / 48,0 % frente a 47,0–49,3 % con 1M); solo suben
   los equipos propios (87 → 91 %). La curva de `val` de C subió hasta 1,75M (35 → 73 %) y bajó en
   2M (64,5 %), pero esa subida no se tradujo en `eval`: el máximo de 8 validaciones ruidosas
   (± 7 puntos por celda) sobreestima, y el preview ya vio los equipos de `val`. **`val` sirve
   para elegir el checkpoint; la cifra que se reporta sale de `eval`.**
3. **Mejor checkpoint frente al último:** iguales dentro del ruido (49,3 / 50,0; 47,0 / 49,0;
   48,7 / 48,0). Elegir con `val` no perjudica.
4. **Por rival, con `eval` (4 diagnósticos, 400 partidas):** solo 47,5 %, joint 50,3 %,
   expectimax 47,0 % (± 2,5): DDQN empata con los heurísticos frente a equipos nuevos; su ventaja
   (75–97 % con los propios) depende de conocer al rival. Expectimax es el más difícil en ambos.
5. Espejo de los rivales: 0 discrepancias (355 690, 354 837 y 702 753 revisiones). Suite
   completa: **747 passed, 1 skipped, 6 xfailed**.

**Recomendación de duración del kick-off: 1M pasos × 5 semillas** (~4,6 h): mismo resultado
en `eval` que 2M y comparaciones más sólidas. 4M no se recomienda (el doble de costo sin ganancia
esperable). **Decisión del usuario (16-11-2026): kick-off de 1M pasos × 5 semillas, por
ahora** (revisable si aparecen configuraciones que aprendan más lento).

### PPO de 1M pasos y nueva evaluación de la simetría (duodécima cola, 16-11-2026)

Aprobado por el usuario, junto con la decisión del punto 7: **PPO empieza desde cero** (sin
`--bc-checkpoint`); la clonación de comportamiento queda solo como posible comparación. Cuatro
corridas de PPO de 1M pasos con la configuración final y el currículo completo, con y sin
simetría (`RL_SYM_AUG=1`), semillas 0 y 1, medidas con `eval`. Preguntas: ¿PPO mantiene la
generalización de los pilotos (74–80 % con `eval`) con joint y expectimax? ¿La simetría mejora la
generalización? (La decisión de apagarla se tomó con validaciones sobre equipos de entrenamiento.)
Script: `scripts/server/kickoff_queue12.sh`.

**Resultados (commit 7f76106, 16/17-11-2026).** 3 heurísticos, 100 partidas por rival:

| Corrida | Propios | `eval` (best.pt) | `eval` (final.pt) | best.pt (paso) |
|---|---|---|---|---|
| PPO sin simetría, semilla 0 | 95,3 % | 77,7 % | 76,7 % | 1M |
| PPO sin simetría, semilla 1 | 96,3 % | 75,7 % | 78,7 % | 750k |
| PPO con simetría, semilla 0 | 96,3 % | **87,0 %** | 81,7 % | 1M |
| PPO con simetría, semilla 1 | 95,7 % | **80,0 %** | 85,0 % | 500k |

1. **PPO mantiene la generalización con el currículo completo** (76–87 % con `eval`, brecha −9
   a −21), frente a ~48 % de DDQN con la misma configuración y presupuesto.
2. **La simetría mejora a PPO contra equipos no vistos:** +9,3 y +4,3 (best.pt); con los 4
   diagnósticos de cada configuración, 83,4 % frente a 77,2 % (+6,2; z ≈ 3,9). Contra los equipos
   propios no cambia (96 %): por eso no se veía validando con equipos de entrenamiento.
3. Contra expectimax con `eval`: 75–85 %. Espejo de los rivales: 0 discrepancias.
4. Coincidiendo best.pt y final.pt (semilla 0 con simetría), los dos diagnósticos dieron 87,0 y
   81,7 %: el diagnóstico de 100 partidas por rival tiene ~± 3 puntos de ruido propio (los
   heurísticos desempatan al azar).

### Máscara del aliado frente a simetría en DDQN (decimotercera cola, 16-11-2026)

Aprobado por el usuario. El ensayo general mejoró +11,8 puntos en `eval` frente a F170, pero
cambiaron dos cosas (máscara activa, simetría apagada). MSYM = máscara activa + simetría activa,
1M, semillas 0 y 1: MSYM − F170 mide la máscara y B − MSYM la simetría. Script:
`scripts/server/kickoff_queue13.sh`.

**Resultados (commit be82f79, 17-11-2026).** Contra `eval`, best.pt (3 heurísticos):

| Semilla | F170 (sin máscara, con simetría) | MSYM (con máscara, con simetría) | B (con máscara, sin simetría) |
|---|---|---|---|
| 0 | 38,7 % | 52,0 % | 49,3 % |
| 1 | 34,0 % | 54,3 % | 47,0 % |

| Efecto | Semilla 0 | Semilla 1 | Media |
|---|---|---|---|
| Máscara del aliado (MSYM − F170) | +13,3 | +20,3 | **+16,8** |
| Apagar la simetría (B − MSYM) | −2,7 | −7,3 | −5,0 |

**La máscara del aliado explica toda la mejora del ensayo general.** En DDQN la simetría
parece ayudar algo con best.pt, pero con final.pt no hay diferencia (MSYM 49,3 / 49,7 frente a B
50,0 / 49,0): no concluyente. Espejo de los rivales: 0 discrepancias.

### Remedios contra la memorización (decimocuarta cola, 16-11-2026)

El análisis local de derrotas ([`memorizacion_equipos.md`](memorizacion_equipos.md) §7) mostró
puntos ciegos estables ante combinaciones de equipo nuevas (con piezas conocidas) que PPO no
comparte. Aprobado por el usuario: (a) regularización (AdamW con weight decay 0,01 + dropout 0,1)
y (b) borrado de los Pokémon rivales no activos de la obs al aprender (p = 0,5), 1M × 2 semillas
cada uno, frente al ensayo general B. Criterio: más de ~5 puntos sobre B en `eval` con ambas
semillas. Diseño en §8. Script: `scripts/server/kickoff_queue14.sh`.

**Resultados (commit 5936dc2, 17-11-2026):** **REG cumple el criterio** (+4,0 y +12,0 sobre B con
best.pt; media +8,0) y con final.pt llega a **62,3 / 65,3 %** contra `eval` (B: 50,0 / 49,0 %),
la primera vez que DDQN supera claramente a los heurísticos con equipos nuevos. FBD no mejora
(+0,3 / −0,7). En REG, `val` eligió el checkpoint de 750k, peor que el final en ~8 puntos: con 30
equipos y 50 partidas por rival, `val` no distingue checkpoints cercanos. Detalle en
[`memorizacion_equipos.md`](memorizacion_equipos.md) §8.

### Configuración final de los kick-offs (decisiones del usuario, 17-11-2026)

| | DDQN | PPO |
|---|---|---|
| Máscara de daño al aliado | activa | activa |
| Simetría | apagada | **activa** (ppo_spec 1.14) |
| Regularización | **weight decay 0,03 + dropout 0,1** (ddqn_spec 1.22; 0,01 hasta el 18-11) | — |
| Validación | reservada (`fit` / `val`) | reservada (`fit` / `val`) |
| Equipos sintéticos | no | no |
| Inicialización | — | desde cero |
| Pasos × semillas | 1M × 5 | 1M × 5 (decisión del usuario, 17-11-2026) |
| Checkpoint principal | **final.pt**; best.pt secundario | **final.pt**; best.pt secundario |

Todo por defecto en el código (sin variables de entorno); rl_env_spec v1.30, protocolo §3c.2.

**Verificación (17-11-2026, commit 1615bef):** suite completa en el servidor 753 passed + 1 test
corregido (`test_ddqn_gpu_replay` comparaba pérdidas con el dropout nuevo; ahora lo apaga) → 754
passed, 1 skipped, 6 xfailed. Prueba en vivo de 12k pasos sin variables de entorno: DDQN registra
`regularization` = weight decay 0,01 / dropout 0,1, sin simetría (`sym_frac_our` ausente) y
validación reservada (`fit`); PPO registra `sym_pi_loss` (simetría activa) y `fit`. Ambos sin
reinicios ni errores.

### Ajuste de REG y kick-offs oficiales (colas 15 y 16, decisiones del usuario, 17-11-2026)

**Decisiones:** PPO con el mismo presupuesto que DDQN (1M × 5 semillas); kick-offs encadenados,
DDQN primero y después PPO; ajustar los hiperparámetros de REG; 200 partidas por rival en la
evaluación de ambos.

**Orden:** el ajuste de REG va antes del kick-off de DDQN, para que el kick-off use el mejor
valor. Para no esperar una decisión intermedia, la elección es automática con una regla fijada de
antemano.

**Cola 15 (ajuste de REG):** tres variantes más fuertes que la actual (weight decay 0,03 +
dropout 0,1; 0,01 + 0,2; 0,03 + 0,2), 1M × 2 semillas. **La elección no usa `eval`:** cada
final.pt se mide contra los 30 equipos `val` con 200 partidas por rival, y la mejor variante se
adopta solo si supera a REG actual por más de 4 puntos (≈ 2 errores estándar); si no, se mantiene
la actual. Motivo: hasta ahora las decisiones de configuración (máscara, simetría, REG) se
tomaron mirando `eval`, lo que puede inflar algo la cifra final medida con esos mismos equipos
(ver la limitación en `memorizacion_equipos.md` §9). Script: `scripts/server/kickoff_queue15.sh`.

**Cola 16 (kick-offs oficiales):** DDQN 1M × 5 semillas con la REG elegida y después PPO 1M × 5
semillas, todo con la configuración final por defecto. Cada semilla se evalúa con final.pt
(principal) y best.pt (secundario) contra `eval`, y final.pt contra sus propios equipos, con 200
partidas por rival. Resumen: `showdown/rl/kickoff_report.py` (media entre semillas con IC de t de
Student, IC por partidas, por rival y PPO frente a DDQN con IC de Welch). Script:
`scripts/server/kickoff_queue16.sh`. Es una evaluación exploratoria respecto de la Conclusión
formal (run_robust_e2e, BO3, protocolo §2–§3), que queda como paso siguiente.

**Resultados de la cola 15 (commit 1efbf5c, 17-11-2026).** final.pt, 3 heurísticos:

| Configuración | `val` s0 | `val` s1 | **`val` media** | `eval` s0 (registro) | `eval` s1 (registro) |
|---|---|---|---|---|---|
| REG actual: weight decay 0,01 + dropout 0,1 | 65,0 % | 70,2 % | 67,6 % | 62,3 % | 65,3 % |
| **R_wd03: 0,03 + 0,1** | 74,7 % | 73,5 % | **74,1 %** | 63,3 % | 66,0 % |
| R_do2: 0,01 + 0,2 | 66,3 % | 67,5 % | 66,9 % | 61,3 % | 61,7 % |
| R_both: 0,03 + 0,2 | 77,7 % | 67,8 % | 72,8 % | 64,0 % | 60,0 % |

`val`: 200 partidas por rival (600 por semilla); `eval`: 100 por rival. **Elección por la regla:
R_wd03** (+6,5 puntos en `val` sobre REG actual, umbral +4), que usan las colas 17 y 18. En el
registro de `eval` la mejora es mucho menor (64,7 frente a 63,8 %, +0,9): parte de la subida en
`val` puede ser ruido o específica de esos 30 equipos (el preview ya los vio). Más dropout (0,2) no
ayuda. El valor por defecto del código sigue siendo 0,01 hasta que el usuario decida la
configuración del kick-off junto con los resultados de las colas 17 y 18.

**17-11-2026, 06:20: el usuario retiró la cola 16 antes de que empezara** (no llegó a ejecutar
nada). El script queda en el repo para lanzarlo cuando lo indique; la cola 15 sigue.

### Duración y simetría en DDQN sobre REG (decimoséptima cola, 17-11-2026)

Aprobado por el usuario (A y D del plan previo al kick-off). **A:** DDQN con la REG elegida en
la cola 15 y PPO con simetría, a 2M pasos (fases ×2), 2 semillas: ¿el kick-off debe ser de 1M o
2M? La prueba de 2M anterior (cola 11) fue sin REG. **D:** DDQN 1M con la REG elegida y simetría,
2 semillas. Elección con `val` (200 partidas por rival, final.pt); se recomienda el cambio si
supera a la referencia de 1M por más de 4 puntos. `eval` y propios quedan como registro. Para PPO,
`val` está cerca del techo (~93–95 %) y es poco sensible. Script:
`scripts/server/kickoff_queue17.sh`; resumen: `showdown/rl/kickoff_summary17.py`. Corre después
de la cola 15. El enfrentamiento directo entre agentes queda para el round robin final (decisión
del usuario).

**Resultados (commit ef53831, 17-11-2026): ningún cambio supera el umbral.** final.pt, 3
heurísticos, 2 semillas agrupadas:

| Pregunta | `val` nuevo | `val` referencia | Diferencia | `eval` nuevo (registro) | `eval` 1M (registro) | Propios nuevo |
|---|---|---|---|---|---|---|
| A: DDQN 2M frente a 1M (REG 0,03 + 0,1) | 75,0 % | 74,1 % | +0,9 (± 1,8) | 61,0 % | 64,7 % | 94,2 % |
| A: PPO 2M frente a 1M (simetría) | 87,8 % | 88,1 % | −0,3 (± 1,3) | 82,8 % | 83,4 % | 97,2 % |
| D: DDQN 1M con simetría frente a sin ella | 72,4 % | 74,1 % | −1,7 (± 1,8) | 65,0 % | 64,7 % | 92,0 % |

**Se mantienen 1M pasos para ambos y la simetría apagada en DDQN.** A 2M, DDQN gana algo más con
sus equipos y algo menos con `eval` (más memorización); PPO queda igual. Espejo de los rivales: 0
discrepancias.

### Valores Q por hueco en DDQN (decimoctava cola, 17-11-2026)

Aprobado por el usuario para antes del kick-off (opción C). Cabeza Q factorizada
V + A1(a1) + A2(a2) (+ interacción de rango r), ddqn_spec 1.21; variantes rango 0 y 8, 1M × 2
semillas, con la REG elegida. Elección con `val` (más de 4 puntos sobre la cabeza conjunta).
Diseño en [`memorizacion_equipos.md`](memorizacion_equipos.md) §10. Script:
`scripts/server/kickoff_queue18.sh`.

**Resultados (18-11-2026): no supera el umbral** (rango 0: +3,2 en `val`, −3,4 en `eval`; rango 8:
+1,4 / −4,2). Se mantiene la cabeza conjunta. Detalle en §10.

### Juego contra el plan del rival (decimonovena cola, 17-11-2026)

Aprobado por el usuario: medir si DDQN y PPO cambian su juego cuando el rival ejecuta su plan
(Trick Room, Tailwind, Fake Out, redirección) y si los heurísticos realmente lo ejecutan.
`policy_diag --context`, con el equipo del agente al azar en cada partida (solo para este
análisis). Diseño en [`memorizacion_equipos.md`](memorizacion_equipos.md) §11–§12. Script:
`scripts/server/kickoff_queue19.sh`.

**Resultados (18-11-2026, §12.1):** con equipo propio al azar ambos se desploman (DDQN 17–22 %,
PPO 28–34 % contra los heurísticos): están especializados en el equipo fijo. Los heurísticos casi
nunca activan Trick Room (0–1 % de las partidas) ni Tailwind (0–4 %), así que la evaluación no
pone a prueba esos arquetipos; Fake Out sí (~27 %). Los agentes no adaptan su juego al plan del
rival (atacan al que activa Trick Room o al redirector como al azar y no se protegen más del Fake
Out).

### Preview y combate por separado; piloto con equipos al azar (vigésima cola, 18-11-2026)

Aprobado por el usuario: la tupla (preview, combate) debe jugar bien en ambas etapas. (1) Mismo
combate con tres previews (v2, formación fija más frecuente, /team 1234) para aislar lo que aporta
el preview; (2) combate según el plan del rival con tu equipo; (3) piloto DDQN y PPO entrenados
con equipo propio al azar. Diseño en [`memorizacion_equipos.md`](memorizacion_equipos.md) §13.
Script: `scripts/server/kickoff_queue20.sh`.

**Parte 1 (18-11-2026, §13.1):** con el combate final, el preview v2 rinde 64,4 % (DDQN) y 85,3 %
(PPO); jugar siempre su formación más frecuente da 60,3 / 82,5 % (adaptación +4,1 / +2,8) y
/team 1234 da 37,3 / 71,3 % (elección +23,0 / +11,2). El preview elige igual para los dos agentes.
Piloto con equipos al azar (§13.2): DDQN no generaliza (19,5 % con equipo propio al azar, +0,6) y
pierde con tu equipo (29,8 %, −34,8); PPO generaliza en parte (45,5 %, +14,0; con tu equipo 72,8 %,
−10,6). Para el kick-off se mantiene el equipo fijo.

### Rival de arquetipos y entropía de PPO (vigesimoprimera cola, 18-11-2026)

Aprobado por el usuario: (1) greedy_joint con modo arquetipo (`PS_JOINT_ARCHETYPE=1`) como rival de
evaluación de los agentes finales; (2) ajuste de entropía de PPO (0,003 y 0,03 frente a 0,01) para
igualar el esfuerzo de ajuste con DDQN; (3) limitaciones documentadas en `protocolo_evaluacion.md`.
Tabla completa por baseline en [`memorizacion_equipos.md`](memorizacion_equipos.md) §14.1. Script:
`scripts/server/kickoff_queue21.sh`.

**Resultados (§14.4):** contra el rival de arquetipos ambos pierden 7–9 puntos (DDQN 62,3 → 53,8 %,
PPO 85,8 → 78,8 %); el rival activa Tailwind en ~29 % y Trick Room en ~11 % de las partidas. PPO
adapta algo su Protect a esos planes; DDQN no. Entropía de PPO: ninguna alternativa supera el
umbral (0,003: −2,1; 0,03: −0,5 en `val`); se mantiene 0,01. Con 0,03 aparecen cambios voluntarios
(3 %) sin perder rendimiento.

### Preview entrenado con su propio combate (vigesimosegunda cola, 18-11-2026)

Aprobado por el usuario tras la parte 1 de la cola 20. Pilotos RL en `preview_live`; un preview por
agente entrenado con su combate congelado y solo con `fit`; comparado con v2 en `val` (más de 4
puntos) y `eval`. Diseño en [`memorizacion_equipos.md`](memorizacion_equipos.md) §15. Script:
`scripts/server/kickoff_queue22.sh`.

**Resultados (§15.1): el preview nuevo es peor** (DDQN −5,8 en `val` y −3,2 en `eval`; PPO −1,4 y
−3,7). Se mantiene el preview v2. Hipótesis: recompensa saturada (el combate ganó el 92–97 % de las
partidas de entrenamiento contra greedy_joint).

### Kick-off oficial de PPO (vigesimotercera cola, 18-11-2026)

**Decisión del usuario:** iniciar primero el kick-off oficial solo de PPO (el de DDQN, después).
Configuración final por defecto: máscara del aliado, simetría, entropía 0,01, desde cero, validación
reservada, preview v2, equipo fijo; 1M pasos × 5 semillas (0–4). Evaluación de cada semilla con 200
partidas por rival: final.pt contra `eval` (principal), best.pt contra `eval` (secundario), final.pt
contra sus propios equipos, y final.pt contra greedy_joint en modo arquetipo (con `--context`;
recomendado tras §14.4, ~2 min por semilla). Resumen: `showdown/rl/kickoff_report.py` (sección 3b
para el rival de arquetipos). Script: `scripts/server/kickoff_ppo_oficial.sh`. Duración estimada
~5,5 h (real: 15:40 → 21:01, commit f671e65, salida `~/iapsvgc_runs/kickoff_ppo1`; todo rc=0).

**Resultados (equipo fijo `ddqn_team.txt`, 200 partidas por rival, 600 contra los 3 heurísticos por
semilla):**

| Medida | Media (IC 95 % entre semillas) | Por semilla (0 / 1 / 2 / 3 / 4) |
|---|---|---|
| **final.pt contra `eval`** (principal) | **81,2 % (± 3,5)**; por partidas ± 1,4 | 81,3 / 81,3 / 84,0 / 82,8 / 76,7 |
| best.pt contra `eval` (secundario) | 81,1 % | 77,7 / 80,0 / 83,0 / 83,7 / 81,0 |
| final.pt contra sus propios equipos (`fit`) | 96,2 % (brecha −15,0) | 96,0 / 96,0 / 96,5 / 97,5 / 95,2 |
| final.pt contra joint arquetipo | 77,1 % (± 2,2); joint normal 82,8 % (−5,7) | 76,0 / 77,5 / 76,5 / 80,0 / 75,5 |

Por rival (final.pt, `eval`): random 99,1 % (± 0,7), greedy_solo 82,7 % (± 4,4), greedy_joint
82,8 % (± 3,9), expectimax 78,2 % (± 4,7).

1. **Reproduce los pilotos:** 81,2 % frente a ~83–85 % de PPO con simetría en 2 semillas (colas 12 y
   22); la diferencia cabe en el IC entre semillas. La semilla 4 es la más débil (76,7 %); en ella
   best.pt (81,0 %) supera a final.pt, pero en la media best.pt y final.pt empatan (81,1 frente a
   81,2), así que la regla de usar final.pt como resultado principal no cambia nada.
2. **Expectimax sigue siendo el rival más difícil** (78,2 %), aunque PPO le gana con holgura en todas
   las semillas (73–83 %).
3. **Brecha de memorización de 15 puntos** (96,2 % con equipos rivales vistos frente a 81,2 % con
   desconocidos), similar a los pilotos.
4. **Rival de arquetipos:** −5,7 puntos (piloto: −7,0), estable entre semillas (± 2,2).

Resumen completo: `~/iapsvgc_runs/kickoff_ppo1/kickoff_summary16.md`. Kick-off de DDQN: pendiente de
la decisión del usuario.

#### Prueba online contra una persona (18-11-2026, exploratoria)

Pedido del usuario: el PPO del kick-off (semilla 2, final.pt, preview v2, equipo fijo) reta desde el
laptop (CPU) a un BO3 en Showdown público (`gen9championsvgc2026regmcbo3`) al usuario
velascoAlvarado. Script: `scripts/online/ppo_online_bo3.py` (mismo `PPOPlayer` de poke-env y misma
observación que el entrenamiento; política determinista). Antes, prueba local en el laptop contra
greedy_joint en BO3: 2-0, 0 errores, el BO3 (`/confirmready`) funciona.

| Partida | Turnos | Resultado | Bajas que causó PPO |
|---|---|---|---|
| 1 | 10 | Derrota (4-0) | 0 |
| 2 | 11 | Derrota (4-1) | 1 (Incineroar) |

Serie perdida 0-2 en 6,3 min; 25 decisiones, 0 excepciones, 0 órdenes de reserva (el agente funcionó
técnicamente; perdió jugando). Rival: equipo de sol (Venusaur + Charizard-Mega-Y, Incineroar,
Garchomp, Farigiraf, Sylveon). Replays y resumen en `logs/online_ppo/`.

Lo que se ve en los replays:
1. **Jugó las dos partidas casi igual** (mismo líder Raichu + Primarina, mismos primeros turnos):
   la política es determinista y no aprende entre partidas; el rival sí se adaptó.
2. **Turno 1 contra doble Protect:** atacó a los dos protegidos (Zap Cannon y Hyper Voice
   desperdiciados) y el rival activó el sol con Charizard-Mega-Y en el turno 2.
3. **Raichu dormido y luego caído** (Sleep Powder + Sludge Bomb); no hubo cambio voluntario para
   salvarlo (el 0 % de cambios voluntarios ya visto, memorizacion_equipos §13.1).
4. **Primarina repitió Hyper Voice** en vez de concentrar daño; **Arcanine encadenó Protect tres
   turnos** al final (el último falló), sin opciones reales.
5. **Patrón de juego no practicado** (corregido tras memorizacion_equipos §16): el sol en sí sí
   aparece en el entrenamiento (39 equipos del pool; el clima de habilidades se activa solo), pero no
   cómo lo juega una persona: doble Protect el turno 1 mientras evoluciona el Mega, Sleep Powder y
   Parting Shot. Los heurísticos casi no usan movimientos de estado ni Tailwind / Trick Room (§14.4).

Conclusión: con 2 partidas no hay estadística, pero sí ilustra las debilidades ya medidas contra un
humano: falta de adaptación entre partidas, 0 % de cambios voluntarios y arquetipos de campo no vistos
en entrenamiento. Refuerza el trabajo futuro (rival de arquetipos en el entrenamiento, más entropía).

### PPO v2: candidatos (vigesimocuarta cola, 18-11-2026)

**Decisión del usuario:** encolar P1, P2, P3 y P4 solo para PPO y reportarlo como **PPO v2**. La v1
(kick-off oficial) queda congelada: tag git `ppo-v1` (f671e65) y `~/iapsvgc_runs/ppo_v1`, en solo
lectura con `MANIFEST.md5`. Diseño y reglas de adopción en
[`memorizacion_equipos.md`](memorizacion_equipos.md) §17. Script: `scripts/server/kickoff_queue24.sh`.
Duración estimada ~8,2 h. Resultados: pendientes.

### P5: corrección de habilidades en la observación (vigesimoquinta cola, 18-11-2026)

Hallazgo del usuario en la prueba online: Primarina usaba Hyper Voice con el sol de Charizard-Mega-Y.
La observación no aplicaba Liquid Voice (Hyper Voice contaba como Normal, no como Agua); tampoco las
habilidades de potencia. **Decisión del usuario: P5 activo en PPO y DDQN** (por defecto en los
entrenamientos nuevos; los checkpoints anteriores conservan su observación). Entra en el kick-off de
DDQN y en la corrida final de PPO v2. Detalle en
[`memorizacion_equipos.md`](memorizacion_equipos.md) §18.

**19-11-2026 (§18.1), decisiones del usuario:**
- Corrección completa con la **calculadora real** (la de los heurísticos) en la observación.
- **Preview v2.1** con la corrección.
- **El kick-off de DDQN usa el preview v2.1, no el v2 congelado.**
- La cola 25 (P5 parcial) se canceló.

Auditoría: 128 habilidades, 173 movimientos y 57 objetos legales con efecto. Primer paso: medir la
velocidad de la calculadora.

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
