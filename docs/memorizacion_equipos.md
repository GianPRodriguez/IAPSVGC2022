# Memorización de los equipos rivales y cómo reducirla

Estado: **diseño e implementación hechos (16-11-2026); resultados pendientes de la octava cola.**
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
siempre la formación más frecuente (+10,4), así que no hay evidencia de que adapte la
elección a rivales que no conoce.

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
  reinicios ni errores, y la validación usó solo equipos de `val`. Diagnóstico: con
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

## 5. Resultados

Pendiente (octava cola).
