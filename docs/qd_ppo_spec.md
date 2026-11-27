# QD-PPO Agent Spec — Quality–Diversity + Masked PPO (VGC Gen 9)

| Campo | Valor |
|---|---|
| **Versión** | **2.0** (I 2.0: desde PPO v2, comparación pareada, reevaluación al desplegar; §I 2.0). 1.7 = diseño original, nunca entrenado |
| **Estado** | FROZEN; `showdown/rl/qd_ppo/` |
| **Training** | **Sí**: MAP-Elites serial con PPO como operador de mejora |
| **Inferencia** | Despliega \(\pi^\*=\arg\max f\) desde `best_elite.pt` |
| **CLI** | `python -m showdown.rl.qd_ppo.train2` (I 2.0; `--control` = PPO v2 sin archivo). `train` = I 1.7 |
| **CLI tune** | `python -m showdown.inference_sim.run_qd_ppo_hp_tune` |

## I 2.0 — QD-PPO desde PPO v2 (25-11-2026)

**Decisiones del usuario (25-11-2026):** arrancar desde PPO v2; medir contra un control "PPO v2 + el
mismo presupuesto"; aplicar después la misma receta a J y K. Motivo: la revisión de la familia QD
(25-11) encontró que, desde cero y con 1M de pasos repartidos en ~100 celdas, el mejor linaje casi no
aprende; que el desplegado se elegía con `argmax f` sobre estimaciones de 20 partidas (maldición del
ganador); y que los descriptores ocupaban ~5 de las 100 celdas. I 1.7 nunca se entrenó en vivo.

**Papel en la tesis.** I es la línea base on-policy de QD: MAP-Elites con PPO como operador de
variación, sin gradientes de los descriptores. Responde si mantener un archivo de estilos mejora a PPO
solo (frente al control) y sirve de ablación para J (crítico compartido) y K (componente diferenciable).
Cada élite guarda su propio crítico, entrenado con sus propios datos (sin el problema de V fuera de
política del crítico compartido).

**Formalización.** Archivo \(\mathcal{A}\) sobre una rejilla de 10 × 10 en \(\mathcal{B}\) con límites
calibrados. Descriptor \(b(\theta)\): fracción de acciones ofensivas frente a estado/preparación y
**foco de ataque** (de los turnos con dos ataques de un solo objetivo contra el rival, fracción en que ambos
apuntan al mismo; rl_env_spec v1.34), promediadas sobre partidas de al menos 4 turnos. El BD2 original
(tasa de cambios voluntarios) quedaba en 0–0,025 con PPO, que casi nunca cambia (prueba en vivo del 25-11):
el archivo habría sido de una dimensión. Decisión del usuario: foco de ataque. Fitness
\(f(\theta)=\mathbb{E}[R_{\text{Cand.1}}]\in[-1,1]\). Variación: \(\theta'=\mathrm{PPO}_{K}(\theta_p)\)
con \(\theta_p\) uniforme entre las celdas ocupadas. \(b\) solo decide dónde compite una política, nunca
cuánto vale: no hay recompensa de novedad.

| Componente | I 1.7 | I 2.0 |
|---|---|---|
| Arranque | 100k pasos de PPO desde cero | `final.pt` de PPO v2 (semilla con mejor `val`, regla previa): actor y crítico, observación P5 del checkpoint, equipo fijo, preview congelado, validación reservada (rivales de `fit`) |
| Mosaico | random / solo / joint 25 / 25 / 50 | proporciones de las fases de PPO v2: random 15, solo 25, joint 30, expectimax 30 (para no olvidar a expectimax) |
| Descriptores | rejilla sobre \([0,1]^2\) | límites = percentiles 5–95 de los descriptores por partida de 512 partidas del arranque (ancho mínimo 0,05); esas partidas dan además la estimación inicial del arranque, primera élite |
| Hijo | 1 actualización PPO (2048 pasos) | 2 actualizaciones PPO (4096 pasos), optimizador nuevo, crítico heredado del padre |
| Inserción | 20 BO1 propios; reemplaza si \(f_{\rm nuevo}>f_{\rm viejo}\) | el hijo juega 32 enfrentamientos nuevos (rival y semilla de equipos fijos); si su celda está ocupada, el ocupante juega **los mismos** y suma esas partidas a su estimación; el hijo reemplaza solo si la diferencia pareada es > 0. Celda vacía: entra directo. Al menos 16 partidas elegibles |
| Despliegue | `argmax f` | las 10 élites con mejor estimación se reevalúan en 300 enfrentamientos comunes con equipos `val` contra los tres heurísticos; π* = la mejor (`best_elite.pt`, `reeval.json`) |
| QD-score | \(\sum \max(0,f)\) | \(\sum (f+1)\) (desplazamiento estándar; todas las celdas cuentan) |
| Presupuesto | 1M desde cero | 1M de pasos extra sobre PPO v2 (~244 hijos); las partidas de evaluación van aparte |

**Control (`--control`).** El mismo entrenador sin archivo: PPO v2 sigue entrenando 1M de pasos con el
mismo mosaico, preview, equipo y semilla; se despliega su `final.pt`. Hipótesis central de la
confirmación: I 2.0 > control.

**Enfrentamientos comunes.** Un enfrentamiento es `{rival, semilla, grupo}`: la semilla fija los equipos
de los dos lados y el grupo los equipos posibles (`val` para el despliegue). El entorno vectorizado los
arranca con `start_matches` y encadena el siguiente al terminar cada partida
(`showdown/rl/qd_ppo/matchups.py`). Una partida caída no cuenta (no es un empate).

**Medición.** Cola `scripts/server/qd_i2.sh` (espera al kick-off de PPO v2): control, I 2.0 y una
exploración descriptiva contra `eval` (I 2.0, control y PPO v2, 200 partidas por rival). La confirmación
se pre-registra aparte.

**Preview (26-11-2026, decisión del usuario).** I 2.0, J 2.0, K 2.0 y el control juegan con el preview
**v2.1b de PPO v2** (`preview_v21b/preview_s0/best.pt`, un `team_preview_ppo`), cargado con
`PREVIEW_KIND=team_preview_ppo`. El preview QD-PGA v1.6 con piloto PPO v2 se lanzó y se detuvo: con el equipo fijo `ddqn_team.txt` (Raichu, Rillaboom, Gholdengo, Arcanine-H, Staraptor, Primarina) ninguna especie está en las listas de soporte, así que \(b_2\) vale siempre 0, y \(b_1\) solo cambia si se lleva a Rillaboom (casi nunca): rejilla calibrada \([0, 0{,}025]^2\), cobertura 0,02 (2 celdas), `best_f` = 1,000 con 20 partidas, y 64 de 256 iteraciones en 2 h 51 min (~11 h en total).
Un archivo de 2 celdas es un escalador de colinas con ruido, no QD. Con el mismo preview que PPO v2, la
comparación con el control y con el arranque aísla el efecto del combate QD. El preview QD-PGA se
rediseñará aparte con descriptores que tengan sentido para el equipo fijo.

```bash
PREVIEW_KIND=team_preview_ppo python -m showdown.rl.qd_ppo.train2 --seed 0 --init-checkpoint <ppo_v2>/final.pt --start-ps --num-envs 32 \
  --device cuda --preview-checkpoint <preview_v21b>/best.pt --require-preview --checkpoint-dir <dir> [--control]
```

Código: `showdown/rl/qd_ppo/train2.py` (I 2.0 y control), `matchups.py`, `archive.py` (límites, estimación
acumulada `n_eval`, QD-score, metadatos), `mosaic.py` (`MOSAICS`, `make_sampler`), `vec_env.py`
(`start_matches`). `train.py` (I 1.7) queda como referencia; nunca produjo un agente.

### Exploración de la familia QD 2.0 (26-11-2026, colas `qd_i2` y `qd_jk2`, commit `7814cb4`)

Contra `eval`, 200 partidas por rival, tasa de victorias en %. La columna `val` es la reevaluación de despliegue.

| Agente (preview v2.1b de PPO v2) | random | solo | joint | expectimax | **3 heur.** | `val` (300, reeval) |
|---|---|---|---|---|---|---|
| J 2.0 PGA-ME (`best_elite`) | 99,5 | 85,0 | 83,0 | 77,5 | **81,8 %** | 91,7 % |
| Control: PPO v2 s1 + 1M | 99,5 | 84,0 | 87,0 | 72,0 | **81,0 %** | 85,3 % |
| I 2.0 QD-PPO (`best_elite`) | 98,5 | 86,0 | 85,5 | 71,0 | **80,8 %** | 93,3 % |
| PPO v2 s1 (arranque) | 99,5 | 82,0 | 83,5 | 75,5 | **80,3 %** | 91,0 % (kick-off) |
| K 2.0 PPGA (`best_elite`) | 98,5 | 80,5 | 82,5 | 72,0 | **78,3 %** | 90,0 % |

| Archivo | Celdas (cobertura) | QD-score | Hijos / aceptados | Tiempo |
|---|---|---|---|---|
| I 2.0 | 18 (0,18) | 35,3 | 245 / 93 | 3 h 04 min (90 pasos/s) |
| J 2.0 | 44 (0,44) | 83,7 | 2880 / 775 (1440 GA, 1260 PG, 180 codicioso) | 2 h 46 min (100 pasos/s) |
| K 2.0 | 87 (0,87) | 162,1 | 520 ramas / 204, 65 iteraciones, 0 reinicios | 1 h 33 min (181 pasos/s) |
| Control | — | — | — | 1 h 53 min (146 pasos/s) |

**Lectura (descriptiva, 600 partidas por agente contra las 3 heurísticas, IC 95 % ≈ ± 3,2 pts por tasa y
≈ ± 4,5 pts por diferencia).**
1. **Ninguno se distingue del control.** J +0,8, I −0,2 y K −2,7 pts. Todo cae dentro del ruido.
2. **Tampoco del arranque:** el control mejora +0,7 pts sobre PPO v2 s1. Con 1M pasos más, PPO v2 ya está
   cerca de su techo con esta receta.
3. **Maldición del ganador en el despliegue.** El mejor del top-10 en `val` (300 partidas) da 90–93 %, pero
   en eval vuelve a ~80 %. Elegir el máximo de 10 estimaciones ruidosas sigue sesgando hacia arriba. `val`
   además es más fácil que eval para s1 (91,0 frente a 79,3 en el kick-off).
4. **Calidad-diversidad:** K cubre el 87 % del mapa con QD-score 162, J el 44 % y I el 18 %. Las élites de
   K y J rinden casi lo mismo que la mejor (K 6,3: 90 % en `val`; J 7,4: 91,7 %). El valor de QD aquí está en
   la variedad de estilos con rendimiento parecido, no en un agente más fuerte.
5. **Por rival:** J es el mejor contra expectimax (77,5 % frente a 72,0 del control, +5,5 pts, en el límite
   del ruido para un rival con 200 partidas, IC ≈ ± 8). Es el único indicio de que otra élite castiga mejor a
   un rival concreto; habría que confirmarlo.
6. **Contraste con la hipótesis a priori (protocolo §4.10):** se esperaba I ≈ control (sí), K algo peor
   (sí, −2,7) y J peor (no: J es el mejor, aunque dentro del ruido). En QD-score se esperaba K ≥ I > J; salió
   K > J > I. El archivo de I queda chico porque cada hijo cuesta 2 actualizaciones de PPO (245 hijos frente a
   2880 de J).

## Por qué existe

Optimizar una única política puede descubrir una sola forma de ganar. QD-PPO
mantiene un archivo 10×10: cada celda representa un estilo y guarda la política
de mayor fitness encontrada allí. PPO mejora copias de elites; MAP-Elites decide
si merecen ocupar o reemplazar una celda. Diversidad y calidad permanecen
separadas: el descriptor ubica, pero no altera la recompensa.

## Glosario

| Término | Significado |
|---|---|
| Elite | Mejor actor-crítico conocido de una celda |
| BD | Dos medidas de estilo en \([0,1]^2\) |
| Coverage | Fracción de las 100 celdas ocupadas |
| QD-score | Suma de fitness no negativo de elites |
| Emitter | Copia de un padre mejorada por PPO |
| Fitness \(f\) | Media Candidate 1 usada para archivo |
| Mosaic | Rivales random/solo/joint con pesos 25/25/50 |

## Algoritmo paso a paso

1. Se crea un actor-crítico PPO y se hace bootstrap con Candidate 2.
2. Se evalúa su fitness Candidate 1 y BD; si hay suficientes episodios elegibles,
   se inserta en el bin correspondiente.
3. Se samplea uniformemente un elite ocupado.
4. Se copia todo su `state_dict` actor-crítico.
5. La copia recoge 2048 pasos y ejecuta mejora Masked PPO con Candidate 2.
6. Se evalúa en 20 BO1 del mosaic: fitness usa todos; BD solo los elegibles.
7. Una celda vacía acepta; una ocupada solo reemplaza si \(f_{\rm new}>f_{\rm old}\).
8. Tras el presupuesto se guarda archivo y mejor elite global.

### Intuición de Quality–Diversity

MAP-Elites conserva
\[
\pi_{ij}=\arg\max_{\pi:\operatorname{bin}(\mathrm{BD}(\pi))=(i,j)}f(\pi).
\]
Dos políticas no compiten si expresan estilos distintos. `coverage` mide nichos
descubiertos; `QD-score` suma \(\max(0,f)\). PPO es un emitter local: mejora
una copia con Candidate 2; la evaluación Candidate 1 decide bin y reemplazo.

### Red y contenido del archivo

Cada elite guarda el `state_dict` completo del `ActorCritic`: torso
codificador estructurado + `1272→512→512`, cabezas auto-regresivas de 40 y value head. No hay parámetros
compartidos entre celdas. `archive.pt` conserva el repertorio y
`best_elite.pt` el máximo fitness desplegable.

## Observación, acción y máscaras

Reutiliza sin bifurcar la ObsSpec v2 (6332D), acción auto-regresiva e IAM de
[`rl_env_spec.md`](rl_env_spec.md) y [`ppo_spec.md`](ppo_spec.md). Candidate 2
optimiza pesos; Candidate 1 mide calidad del archivo. BD1 mide agresividad frente
a status/setup y BD2 hard switches. BD nunca entra en reward ni fitness.
Tera id 22 permanece siempre ilegal.

## Entrenamiento

El data loop es bootstrap → evaluar/sembrar → samplear padre → PPO corto →
evaluar → insertar. Cada elite guarda actor y crítico completos, de modo que no
hay parámetros compartidos ni congelados entre celdas. Durante un emit, el padre
queda intacto y solo cambia la copia.

```bat
python -m showdown.rl.qd_ppo.train --seed 0 --smoke
python -m showdown.rl.qd_ppo.train --seed 0 --start-ps --suite-games 200
python -m showdown.rl.qd_ppo.train --seed 0 --bc-checkpoint logs/ppo_bc.pt --checkpoint-dir logs/qd_ppo_archive
```

`--smoke` reduce internamente presupuesto, rollout y evaluaciones. Sin
`--start-ps`, el loop usa `MockBattleBackend`; para resultados live se requiere
Showdown local con `--no-security`. Artefactos: `archive.pt`,
`best_elite.pt`, `summary.json` y `reporte.txt`.

### Loop de datos completo

1. Bootstrap mejora una política durante 100k pasos Candidate 2.
2. Se evalúan semilla y hasta cuatro auxiliares.
3. Un padre se samplea uniformemente entre celdas ocupadas.
4. Se clona actor-crítico y se recogen 2048 pasos contra el mosaic.
5. PPO usa los defaults completos del contrato base.
6. Veinte BO1 Candidate 1 producen fitness y BD elegible.
7. Con al menos diez elegibles, el hijo intenta insertar/reemplazar.
8. Cada outer guarda archivo y mejor elite; nominalmente hay 109.

La evaluación no consume `T_total`; el smoke recorta valores solo para probar
integración, no redefine el contrato.

## Hiperparámetros

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| Grid | `10×10` | Discretizar BD | Más bins dan detalle y menor densidad |
| \(T_{\rm total}\) | `1_000_000` | Presupuesto env | Más permite más emits |
| Bootstrap | `100_000` | Crear política base | Mayor mejora semilla y resta QD |
| Emitters | `4` | Hijos por iteración | Agrupa checkpoints y presupuesto |
| Pasos/emit | `2048` | Mejora PPO | Más cambia más al hijo |
| \(N_{\rm outer}\) | `109` nominal | Iteraciones | Derivado del presupuesto congelado |
| Fitness \(G\) | `20` BO1 | Estimar calidad/BD | Más reduce ruido y cuesta evaluación |
| \(g_{\min}\) | `10` | Validez de BD | Evita bins basados en partidas cortas |
| Padres | uniforme filled | Cobertura neutral | No favorece elites de alto fitness |
| Replace | \(f_{\rm new}>f_{\rm old}\) | Elitismo | Empates conservan incumbent |
| Workers | `1` | Orden reproducible | Limita throughput |
| Semillas auxiliares | `4` | Poblar bins iniciales | Consume budget para diversidad inicial |
| Mosaic | `25%/25%/50%` | random/solo/joint | Pondera más greedy_joint |
| Log pasos/tiempo | `5_000` / `60 s` | Heartbeat | Observabilidad, no gradiente |
| Max turns | `40` | Acotar episodios | Aplica semántica PPO de truncation |
| Seed | `0` | Reproducibilidad | Cambia padres, datos y pesos |

Los hiperparámetros internos PPO son los de [`ppo_spec.md`](ppo_spec.md).
El CLI de train acepta `--lr` y `--clip-eps` para overrides (tune / ablaciones).

### Tune de hiperparámetros (independiente del Preview QD-PGA)

Fase mínima: solo knobs del **actor PPO** dentro de cada emit. Grid, bootstrap,
`n_emit`, mosaic y Preview QD-PGA quedan congelados (mejora posterior puede
añadir knobs de archivo).

```bat
python -m showdown.inference_sim.run_qd_ppo_hp_tune --dry-run
python -m showdown.inference_sim.run_qd_ppo_hp_tune --seed 0 --max-trials 6 --suite-games 4
python -m showdown.inference_sim.run_qd_ppo_hp_tune --seed 0 --max-trials 6 --start-ps --no-smoke --t-total 50000
```

| Ítem | Contrato |
|------|----------|
| Knobs variables | `lr ∈ {1e-4,3e-4,1e-3}`, `clip_eps ∈ {0.10,0.20}` |
| Congelados | grid 10×10, \(\gamma\), mosaic, `n_emit`, BC opcional fijo, … |
| Score | `MEAN_WR` de \(\pi^\*\) (suite Cand. 1); QD-score solo en reporte de train |
| Criterio | Aceptar trial si mejora el mejor score |
| Independencia | No entrena ni afina `team_preview_qd_pga` |

**Tracking** (cómo cambian los HP trial a trial):

| Artefacto | Contenido |
|-----------|-----------|
| `logs/qd_ppo_hp_tune_history.jsonl` | `prev_overrides` → `overrides` + `param_change` (from/to/delta) |
| `logs/qd_ppo_hp_tune_progress.json` | estado vivo del trial actual |
| `logs/qd_ppo_hp_tune_tracking.txt` | log humano con `param_change` |
| `logs/qd_ppo_best_hp.json` | mejores overrides + score |
| `logs/figures/qd_ppo_hp_tune_latest.svg` | score vs trial; ledger en `hp_tune_ledger*.svg` |

### Ejemplo de inserción

Un hijo con BD `(0.34,0.81)` cae en `(3,8)` y obtiene fitness `0.25`. Entra si
la celda está vacía o contiene `0.20`; no reemplaza `0.25` porque la mejora debe
ser estricta. Candidate 2 produjo los pesos, pero no se suma al QD-score.

## Ilegales, máscaras y fallos

- IAM excluye ilegales en sampling, greedy y ratio PPO.
- Máscara vacía usa `pass` defensivo y debe auditarse.
- L1 permite fallback local observable.
- L2/crash se descarta del rollout y no define conducta del BD.
- Turn-cap puede bootstrappear en PPO; evaluación usa Candidate 1.
- Menos de `g_min` episodios elegibles impide inserción.
- L3 aborta sin fabricar fitness ni ocupar una celda.

## Training frente a inferencia

Training mantiene muchas políticas, recopila datos, actualiza por gradiente y
modifica el archivo. Inferencia usa solo `best_elite.pt`, greedy masked, sin
archivo activo ni mejora. Coverage/QD-score describen búsqueda; win rate de
\(\pi^\*\) describe el bot desplegado.

La suite de training acompaña coverage, QD-score, aceptación y fitness para
diagnóstico y selección. La **Conclusion eval** carga el artefacto congelado,
agrega el preview familiar y usa el round-robin común; no consulta el archivo
para cambiar de estilo entre partidas.

## Team Preview y Conclusión

**No hay Team Preview en el MDP de combate.** Flujo familiar: (1) entrenar
Preview QD-PGA → `archive.pt` / `best_elite.pt`; (2) train/infer QD-PPO **bajo**
ese preview congelado (`frozen_preview`, `combat_family='qd_ppo'`). Artefactos
de preview y de combate son distintos. Véase
[`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

## Mapa de código

| Archivo | Responsabilidad |
|---|---|
| `showdown/rl/qd_ppo/train.py` | presupuesto, outer loop y artefactos |
| `showdown/rl/qd_ppo/archive.py` | bins, elitismo y métricas |
| `showdown/rl/qd_ppo/improve.py` | emitter PPO y mosaic |
| `showdown/rl/qd_ppo/evaluate.py` | fitness y BD |
| `showdown/rl/qd_ppo/agent.py` | wrapper actor-crítico |
| `showdown/rl/qd_ppo/player.py` | despliegue |
| `showdown/rl/ppo/network.py` | arquitectura |
| `showdown/battle_bots/qd_ppo/pokeenv_player.py` | integración live |

## Limitaciones

- Guardar actor-crítico completo por celda consume memoria.
- El presupuesto se divide entre estilos; el mejor puede recibir menos updates.
- Mosaic fijo no representa todos los rivales finales.
- Fitness de 20 juegos es ruidoso y puede provocar reemplazos fortuitos.
- Despliegue por máximo fitness no usa la diversidad durante una batalla.
- Tera está desactivada.

## Cómo leer las métricas

Coverage alto sin QD-score alto indica variedad débil; QD-score alto con poca
coverage indica concentración en pocos nichos. `best_f` puede mejorar aunque
ambas métricas globales apenas cambien. La tasa de aceptación mezcla inserciones
en celdas vacías y reemplazos, por lo que debe leerse junto al mapa.

Los BD describen comportamiento observado, no intención ni habilidad. Con solo
diez partidas elegibles pueden fluctuar entre bins vecinos. Para reportes se
conservan seed, presupuesto y mosaic, y la métrica de juego principal sigue
siendo la suite de \(\pi^\*\), no coverage por sí sola.

## Changelog

| Versión | Fecha | Notas |
|---|---|---|
| **2.1.1** | 2026-11-26 | F1 (protocolo §5.10): I 2.0 rinde menos que el control (79,5 frente a 82,2 %, −2,64 pts, no inferioridad no confirmada); J 2.0 confirma J > control contra expectimax (+3,0 pts); K 2.0 −3,14 pts. Checkpoints de PPO v2 y del control en `checkpoints/ppo_v2/`. |
| **2.1** | 2026-11-26 | Herramientas de F1–F4 (protocolo §4.11–§4.14): `confirm_f1.py` (despliegue con 1000 partidas `val`, partidas pareadas, no inferioridad y Holm, fusión de resultados, exportación de élites), `selector_f2.py` (selector de élite por rival), `team_roles.py` (roles por set), mosaico `pool` (élites como rivales `ppo_v1`). |
| **2.0.2** | 2026-11-26 | Exploración de la familia QD 2.0 (§I 2.0): contra eval J 81,8 %, control 81,0 %, I 80,8 %, PPO v2 s1 80,3 %, K 78,3 %; ninguno se distingue del control; QD-score K 162 > J 84 > I 35; maldición del ganador en el despliegue por `val` (90–93 % → ~80 %). |
| **2.0.1** | 2026-11-26 | I 2.0, J 2.0, K 2.0 y el control usan el preview v2.1b de PPO v2 (`PREVIEW_KIND=team_preview_ppo`, nuevo en `frozen_preview`): el preview QD-PGA era degenerado con el equipo fijo (cobertura 0,02, \(b_2\) siempre 0). Decisión del usuario. Ver §I 2.0, Medición. |
| **2.0** | 2026-11-25 | **I 2.0** (decisiones del usuario tras la revisión de la familia QD): arranque desde PPO v2, mosaico con las proporciones de PPO v2, descriptores calibrados, hijos con 2 actualizaciones PPO, comparación pareada con el ocupante en enfrentamientos comunes, reevaluación de las 10 mejores en `val` antes de desplegar, QD-score \(\sum(f+1)\), control `--control`. Ver §I 2.0. |
| **1.7** | 2026-11-15 | Codificador estructurado (rl_env_spec §3.c) dentro del actor; consistencia por simetría (§3.d) en las actualizaciones PPO del actor (vía `PPOAgent.update`). Objetivo aliado del slot 0 corregido. Checkpoints/archivos anteriores incompatibles. |
| **1.6** | 2026-11-14 | ObsSpec v2 (6332D, OTS completo + estado + historial) y ActionSpec v2 (40 ids por slot: mega como flag de cada movimiento×objetivo; máscara de entrenamiento = `valid_orders`). Checkpoints anteriores incompatibles. |
| **1.5** | 2026-10-11 | Train/infer bajo Preview QD-PGA congelado (`frozen_preview`). |
| **1.4** | 2026-10-11 | CLI `--lr/--clip-eps`; tune `run_qd_ppo_hp_tune` + tracking prev→nuevo; independencia vs Preview QD-PGA. |
| **1.3** | 2026-10-11 | QD/PPO, red, loop, fallos, evaluación, code map y ejemplo ampliados. |
| **1.2** | 2026-10-11 | Reescritura dual-audience y pairing QD-PGA-preview. |
| **1.1** | 2026-10-10 | Lectura pedagógica. |
| **1.0** | 2026-10-10 | Contrato Grid MAP-Elites + PPO. |
