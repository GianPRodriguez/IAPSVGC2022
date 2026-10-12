# QD-PPO Agent Spec — Quality–Diversity + Masked PPO (VGC Gen 9)

| Campo | Valor |
|---|---|
| **Versión** | **1.5** (alineada con [`rl_env_spec.md`](rl_env_spec.md) v1.16) |
| **Estado** | FROZEN; `showdown/rl/qd_ppo/` |
| **Training** | **Sí**: MAP-Elites serial con PPO como operador de mejora |
| **Inferencia** | Despliega \(\pi^\*=\arg\max f\) desde `best_elite.pt` |
| **CLI** | `python -m showdown.rl.qd_ppo.train` |
| **CLI tune** | `python -m showdown.inference_sim.run_qd_ppo_hp_tune` |

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
`1024→512→512`, cabezas auto-regresivas de 24 y value head. No hay parámetros
compartidos entre celdas. `archive.pt` conserva el repertorio y
`best_elite.pt` el máximo fitness desplegable.

## Observación, acción y máscaras

Reutiliza sin bifurcar la Obs 1024D, acción auto-regresiva e IAM de
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
| **1.5** | 2026-10-11 | Train/infer bajo Preview QD-PGA congelado (`frozen_preview`). |
| **1.4** | 2026-10-11 | CLI `--lr/--clip-eps`; tune `run_qd_ppo_hp_tune` + tracking prev→nuevo; independencia vs Preview QD-PGA. |
| **1.3** | 2026-10-11 | QD/PPO, red, loop, fallos, evaluación, code map y ejemplo ampliados. |
| **1.2** | 2026-10-11 | Reescritura dual-audience y pairing QD-PGA-preview. |
| **1.1** | 2026-10-10 | Lectura pedagógica. |
| **1.0** | 2026-10-10 | Contrato Grid MAP-Elites + PPO. |
