# Entrenamiento en el servidor (phantom)

Guía para correr entrenamiento y experimentación en el servidor de la tesis,
aprovechando sus 32 hilos y 3 GPU. La laptop ya no se usa para experimentar.

| Recurso | Servidor |
|---|---|
| CPU | AMD Threadripper PRO 5955WX, 16 núcleos / 32 hilos |
| RAM | 251 GiB |
| GPU | 3 × NVIDIA RTX A5500 (24 GiB c/u), driver CUDA 13.0 |
| SO | Ubuntu 24.04, sin sudo |
| Python | conda `kuroneko` (Python 3.14) |
| Node | v26 |

## Principio: un método a la vez, paralelismo por dentro

Nunca se entrenan dos métodos distintos al mismo tiempo. Cada entrenamiento
dispone de todo el servidor y lo aprovecha internamente:

| Qué | Cómo se paraleliza | Opción |
|---|---|---|
| DDQN / PPO (combate) | N batallas en N procesos; la red decide en lote en GPU | `--num-envs`, `--envs-per-server`, `--device` |
| Tuning de hiperparámetros (mismo método) | Trials simultáneos, uno por GPU, con puertos separados | `--parallel-trials` (+ `--num-envs` por trial en DDQN/PPO) |
| Heurísticos: random, greedy_solo, greedy_joint, expectimax (`run_robust_e2e`) | Los matchups se reparten en procesos; un reporte combinado | `--procs`, `--workers` |
| Tuning de expectimax / barrido MCTS | Cada corrida E2E usa `--procs` | `--procs` |
| Secuencia completa | Cola que ejecuta un paso tras otro, con reanudación | `run_train_queue` |

Por qué procesos y no hilos: las decisiones de los bots (greedy_joint,
expectimax, la inferencia de stats) son Python puro. En un solo proceso
compiten por un núcleo aunque se usen corrutinas. Con procesos, cada batalla
usa su propio núcleo.

Por qué la GPU ayuda poco por sí sola: las redes son chicas (observación de 1024
dimensiones, MLP). El cuello de botella es simular batallas en CPU. La GPU
sirve para decidir en lote y para correr trials de tuning en paralelo.

## Puesta en marcha (una vez)

Estructura esperada (ya existe en el servidor):

```
~/20152948-GIANPIERRERODRIGUEZ/
  IAPSVGC2022/        <- este repo
  pokemon-showdown/   <- con npm install hecho
  poke-env/
  damage-calc/
```

```bash
conda activate kuroneko
cd ~/20152948-GIANPIERRERODRIGUEZ/IAPSVGC2022
pip install -r requirements-dev.txt          # incluye -e ../poke-env
pip install "torch>=2.0,<3" --index-url https://download.pytorch.org/whl/cu130
pip install scipy matplotlib pytest

# Repos hermanos en los commits validados y compilados (ver "Versiones")
B=~/20152948-GIANPIERRERODRIGUEZ
git -C $B/pokemon-showdown checkout 5b01be810 && (cd $B/pokemon-showdown && node build)
git -C $B/poke-env checkout 7ff6212
git -C $B/damage-calc checkout bacbbfc && (cd $B/damage-calc/calc && rm -rf dist .tsbuildinfo && npm run compile)

bash scripts/server/check_env.sh             # debe terminar con 0 faltantes
```

`node` viene de nvm y solo entra al PATH en shells interactivos. El código lo
busca solo (`NODE_BINARY` > PATH > `~/.nvm/versions/node/*`) y lo pasa a los
procesos hijos, así que también funciona desde SSH, cron o la cola.

El límite de archivos abiertos (`ulimit -n` = 1024) se sube solo hasta 65536 al
arrancar cada entrenamiento. No hace falta sudo, porque el límite duro es 1048576.

## Flujo de trabajo con la laptop

- **Laptop**: única fuente de código. Allí se edita y se hace commit y push.
- **Servidor**: solo `git pull`, mediante el script:

```bash
bash scripts/server/sync.sh
```

Los entrenamientos escriben en archivos versionados de `logs/`. El script
respalda esos cambios en `~/iapsvgc_runs/_logs_backup/<fecha>/`, restaura
`logs/` y hace `git pull --ff-only`. Si encuentra cambios fuera de `logs/` se
detiene sin tocar nada.

## Corridas largas: usar tmux

Las corridas duran horas. Hay que lanzarlas dentro de `tmux` para que sigan
corriendo si se cae la conexión SSH:

```bash
tmux new -s tesis            # crear sesión
conda activate kuroneko
# ... lanzar el entrenamiento ...
# Ctrl+b, luego d            -> desacoplar (sigue corriendo)
tmux attach -t tesis         # volver a verla
```

## Comandos recomendados para este servidor

`{N}` = entornos o procesos paralelos. Por defecto son núcleos menos 4, con
tope de 28; en este servidor da 28. Pasar `0` en `--num-envs` o `--procs` lo
calcula solo.

### DDQN / PPO (combate)

```bash
python -m showdown.rl.ddqn.train --seed 0 --opponent curriculum --start-ps \
  --num-envs 24 --envs-per-server 4 --device cuda \
  --checkpoint-dir ~/iapsvgc_runs/ddqn_s0

python -m showdown.rl.ppo.train --seed 0 --opponent curriculum --start-ps \
  --num-envs 24 --envs-per-server 4 --device cuda \
  --checkpoint-dir ~/iapsvgc_runs/ppo_s0
```

- `--steps` y las fases cuentan pasos de entorno **totales** (suma de los N
  entornos). El presupuesto no cambia al variar `--num-envs`; solo termina antes.
- `--envs-per-server 4`: 24 entornos sobre 6 servidores Showdown (puertos
  8000-8005). Cada servidor ya reparte sus batallas en subprocesos propios.
- Los pasos son asíncronos: la red decide en lote para los entornos que ya
  respondieron, sin esperar al más lento. En PPO se esperan todos antes de cada
  actualización, así que el lote es on-policy y el GAE es exacto por entorno.
- Si un entorno se cuelga o muere, se relanza solo. Ese paso se marca como
  `crash` y no entra al buffer. El contador `restarts=` aparece en el heartbeat.

#### Preview congelado de la familia (obligatorio en runs oficiales)

El combate juega con el preview ya entrenado de su familia: PPO-preview para
DDQN/PPO, Nash para MCTS y QD-PGA para la familia QD. Si el checkpoint no
aparece, el código usa la formación fija `/team 1234`. Para que eso no pase en
silencio en un run oficial:

```bash
python -m showdown.rl.ddqn.train ... \
  --preview-checkpoint ~/iapsvgc_runs/<run>/preview_ppo/final.pt --require-preview
```

Al arrancar, el entrenador imprime `preview=... cargado=sí|NO`. Con
`--require-preview`, un preview que no carga detiene el entrenamiento. La ruta
se propaga a los procesos de los entornos.

### Tuning de hiperparámetros (DDQN / PPO)

```bash
python -m showdown.inference_sim.run_ddqn_hp_tune --seed 0 --max-trials 6 \
  --opponent greedy_joint --start-ps --steps 50000 --suite-games 20 \
  --parallel-trials 3 --num-envs 8 --device cuda
```

Tres trials simultáneos, cada uno en su GPU y con 8 entornos (24 en total). La
regla "se acepta si mejora el mejor" se aplica después en el orden original, así
que el resultado es idéntico al de la corrida secuencial. Los demás tuners
(preview PPO/Nash/QD-PGA y QD-PPO/PGA/PPGA) aceptan `--parallel-trials`.

### Heurísticos y evaluación (robust E2E)

```bash
python -m showdown.inference_sim.run_robust_e2e --teams 100 --seed 10 \
  --player-a expectimax_bayesian --player-b greedy_joint \
  --procs 24 --workers 1 --start-ps
```

- `--procs 24 --workers 1`: 24 procesos, cada uno con su servidor Showdown
  (puertos 8000-8023). Los 50 matchups se reparten entre ellos.
- El reporte (`logs/robust_e2e_report.txt`), el fixplan y los diagnósticos de
  greedy_joint y expectimax se combinan en uno solo, igual que en una corrida de
  un proceso.
- Tuning de expectimax: `run_expectimax_tune ... --procs 24 --workers 1`.
- Barrido MCTS: `run_mcts_sims_sweep ... --procs 24`.

### Team Preview live (PPO, Nash, QD-PGA)

Las specs de preview exigen que cada etiqueta salga de un **BO1 real** jugado
por `greedy_joint` congelado (π_c), con un máximo de 40 turnos. El backend `mock`
es solo un proxy de cobertura y no debe reportarse como live.

```bash
python -m showdown.rl.team_preview_ppo.train --seed 0 --updates 10   --backend live --num-workers 28 --start-ps --device cuda   --checkpoint-dir ~/iapsvgc_runs/preview_ppo_s0
```

- `--num-workers 28`: 28 BO1 a la vez, con dos pilotos por worker y 4 workers
  por servidor Showdown. Cada update de PPO juega 2048 BO1.
- El agente elige la formación (`/team 1234`) y Showdown la aplica. Está
  verificado que los leads jugados coinciden con el orden pedido, en ambos lados.
- Contextos: equipos reales del pool MetaVGC Reg M-C, ya validados.
- `train_meta.json` y `reporte.txt` registran `provenance: live` y las
  estadísticas de los BO1: W/L/T, topes de turnos, fallos y batallas por segundo.
- Ablación de la spec: `--pilot expectimax_bayesian`.
- Rendimiento: los lotes grandes (PPO y Nash, 2048 por update) rinden unos 10–13
  BO1/s con 16 workers. Los lotes chicos (evaluación G=20 de QD-PGA) quedan
  limitados por la batalla más lenta del lote.

### Familia QD (QD-PPO, QD-PGA, QD-PPGA)

```bash
python -m showdown.rl.qd_ppo.train --seed 0 --start-ps   --num-envs 24 --envs-per-server 4 --device cuda   --checkpoint-dir ~/iapsvgc_runs/qd_ppo_s0
```

- La mejora (Cand. 2) y la evaluación de G partidas por candidato (Cand. 1) se
  reparten entre los N entornos. El rival de cada episodio sale del mosaico
  25/25/50 y es independiente por entorno.
- Con `--num-envs 1` el comportamiento es el de siempre. En `mock` las
  estadísticas son idénticas entre 1 y N entornos (verificado).

### Validación de equipos Reg M-C

Activa por defecto. `active_teams()` valida el pool con el `TeamValidator` de
Showdown (`showdown/tools/validate_regmc_batch.js`), con caché por proceso y
equipo. Verificado: 200/200 equipos válidos; se rechazan movimientos ilegales,
Species Clause y legendarios restringidos.
`python -m showdown.inference_sim.validate_regmc_teams` genera
`logs/regmc_team_validation.txt`.

### Todo en secuencia: cola de entrenamiento

```bash
python -m showdown.inference_sim.run_train_queue scripts/server/queue.example.json --dry-run
python -m showdown.inference_sim.run_train_queue scripts/server/queue.example.json
```

- Ejecuta los pasos uno detrás de otro. Cada paso deja su log y un marcador
  `.done` en `~/iapsvgc_runs/<nombre>/`.
- Si se corta, se relanza el mismo comando y retoma en el primer paso pendiente.
  `--only paso --force` repite un paso concreto.
- Marcadores en los argumentos: `{num_envs}`, `{procs}`, `{gpus}`, `{device}` y
  `{run_dir}`.

## Estado de la paralelización por método

| Método | Paralelizado |
|---|---|
| DDQN, PPO (combate) | Sí: entornos vectorizados asíncronos (`--num-envs`) |
| QD-PPO, QD-PGA, QD-PPGA | Sí: entornos vectorizados (`--num-envs`) |
| Team Preview PPO / Nash / QD-PGA | Sí: pool de BO1 live (`--backend live --num-workers`) |
| Tuners DDQN/PPO/QD | Sí: trials paralelos + entornos por trial |
| Tuners preview | Sí: trials paralelos + BO1 live por trial |
| random, greedy_solo, greedy_joint, expectimax (E2E) | Sí: procesos (`--procs`) |
| MCTS / MCTS-Nash (E2E y barrido) | Sí: procesos |

## Rendimiento medido en el servidor

Mediciones con Showdown real (`greedy_joint` como rival), 2026-11-13. Son cortas
(~1500 pasos, incluyen ~10 s de arranque), así que sirven para comparar entre sí
y subestiman el ritmo sostenido.

| Prueba | Resultado |
|---|---|
| PPO, 8 / 16 / 24 / 28 / 32 entornos | 36 / **48** / 47 / 45 / 43 pasos/s |
| PPO, 24 entornos con 2 / 4 / 8 entornos por servidor | 47 / 47 / 45 pasos/s |
| Pool preview live, 8 / 16 / 28 workers | 6,7 / 7,7 / 4,6 BO1/s (el de 28 se midió aparte) |
| E2E robust `--procs` 10 / 20 / 28 (56 BO3, sin sync) | 457 / 309 / 205 s |
| 20 procesos, 160 BO3 con sync (4 cruces) | 145–197 s por cruce, 0 errores |
| 24 servidores Showdown a la vez | arrancan en 2,8 s |

**Cómo leerlo**

- PPO y DDQN **dejan de escalar a ~16 entornos**: coincide con los 16 núcleos
  físicos (los 32 hilos son SMT). Con 16–24 entornos se obtiene lo mismo; usar
  más no ayuda con rivales rápidos. Para rivales lentos (expectimax) los
  entornos pasan más tiempo esperando y compensa subir. Recomendación inicial:
  `--num-envs 16` a `24`; ajustar con una corrida corta del método real.
- El E2E de heurísticos sí sigue mejorando hasta 28 procesos: usa `--procs 28`.
- **El proceso que entrena no es el cuello de botella**: perfilado con cProfile,
  pasa el 85–90 % del tiempo esperando a los workers (`poll`). La GPU queda por
  debajo del 1 % (redes pequeñas); sirve para correr varios trials a la vez.
- **El cálculo de daño tampoco**: PPO contra `greedy_joint` no arranca el
  servidor de damage-calc, y forzar un worker por proceso
  (`DAMAGE_CALC_FORCE_LOCAL=1`) no cambió nada (50 contra 51 pasos/s). Los
  procesos comparten un servidor Node de damage-calc (puerto 8765); si algún día
  se mide contención con expectimax o con la inferencia por daño, ese es el
  primer sitio a revisar.
- Un BO1 de preview dura ~5 s, pero se vio una vez un lote que tardó 1170 s por
  batallas colgadas. El timeout por BO1 ahora es de 180 s (600 s con
  expectimax) y cada reintento se imprime (`[preview_live] ... reintento`).
  Un `retries` alto en `reporte.txt` indica el problema.
- **Reproducibilidad:** `--seed` fija los equipos y los emparejamientos, no el
  azar interno de Showdown (críticos, rolls, empates de velocidad). Dos corridas
  con la misma semilla no dan exactamente el mismo win rate.

## Reproducibilidad y diagnóstico de batallas

**Semilla fija.** Con `--seed N` (o `--battle-seed N`) `run_robust_e2e` arranca los
servidores con `showdown/tools/ps_seeded_start.js`, que fija la semilla de cada
batalla de Showdown a partir del contenido (los dos equipos y cuántas veces se ha
jugado ese emparejamiento), no del orden de llegada. Además se fija
`PYTHONHASHSEED` en los procesos hijos. Verificado: 40 BO3 idénticos en las 40
series con 4 procesos, con 2 procesos × 2 workers y con 1 proceso; sin semilla de
batalla (`--no-battle-seed`) difieren 38 de 40.

- Solo aplica a servidores que arranca la propia corrida. Si el puerto ya estaba
  ocupado se imprime un aviso (`--restart-ps` lo reinicia con semilla).
- Expectimax usa plazos de reloj, así que su decisión puede variar con la carga
  de la máquina; los demás bots y Showdown son deterministas con la semilla.
- Entrenamiento de RL: la política que aprende depende además del orden en que
  responden los workers; la semilla de batalla fija el azar de Showdown, no el
  planificador del sistema operativo.

**Por qué las partidas eran lentas y se colgaban.** Showdown rechazaba órdenes
inválidas y poke-env repetía la misma orden (el bot es determinista) hasta que
`random() < 1/1000` aceptaba la orden por defecto. Tres defectos lo provocaban,
ya corregidos:

1. Doble mega: los bots pedían mega para los dos huecos y Showdown solo admite una
   por combate (`enforce_single_mega`; la máscara de RL tampoco lo permite).
2. Cambio forzado de un solo hueco: el otro hueco mandaba `default` en vez de
   `pass` (también en la máscara de RL).
3. El guard de salas se tragaba el `/confirmready` de la sala padre del BO3, así
   que Showdown esperaba 40 s entre juegos. Una corrida de 2 series pasó de 81,9 s
   a 1,8 s.

Una corrida sana no debe tener `[Invalid choice]`. Para comprobarlo:

```bash
PS_PLAYER_LOG_LEVEL=25 python -m showdown.inference_sim.run_robust_e2e ... --verbose   | grep -c "Invalid choice"        # debe dar 0
```

Herramientas de diagnóstico (todas inertes si no se activan):

| Variable | Qué hace |
|---|---|
| `PS_PLAYER_LOG_LEVEL=25` | muestra los `\|error\|` que Showdown devuelve a una orden |
| `PS_DECISION_LOG=<ruta>` | una línea por decisión (turno, huella del estado, orden); para comparar corridas |
| `PS_SEED_LOG=<archivo>` | una línea por batalla con su semilla (para reproducirla) |
| `PS_HANG_LOG=<archivo>` | estado de ambos lados de un BO1 de preview colgado |
| `PS_CALIB_LOG=<archivo>` | residuos del cálculo de daño contra la verdad terreno |

## Calibración de la inferencia por daño

`showdown/inference_sim/calibration.py` registra, en cada observación de daño, la
distancia entre el rango del cálculo (con los SP verdaderos del rival) y el daño
que Showdown dibujó. `python -m showdown.inference_sim.analyze_calibration` resume
las distribuciones y comprueba tolerancias contra datos independientes.

Hallazgos (158 mil observaciones de calibración, 59 mil independientes, semillas
distintas):

- **Arreglado:** cuando faltaba una stat propia (Pokémon recién entrado) se
  sustituía por 1, y el cálculo salía 100 veces fuera de escala. Esas
  observaciones ya no se usan: los `belief_skips` bajan de ~800 a ~150 por cruce
  sin cambiar decisiones ni soundness.
- **El 90 % de los golpes coincide exactamente con el cálculo**, pero la cola es
  pesada y se reparte por casi todos los movimientos (críticos no detectados,
  objeto desconocido, golpes múltiples, potencia según HP...).
- **No existe una tolerancia estrecha segura basada en una sola observación.**
  Se probaron tolerancias por movimiento calculadas con datos: la cota superior
  descartó el SP verdadero en datos nuevos (6 de 3.624), y la inferior tampoco
  llega a 0 violaciones en ambos sentidos con ningún piso ni margen probado (hasta
  piso 10/12 y margen ×3). Por eso **la tolerancia amplia (28 puntos de HP / 32 %)
  se mantiene**: con tu criterio de podar solo cuando hay certeza, no se
  recorta.
- El código para activar una tolerancia por movimiento sigue disponible
  (`API/JSONs/damage_narrow_tolerance.json`, `DAMAGE_NARROW_TOL_FILE`,
  `analyze_calibration --emit-moves`) pero **no se distribuye ninguna tabla**.
  Con una tabla de cota inferior se vieron 31–50 estrechamientos por cada 200 BO3
  (frente a 1–2) con 0 fallos de soundness, pero a nivel de observación no pasó
  la validación, así que se dejó apagado.
- Camino para estrechar de verdad: acumular evidencia de varias observaciones
  independientes (excluir un SP solo si varias lo contradicen) en lugar de decidir
  con una sola. Requiere rediseñar la inferencia y medir su riesgo.

## Versiones de los repos hermanos

Para que los resultados sean comparables, el servidor usa los mismos commits que
la laptop:

| Repo | Commit |
|---|---|
| pokemon-showdown | `5b01be810` (compilado: `node build`) |
| poke-env | `7ff6212` |
| damage-calc | `bacbbfc` (compilado: `cd calc && rm -f .tsbuildinfo && npm run compile`) |

`damage-calc` debe estar compilado (`calc/dist/index.js`). Si no lo está, el
cálculo de daño devuelve `None` y la inferencia de stats por daño no poda nada,
sin dar error.
