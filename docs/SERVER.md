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
