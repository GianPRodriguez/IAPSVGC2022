# QD-PPGA Agent Spec — Proximal Policy Gradient Arborescence (VGC Gen 9)

| Campo | Valor |
|---|---|
| **Versión** | **1.7** (alineada con [`rl_env_spec.md`](rl_env_spec.md) v1.20) |
| **Estado** | FROZEN; `showdown/rl/qd_ppga/` |
| **Training** | **Sí**: emits PPO-only con crítico compartido y padres por arborescencia |
| **Inferencia** | `ValidatedQDPPGAPlayer` con el mejor elite actor-only |
| **CLI** | `python -m showdown.rl.qd_ppga.train` |
| **CLI tune** | `python -m showdown.inference_sim.run_qd_ppga_hp_tune` |

## K 2.0 — PPGA fiel a la teoría (diseño del 25-11-2026)

**Decisión del usuario (25-11-2026):** implementación fiel de PPGA (Batra et al., ICLR 2024), con el
mismo arranque desde PPO v2, el mismo control y el mismo despliegue que I 2.0. K 1.7 (PPO sobre el
fitness y selección de padres hacia celdas vacías) solo tomaba el nombre: no tenía gradientes de los
descriptores, ni ramificación, ni archivo suave.

**Fuente verificada:** arXiv 2305.13795 v2 (versión HTML), Algoritmos 1–3, §3–§4 y apéndices, leída el
25-11-2026. Lo que el paper no fija se marca "no lo fija el paper" y se decide aquí.

### Algoritmo (Algoritmo 1 del paper)

Un único **punto de búsqueda** \(\theta_\mu\) y una distribución de coeficientes \(c \sim \mathcal N(\mu, \Sigma)\)
en \(\mathbb R^{k+1}\) (fitness + \(k = 2\) descriptores), mantenida con **xNES** (el paper la cambió de
CMA-ES porque su tamaño de paso divergía con objetivos ruidosos). Por iteración:

1. **Jacobiano por PPO (VPPO):** el actor se copia \(k+1\) veces, una por objetivo, cada copia con su
   propia función de valor; cada copia se entrena `N1` iteraciones de PPO con su recompensa. Los
   "gradientes" son **diferencias de parámetros**: \(\nabla f = \theta'_f - \theta_\mu\),
   \(\nabla m_j = \theta'_{m_j} - \theta_\mu\). Se normalizan.
2. \(\theta_\mu\) se evalúa y se intenta insertar.
3. **Ramificación:** \(\lambda\) coeficientes \(c_i \sim \mathcal N(\mu,\Sigma)\);
   \(\theta'_i = \theta_\mu + c_{i,0}\nabla f + \sum_j c_{i,j}\nabla m_j\); cada rama se evalúa y se inserta, y
   devuelve su mejora del archivo \(\Delta_i\).
4. **xNES** actualiza \(\mu,\Sigma\) con el ranking de los \(\Delta_i\).
5. **Paseo:** \(\theta_\mu\) se entrena `N2` iteraciones de PPO con la recompensa combinada
   \(c_{\mu,0}\,r_f + \sum_j c_{\mu,j}\,\delta_j\), con \(c_\mu = \mu\) nuevo, fija durante las `N2` iteraciones.
6. **Reinicio:** si el archivo no cambió, xNES vuelve a \(\mu = 0\), \(\Sigma = \sigma_g I\) y \(\theta_\mu\) pasa a
   una élite al azar.

**Archivo suave (CMA-MAE, Algoritmo 2):** cada celda tiene un umbral \(t\); si \(f > t\):
\(\Delta = f - t\) y \(t \leftarrow (1-\alpha)t + \alpha f\); si no, \(\Delta = 0\).

**Proxies markovianos de los descriptores (§3.1):** el descriptor se vuelve una recompensa por paso.
Aquí: \(\delta_1\) = fracción de acciones ofensivas del turno (sobre ofensivas + estado/preparación;
0,5 si no hay); \(\delta_2\) = 1 si los dos activos atacan al mismo rival con ataques de un solo objetivo,
0 si a rivales distintos, 0,5 si el turno no tiene dos de esos ataques. Su promedio por partida
reproduce BD1 y BD2 (rl_env_spec v1.34) salvo en la ponderación de los turnos.

### Hiperparámetros

| Parámetro | Paper | K 2.0 | Motivo |
|---|---|---|---|
| `N1`, `N2` | 10, 10 | **3, 3** | Presupuesto de 1M de pasos (el paper usa cientos de millones); su ablación muestra que bajar `N1/N2` reduce el rendimiento pero conserva el método ((1, 1) ≈ 50 % del de referencia) |
| Rollout por iteración de PPO | 128 pasos × E entornos (E no lo fija) | 32 pasos × 32 entornos = 1024 por copia | Mismo paralelismo que el resto de la familia |
| Épocas / minibatches de PPO | 4 / 8 | 4 / 4 (lote 256) | Los de PPO v2 |
| \(\lambda\) (ramas) | no lo fija (sugiere 300 políticas por iteración) | **8** | Cada rama cuesta 32 partidas |
| \(\sigma_g\) inicial | no lo fija | 1,0, con \(\nabla\) normalizados y luego escalados por la norma media de los tres pasos de PPO | Así \(c \approx 1\) equivale a un paso de PPO de tamaño típico |
| Normalización de \(\nabla\) | `normalize()`, sin detalle | norma euclídea unitaria sobre todos los parámetros | — |
| \(\alpha\) del archivo | 0,1 (Humanoid, Ant) | 0,1 | — |
| Umbral inicial de celda vacía | 0 | \(-1\) (mínimo de Cand. 1) | Con fitness en [−1, 1], 0 rechazaría políticas que pierden más de lo que ganan |
| xNES | Glasmachers et al. (2010), sin detalle | xNES estándar: utilidades por rango, \(\eta_\mu = 1\), \(\eta_\sigma = \eta_B = (9+3\ln d)/(5d\sqrt d)\), \(d = 3\) | Valores por defecto del método |
| Funciones de valor | una por objetivo | la cabeza de valor de cada copia del actor-crítico (la de \(r_f\) viene de PPO v2; las de \(\delta_j\) se reentrenan en cada VPPO) | — |

**Presupuesto:** 1M de pasos de entorno en total (VPPO, paseo y evaluación de ramas), ~60 iteraciones.

**Común con I 2.0:** arranque desde PPO v2, mosaico, descriptores calibrados, inserción en el archivo de
resultados por comparación pareada, reevaluación de las 10 mejores en `val`, QD-score \(\sum(f+1)\) y el
mismo control. El archivo suave solo se usa para ordenar las ramas (\(\Delta\)), como en CMA-MAE.

## Por qué existe

QD-PPGA estudia si se puede cubrir el mapa sin mutaciones genéticas. Todos los
hijos se obtienen con pasos PPO proximales, y una regla de arborescencia dirige
parte de los padres hacia celdas vacías o débiles. El resultado combina mejora
estable de política con presión explícita de cobertura.

## Glosario

| Término | Significado |
|---|---|
| Proximal-only | \(p_{\rm mut}=0\); toda variación usa PPO |
| Arborescencia | Regla para escoger padres cerca de una celda objetivo |
| \(\alpha\) | Probabilidad de padre uniforme frente a rama frontera |
| Frontera | Celda vacía; si no quedan, la de menor fitness |
| Distancia \(L_1\) | Cercanía BD entre elite y celda objetivo |
| Actor-only | El archivo excluye pesos de valor |
| Crítico compartido | \(V_\phi\) entrenado con replay común |

## Algoritmo paso a paso

1. Bootstrappea actor y crítico con Candidate 2.
2. Evalúa actores semilla y llena el archivo 10×10.
3. Con probabilidad \(\alpha=0.5\), toma un elite uniforme.
4. En otro caso elige una celda vacía al azar; si el grid está lleno, una de
   fitness mínimo. Toma el elite ocupado más cercano en distancia \(L_1\).
5. Copia el actor y recoge 2048 pasos Candidate 2.
6. Escribe replay, calcula GAE con el crítico compartido y hace cuatro épocas PPO.
7. Actualiza el crítico por TD(0), evalúa fitness/BD e intenta insertar.
8. Repite 109 iteraciones nominales y despliega el máximo fitness.

No existe rama `mutate`: `P_MUT=0.0` es una aserción contractual.

### Intuición de arborescencia

Una selección uniforme explota todo el repertorio por igual. La rama frontera
elige primero un objetivo: celda vacía, o la de menor fitness si el grid está
lleno. Después toma el elite ocupado más cercano en distancia Manhattan de
bins. Esto construye una «arborescencia» implícita desde nichos conocidos hacia
zonas que necesitan cobertura.

\(\alpha=0.5\) mezcla ambas presiones. Con probabilidad \(\alpha\) el padre es
uniforme; en caso contrario lo dirige la frontera. La distancia en BD no
garantiza cercanía en parámetros ni alcanzabilidad por PPO: es una heurística
de selección, no una restricción matemática.

### Arquitectura proximal-only

Los elites guardan solo el actor PPO: codificador estructurado y torso `1272→512→512` y cabezas AR 40/40.
El value head se excluye del archivo. Un `SharedCritic` separado aporta valores
a todos los hijos y aprende TD(0) desde replay. Cada hijo se deriva por PPO
clipped; `P_MUT=0.0` se verifica con una aserción y no existe fallback genético.

## Observación, acción y máscaras

ObsSpec v2 (6332D), Candidate 1/2, BD, acción auto-regresiva, IAM, fallbacks y Tera off
provienen de [`rl_env_spec.md`](rl_env_spec.md). La máscara se aplica al recoger
y al recalcular PPO. Candidate 2 mueve actor/crítico; Candidate 1 mide el elite;
BD solo selecciona bin y padres.

## Entrenamiento

Cada padre del archivo queda congelado y genera una copia. Solo la copia recibe
gradientes PPO. El crítico y replay se comparten entre emits; el archivo guarda
actores, no valores. L2/crash se descartan y turn-cap puede bootstrappear.

```bat
python -m showdown.rl.qd_ppga.train --seed 0 --smoke
python -m showdown.rl.qd_ppga.train --seed 0 --start-ps --suite-games 200
python -m showdown.rl.qd_ppga.train --seed 0 --alpha 0.5 --bc-checkpoint logs/ppo_bc.pt
```

Sin `--start-ps` se usa mock. Live inicia Showdown con `--no-security`.
Artefactos: `archive.pt`, `best_elite.pt`, `summary.json`, `reporte.txt`.

### Loop de datos completo

1. Bootstrap PG entrena actor/crítico durante 100k pasos.
2. Semilla y auxiliares intentan ocupar bins.
3. La moneda \(\alpha\) elige padre uniforme o de frontera.
4. Se clona el actor y se recogen 2048 pasos Candidate 2.
5. Cuatro épocas PPO mejoran el hijo; replay actualiza el crítico.
6. Veinte BO1 Candidate 1 generan fitness y BD.
7. Diez episodios elegibles habilitan inserción estricta.
8. Archivo, mejor elite y conteos de padres se guardan cada outer (109 nominal).

## Hiperparámetros

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| Grid | `10×10` | Espacio de estilos | Más resolución dificulta coverage |
| \(T_{\rm total}\) | `1_000_000` | Presupuesto | Limita steps de gradiente |
| Bootstrap | `100_000` | Sembrar actor/crítico | Compite con outer loop |
| \(T_{\rm emit}\) | `2048` | Rollout del hijo | Más estable y costoso |
| \(K_{\rm pg}\) | `4` | Épocas PPO | Mayor cambio por hijo |
| \(n_{\rm emit}\) | `4` | Hijos/outer | Define agrupación serial |
| \(N_{\rm outer}\) | `109` nominal | Iteraciones | Derivado del presupuesto |
| \(\alpha\) | `0.5` | Uniforme/frontera | Menor empuja más la cobertura |
| \(p_{\rm mut}\) | `0` | Prohibir mutación | Aísla variación proximal |
| Clip PPO | `0.2` | Limitar update | Menor conserva más al padre |
| Replay crítico | `200_000` | TD común | Mezcla estilos visitados |
| Fitness \(G/g_{\min}\) | `20/10` | Evaluar inserción | Balancea ruido y costo |
| Crítico LR/\(\gamma\)/\(c_v\) | `3e-4` / `0.95` / `0.5` | Valor compartido | Escala y horizonte TD |
| Critic updates/batch | `8` / `256` | Ajuste por emit | Más reduce error y aumenta costo |
| GAE/entropía | `0.95` / `0.01` | Ventajas/exploración | Controlan trazas y diversidad |
| Semillas auxiliares | `4` | Poblar mapa inicial | Consume parte del presupuesto |
| Mosaic | `25%/25%/50%` | random/solo/joint | Define distribución de búsqueda |
| Log pasos/tiempo | `5_000` / `60 s` | Heartbeat | Solo observabilidad |
| Max turns | `40` | Acotar episodios | Aplica truncation común |

CLI de train acepta `--lr` y `--clip-eps` (actor). Crítico compartido y
\(\alpha\) / \(p_{\rm mut}=0\) quedan fuera del grid mínimo.

### Tune de hiperparámetros (independiente del Preview QD-PGA)

```bat
python -m showdown.inference_sim.run_qd_ppga_hp_tune --dry-run
python -m showdown.inference_sim.run_qd_ppga_hp_tune --seed 0 --max-trials 6 --suite-games 4
```

| Ítem | Contrato |
|------|----------|
| Knobs variables | `lr`, `clip_eps` del actor |
| Congelados | crítico LR, \(\alpha\), \(p_{\rm mut}=0\), grid, mosaic, preview |
| Score | `MEAN_WR` de \(\pi^\*\) |
| Criterio | Aceptar si mejora el mejor score |
| Independencia | No co-afina `team_preview_qd_pga` |

**Tracking:** `logs/qd_ppga_hp_tune_{history.jsonl,progress.json,tracking.txt}` y
`qd_ppga_best_hp.json` con cambios `prev → new` por knob.
**Figuras:** `logs/figures/qd_ppga_hp_tune_latest.svg` + ledger histórico.

### Ejemplo de elección de padre

Si la celda objetivo vacía es `(8,2)` y hay elites en `(6,2)`, `(8,6)` y
`(4,1)`, sus distancias L1 son 2, 4 y 5: la rama frontera escoge `(6,2)`.
Luego PPO puede terminar en cualquier bin medido; no se fuerza al hijo a llegar
a `(8,2)`. Con la rama uniforme, esos tres elites tendrían igual probabilidad.

## Ilegales, máscaras y fallos

- IAM se conserva durante collect y las cuatro épocas PPO.
- `P_MUT=0` impide introducir ruido que pudiera eludir el experimento.
- Máscara vacía produce `pass` defensivo y telemetría.
- L1 permite fallback seguro según el contrato común.
- L2/crash no entra a rollout/replay ni cuenta como estilo.
- Turn-cap puede bootstrappear; fitness sigue Candidate 1.
- L3 aborta sin crear una inserción o fitness artificial.

## Training frente a inferencia

Training recorre el archivo y modifica copias mediante PPO, además de aprender
el crítico. Inferencia carga el actor ganador, actúa greedy masked y descarta
arborescencia, replay y crítico. Coverage es propiedad del proceso de búsqueda,
no del player desplegado.

Durante training se comparan coverage, QD-score, mejor fitness, aceptación,
conteos uniforme/frontera y suite de \(\pi^\*\). La **Conclusion eval** carga el
actor congelado y preview familiar bajo el round-robin común; arborescencia,
crítico, replay y archivo quedan fuera de inferencia.

## Team Preview y Conclusión

**No hay Team Preview en el MDP de combate.** Preview familiar congelado =
QD-PGA (`frozen_preview`, `combat_family='qd_ppga'`) → combate PPGA. No implica
que PPGA entrene el preview. Protocolo:
[`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

## Mapa de código

| Archivo | Responsabilidad |
|---|---|
| `showdown/rl/qd_ppga/train.py` | presupuesto y loop proximal |
| `showdown/rl/qd_ppga/arborescence.py` | selección uniforme/frontera |
| `showdown/rl/qd_ppga/archive.py` | elites actor-only |
| `showdown/rl/qd_ppga/improve.py` | PPO y crítico |
| `showdown/rl/qd_ppga/critic.py` | value compartido |
| `showdown/rl/qd_ppga/replay.py` | transiciones TD |
| `showdown/rl/qd_ppga/evaluate.py` | fitness y BD |
| `showdown/battle_bots/qd_ppga/pokeenv_player.py` | integración live |

## Limitaciones

- Sin mutación puede no cruzar valles entre estilos distantes.
- La arborescencia depende de que los BD representen cercanía útil.
- El crítico compartido puede sufrir interferencia entre elites.
- Seleccionar frontera mejora coverage, no necesariamente fitness global.
- Evaluación de 20 juegos mantiene ruido de inserción.
- Tera está desactivada.

## Cómo leer las métricas

`n_uniform` y `n_frontier` comprueban el efecto de \(\alpha\); con una seed
finita no tienen que ser exactamente iguales. Coverage creciente junto con
fitness plano es coherente con presión de frontera. Si el grid se llena, la
rama frontera pasa a apuntar a celdas de fitness mínimo.

QD-score resume repertorio, mientras `best_f` y `MEAN_WR_star` describen el
artefacto desplegado. Replay length informa cobertura de datos del crítico, no
diversidad de elites. Comparaciones válidas congelan seed set, mosaic,
presupuesto, \(\alpha\) y protocolo de evaluación.

El metadata de cada elite conserva el modo de padre para auditoría.

## Changelog

| Versión | Fecha | Notas |
|---|---|---|
| **2.0** (diseño) | 2026-11-25 | **K 2.0 = PPGA fiel** (decisión del usuario): VPPO con k+1 copias y gradientes como diferencias de parámetros, ramificación con xNES, archivo suave CMA-MAE (\(\alpha = 0{,}1\)), paseo por PPO con la recompensa combinada, reinicio sin mejora, proxies markovianos de BD1/BD2; arranque desde PPO v2 y regla común de I 2.0. Fuente: arXiv 2305.13795 v2. Ver §K 2.0. |
| **1.7.1** | 2026-11-25 | Archivo compartido con I 2.0: QD-score \(\sum(f+1)\) y límites/metadatos opcionales (por defecto la rejilla sigue sobre \([0,1]^2\)). El rediseño de K (implementación fiel o renombre, arranque desde PPO v2) queda pendiente de decisión del usuario; ver la revisión de la familia QD del 25-11-2026. |
| **1.7** | 2026-11-15 | Codificador estructurado (rl_env_spec §3.c) dentro del actor; consistencia por simetría (§3.d) en las actualizaciones PPO del actor y aumentación exacta en el TD del crítico compartido. Objetivo aliado del slot 0 corregido. Checkpoints/archivos anteriores incompatibles. |
| **1.6** | 2026-11-14 | ObsSpec v2 (6332D, OTS completo + estado + historial) y ActionSpec v2 (40 ids por slot: mega como flag de cada movimiento×objetivo; máscara de entrenamiento = `valid_orders`). Checkpoints anteriores incompatibles. |
| **1.5** | 2026-10-11 | Train/infer bajo Preview QD-PGA congelado (`frozen_preview`). |
| **1.4** | 2026-10-11 | CLI `--lr/--clip-eps`; tune `run_qd_ppga_hp_tune` + tracking prev→nuevo. |
| **1.3** | 2026-10-11 | Arborescencia, arquitectura, loop, fallos, evaluación, code map y ejemplo. |
| **1.2** | 2026-10-11 | Reescritura dual-audience, freeze/data loop y pairing QD-PGA-preview. |
| **1.1** | 2026-10-10 | Lectura pedagógica. |
| **1.0** | 2026-10-10 | Contrato proximal-only con arborescencia. |
