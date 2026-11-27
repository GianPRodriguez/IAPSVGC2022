# QD-PGA-MAP-Elites Agent Spec (VGC Gen 9)

| Campo | Valor |
|---|---|
| **Versión** | **1.7** (alineada con [`rl_env_spec.md`](rl_env_spec.md) v1.20) |
| **Estado** | FROZEN; `showdown/rl/qd_pga/` |
| **Training** | **Sí**: variación híbrida mutación/PPO y crítico compartido |
| **Inferencia** | `ValidatedQDPGAPlayer` con el mejor elite actor-only |
| **CLI** | `python -m showdown.rl.qd_pga.train` |
| **CLI tune** | `python -m showdown.inference_sim.run_qd_pga_hp_tune` |

## J 2.0 — PGA-ME fiel a la teoría, adaptado a acciones discretas (diseño del 25-11-2026)

**Decisión del usuario (25-11-2026):** implementación fiel de PGA-ME (Nilsson & Cully, 2021), con el
mismo arranque desde PPO v2, el mismo control y el mismo despliegue que I 2.0. J 1.7 (V(s) compartido
por TD(0), PPO como operador, mutación isotrópica, episodios "burn") no era PGA-ME y nunca se entrenó.

**Fuentes verificadas.** El paper no se pudo descargar (bloqueo de ACM/HAL); los detalles salen de la
implementación de referencia del mismo laboratorio, QDax (`qdax/core/emitters/pga_me_emitter.py`,
`qpg_emitter.py`, `mutation_operators.py`, `losses/td3_loss.py`, `examples/pgame.ipynb`), leída el
25-11-2026.

### Algoritmo (por generación)

1. **Padres:** `b` élites uniformes del archivo.
2. **Mitad genética (Iso+LineDD, Vassiliades & Mouret, 2018):** para cada par \((\theta_1,\theta_2)\),
   \(\theta' = \theta_1 + \sigma_{\rm iso}\,\varepsilon + \sigma_{\rm line}\,\xi\,(\theta_2-\theta_1)\), con
   \(\varepsilon \sim \mathcal N(0, I)\) por peso y \(\xi \sim \mathcal N(0,1)\) por par.
3. **Mitad de gradiente:** cada padre se copia y recibe `n_pg` pasos de gradiente con Adam nuevo, sobre
   minibatches del replay, maximizando el valor del crítico Q₁ bajo la política. Más el **actor codicioso**
   (el actor del crítico), que entra como un hijo más en cada generación.
4. **Evaluación:** todos los hijos se juegan; **todas** sus transiciones van al replay del crítico.
5. **Inserción** en el archivo (regla común de I/J/K: comparación pareada con el ocupante).
6. **Crítico:** `n_critic` pasos TD3 y `n_critic/2` pasos del actor codicioso (retardo 2) por generación.

### Adaptación a acciones discretas (la única desviación de fondo)

PGA-ME usa TD3, que supone acciones continuas y un actor determinista (\(\nabla_\theta Q(s,\pi_\theta(s))\)).
Aquí la acción es discreta y auto-regresiva: \(\pi(a_1,a_2\mid s)=\pi_1(a_1\mid s)\,\pi_2(a_2\mid s,a_1)\), con
1600 acciones conjuntas y máscara de legalidad. Se usa el análogo discreto estándar:

| PGA-ME (continuo) | J 2.0 (discreto) |
|---|---|
| Crítico gemelo \(Q_{1,2}(s,a)\) | Crítico gemelo con cabeza conjunta de 1600 salidas (la de DDQN v1) |
| Objetivo de la rama de gradiente: \(\max_\theta \mathbb E_s[Q_1(s,\pi_\theta(s))]\) | **Gradiente de política esperado** (Ciosek & Whiteson, 2018): \(\max_\theta \mathbb E_s\big[\sum_{a} \pi_\theta(a\mid s)\,Q_1(s,a)\big]\) sobre las acciones legales; se calcula exacto con las 40 × 40 combinaciones |
| Objetivo del crítico: \(r + \gamma \min_k Q'_k(s', \pi'(s') + \text{ruido})\) | **SARSA esperado**: \(r + \gamma \sum_{a'} \pi'(a'\mid s') \min_k Q'_k(s',a')\) con el actor codicioso objetivo \(\pi'\); la expectativa sobre la política estocástica cumple el papel del suavizado del objetivo de TD3 |
| Redes objetivo con \(\tau = 0{,}005\) | igual |

### Hiperparámetros

| Parámetro | QDax (referencia) | J 2.0 | Motivo de la diferencia |
|---|---|---|---|
| Hijos por generación | 100 (50 GA + 49 PG + codicioso) | 16 (8 GA + 7 PG + codicioso) | Cada hijo cuesta 32 partidas; con 1M de pasos da ~140 generaciones |
| Pasos de la rama de gradiente | 100, Adam 1e-3, lote 256 | 100, Adam **1e-4**, lote 256 | La red es mucho mayor y viene entrenada (PPO v2); 1e-3 la desarma |
| Crítico | 300 pasos/generación, Adam 3e-4, lote 256, \(\tau\) 0,005 | igual | — |
| Actor codicioso | 150 pasos/generación (retardo 2), Adam 3e-4, estado persistente | igual, Adam **1e-4**, inicializado con PPO v2 | Mismo motivo que la rama de gradiente |
| \(\gamma\) | 0,99 | **0,95** | El de PPO v2 y del resto de agentes de la tesis |
| Replay | 1M transiciones | **200 000** (obs. en float16, máscaras conjuntas empaquetadas) | Memoria: cada transición guarda dos observaciones de 6332 dimensiones |
| Iso / Line | 0,005 / 0,05 | igual | — |
| Recompensa del crítico | la del entorno | Cand. 2 (densa), como PPO; el fitness sigue siendo Cand. 1 | — |
| Calentamiento | transiciones de la población inicial | las 512 partidas de calibración del arranque y 2000 pasos del crítico antes de la primera rama de gradiente | El crítico parte de cero |

**Presupuesto:** 1M de pasos de entorno en total. En PGA-ME toda la interacción es evaluación de hijos
(no hay rollouts de entrenamiento aparte), así que las partidas de evaluación sí cuentan aquí, a
diferencia de I.

**Evaluación por lotes (`showdown/rl/qd_ppo/batch_insert.py`).** Todos los candidatos de una generación (o el punto de búsqueda y sus ramas) juegan sus enfrentamientos a la vez, cada entorno con la política de su partida y sin esperar a la partida más lenta; después, los ocupantes de las celdas afectadas juegan los mismos enfrentamientos, también a la vez. Si varios candidatos caen en la misma celda ocupada entra el de mayor diferencia pareada positiva; en una celda vacía, el de mayor fitness. La prueba en vivo de J evaluando de a uno daba ~11 pasos/s con 8 entornos (más de 7 h para 1M).

**Común con I 2.0:** arranque desde PPO v2, mosaico de PPO v2, descriptores calibrados (BD2 = foco de
ataque), comparación pareada con el ocupante, reevaluación de las 10 mejores en `val` antes de
desplegar, QD-score \(\sum(f+1)\), y el mismo control (PPO v2 + 1M pasos con el mismo preview y mosaico).

**Exploración (26-11-2026):** ver la tabla común de I, J y K en qd_ppo_spec §I 2.0. J 2.0: **81,8 % contra eval** (control 81,0 %, +0,8 pts, dentro del ruido); 77,5 % contra expectimax (+5,5 frente al control); archivo de 44 celdas, QD-score 83,7, 2880 hijos en 2 h 46 min. El crítico funcionó lo bastante para no degradar al arranque, contra lo esperado (§4.10 del protocolo).

## Por qué existe

QD-PGA explora un mapa de estilos con dos herramientas complementarias. La
mutación Gaussiana puede saltar a comportamientos nuevos sin seguir el gradiente;
la rama policy-gradient mejora de forma dirigida con PPO. Para no guardar un
crítico por celda, todos los emits aprenden contra un crítico común mientras el
archivo conserva solo actores.

## Glosario

| Término | Significado |
|---|---|
| Actor-only elite | Pesos de política, sin `value.*` |
| Crítico compartido | \(V_\phi\) usado por todos los actores |
| Mutación | Ruido Gaussiano sobre parámetros float del actor |
| Burn-rollout | Interacción de rama mut para medir conducta, sin replay |
| PG emit | Rollout + GAE + PPO sobre el actor |
| Value replay | Datos de bootstrap/rama PG para TD(0) del crítico |
| Fitness/BD | Candidate 1 para calidad; descriptor solo para bin |

## Algoritmo paso a paso

1. Bootstrappea un actor y el crítico con Candidate 2.
2. Evalúa e inserta actores semilla en el grid 10×10.
3. Samplea un padre uniformemente entre celdas llenas.
4. Con probabilidad 0.5:
   - **mut:** añade ruido \(\mathcal N(0,0.02^2)\), hace burn-rollout y no escribe replay;
   - **PG:** recoge 1024 pasos, escribe replay, calcula GAE y hace dos épocas PPO.
5. La rama PG también actualiza el crítico compartido desde replay TD(0) Huber.
6. Evalúa fitness Candidate 1 y BD sobre el mosaic.
7. Inserta solo si hay al menos 10 episodios elegibles y mejora la celda.
8. Despliega el actor con mayor fitness.

### Intuición híbrida

MAP-Elites separa calidad dentro de un nicho y exploración entre nichos. La
mutación \(\theta'=\theta+\epsilon,\ \epsilon\sim\mathcal N(0,\sigma^2)\)
puede atravesar zonas donde el gradiente de reward no apunta a un BD nuevo.
La rama PG, en cambio, usa ventajas y clip PPO para elevar Candidate 2 sin
alejarse bruscamente del padre. `p_mut=0.5` estudia ambas fuentes con la misma
probabilidad, no garantiza exactamente mitad de emits en una seed finita.

El crítico compartido aproxima valor para muchas políticas. Ahorra un value
head por elite y transfiere señal, pero su replay mezcla distribuciones. Solo la
rama PG y bootstrap escriben ese replay; el burn de mutación observa el estilo
sin enseñar al crítico con una política perturbada accidentalmente.

### Arquitectura

El actor usa el codificador estructurado y el torso PPO `1272→512→512` y cabezas AR 40/40. `actor_state_dict`
excluye claves `value.*`; cada hijo recibe actor del padre y usa el
`SharedCritic` separado. Este crítico es un MLP de valor entrenado por TD(0)
Huber. `best_elite.pt` se guarda en formato de agente para inferencia greedy.

## Observación, acción y máscaras

Usa ObsSpec v2 (6332D), acción auto-regresiva e IAM de
[`rl_env_spec.md`](rl_env_spec.md). Candidate 2 alimenta bootstrap, burn y PG;
Candidate 1 define \(f\). Los BD canónicos ubican el elite y no alteran el
gradiente. Tanto actor mutado como actor PG aplican máscaras derivadas de
`valid_orders`. Mega sigue disponible; Tera id 22 está siempre apagada.

## Entrenamiento

El actor padre se congela como entrada: se modifica una copia. Los elites no
guardan crítico; \(V_\phi\) se comparte y cambia solo con replay de bootstrap y
rama PG. La rama mutación consume pasos para observar comportamiento, pero sus
transiciones no contaminan el replay.

```bat
python -m showdown.rl.qd_pga.train --seed 0 --smoke
python -m showdown.rl.qd_pga.train --seed 0 --start-ps --suite-games 200
python -m showdown.rl.qd_pga.train --seed 0 --bc-checkpoint logs/ppo_bc.pt --checkpoint-dir logs/qd_pga_archive
```

Sin `--start-ps` se usa backend mock. Live requiere Showdown local
`node pokemon-showdown start --no-security`. El resultado desplegable es
`best_elite.pt`; `archive.pt` conserva el repertorio.

### Loop de datos completo

1. Bootstrap alterna chunks de 1024 pasos PG hasta 100k.
2. Semilla y auxiliares intentan poblar el grid.
3. Se samplea uniformemente una celda ocupada.
4. Una moneda elige mutación o PG.
5. Mutación añade ruido y hace burn de 1024 pasos sin replay.
6. PG recoge 1024 pasos, hace dos épocas y ocho updates de crítico.
7. Veinte BO1 calculan fitness/BD; diez elegibles son el mínimo.
8. Inserción estricta, checkpoint y métricas cierran el outer nominal 219.

## Hiperparámetros

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| Grid | `10×10` | Archivo de estilos | Resolución frente a cobertura |
| \(T_{\rm total}\) | `1_000_000` | Presupuesto | Limita interacciones de variación |
| Bootstrap | `100_000` | Semilla/crítico | Más estabilidad, menos emits |
| \(n_{\rm emit}\) | `4` | Hijos/outer | Agrupa trabajo serial |
| \(T_{\rm emit}\) | `1024` | Datos por hijo | Menor que QD-PPO, permite más emits |
| \(N_{\rm outer}\) | `219` nominal | Iteraciones | Derivado de presupuesto |
| \(p_{\rm mut}\) | `0.5` | Balance exploración/PG | Mayor prioriza saltos ciegos |
| \(\sigma\) | `0.02` | Escala de mutación | Mayor cambia más la política |
| \(K_{\rm pg}\) | `2` | Épocas PPO | Más explota cada rollout |
| Replay crítico | `200_000` | TD compartido | Mezcla datos de múltiples actores |
| Critic updates | `8×256` | Aprender valor | Más costo y ajuste por emit |
| Fitness \(G/g_{\min}\) | `20/10` | Calidad del archivo | Controla ruido y validez de BD |
| Crítico LR/\(\gamma\)/\(c_v\) | `3e-4` / `0.95` / `0.5` | TD compartido | Escala y horizonte del valor |
| Clip/GAE/entropía | `0.2` / `0.95` / `0.01` | Mejora PPO | Conservadurismo y exploración |
| Semillas auxiliares | `4` | Diversidad inicial | Puede usar rama mut o PG |
| Mosaic | `25%/25%/50%` | Rivales de búsqueda | Pondera random/solo/joint |
| Log pasos/tiempo | `5_000` / `60 s` | Heartbeat | No altera aprendizaje |
| Max turns | `40` | Acotar episodio | Define truncation común |

CLI de train acepta `--lr` y `--clip-eps` (actor). El LR del crítico compartido
sigue fijo en `3e-4` en esta fase de tune.

### Tune de hiperparámetros (independiente del Preview QD-PGA)

```bat
python -m showdown.inference_sim.run_qd_pga_hp_tune --dry-run
python -m showdown.inference_sim.run_qd_pga_hp_tune --seed 0 --max-trials 6 --suite-games 4
```

| Ítem | Contrato |
|------|----------|
| Knobs variables | `lr`, `clip_eps` del actor (mismo grid que PPO combate) |
| Congelados | crítico LR, \(p_{\rm mut}\), \(\sigma\), grid, mosaic, preview |
| Score | `MEAN_WR` de \(\pi^\*\) |
| Criterio | Aceptar si mejora el mejor score |
| Independencia | No co-afina `team_preview_qd_pga` |

**Tracking:** `logs/qd_pga_hp_tune_{history.jsonl,progress.json,tracking.txt}` y
`qd_pga_best_hp.json` con `prev_overrides` → `overrides` / `param_change`.
**Figuras:** `logs/figures/qd_pga_hp_tune_latest.svg` + ledger histórico.

### Ejemplo de las dos ramas

Desde el mismo padre, la rama mut añade ruido `σ=0.02`; si cambia el estilo de
BD `(0.2,0.4)` a `(0.7,0.4)`, puede llenar un nicho lejano aunque su fitness sea
modesto. La rama PG probablemente queda cerca del BD original, pero puede
reemplazar su incumbent al elevar fitness. Ninguna rama inserta si no reúne diez
episodios elegibles.

## Ilegales, máscaras y fallos

- Actor mutado y actor PG conservan IAM; ruido no habilita ids ilegales.
- Máscara vacía usa `pass` como defensa y se audita.
- L1 permite fallback seguro y observable.
- L2/crash se descarta tanto del rollout como del replay.
- Turn-cap puede bootstrappear en PG; Candidate 1 rige evaluación.
- Burn mut no escribe replay incluso si la transición es válida.
- L3 aborta sin crear un elite artificial.

## Training frente a inferencia

Training necesita archivo, replay, crítico, actores hijos y dos operadores.
Inferencia carga únicamente el actor de `best_elite.pt`; no muta, no consulta el
archivo y no usa el crítico compartido para actuar.

La evaluación durante training sigue coverage, QD-score, fitness, aceptación,
conteos mut/PG y suite de \(\pi^\*\). La **Conclusion eval** carga solo el actor
congelado, lo combina con preview familiar y aplica el protocolo común; no usa
crítico, replay ni selección dinámica de elites.

## Team Preview y Conclusión

**No hay Team Preview en el MDP de combate.** Flujo: Preview QD-PGA congelado
(`frozen_preview`, `combat_family='qd_pga'`) → combate QD-PGA. Comparten
familia, no pesos/obs/archivo. Fallback `/team 1234` si falta checkpoint.
[`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

## Mapa de código

| Archivo | Responsabilidad |
|---|---|
| `showdown/rl/qd_pga/train.py` | presupuesto y mezcla de ramas |
| `showdown/rl/qd_pga/archive.py` | elites actor-only y bins |
| `showdown/rl/qd_pga/mutate.py` | ruido Gaussiano |
| `showdown/rl/qd_pga/improve.py` | burn, PPO y updates del crítico |
| `showdown/rl/qd_pga/critic.py` | valor compartido TD(0) |
| `showdown/rl/qd_pga/replay.py` | replay de valor |
| `showdown/rl/qd_pga/evaluate.py` | fitness y BD |
| `showdown/battle_bots/qd_pga/pokeenv_player.py` | integración live |

## Limitaciones

- Mutación es ciega y puede destruir una política competente.
- El crítico compartido mezcla distribuciones de muchos estilos.
- No guardar crítico por elite reduce memoria, pero pierde especialización.
- Fitness corto y mosaic fijo pueden sobreajustar el archivo.
- Desplegar solo \(\pi^\*\) no aprovecha adaptación entre elites.
- Tera está desactivada.

## Cómo leer las métricas

Los conteos `n_mut` y `n_pg` verifican la mezcla realizada; no tienen por qué
ser idénticos. Coverage atribuido a una rama requiere mirar metadata del elite,
no inferirlo solo del conteo. Replay length mide datos del crítico y excluye
deliberadamente burn-rollouts mutados.

Coverage y QD-score describen el archivo; `best_f` y `MEAN_WR_star` describen
mejor el despliegue. Una subida de coverage con caída temporal de aceptación es
posible cuando quedan nichos difíciles. Seed, mosaic y presupuesto deben
acompañar cualquier comparación.

## Changelog

| Versión | Fecha | Notas |
|---|---|---|
| **2.0.5** | 2026-11-27 | F3a (protocolo §5.12): J 2.0 desde s4 rinde 83,8 % frente a 85,8 % de control4 (−1,95 pts); **la ventaja contra expectimax de F1 no se replica** (−1,0 pts). |
| **2.0.4** | 2026-11-26 | F1 (protocolo §5.10): **J 2.0 > control contra expectimax confirmada** (78,9 frente a 75,9 %, +3,0 pts, p Holm 2,2·10⁻⁴); en el promedio −0,64 pts (no inferioridad no confirmada por poco). |
| **2.0.3** | 2026-11-26 | F3a (protocolo §4.13): réplica de J 2.0 desde PPO v2 s4. |
| **2.0.2** | 2026-11-26 | Exploración: J 2.0 81,8 % contra eval (control 81,0 %), 44 celdas, QD-score 83,7. Ver qd_ppo_spec §I 2.0. |
| **2.0.1** | 2026-11-26 | J 2.0 juega con el preview v2.1b de PPO v2 (`PREVIEW_KIND=team_preview_ppo`), igual que I 2.0 y el control: el preview QD-PGA era degenerado con el equipo fijo (qd_ppo_spec 2.0.1). Decisión del usuario. |
| **2.0** (diseño) | 2026-11-25 | **J 2.0 = PGA-ME fiel** (decisión del usuario): Iso+LineDD, rama de gradiente con crítico gemelo Q (cabeza conjunta 1600) y gradiente de política esperado, SARSA esperado con el actor codicioso objetivo, actor codicioso inyectado por generación, todas las transiciones al replay; arranque desde PPO v2 y regla común de I 2.0. Fuente: QDax. Ver §J 2.0. |
| **1.7.1** | 2026-11-25 | Archivo compartido con I 2.0: QD-score \(\sum(f+1)\) y límites/metadatos opcionales (por defecto la rejilla sigue sobre \([0,1]^2\)). El rediseño de J (implementación fiel o renombre, arranque desde PPO v2) queda pendiente de decisión del usuario; ver la revisión de la familia QD del 25-11-2026. |
| **1.7** | 2026-11-15 | Codificador estructurado (rl_env_spec §3.c) dentro del actor; consistencia por simetría (§3.d) en las actualizaciones PPO del actor y aumentación exacta en el TD del crítico compartido; la mutación gaussiana también perturba el codificador. Objetivo aliado del slot 0 corregido. Checkpoints/archivos anteriores incompatibles. |
| **1.6** | 2026-11-14 | ObsSpec v2 (6332D, OTS completo + estado + historial) y ActionSpec v2 (40 ids por slot: mega como flag de cada movimiento×objetivo; máscara de entrenamiento = `valid_orders`). Checkpoints anteriores incompatibles. |
| **1.5** | 2026-10-11 | Train/infer bajo Preview QD-PGA congelado (`frozen_preview`). |
| **1.4** | 2026-10-11 | CLI `--lr/--clip-eps`; tune `run_qd_pga_hp_tune` + tracking prev→nuevo. |
| **1.3** | 2026-10-11 | Híbrido, arquitectura, loop, tabla, fallos, evaluación, code map y ejemplo. |
| **1.2** | 2026-10-11 | Guía dual-audience, data loop/freeze y pairing de Conclusión. |
| **1.1** | 2026-10-10 | Lectura pedagógica. |
| **1.0** | 2026-10-10 | Contrato híbrido PGA-MAP-Elites. |
