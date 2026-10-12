# QD-PGA-MAP-Elites Agent Spec (VGC Gen 9)

| Campo | Valor |
|---|---|
| **Versión** | **1.5** (alineada con [`rl_env_spec.md`](rl_env_spec.md) v1.16) |
| **Estado** | FROZEN; `showdown/rl/qd_pga/` |
| **Training** | **Sí**: variación híbrida mutación/PPO y crítico compartido |
| **Inferencia** | `ValidatedQDPGAPlayer` con el mejor elite actor-only |
| **CLI** | `python -m showdown.rl.qd_pga.train` |
| **CLI tune** | `python -m showdown.inference_sim.run_qd_pga_hp_tune` |

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

El actor usa el torso PPO `1024→512→512` y cabezas AR 24/24. `actor_state_dict`
excluye claves `value.*`; cada hijo recibe actor del padre y usa el
`SharedCritic` separado. Este crítico es un MLP de valor entrenado por TD(0)
Huber. `best_elite.pt` se guarda en formato de agente para inferencia greedy.

## Observación, acción y máscaras

Usa Obs 1024D, acción auto-regresiva e IAM de
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
| **1.5** | 2026-10-11 | Train/infer bajo Preview QD-PGA congelado (`frozen_preview`). |
| **1.4** | 2026-10-11 | CLI `--lr/--clip-eps`; tune `run_qd_pga_hp_tune` + tracking prev→nuevo. |
| **1.3** | 2026-10-11 | Híbrido, arquitectura, loop, tabla, fallos, evaluación, code map y ejemplo. |
| **1.2** | 2026-10-11 | Guía dual-audience, data loop/freeze y pairing de Conclusión. |
| **1.1** | 2026-10-10 | Lectura pedagógica. |
| **1.0** | 2026-10-10 | Contrato híbrido PGA-MAP-Elites. |
