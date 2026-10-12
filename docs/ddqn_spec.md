# DDQN Agent Spec — Double Deep Q-Network joint (VGC Gen 9)

| Campo | Valor |
|---|---|
| **Versión** | **1.7** (alineada con [`rl_env_spec.md`](rl_env_spec.md) v1.16) |
| **Estado** | FROZEN; implementado en `showdown/rl/ddqn/` y `battle_bots/ddqn/` |
| **Training** | **Sí**: off-policy con replay; entrenamiento live requiere Showdown local |
| **Inferencia** | `ValidatedDDQNPlayer`, acción greedy legal desde un checkpoint |
| **CLI** | `python -m showdown.rl.ddqn.train` |
| **Dependencias** | `requirements-rl.txt` (`torch>=2.0`) |

Esta especificación explica el agente de combate. Observación, recompensa, acciones,
fallos y Tera off pertenecen al contrato común de
[`rl_env_spec.md`](rl_env_spec.md); no deben duplicarse con otra semántica aquí.

## Por qué existe

DDQN aprende cuánto conviene cada orden conjunta de los dos Pokémon activos. Una
red *online* elige la mejor continuación y una red *target* la valora. Separar
esas tareas reduce la sobreestimación del DQN clásico. Es útil como contraste
basado en valor frente a PPO: usa experiencias antiguas y una salida joint de
576 valores, en vez de aprender directamente una distribución de acciones.

## Glosario

| Término | Significado |
|---|---|
| \(Q(s,a)\) | Retorno futuro estimado al ejecutar \(a\) en \(s\) |
| Acción joint | Un id \(a=24a_1+a_2\), con \(a\in[0,575]\) |
| Red online | Red que actúa y recibe gradientes |
| Red target | Copia lenta que estabiliza el objetivo TD |
| Replay | Memoria de transiciones reutilizables; DDQN es off-policy |
| \(\varepsilon\)-greedy | Mejor acción legal salvo exploración uniforme legal |
| \(\Omega\) | Conjunto acumulativo de oponentes del curriculum |

## Algoritmo paso a paso

1. `VGCEnv` entrega \(s_t\) y la máscara joint legal.
2. Durante warm-up se elige uniformemente entre acciones legales; después se usa
   \(\varepsilon\)-greedy.
3. Se ejecuta \((a_1,a_2)\) y se obtiene \(r_t,s_{t+1}\).
4. Si hubo L2 (`skip_buffer`), crash o cualquier truncation, la transición se
   descarta. En otro caso se guarda con la máscara siguiente.
5. Cada cuatro pasos de entorno se muestrea un minibatch del replay.
6. La online selecciona
   \(a^\*=\arg\max_{a\in L(s')}Q_\theta(s',a)\); la target calcula
   \(y=r+\gamma Q_{\theta^-}(s',a^\*)\), o \(y=r\) si terminó.
7. Se minimiza Huber y se actualiza suavemente la target.
8. En inferencia se fija \(\varepsilon=0\) y se toma el máximo legal.

La red es `1024→512→512→576` con ReLU. Dueling, PER, n-step y Rainbow están fuera.

### Intuición matemática

El retorno \(G_t=\sum_{k\ge0}\gamma^k r_{t+k}\) resume rewards futuros. La red
aproxima \(\mathbb E[G_t\mid s_t,a_t]\), no solo el reward inmediato. DDQN
separa «qué acción elegir» de «cuánto vale»:

\[
a^\*=\arg\max_{a\in L(s')}Q_\theta(s',a),\qquad
y=r+\gamma(1-d)Q_{\theta^-}(s',a^\*).
\]

Una única red tendería a seleccionar también el ruido positivo de sus propias
estimaciones. La copia lenta reduce esa realimentación. Huber es cuadrática
cerca de cero y lineal ante errores TD grandes.

### Arquitectura y flujo tensorial

La observación `float32[1024]` atraviesa dos capas densas de 512 unidades con
ReLU. La salida contiene 576 escalares, uno por joint; no es una probabilidad
ni usa softmax. `apply_q_mask` vuelve ineligibles los Q de acciones ilegales.
Online y target son MLP idénticas, pero solo online pertenece a Adam. Después
de cada gradiente se aplica
\(\theta^-\leftarrow\tau\theta+(1-\tau)\theta^-\). El checkpoint guarda ambas
redes y contadores; el replay no se serializa.

## Observación, acción y máscaras

Se heredan Obs 1024D, Candidate 2, ActionSpec y protocolos L1/L2/L3 de
[`rl_env_spec.md`](rl_env_spec.md). El segmento belief es cero/prior plano.
`legal_joint_mask` se deriva de `valid_orders`:

\[
M(a_1,a_2)=M_1(a_1)M_2(a_2\mid a_1).
\]

Los ids 0–15 son movimiento×objetivo, 16–19 switch, 20 pass, 21 Mega,
22 Tera reservado y 23 reservado. En Reg M-C, Tera está desactivada y el id 22
siempre queda enmascarado. Ni exploración ni inferencia pueden elegir ilegales.

## Entrenamiento

El bucle real está en `ddqn/train.py`: collect → replay → minibatch TD → checkpoint.
La red target no recibe optimización directa; se mueve por soft update. El
entorno y los rivales están congelados respecto de los gradientes. El replay se
mantiene entre fases de \(\Omega\).

```bat
python -m showdown.rl.ddqn.train --seed 0 --opponent mock --steps 20000 --smoke
python -m showdown.rl.ddqn.train --seed 0 --opponent curriculum --start-ps
python -m showdown.rl.ddqn.train --seed 0 --start-ps --expand-omega --phases random:100k,greedy_solo:200k,greedy_joint:300k,expectimax_bayesian:50k
```

`--start-ps` usa el servidor local con `node pokemon-showdown start --no-security`.
Artefactos: `final.pt`, checkpoints por fase, `train_meta.json` y `reporte.txt`.
El curriculum por defecto desbloquea random → greedy_solo → greedy_joint →
expectimax_bayesian; este último puede omitirse con `--no-expectimax`.

### Ciclo de datos y curriculum

1. El actor genera una transición Candidate 2 bajo una máscara legal.
2. Replay rechaza L2, crash y truncation antes de contaminar datos.
3. `opponent_code` audita la procedencia; no entra como feature.
4. Tras warm-up, cada cuarto paso activa como máximo un gradiente.
5. `--expand-omega` conserva acumulativamente rivales desbloqueados.
6. Los checkpoints por fase permiten observar progreso y olvido.

Los `20_000` pasos del CLI son un default práctico, no la fase contractual de
`500_000`. `--smoke` valida cableado, no calidad.

## Hiperparámetros

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| \(\gamma\) | `0.95` | Descuento TD | Mayor valor mira más lejos |
| LR / optimizador | `3e-4` / Adam | Actualizar online | Alto acelera pero puede desestabilizar |
| \(\tau\) | `0.005` por grad step | Soft update target | Menor separa más ambas redes |
| Batch | `256` | Estimar gradiente | Mayor suaviza y consume memoria |
| Replay | `200_000` | Reutilizar datos | Amplía diversidad y antigüedad |
| Warm-up | `5_000` pasos | Llenar replay | Evita aprender de una muestra mínima |
| Frecuencia | 1 grad / 4 env steps | Relación aprendizaje/datos | Más updates elevan costo y correlación |
| \(\varepsilon\) | `1.0→0.05` en `500_000` | Exploración | Decae de aleatorio legal a casi greedy |
| Loss / clip | Huber \(\delta=1\), norm `10` | Robustez | Limita outliers y gradientes |
| Fase curriculum | `500_000` pasos | Desbloquear rivales | Controla exposición a dificultad |
| Max turns | `40` | Acotar episodio | Truncation se descarta en DDQN |
| Device | `cpu` | Dispositivo por defecto | GPU acelera batches, no cambia la regla |
| Seed | `0` | Reproducibilidad | Cambia pesos, replay y exploración |

CLI de train también acepta `--lr` y `--gamma` para overrides (tune / ablaciones).

### Tune de hiperparámetros (independiente del Preview PPO)

Los **pesos** se actualizan dentro de `ddqn.train`. Los **hiperparámetros** `lr`
y `gamma` se pueden barrer con un CLI aparte que **no** entrena el preview:

```bat
python -m showdown.inference_sim.run_ddqn_hp_tune --dry-run
python -m showdown.inference_sim.run_ddqn_hp_tune --seed 0 --max-trials 6 --steps 8000 --suite-games 4
python -m showdown.inference_sim.run_ddqn_hp_tune --seed 0 --max-trials 6 --opponent greedy_joint --start-ps
```

| Ítem | Contrato |
|------|----------|
| Knobs variables | `lr ∈ {1e-4,3e-4,1e-3}`, `gamma ∈ {0.90,0.95}` |
| Congelados | \(\tau\), batch, replay, ε-schedule, red, … |
| Score | `MEAN_WR` del suite offline en `reporte.txt` (Cand. 1) |
| Criterio | Aceptar trial si mejora el mejor score hasta ahora |
| Independencia | No lanza ni modifica Team Preview PPO |

**Tracking:** `logs/ddqn_hp_tune_history.jsonl`, `_progress.json`,
`_tracking.txt`, `ddqn_best_hp.json`. Cada evento registra overrides prev→nuevos
y si se aceptó.

**Figuras** (al `DONE` del tune; también regenerables):

| Artefacto | Contenido |
|-----------|-----------|
| `logs/figures/ddqn_hp_tune_latest.svg` (+ `.png` si matplotlib) | Score y best_so_far vs trial |
| `logs/figures/hp_tune_ledger.jsonl` | Append histórico de mejores scores entre runs |
| `logs/figures/hp_tune_ledger_ddqn.svg` | Curva de mejoras históricas |

```bat
python -m showdown.inference_sim.run_plot_tune_history --agent ddqn
```

Tras el tune, reentrena (o reutiliza el checkpoint del mejor trial) con esos
hiperparámetros para el artefacto de Conclusión. El preview familiar sigue
siendo un train/tune **separado**.

### Ejemplo mínimo de un objetivo

Con \(r=0.2\), episodio no terminal y
\(Q_{\theta^-}(s',a^\*)=0.6\), resulta
\(y=0.2+0.95(0.6)=0.77\). Si terminó, \(y=0.2\). Si una acción tiene mayor Q
pero es ilegal, la máscara impide que sea \(a^\*\).

## Ilegales, máscaras y niveles de fallo

- Una ilegal prevenible queda fuera de exploración, inferencia y bootstrap.
- Una máscara vacía inesperada produce `(pass, pass)` como defensa, no como
  redefinición del espacio legal.
- L1 admite la corrección segura definida por el contrato común y se registra.
- L2, crash o truncation marcan la transición para no entrenar con ella.
- L3 aborta; no se convierte artificialmente en derrota o reward.
- Una máscara siguiente desfasada sería crítica: haría bootstrap sobre una
  acción imposible. La validación de sync debe detectarlo.

## Training frente a inferencia

Training usa Candidate 2, \(\varepsilon>0\), replay, online+target y gradientes.
Inferencia carga un checkpoint, fija modo evaluación y \(\varepsilon=0\); no
escribe replay ni actualiza pesos. OTS habilita sincronización del harness, no
añade inferencia bayesiana al belief.

La evaluación durante training diagnostica aprendizaje, composición del replay
y estabilidad en suites de control; puede orientar el checkpoint. La
**Conclusion eval** ocurre después, con pesos congelados, pairing familiar,
equipos comunes y métricas de round-robin. No actualiza replay ni redes. Una
suite interna favorable no equivale a un resultado de Conclusión.

## Team Preview y Conclusión

**No hay Team Preview en el MDP de combate** (DDQN no aprende `/team`).
Seleccionar cuarteto/orden es un componente **externo congelado**:

1. Entrenar/afinar primero [`team_preview_masked_ppo_spec.md`](team_preview_masked_ppo_spec.md)
   (\(\pi_c\)=`greedy_joint`) → `logs/team_preview_ppo_checkpoints/final.pt`.
2. Entrenar DDQN **bajo** ese preview congelado (live: `PokeEnvBattleBackend`
   con `combat_family='ddqn'`; inferencia: `DDQNPlayer.teampreview`).
3. Si falta el checkpoint de preview → fallback `/team 1234` (solo smoke/dev).

Pipeline familiar: **`team_preview_ppo` → DDQN**. Código compartido:
`showdown/rl/frozen_preview.py`. Conclusión:
[`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

## Mapa de código

| Archivo | Responsabilidad |
|---|---|
| `showdown/rl/ddqn/network.py` | MLP, copia y soft update |
| `showdown/rl/ddqn/agent.py` | epsilon-greedy, DDQN, Huber, checkpoint |
| `showdown/rl/ddqn/replay.py` | filtros y minibatches |
| `showdown/rl/ddqn/curriculum.py` | fases y mezcla de \(\Omega\) |
| `showdown/rl/ddqn/train.py` | loop, CLI, reporting y artefactos |
| `showdown/rl/ddqn/player.py` | inferencia validada |
| `showdown/battle_bots/ddqn/pokeenv_player.py` | integración poke-env |
| `showdown/rl/action_space.py` | joints y máscaras |

## Limitaciones

- La salida 576 coordina slots, pero es grande y no factoriza estructura.
- Replay antiguo puede quedar desalineado con el curriculum.
- No hay self-play en este contrato ni belief oculto sofisticado.
- La calidad queda condicionada al simulador, al reward denso y al preview usado.
- No se entrena con fallbacks ni truncations; eso protege el dataset pero pierde datos.
- Tera no puede aprenderse ni ejecutarse porque está desactivada por formato.

## Changelog

| Versión | Fecha | Notas |
|---|---|---|
| **1.7** | 2026-10-12 | Figuras reales al cerrar tune (`logs/figures/`) + ledger histórico tesis. |
| **1.6** | 2026-10-11 | Train/infer bajo Preview PPO congelado (`frozen_preview`); fallback `/team 1234`. |
| **1.5** | 2026-10-11 | CLI `--lr/--gamma`; tune `run_ddqn_hp_tune` + tracking; independencia vs Preview PPO. |
| **1.4** | 2026-10-11 | Arquitectura, loop, fallos, evaluación, code map y ejemplo ampliados. |
| **1.3** | 2026-10-11 | Reescritura dual-audience; training/inferencia y pairing `team_preview_ppo` para Conclusión. |
| **1.2** | 2026-10-11 | Tera id 22 reservado/off; Mega 21 intacto. |
| **1.1** | 2026-10-10 | Glosario y lectura pedagógica. |
| **1.0** | 2026-10-10 | Contrato inicial DDQN. |
