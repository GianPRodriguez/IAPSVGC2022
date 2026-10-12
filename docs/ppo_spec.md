# Masked PPO Agent Spec — auto-regresivo Slot1→Slot2 (VGC Gen 9)

| Campo | Valor |
|---|---|
| **Versión** | **1.4** (alineada con [`rl_env_spec.md`](rl_env_spec.md) v1.15) |
| **Estado** | FROZEN; `showdown/rl/ppo/` y `battle_bots/ppo/` |
| **Training** | **Sí, parcial**: PPO/curriculum operativos; self-play tiene infraestructura básica, no un rival de checkpoint plenamente cableado |
| **Inferencia** | `ValidatedPPOPlayer`, greedy masked desde checkpoint |
| **CLI** | `python -m showdown.rl.ppo.train` |

Obs, reward, acciones, truncations y Tera off se heredan de
[`rl_env_spec.md`](rl_env_spec.md).

## Por qué existe

PPO aprende directamente una política. Primero decide la orden del slot 1 y
después la del slot 2 condicionada a la primera. Esta factorización evita una
cabeza plana de 576 probabilidades y representa el conflicto entre órdenes. El
clip de PPO limita cambios bruscos entre la política que recogió el rollout y la
que se está optimizando.

## Glosario

| Término | Significado |
|---|---|
| Política \(\pi\) | Distribución sobre acciones legales |
| Auto-regresiva | \(\pi(a_1,a_2|s)=\pi(a_1|s)\pi(a_2|s,a_1)\) |
| IAM | Invalid Action Masking antes de samplear y recalcular log-prob |
| Rollout | Bloque reciente de experiencia on-policy |
| GAE | Estimador de ventaja con menor varianza |
| Ratio PPO | \(\pi_\theta/\pi_{\theta_{\rm old}}\) |
| BC | Behavioral Cloning opcional para iniciar el actor |

## Algoritmo paso a paso

1. La red procesa la observación y produce logits del slot 1 y \(V(s)\).
2. IAM elimina acciones ilegales; se samplea \(a_1\).
3. La segunda cabeza recibe features más `onehot(a1)`, aplica la máscara
   condicionada y samplea \(a_2\).
4. Se guardan acción, log-prob conjunta, valor, reward y máscaras.
5. L2 y crash se descartan. Turn-cap se conserva y bootstrappea \(V(s')\).
6. Al llenar el rollout se calculan GAE y retornos, y se normalizan ventajas.
7. Durante cuatro épocas se recalculan log-probs con las mismas máscaras y se
   optimizan surrogate clipped, error de valor y entropía.
8. Se vacía el buffer: PPO no reutiliza replay antiguo.

Red: torso `1024→512→512`, cabeza slot 1 `512→24`, cabeza slot 2
`[512+onehot(24)]→24` y value `512→1`.

### Intuición de PPO y GAE

La ventaja \(A_t\) pregunta cuánto mejor fue la acción que lo esperado por
\(V(s_t)\). GAE combina residuos
\(\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)\):

\[
\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}.
\]

PPO compara la política actual con la que recogió el dato mediante
\(\rho_t=\exp(\log\pi_\theta-\log\pi_{\rm old})\), y optimiza
\(\min(\rho_tA_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)\).
El clip no prohíbe todo cambio: evita que una muestra obtenga incentivo extra
por alejar demasiado su probabilidad. El término de valor enseña el baseline y
la entropía evita colapsar prematuramente entre las opciones legales.

### Arquitectura y condicionamiento

El torso compartido comprime 1024 features a un embedding de 512. `policy1`
produce 24 logits. Tras elegir \(a_1\), su one-hot de 24 posiciones se concatena
al embedding: `policy2` puede aprender que una orden para slot 2 deja de ser
válida o conveniente por la orden del slot 1. `value` produce un escalar.

Actor y crítico comparten torso; ninguna cabeza está congelada. Los logits
ilegales reciben `-1e9` antes de construir `Categorical`. La misma máscara
guardada durante collect se reutiliza en update: recalcularla desde un estado
que ya cambió rompería el ratio PPO.

## Observación, acción y máscaras

El contrato canónico es [`rl_env_spec.md`](rl_env_spec.md): Obs `float32[1024]`,
Candidate 2 durante bootstrap y `MultiDiscrete([24,24])`. Los ids son 0–15
movimiento×objetivo, 16–19 switch, 20 pass, 21 Mega, 22 Tera reservado y 23
reservado. Tera está off y su máscara siempre vale cero.

IAM debe ser idéntica en collect y update:

\[
\operatorname{softmax}(z+(1-M)(-10^9)).
\]

La log-prob conjunta es la suma de ambos slots. El belief de combate es plano;
OTS no activa inferencia oculta adicional.

## Entrenamiento

El flujo implementado es collect → GAE → minibatches PPO → checkpoint. Si se
carga BC, se transfieren sus pesos y se ajusta toda la red: no se congela el
torso ni las cabezas. El oponente y el entorno no reciben gradientes.

```bat
python -m showdown.rl.ppo.train --seed 0 --opponent mock --steps 20000 --rollout-steps 256 --smoke
python -m showdown.rl.ppo.train --seed 0 --opponent curriculum --start-ps --expand-omega
python -m showdown.rl.ppo.train --seed 0 --opponent self --reward anneal --bc-checkpoint logs/ppo_bc.pt --start-ps
```

El último comando existe, pero el path actual de self-play conserva el pool y
checkpoints sin reemplazar plenamente al oponente live por pesos del pool; por
eso el estado se declara parcial. No usarlo como evidencia de self-play completo
sin validar ese cableado. Showdown local se inicia con `--no-security`.

### Curriculum y ciclo de rollout

1. Candidate 2 produce rewards densos mientras la política samplea.
2. Cada transición guarda ambas máscaras, log-prob vieja y \(V(s)\).
3. Al reunir 2048 pasos se calcula GAE, se normalizan ventajas y se forman
   minibatches aleatorios.
4. Cuatro épocas reutilizan únicamente ese rollout; después se vacía.
5. El curriculum puede expandir \(\Omega\) de random a heurísticos más fuertes.
6. BC opcional carga todo el actor-crítico y cambia LR a `1e-4`.

`num_envs=8` pertenece al contrato de configuración aunque el backend concreto
puede recolectar de forma serial. El default CLI de `20_000` pasos y un smoke
con rollout 256 son controles operativos, no nuevos hiperparámetros del agente.

## Hiperparámetros

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| \(\gamma\) | `0.95` | Descuento | Mayor extiende crédito temporal |
| \(\lambda\) GAE | `0.95` | Sesgo/varianza | Mayor usa trazas más largas |
| Clip PPO | `0.2` | Limitar cambio | Menor hace updates conservadores |
| LR | `3e-4` | Optimización normal | Alto aprende rápido y puede divergir |
| LR post-BC | `1e-4` | Fine-tune suave | Reduce borrado del warm-start |
| Rollout | `2048` | Datos por update | Mayor estabiliza y retrasa updates |
| Minibatch / épocas | `256` / `4` | Reuso on-policy | Más épocas extraen más, con más drift |
| \(c_v\) | `0.5` | Peso value loss | Afecta calidad del crítico |
| \(c_e\) | `0.01` | Entropía | Mayor mantiene exploración |
| Grad clip | `0.5` | Estabilidad | Limita updates extremos |
| Value clip | off | Valor sin clipping | Simplifica, puede variar más |
| Max turns | `40` | Acotar episodio | Turn-cap sí bootstrappea |
| Entornos configurados | `8` | Paralelismo nominal | Afecta diversidad/throughput del collect |
| Device | `cpu` | Ejecución por defecto | GPU acelera updates, no cambia IAM |
| Seed | `0` | Repetibilidad | Controla inicialización y sampling |

### Mini-ejemplo auto-regresivo

Supóngase que slot 1 tiene legales `{movimiento 0, switch 16}`. IAM normaliza
solo esos dos logits y samplea `16`. La máscara condicionada de slot 2 se
construye entonces sabiendo que ese switch ya ocupa su destino; si `switch 16`
conflictúa, queda a cero. La log-prob conjunta es
\(\log\pi(a_1|s)+\log\pi(a_2|s,a_1)\). En el update se usan exactamente esas
dos máscaras, aunque la política nueva prefiera otra combinación.

## Ilegales, máscaras y fallos

- Si un slot no tiene legales, la defensa selecciona `pass`; con legales,
  cualquier sample ilegal se corrige al primer id legal y debe investigarse.
- Una máscara errónea afecta acciones, entropía y ratio; no es un fallo inocuo.
- L1 usa el fallback seguro del contrato y deja telemetría.
- L2/crash usa `skip_buffer`: no entra al rollout.
- Turn-cap es truncation bootstrapeable con \(V(s')\), no terminal artificial.
- L3 aborta y no se transforma en una muestra negativa.
- En inferencia, el adaptador validado vuelve a comprobar legalidad antes de
  enviar la orden a Showdown.

## Training frente a inferencia

Training samplea, almacena rollout, calcula ventajas y cambia actor-crítico.
Inferencia carga pesos, usa greedy masked y no conserva buffer ni optimiza. La
política de combate no decide qué cuatro Pokémon entran a la partida.

La evaluación intermedia mide pérdidas, KL aproximado, fracción de clip,
entropía, explained variance y suites frente a controles. Sirve para detectar
colapso o regresión y elegir checkpoint. La **Conclusion eval** usa el checkpoint
ya congelado, greedy masked, protocolo compartido y pairing de preview; no
samplea para explorar, no calcula GAE y no actualiza con resultados del
round-robin.

## Team Preview y Conclusión

**No hay Team Preview en el MDP de combate.** El pairing familiar es
**`team_preview_ppo` → PPO de combate**. Son dos MDP y
dos checkpoints: el preview es un bandit que selecciona equipo/orden; PPO recibe
la batalla ya iniciada y elige turnos. Se congelan antes de Conclusión. Véase
[`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

## Mapa de código

| Archivo | Responsabilidad |
|---|---|
| `showdown/rl/ppo/network.py` | torso, cabezas AR, value e IAM |
| `showdown/rl/ppo/agent.py` | sampling, GAE/PPO y checkpoint |
| `showdown/rl/ppo/buffer.py` | rollout, truncations y minibatches |
| `showdown/rl/ppo/bc.py` | warm-start por imitación |
| `showdown/rl/ppo/selfplay.py` | pool experimental de self-play |
| `showdown/rl/ppo/train.py` | curriculum, CLI y reporting |
| `showdown/rl/ppo/player.py` | inferencia validada |
| `showdown/battle_bots/ppo/pokeenv_player.py` | integración live |

## Limitaciones

- On-policy: consume más partidas y no reaprovecha replay histórico.
- Una mala máscara invalida tanto muestreo como ratio PPO.
- Self-play está parcial en la implementación actual.
- Belief plano y preview externo limitan adaptación conjunta.
- Candidate 2 puede inducir preferencias distintas al win rate final.
- Tera está deshabilitada; no es una opción aprendible.

## Changelog

| Versión | Fecha | Notas |
|---|---|---|
| **1.4** | 2026-10-11 | Matemática, arquitectura, curriculum, fallos, evaluación, code map y ejemplo ampliados. |
| **1.3** | 2026-10-11 | Guía dual-audience, estado real de self-play y pairing de Conclusión. |
| **1.2** | 2026-10-11 | Tera reservado/off. |
| **1.1** | 2026-10-10 | Lectura pedagógica. |
| **1.0** | 2026-10-10 | Contrato inicial Masked PPO. |
