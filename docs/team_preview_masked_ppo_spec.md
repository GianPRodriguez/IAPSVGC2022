# Masked PPO — Team Preview Exclusive Spec (VGC Gen 9 Regulation M-C)

| Campo | Valor |
|-------|-------|
| **Documento** | Contrato exclusivo — Masked PPO solo Team Preview |
| **Versión** | **1.6** (alineado con `docs/rl_env_spec.md` v1.20 §6.i) |
| **Estado** | **FROZEN** — implementado en `showdown/rl/team_preview_ppo/` |
| **Algoritmo** | PPO + Invalid Action Masking (IAM), acción Flat-360 |
| **Formato** | Gen 9 VGC **Regulation M-C**, OTS, Mega on, **Tera off** |
| **Código** | `showdown/rl/team_preview_ppo/` y `team_preview_{env,obs,action}.py` |
| **CLI** | `python -m showdown.rl.team_preview_ppo.train --seed N` → `final.pt` |
| **Consumidores** | DDQN / PPO combate (`frozen_preview`) |

Este agente solo elige cuatro Pokémon, con los dos leads primero y los dos de
banca después. No decide movimientos durante el combate.

## 1. Por qué existe

Team Preview condiciona toda la partida. Si se aprende junto con los turnos, no
se puede atribuir una mejora a uno u otro componente. Este módulo lo convierte
en una sola decisión: observa ambas plantillas OTS, elige una de 360 formaciones
y recibe el resultado final del BO1.
La observación contiene toda la información de la hoja abierta (§4.3).

La analogía es un director técnico que elige la alineación y entrega el partido
a un entrenador fijo. Ese entrenador es \(\pi_c\). Congelarlo hace comparables
los episodios: cambia el selector de preview, no el piloto de combate.

## 2. Glosario

| Término | Explicación |
|---------|-------------|
| Política \(\pi(a\mid s)\) | Probabilidad con que el actor elige cada formación dado el estado |
| Valor \(V(s)\) | Retorno esperado antes de elegir; referencia para juzgar la acción |
| Ventaja | En este bandit, señal derivada de \(R-V(s)\) que guía al actor |
| PPO | Mejora la política limitando cuánto puede cambiar en una actualización |
| IAM | Enmascarado que da probabilidad cero a acciones ilegales antes del muestreo |
| OTS | Open Team Sheets: información pública disponible en preview |
| Bandit contextual | MDP de horizonte \(T=1\): contexto, una acción y un resultado |
| Flat-360 | Las \(P(6,4)=360\) selecciones ordenadas de cuatro miembros distintos |
| \(\pi_c\) | Piloto `greedy_joint`, congelado al entrenar preview |
| Cand. 1 | \(+1\) victoria, \(-1\) derrota, \(0\) empate |
| Checkpoint | Pesos aprendidos y metadatos necesarios para reproducir inferencia |

## 3. Algoritmo: PPO con IAM, paso a paso

La red comparte un tronco MLP. La **policy head** produce 360 logits
(preferencias sin normalizar); la **value head** produce \(V(s)\), un escalar.

1. `env.reset()` construye la observación OTS \(s\in\mathbb{R}^{D}\) (§4.3, `OBS_DIM`) y la máscara legal \(M\).
2. La red calcula logits \(z\) y valor \(V(s)\).
3. IAM aplica
   \[
   \tilde z_i=z_i+(1-M_i)(-10^9),\qquad
   \pi(a\mid s)=\operatorname{Softmax}(\tilde z).
   \]
   Así no se desperdician episodios ni gradiente en acciones inexecutable.
4. Durante entrenamiento se muestrea \(a\); en evaluación se usa el argmax
   masked. La acción se decodifica a `/team t1t2t3t4`.
5. \(\pi_c\) juega el BO1 sin actualizar parámetros y devuelve Cand. 1.
6. El rollout guarda observación, acción, recompensa, log-probabilidad antigua,
   valor y máscara.
7. PPO calcula \(\rho=\exp(\log\pi_{\rm new}-\log\pi_{\rm old})\) y minimiza
   \[
   -\min(\rho A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A).
   \]
   El clip evita saltos destructivos; la pérdida de valor enseña \(V\), y la
   entropía conserva exploración.
8. La misma máscara almacenada se reutiliza en todas las épocas PPO.

## 4. MDP, acción y observación

### 4.1 MDP y recompensa

| Elemento | Contrato |
|----------|----------|
| Horizonte | \(T=1\), una acción de preview por episodio |
| Episodio producción | Preview + un BO1, máximo 40 turnos |
| Política de combate | `greedy_joint` congelada; ablación opcional `expectimax_bayesian` |
| Recompensa | Cand. 1, \(R\in\{-1,0,+1\}\), sin \(\Delta HP\) |
| Self-play preview / BC | Off / fuera de alcance |

### 4.2 Acción Flat-360

\[
|A|=P(6,4)=360,\qquad a\in\{0,\ldots,359\}.
\]

Gym y logits son 0-based; Showdown recibe índices 1-based. La codificación
Lehmer usa factores \((60,12,3,1)\), contando en cada posición el índice dentro
de los miembros aún disponibles. `decode_preview_action` invierte el proceso y
`action_to_team_string` suma uno. Ejemplo:
\((0,1,2,3)\leftrightarrow a=0\leftrightarrow\)`/team 1234`.

Con seis miembros todas las acciones son legales. `legal_mask(pool_size)`
mantiene el contrato para pools reducidos.

### 4.3 Observación (v2, OTS completo)

En preview con Open Team Sheets se conoce **todo** el equipo propio (especie,
item, habilidad, naturaleza, SP, movimientos) y la hoja pública del rival
(especie, item, habilidad, naturaleza, movimientos). Los SP del rival **no** son
públicos. La observación v2 codifica toda esa información
(`showdown/rl/team_preview_obs.py`, `OBS_VERSION = 2`):

\[
D = 2\cdot 6\cdot \underbrace{(193 + V_{esp} + V_{item} + V_{hab})}_{\text{bloque por Pokémon}} + 3\cdot 36 = 5676
\]

| Bloque | Dimensión | Contenido |
|--------|-----------|-----------|
| Presencia / stats exactos | 2 | 1 si el Pokémon existe; 1 si sus stats son exactos (propio) |
| Stats | 6+6+6 | Base de la forma efectiva; stat mínimo y máximo a nivel 50 (propio: exacto; rival: rango SP 0–32 con su naturaleza) |
| Tipos | 18 | Forma efectiva (mega si lleva su piedra) |
| Mega | 1+6+18 | Puede megaevolucionar; base y tipos pre-mega |
| Item | 22+18 | Categoría (Choice, Sash, Life Orb, potenciador de tipo, baya de resistencia, semilla, megapiedra…) y su tipo asociado |
| Habilidad | 22+18 | Intimidate, clima/terreno, inmunidades, redirección, Prankster, -ate, potencia, velocidad, bulk… y su tipo asociado |
| Movimientos | 18+4+28 | Mejor poder efectivo por tipo (STAB, precisión, -ate, multigolpe); mejor físico/especial/spread/prioridad; 28 utilidades (Protect, Fake Out, Tailwind, Trick Room, redirección, Helping Hand, guardias, bajadas de velocidad/ataque, sueño, Taunt, Encore, pivote, setup, recuperación, pantallas, clima/terreno, Perish Song, Knock Off, flinch, crítico…) |
| Identidad | 105, 76, 90 | One-hot de especie, item y habilidad (vocabulario congelado `showdown/rl/data/preview_vocab.json`; índice 0 = desconocido) |
| \(D_{us\to foe}\) | 36 | Daño esperado del mejor movimiento de \(i\) sobre \(j\) como fracción de HP (fórmula nivel 50, STAB, tipos, inmunidades por habilidad, items, spread, precisión), /2 |
| \(D_{foe\to us}\) | 36 | Lo mismo en sentido contrario |
| \(S\) | 36 | +1 siempre más rápido, −1 siempre más lento, intermedio si depende de SP (incluye Choice Scarf) |

Los datos salen del snapshot congelado del dex Champions de Showdown
(`API/JSONs/champions_dex_snapshot.json`, `scripts/export_champions_dex.js`,
commit PS `5b01be810`). El rival se reordena de forma canónica (por especie),
así que la observación no depende del orden en que llega su hoja; el orden
propio se conserva porque `/team ####` indexa esas posiciones. Las matrices se
calculan una vez en `reset()` (~16 ms), nunca dentro del combate. Tera está
desactivada en Reg M-C y no se codifica.

Fuentes por camino: entrenamiento (`team_factory.MonSet`; propio con SP, rival
`public_only`), poke-env (`teampreview_team` del teambuilder + `opponent_team`
tras `showteam`; los `evs` del rival se ignoran) e IAP (`side.pokemon` del
|request|).

## 5. Entrenamiento detallado

### 5.1 Loop, datos y CLI

Cada update recolecta 2048 episodios (64 en smoke). Cada fila contiene
`obs[512]`, acción, Cand. 1, `old_logp`, valor y `action_mask[360]`. Luego se
baraja y recorre cuatro veces en minibatches de 128.

En producción, `greedy_joint` juega todos los combates congelado. No recibe
gradientes, replay ni cambios de parámetros desde preview. Esto evita una diana
móvil y permite atribuir la señal al selector bajo una competencia fija.

`--backend mock` (default) usa el proxy de daño; `--backend live` juega BO1
reales con `greedy_joint` congelado (`LivePreviewPool`, equipos MetaVGC). Un
run mock no debe reportarse como live (`provenance` en el reporte).

```bat
python -m showdown.rl.team_preview_ppo.train --seed 42 --updates 10
python -m showdown.rl.team_preview_ppo.train --seed 42 --smoke
python -m showdown.rl.team_preview_ppo.train --seed 0 --updates 50 --backend live --num-workers 28 --start-ps
```

### 5.1b Evaluación y logs de entrenamiento (tesis)

Al final (y cada `--eval-every` updates) se evalúa la política **greedy** sobre
contextos **fijos** (`--eval-seed`, iguales en todos los trials) contra una
línea base en los mismos contextos (`showdown/rl/preview_eval.py`):
`fixed` = `/team 1234`, `pilot` = preview propio de \(\pi_c\) (live), `random`.
Se reporta `MEAN_WR`, `BASE_WR` y `DELTA_WR` ± SE pareado. La evaluación
restaura el RNG de contextos, así que no altera el entrenamiento. El mejor
checkpoint periódico se guarda en `best.pt`.

`train_metrics.jsonl` (una línea por evento) registra:

| `kind` | Contenido |
|--------|-----------|
| `update` | WR on-policy del lote, pg/v loss, entropía, acciones efectivas \(e^{H}\), prob. máxima media, KL aproximada, fracción clipeada, varianza explicada de \(V\), sesgo posicional de leads, fracción igual a `/team 1234`, y por rol (Fake Out, Tailwind, Trick Room, redirección, Intimidate, megas…) la tasa de uso entre contextos donde está disponible y el WR con/sin él |
| `eval` | Evaluación pareada periódica y final |

Resumen legible: `python -m showdown.rl.train_metrics <dir>/train_metrics.jsonl`.

### 5.1c Medición previa: cuánto importa el preview

Antes de entrenar, `python -m showdown.inference_sim.run_preview_value_probe`
juega los mismos contextos con el preview propio de \(\pi_c\), `/team 1234` y
uno aleatorio. La diferencia acota cuánto puede aportar un preview aprendido
con este piloto y dimensiona los episodios necesarios.

`--seed` es obligatorio; `--device` vale `cpu` por defecto.

### 5.2 Qué significa un checkpoint

Un checkpoint de inferencia contiene los pesos completos de
`PreviewActorCritic` (tronco, policy head y value head) y debe registrar versión
del spec, seed, hiperparámetros, episodios/updates y procedencia `mock` o `live`.
Para reanudar exactamente también requiere optimizador y estados RNG.

El CLI escribe `final.pt`, `best.pt` (mejor evaluación periódica),
`train_meta.json`, `reporte.txt` y `train_metrics.jsonl`.

## 6. Hiperparámetros

| Parámetro | Valor | Propósito | Cómo influye |
|-----------|-------|-----------|--------------|
| Red | `codificador(D→876)→256→256→(360,1)`, ReLU (\(D\)=`OBS_DIM`; codificador: rl_env_spec §3.c) | Representación compartida de policy/value | Más capacidad modela interacciones OTS, con riesgo de sobreajuste |
| \(\gamma\) | `1.0` | Descuento | En \(T=1\), conserva toda la recompensa |
| \(\lambda\) GAE | `1.0` | Construir ventaja | En este bandit usa el retorno terminal |
| \(\epsilon_{\rm clip}\) | `0.2` | Limitar cambios PPO | Menor valor hace updates más conservadores |
| \(c_v\) | `0.5` | Peso de pérdida de valor | Alto prioriza calibrar \(V\) |
| \(c_e\) | `0.02` | Bono de entropía | Alto mantiene exploración |
| Learning rate | `3e-4` | Paso de Adam | Alto aprende rápido, pero puede inestabilizar |
| Gradient clip | `0.5` | Acotar gradiente | Reduce saltos por batches ruidosos |
| Episodios/update | `2048`; smoke `64` | Tamaño del rollout | Más reduce ruido a mayor coste |
| Minibatch \(B\) | `128` | Tamaño de SGD | Controla ruido y memoria |
| Épocas \(K\) | `4` | Reutilizar rollout | Más señal, pero mayor deriva de la policy antigua |
| Seed | Obligatoria | Reproducibilidad | Fija inicialización y muestreo |

CLI de train acepta `--lr`, `--clip-eps`, `--checkpoint-dir`, `--eval-*` y
escribe `reporte.txt` / `train_meta.json` con `MEAN_WR` (evaluación pareada).

### Tune de hiperparámetros (independiente de DDQN / PPO combate)

Este módulo se afina **aparte**. Un solo mejor checkpoint se reutiliza en
Conclusión para **DDQN y PPO combate** (no se entrena dos veces ni “dentro” de
esos trains).

```bat
python -m showdown.inference_sim.run_team_preview_ppo_hp_tune --dry-run
python -m showdown.inference_sim.run_team_preview_ppo_hp_tune --seed 0 --max-trials 6 --smoke --updates 3
```

| Ítem | Contrato |
|------|----------|
| Knobs variables | `lr ∈ {1e-4,3e-4,1e-3}`, `clip_eps ∈ {0.10,0.20}` |
| Congelados | \(\gamma=1\), \(\lambda=1\), `ent_coef`, \(\pi_c\)=greedy_joint, … |
| Score | `MEAN_WR` greedy sobre contextos fijos (misma `--eval-seed` en todos los trials), pareado vs línea base. Antes: `-pg_loss`, que con ventajas normalizadas ronda 0 sin medir calidad |
| Criterio | Aceptar si mejora el mejor score |
| Independencia | No lanza `ddqn.train` ni `ppo.train` de combate |

**Tracking:** `logs/team_preview_ppo_hp_tune_history.jsonl`, `_progress.json`,
`_tracking.txt`, `team_preview_ppo_best_hp.json`.

**Figuras:** `logs/figures/team_preview_ppo_hp_tune_latest.svg` + ledger
histórico `hp_tune_ledger_team_preview_ppo.svg`. Regenerar:
`python -m showdown.inference_sim.run_plot_tune_history --agent team_preview_ppo`.

## 7. Inferencia y despliegue en batalla

1. Construir exactamente la observación OTS v2 desde ambas hojas.
2. Cargar una red compatible y calcular logits.
3. Aplicar la máscara legal antes de seleccionar.
4. Usar argmax masked para evaluación o muestrear la categórica masked para
   juego estocástico.
5. Decodificar y enviar `/team ####`.
6. Transferir el control al combatiente; preview no vuelve a actuar.

La suite propia usa 200 BO1 por baseline y siempre declara checkpoint, modo de
selección y \(\pi_c\).

## 8. Pipeline de Conclusión

Este preview alimenta dos familias:

- **DDQN + PPO-preview**
- **PPO + PPO-preview**

Selector y combatiente son checkpoints separados, fijados antes del torneo y
sin reentrenamiento durante él. Véase el protocolo de snapshot, pares, swap y
métricas en [`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

## 9. Limitaciones

- El aprendizaje hereda competencia y sesgos de `greedy_joint`.
- El backend mock es un proxy de daño; los resultados de tesis deben ser live.
- Cand. 1 es una señal escasa y no explica una derrota.
- Flat-360 distingue orden de banca aunque varios baselines no lo hagan.
- No incluye self-play ni BC.
- Los SP del rival se modelan como rango; el daño de las matrices es una
  aproximación (sin clima/terreno, críticos ni boosts).
- Tera está desactivada; activarla cambia observación y dinámica.

## 10. Referencias

- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*.
- Huang & Ontañón (2022), *A Closer Look at Invalid Action Masking*.
- [`rl_env_spec.md`](rl_env_spec.md) §4 y §6.i.
- [`ppo_spec.md`](ppo_spec.md), combatiente PPO separado.
- [`team_preview_nash_spec.md`](team_preview_nash_spec.md) y
  [`team_preview_qd_pga_spec.md`](team_preview_qd_pga_spec.md).

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.6** | 2026-11-15 | Codificador estructurado compartido (rl_env_spec §3.c): mismo sub-MLP `464→128→64` para los 12 Pokémon + matrices crudas → 876 → tronco 256 (1,61M → 0,45M parámetros). `RL_OBS_ENCODER=flat` restaura el MLP plano. Checkpoints anteriores incompatibles. |
| **1.5** | 2026-11-14 | Observación v2 con OTS completo (5676D: stats exactos/rango, mega, item, habilidad, movimientos, identidad y matrices daño/velocidad) desde el dex Champions congelado; rival en orden canónico. Score de checkpoint/tune = `MEAN_WR` pareado sobre contextos fijos (antes `-pg_loss`). Evaluación periódica + `best.pt`. `train_metrics.jsonl` (diagnóstico PPO y estrategia por rol). Barajado de minibatches distinto por update. Probe `run_preview_value_probe`. |
| **1.4** | 2026-10-11 | Persiste `final.pt`; consumido por DDQN/PPO vía `frozen_preview`. |
| **1.3** | 2026-10-11 | Tune `run_team_preview_ppo_hp_tune` + tracking; CLI lr/clip; independencia vs combate DDQN/PPO. |
| **1.2** | 2026-10-11 | Algoritmo, loop, checkpoint, despliegue y Conclusión; alineación documental actualizada a rl_env v1.15. |
| **1.1** | 2026-10-11 | Redacción dual inicial, glosario y límites. |
| **1.0** | 2026-10-11 | Flat-360 IAM, obs 512D, Cand. 1 y \(\pi_c\)=`greedy_joint`. |

*Fin del contrato Team Preview Masked PPO v1.5.*
