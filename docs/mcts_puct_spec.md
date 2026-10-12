# MCTS + PUCT Agent Spec — planificación open-loop (VGC Gen 9)

| Campo | Valor |
|---|---|
| **Documento** | Contrato exclusivo — MCTS+PUCT open-loop (solo turnos) |
| **Versión** | **1.5** (alineada con [`rl_env_spec.md`](rl_env_spec.md) v1.16) |
| **Estado** | FROZEN; `showdown/rl/mcts_puct/` y `battle_bots/mcts_puct/` |
| **Training** | **No**: sin gradientes ni `.pt` de combate |
| **Ajuste** | Hiperparámetros de búsqueda (sobre todo \(N_{\rm sim}\)); barrido automatizado |
| **CLI smoke** | `python -m showdown.rl.mcts_puct --seed 0 --sims 16 --smoke` |
| **CLI barrido** | `python -m showdown.inference_sim.run_mcts_sims_sweep --agent mcts_puct ...` |
| **Env presupuesto** | `MCTS_SIMS` (default config `200`) |
| **Dependencias** | NumPy; no requiere Torch |

> **Para quien aprende IA.** Este agente no “estudia” partidas previas. En cada
> turno imagina muchas continuaciones (simulaciones), anota qué acciones
> parecieron buenas y juega la más visitada. Es planificación con presupuesto,
> no aprendizaje por gradiente.

---

## 1. Por qué existe

MCTS cambia costo de entrenamiento por costo de decisión. Desde el estado
actual simula futuros con un modelo local S2, equilibra explorar acciones nuevas
y explotar las prometedoras mediante PUCT, y juega la raíz más visitada. Es un
baseline interpretable frente a DDQN/PPO/QD, que comprimen experiencia en pesos.

**Analogía:** un jugador que, con reloj, juega mentalmente muchas variantes
antes de mover; cuando se acaba el tiempo, elige la línea que más veces “sobrevivió”
en su análisis — no una red entrenada ayer.

Hermano: [`mcts_puct_nash_spec.md`](mcts_puct_nash_spec.md) (matriz simultánea).
Preview familiar en Conclusión: [`team_preview_nash_spec.md`](team_preview_nash_spec.md).

---

## 2. Glosario

| Término | Significado |
|---|---|
| Simulación | Una trayectoria imaginada dentro de S2 |
| Open-loop | El rival se samplea; no tiene nodo estratégico propio |
| Arista / acción | Un joint legal \(a=24a_1+a_2\) desde el nodo |
| \(N\) | Visitas: cuántas sims pasaron por esa arista |
| \(W\) | Suma de retornos Candidate 1 acumulados en la arista |
| \(Q=W/N\) | Valor medio estimado de la arista |
| \(P\) | Prior (aquí uniforme sobre legales) |
| PUCT | Score \(Q\) + bono de exploración con \(N\) y \(P\) |
| Rollout | Política `greedy_joint_snapshot` desde la hoja |
| Backup | Propagar el retorno \(z\) hacia la raíz |
| \(N_{\rm sim}\) | Presupuesto: sims por **decisión de turno**, no épocas de train |
| Meseta | Zona donde subir \(N_{\rm sim}\) ya casi no mejora WR |

---

## 3. Qué son \(Q\) y \(N\) (y qué no son)

### 3.1 No son parámetros de entrenamiento

\(Q\) y \(N\) **no** se “aprenden” entre partidas ni se guardan en un checkpoint.
Son **estadísticas temporales del árbol** de la decisión actual:

1. Empiezan en cero al crear el árbol del turno.
2. Cada simulación que elige una arista hace \(N \leftarrow N+1\) y \(W \leftarrow W+z\).
3. Se recalcula \(Q=W/N\).
4. Al jugar la orden real, **el árbol se descarta**. El siguiente turno vuelve a cero.

Lo que sí fijas a mano (o con el barrido) es \(N_{\rm sim}\), \(c_{\rm puct}\), etc.

### 3.2 Intuición numérica

| Símbolo | Pregunta que responde | Fiabilidad |
|---|---|---|
| \(Q\) | “¿Qué tan bien me fue *en promedio* cuando elegí esto?” | Baja si \(N\) es pequeño |
| \(N\) | “¿Cuántas veces miré esto?” | Es el contador de evidencia |

Escala de retorno aquí: Candidate 1, \(z\in\{-1,0,+1\}\) (derrota / empate-cap / victoria).
Así \(Q\in[-1,1]\) aproximadamente.

### 3.3 Durante la búsqueda vs al final

**Durante las sims** PUCT elige la arista con mayor

\[
\text{score}(a)=Q(s,a)+c_{\rm puct}\,P(s,a)\frac{\sqrt{N(s)}}{1+N(s,a)}.
\]

El segundo término (bono) es grande cuando \(N(s,a)\) es pequeño → obliga a
probar acciones poco vistas.

**Al cerrar el turno** se juega \(\arg\max_a N(s,a)\) (desempate \(Q\)), no el
máximo \(Q\) aislado. Motivo: un \(Q\) alto con \(N=2\) puede ser suerte; un
\(N\) grande resume el proceso acumulado de selección.

### 3.4 Ejemplo trabajado

Con \(Q_A=0.30,\,N_A=20\) y \(Q_B=0.10,\,N_B=2\):

1. **PUCT (mitad de la búsqueda):** B puede ganar el score temporalmente porque
   su bono de exploración es alto (\(N_B=2\)).
2. **Tras más visitas a B:** el bono cae; si B sigue mediocre, A recupera el score.
3. **Decisión final:** gana A (\(N_A=20>2\)), aunque en algún momento B haya sido
   “la más interesante” para explorar.

Para un aprendiz: \(Q\) es la nota media; \(N\) es cuántos exámenes; PUCT manda
hacer exámenes a los poco evaluados; al entregar la respuesta final, confías en
quien tiene más exámenes, no en una nota 10 con una sola prueba.

---

## 4. Algoritmo paso a paso

1. Convertir la batalla sincronizada en `BattleSnapshot`.
2. Expandir en la raíz todos los joints legales.
3. Repetir \(N_{\rm sim}\) veces:
   - Bajar con PUCT (priorizando aristas no visitadas).
   - Samplear acción rival uniforme entre sus legales (open-loop).
   - `step_sim` en S2 (accuracy, daño, azar).
   - En hoja nueva: rollout greedy joint ambos lados hasta terminal o cap.
   - Backup: \(N{+}{=}1,\,W{+}{=}z,\,Q=W/N\) en el camino (sin invertir signo).
4. Jugar \(\arg\max N\) en la raíz (desempate \(Q\)).
5. Descartar el árbol.

### Ciclo de vida del presupuesto

Cada turno real: raíz nueva → sims con RNG `seed|turn|k` → decisión → telemetría
(`sims`, `nodes`, `avg_depth`, `failures`) → árbol muerto. Una excepción de
simulación aporta \(z=0\) y se registra; no inventa una orden ilegal.

Open-loop es barato, pero mezcla la incertidumbre del rival dentro del retorno
de la arista propia (no hay matriz simultánea; eso es MCTS–Nash).

---

## 5. Observación, acción y máscaras

Contrato live: [`rl_env_spec.md`](rl_env_spec.md). La búsqueda usa snapshots, no
el vector 1024D. Aristas = joints legales; Mega puede ser legal; Tera id 22
siempre off (Reg M-C). Candidate 1 en hojas.

**No hay Team Preview en este MDP de combate** (MCTS no elige `/team` como
acción de búsqueda). El preview familiar **congelado** es
[`team_preview_nash`](team_preview_nash_spec.md) vía `frozen_preview`
(`MCTSPuctPlayer.teampreview`); fallback `/team 1234` si falta checkpoint.

---

## 6. “Mejorar” el agente sin entrenamiento

No hay `final.pt`. Mejorar = ajustar presupuesto/reglas y medir:

| Palanca | Efecto típico |
|---|---|
| Subir \(N_{\rm sim}\) | Menos ruido, más latencia; luego **meseta** |
| \(c_{\rm puct}\) | Más/menos exploración durante el árbol |
| Rollout / rival en árbol | Cambian el sesgo del valor de hoja (cambio fuerte de agente) |

La palanca principal y limpia para tesis es \(N_{\rm sim}\).

### 6.1 Meseta de \(N_{\rm sim}\)

Subir sims no mejora el WR de forma lineal. Se busca el **menor** \(N_{\rm sim}\)
tal que el siguiente escalón ya no paga:

- mejora de WR &lt; umbral (pp), **y**
- el tiempo crece al menos un factor dado.

Defaults de config: **200** sims (`16` en smoke). Env: `MCTS_SIMS`.

### 6.2 Barrido automatizado

```bat
python -m showdown.inference_sim.run_mcts_sims_sweep --agent mcts_puct ^
    --player-b greedy_joint --seed 10 --teams 40 ^
    --delta-wr-pp 1.5 --time-ratio 1.8 --metric wr_raw --start-ps
```

| Parámetro CLI | Default | Propósito | Cómo influye |
|---|---:|---|---|
| `--agent` | `both` / `mcts_puct` | Qué árbol barrer | Open-loop y Nash tienen grillas distintas |
| `--player-b` | `greedy_joint` | Rival del robust e2e | Cambia la dificultad del WR; debe fijarse al citar meseta |
| `--seed` | `10` | Repro batch + MCTS_SEED | Otra seed → otra curva (misma metodología) |
| `--teams` | `40` | Tamaño del batch robusto | Más → WR más estable, más horas |
| `--sims` | `50,100,200,400` | Rejilla de presupuestos | Define los puntos de la curva |
| `--delta-wr-pp` | `1.5` | Criterio de meseta (pp) | Más estricto → exige más ganancia para subir N |
| `--time-ratio` | `1.8` | Factor de tiempo | Más bajo → declara meseta antes |
| `--metric` | `wr_raw` | WR crudo o de pares | `pair_wr` alinea con métrica BO3 del robusto |
| `--dry-run` | off | Solo lógica/tabla fake | Valida CLI sin PS |

Por cada punto: `MCTS_SIMS=<N>` + `run_robust_e2e --player-a mcts_puct`.

#### Tracking del barrido (cómo cambia \(N_{\rm sim}\))

El hiperparámetro que **varía** es solo `n_sims` (vía `MCTS_SIMS`). El resto
(`c_puct`, prior, rollout, …) queda **congelado** y se registra en cada evento
para auditar que no se movió a escondidas.

| Artefacto | Contenido |
|-----------|-----------|
| `logs/mcts_sims_sweep_mcts_puct_history.jsonl` | Un evento por paso: `sweep_start`, `hyperparam_step` (prev→nuevo N, ΔWR, time_ratio), `plateau_decision` |
| `logs/mcts_sims_sweep_mcts_puct_progress.json` | Estado vivo (`running`/`done`, N actual, recomendación interim) |
| `logs/mcts_sims_sweep_mcts_puct_tracking.txt` | Log humano paso a paso hasta la meseta |
| `logs/mcts_sims_sweep_mcts_puct.{txt,json}` | Tabla final + recomendación + copia del tracking |
| `logs/figures/mcts_puct_sims_sweep_latest.svg` | WR vs paso del barrido (real) |
| `logs/figures/hp_tune_ledger.jsonl` | Append histórico de recomendaciones |

En consola cada paso imprime `TRACK step i/n N_sim prev->cur ...`. La
recomendación **interim** se recalcula con los puntos ya vistos (útil si el
job se corta).

Lógica pura (testeable): `showdown/inference_sim/mcts_sims_sweep.py`.  
Tests: `tests/test_mcts_sims_sweep.py`.

El valor recomendado se **congela** antes de Conclusión (igual que un checkpoint).

```bat
python -m showdown.rl.mcts_puct --seed 0 --sims 16 --smoke
```

---

## 7. Hiperparámetros de búsqueda

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| \(N_{\rm sim}\) | `200` (`16` smoke) | Presupuesto por decisión | Más reduce varianza y aumenta latencia; barrer hasta meseta |
| \(c_{\rm puct}\) | `1.5` | Exploración PUCT | Mayor visita acciones poco probadas |
| Prior \(P\) | uniforme legal | Iniciar aristas | Sin red guía |
| Rival en árbol | uniforme legal | Transición open-loop | Barato; no anticipa estrategia |
| Rollout | greedy joint ambos | Valorar hojas | Sesgo dominante del valor |
| Max turns | `40` | Acotar rollout | Caps → señal no ganadora |
| Descuento | `1` | Backup terminal | Escala Candidate 1 |
| Dirichlet | off | Ruido en raíz | Inferencia estable dado RNG |
| RNG | seed+turn+sim | Repro | Azar S2 / accuracy |
| Acción final | máx. \(N\), desempate \(Q\) | Decidir raíz | Prefiere evidencia acumulada |
| Expansión | primera no visitada | Cobertura | Prueba legales antes de reusar |

---

## 8. Ilegales, máscaras y fallos

- Joints legales desde el snapshot; Tera 22 ausente.
- Mega solo si el estado simulado la permite.
- Sin aristas → `(pass, pass)`.
- L1: fallback seguro del player live.
- L2: desync → decisión inválida en harness validated.
- Fallo interno S2 → \(z=0\) + `failures`.
- Más sims **no** corrigen sesgo sistemático de S2.
- L3/crash según `rl_env_spec`.

---

## 9. Training frente a inferencia

No hay training. Toda la cómputo útil es inferencia. Cambiar `MCTS_SIMS` cambia
calidad/latencia al instante; no produce modelo reutilizable.

Desarrollo: Candidate 1, fidelidad S2↔S1, latencia, profundidad, fallos, barrido
\(N_{\rm sim}\).  
**Conclusión:** congela presupuesto, S2, rollout y seed policy; no adapta el
árbol entre partidas. Protocolo:
[`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

---

## 10. Team Preview y Conclusión

Pipeline familiar: **`team_preview_nash` → MCTS+PUCT combate**.
No comparten árbol ni parámetros. El \(N_{\rm sim}\) congelado del barrido es
parte del artefacto de evaluación.

---

## 11. Mapa de código

| Archivo | Responsabilidad |
|---|---|
| `showdown/rl/mcts_puct/agent.py` | `MCTSPuctConfig` / API |
| `showdown/rl/mcts_puct/search.py` | select, expand, rollout, backup |
| `showdown/rl/mcts_puct/tree.py` | nodos, aristas, PUCT, \(Q,N,W\) |
| `showdown/rl/mcts_puct/simulator.py` | S2 |
| `showdown/rl/mcts_puct/greedy_snapshot.py` | rollout |
| `showdown/rl/mcts_puct/player.py` | poke-env + `MCTS_SIMS` |
| `showdown/inference_sim/mcts_sims_sweep.py` | meseta (lógica) |
| `showdown/inference_sim/run_mcts_sims_sweep.py` | CLI barrido |
| `showdown/battle_bots/mcts_puct/` | integración live |

---

## 12. Limitaciones

- Rival uniforme open-loop subestima respuestas estratégicas.
- S2 aproximado ≠ Showdown perfecto.
- Costo ~ sims × longitud de rollout.
- Prior uniforme no aporta conocimiento neural.
- Rollout greedy sesga hojas.
- Tera off por formato.

---

## 13. Lectura de telemetría

`sims` debe coincidir con el presupuesto. `nodes` / `avg_depth` describen la
forma del árbol. `failures>0` investiga S2. La acción más visitada puede tener
\(Q\) menor que otra poco probada: no es contradicción (ver §3). Conservar seed
y turno para reproducir RNG.

---

## Changelog

| Versión | Fecha | Notas |
|---|---|---|
| **1.5** | 2026-10-11 | Inferencia bajo Preview Nash congelado (`frozen_preview`). |
| **1.4** | 2026-10-11 | §Q/N pedagógico; meseta \(N_{\rm sim}\); CLI `run_mcts_sims_sweep`; rival/seed/criterio; **tracking** history/progress de cambios de N_sim. |
| **1.3** | 2026-10-11 | PUCT, lifecycle, fallos, evaluación, code map. |
| **1.2** | 2026-10-11 | Dual-audience; pairing Nash-preview. |
| **1.1** | 2026-10-10 | Lectura pedagógica. |
| **1.0** | 2026-10-10 | Contrato open-loop PUCT. |
