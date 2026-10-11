# MCTS + PUCT Agent Spec — Planificación open-loop (VGC Gen 9)

| Campo | Valor |
|-------|--------|
| **Documento** | **Contrato exclusivo — MCTS + PUCT (estándar)** |
| **Versión** | **1.1** (alineado con `docs/rl_env_spec.md` **v1.10** §6.c) |
| **Estado** | **FROZEN — implementado en `showdown/rl/mcts_puct/` + `battle_bots/mcts_puct`** |
| **Algoritmo** | Monte Carlo Tree Search + PUCT (Rosin 2011; forma Silver et al. 2017) |
| **Rol** | Agente de **planificación pura** bajo Cand. 1 (no aprende pesos) |
| **Code root** | `showdown/rl/mcts_puct/` |
| **CLI** | `python -m showdown.rl.mcts_puct` |
| **Deps** | **Sin torch** en v1.1 (`numpy`). Heuristics intactos. |

> **Alcance.** SoT de MCTS+PUCT. Env/Cand. 1 en [`rl_env_spec.md`](rl_env_spec.md).
> Sibling simultáneo: [`mcts_puct_nash_spec.md`](mcts_puct_nash_spec.md).
>
> **Inmutabilidad.** Bump **1.2+** + `rl_env_spec` §6.c juntos.

---

## 0. Por qué existe (lectura no técnica)

En vez de entrenar una red durante horas, este agente **piensa en el momento**:
simula muchas partidas imaginarias desde la posición actual, anota qué jugadas
propias llevan a mejores resultados (+1 / −1 / 0) y al final juega la acción
más visitada en la raíz. PUCT equilibra “explotar lo que ya parece bueno” y
“explorar lo poco visitado”, guiado por un prior (aquí: uniforme sobre legales).

El rival en el árbol se modela de forma **open-loop**: se muestre su acción al azar
entre legales (no se construye un árbol minimax de dos jugadores). En las hojas,
ambos bandos juegan `greedy_joint_snapshot` hasta el final.

**Analogía:** un jugador que, con reloj, juega mentalmente 200 finales rápidos con
un compañero “greedy”, y elige la apertura que más veces ganó en esas fantasías.

| Término | Significado aquí |
|---------|------------------|
| Simulación | Una trayectoria imaginaria en S2 |
| Open-loop | El rival se samplea; no es nodo de decisión propio |
| Prior \(P\) | Sesgo de exploración PUCT (uniforme legal) |
| Rollout | Política de hoja hasta terminal |
| S2 | Simulador aproximado in-process (sin Node) |
| Cand. 1 | \(z\in\{+1,-1,0\}\) (tie = 0) |

---

## 1. Entorno heredado

| Contrato | Fuente | Uso |
|----------|--------|-----|
| `VGCEnv` | §1 | I/O partida real; búsqueda en S2 |
| Reward **Cand. 1** | §4 | Terminal \(z\); nunca Cand. 2 −0.25 |
| Máscara joint | §2 | Solo \(a\in L(s)\) |
| No `@smogon/calc` | §1 | También prohibido en pasos S2 |

**Belief:** plano / no usado (paridad con DDQN/PPO).

---

## 2. Delimitación MDP

| Ítem | Contrato |
|------|----------|
| Episodio | BO1; max turns **40**; truncate ≠ loss |
| Decisión | Joint \(a\in\{0,\ldots,575\}\) legal |
| Preview | Fuera: `/team 1234` |
| Training | **Ninguno** (pure search). Prior neural → v1.2+ |
| Player | `ValidatedMCTSPuctPlayer` |

---

## 3. Espacio de acción

Joint legal (paridad DDQN): \(a\in L(s)\subseteq\{0,\ldots,575\}\).
Sin nodos intermedios AR Slot1→Slot2 (rompería paridad MDP).

---

## 4. Búsqueda (single-agent, open-loop)

### 4.1 Estructura

- Nodos de decisión = solo el agente que busca.
- Aristas = \(a_{\mathrm{self}}\) legal.
- Transición: \(a_{\mathrm{opp}}\sim U(L_{\mathrm{opp}})\), luego S2 con True MC.
- **No** es árbol 2P alternado / minimax.

### 4.2 Selección PUCT

\[
a^\star=\arg\max_{a\in L(s)}\Big[
Q(s,a)+c_{\mathrm{puct}}\,P(s,a)\,\frac{\sqrt{N(s)}}{1+N(s,a)}
\Big]
\]

\(Q=W/N\) (0 si \(N=0\)). Dirichlet: **off** en eval.

### 4.3 Expansión

Primera visita no terminal: aristas \(\forall a\in L(s)\) con \(P(a)=1/|L(s)|\).

### 4.4 Hoja / rollout

Rollout a terminal (cap 40) con **`greedy_joint_snapshot` ambos**. Retorna Cand. 1 \(z\).

### 4.5 Backup

\(z\) se propaga **sin** invertir signo. Por arista del path:
\(N{+}{=}1\), \(W{+}{=}z\) (\(\gamma=1\)).

### 4.6 Acción real

\(a_{\mathrm{play}}=\arg\max_{a\in L(s_0)} N(s_0,a)\) (visit-greedy; **no** es la política de rollout).

---

## 5. Simulador S2

**Default:** `BattleSnapshot` + heurística de daño local (sin Node / poke-env).
**S1** (PS live): ablación de fidelidad offline.
**`greedy_joint_snapshot`:** greedy joint nativo en snapshot; **solo** rollouts.

**Riesgo tesis:** si S2 diverge de S1, MCTS optimiza una realidad alucinada → cuantificar con ablaciones.

**Estocasticidad:** True MC (rolls, accuracy). Sin nodos chance Expectimax.

---

## 6. Tres políticas (no confundir)

| Rol | Política | Dónde |
|-----|----------|-------|
| Prior \(P\) | Uniforme legal | Fórmula PUCT |
| Modelo rival (árbol) | Uniforme legal | Transiciones |
| Rollout (hoja) | `greedy_joint_snapshot` | Hasta terminal / 40 |
| Acción jugada | \(\arg\max N\) | Showdown / `VGCEnv` |

Prior neural (PPO/DDQN): **fuera de v1.1**.

---

## 7. RNG

Cada sim: RNG local de `(global_seed, root_turn, sim_id)` → misma secuencia de opp + daño.

---

## 8. Hiperparámetros congelados

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| \(N_{\mathrm{sim}}\) | `200` | Smoke `16`. Curva tesis `{50,200,800}` |
| \(c_{\mathrm{puct}}\) | `1.5` | Familia AlphaZero |
| Árbol | Single-agent open-loop | |
| Hoja | Rollout cap 40 | `greedy_joint_snapshot`; Cand. 1 |
| Sim | S2 | S1 = ablación |
| Prior / opp | Uniforme legal | |
| Dirichlet | Off | |
| Estocasticidad | True MC | |
| Seed | CLI obligatorio | |

---

## 9. Módulos y CLI

`simulator.py`, `greedy_snapshot.py`, `tree.py`, `search.py`, `agent.py`, `player.py`,
`bench.py` / `__main__.py`, tests `tests/test_rl_mcts_puct.py`.

```bat
python -m showdown.rl.mcts_puct --seed 0 --sims 16 --smoke
```

---

## 10. Suite de evaluación

200 BO1 vs `random`, `greedy_joint`, `expectimax_bayesian`; `/team 1234`; sims = default congelado.

---

## 11. Telemetría

`sims_per_decision`, `tree_nodes`, `avg_depth`, `sim_failures`, `eval_wr_<opp>`, wall time.

---

## 12. Tests de aceptación

Legales; PUCT mock; backup \(W{+}{=}z\) sin flip; Cand. 1; RNG reproducible; smoke `--sims 16`.

---

## 13. Fortalezas y límites (tesis)

| Fortaleza | Límite |
|-----------|--------|
| Sin train; interpretable en el turno | Costo \(\propto N_{\mathrm{sim}}\) en wall-clock |
| Paridad joint 576 con DDQN | Open-loop subestima rival estratégico |
| Rollout Joint ancla heurística fuerte | S2 ≠ PS exacto |

---

## 14. Non-goals

MCTS+Nash (sibling); prior/value neural v1.1; QD/BD en búsqueda; calc/Node en S2;
minimax 2P; preview en MDP; nodos AR Slot1→Slot2.

---

## 15. Referencias

Rosin (2011); Silver et al. (2017); Kocsis & Szepesvári (2006);
[`rl_env_spec.md`](rl_env_spec.md); [`mcts_puct_nash_spec.md`](mcts_puct_nash_spec.md);
[`greedy_joint_spec.md`](greedy_joint_spec.md).

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.1** | 2026-10-10 | Redacción amena (ES); semántica intacta. |
| **1.0** | 2026-10-10 | Frozen: joint 576, S2, open-loop, \(N_{\mathrm{sim}}=200\), \(c=1.5\). |

---

*Fin del contrato MCTS+PUCT v1.1.*
