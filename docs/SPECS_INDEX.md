# Índice de especificaciones (tesis IAPSVGC2022)

Guía de lectura para alguien que aprende IA y para quien implementa.
Cada contrato exclusivo vive en su propio `*_spec.md`. El entorno compartido
está en [`rl_env_spec.md`](rl_env_spec.md). La evaluación final está en
[`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

## Cómo leer (orden sugerido)

1. **Formato y reglas** — Reg M-C, Mega on, Tera off → `rl_env_spec.md` §formato / Tera.
2. **Qué ve y qué recibe el agente** — observación, acciones, recompensa → `rl_env_spec.md`.
3. **Baselines sin aprendizaje** — Random → Greedy-Solo → Greedy-Joint → Expectimax.
4. **Team Preview aprendido** — Masked PPO → Payoff Nash → QD-PGA (solo formación).
5. **Combate aprendido / planificado** — DDQN, PPO, MCTS, MCTS–Nash, QD-*.
6. **Conclusión** — 11 pipelines (preview + turnos) en un round-robin fijo.

## Mapa rápido

| Spec | Fase | ¿Entrena? | Rol en Conclusión |
|------|------|-----------|-------------------|
| [`random_spec`](random_spec.md) | Preview + turnos | No | Baseline piso |
| [`greedy_solo_spec`](greedy_solo_spec.md) | Preview + turnos | No | Baseline factored |
| [`greedy_joint_spec`](greedy_joint_spec.md) | Preview + turnos | No | Baseline joint; también \(\pi_c\) de preview |
| [`expectimax_bayesian_spec`](expectimax_bayesian_spec.md) | Preview + turnos | No | Mejor baseline heurístico |
| [`team_preview_masked_ppo_spec`](team_preview_masked_ppo_spec.md) | Solo preview | Sí | Preview de DDQN y PPO |
| [`team_preview_nash_spec`](team_preview_nash_spec.md) | Solo preview | Sí (red) + LP | Preview de MCTS / MCTS–Nash |
| [`team_preview_qd_pga_spec`](team_preview_qd_pga_spec.md) | Solo preview | Sí (archive) | Preview de QD-PPO / QD-PGA / QD-PPGA |
| [`ddqn_spec`](ddqn_spec.md) | Solo turnos (MDP) | Sí | Combate + preview PPO |
| [`ppo_spec`](ppo_spec.md) | Solo turnos (MDP) | Sí | Combate + preview PPO |
| [`mcts_puct_spec`](mcts_puct_spec.md) | Solo turnos | No (búsqueda) | Combate + preview Nash |
| [`mcts_puct_nash_spec`](mcts_puct_nash_spec.md) | Solo turnos | No (búsqueda) | Combate + preview Nash |
| [`qd_ppo_spec`](qd_ppo_spec.md) | Solo turnos | Sí | Combate + preview QD-PGA |
| [`qd_pga_map_elites_spec`](qd_pga_map_elites_spec.md) | Solo turnos | Sí | Combate + preview QD-PGA |
| [`qd_ppga_spec`](qd_ppga_spec.md) | Solo turnos | Sí | Combate + preview QD-PGA |
| [`rl_env_spec`](rl_env_spec.md) | Contrato común | — | Obs / Reward / Action / Tera |
| [`heuristic_agents_spec`](heuristic_agents_spec.md) | Umbrella | — | Escalera heurística |
| [`conclusion_eval_protocol`](conclusion_eval_protocol.md) | Eval final | Freeze | Round-robin 11 |

## Pipelines familiares (Conclusión)

| Preview | Combates |
|---------|----------|
| `team_preview_ppo` | `ddqn`, `ppo` |
| `team_preview_nash` | `mcts_puct`, `mcts_puct_nash` |
| `team_preview_qd_pga` | `qd_ppo`, `qd_pga`, `qd_ppga` |
| Preview propio §3 | `random`, `greedy_solo`, `greedy_joint`, `expectimax_bayesian` |

## Convención de tablas de parámetros

En las specs enriquecidas, los hiperparámetros usan:

| Columna | Significado |
|---------|-------------|
| **Parámetro** | Nombre / símbolo |
| **Valor** | Default del contrato |
| **Propósito** | Para qué existe en el algoritmo |
| **Cómo influye** | Qué pasa si sube / baja / se apaga |

## Entrenamiento vs inferencia (resumen)

- **Heurísticas:** no entrenan; la “calidad” es el diseño de reglas.
- **Preview RL / QD:** entrenan con \(\pi_c\) de combate **congelado** (default `greedy_joint`).
- **DDQN / PPO / QD combate:** entrenan pesos; en Conclusión se carga el último checkpoint.
- **MCTS / MCTS–Nash:** no hay gradientes; se congela el **presupuesto de búsqueda** (`n_sims`, etc.).
