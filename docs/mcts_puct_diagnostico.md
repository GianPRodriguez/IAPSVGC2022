# MCTS+PUCT (agente G) — diagnóstico y plan

Spec: [`mcts_puct_spec.md`](mcts_puct_spec.md) (v2.0). Pipeline G = `team_preview_nash` +
`mcts_puct`. Protocolo de afirmaciones: [`protocolo_evaluacion.md`](protocolo_evaluacion.md).

## 1. Estado antes de este diagnóstico (2026-11-19)

- G nunca se había medido en combate real. El único barrido de \(N_{\rm sim}\)
  (`logs/mcts_sims_sweep_mcts_puct.*`, 2026-10-12) tiene `dry_run: True` y W-L `0-0`:
  los WR 41–44 % y la "recomendación 50 sims" son datos falsos del modo de prueba.
- El preview Nash congelado (`team_preview_nash_best_hp.json`) tiene `score 0.5`; está
  pendiente comprobar que aprendió algo (fase posterior).

## 2. Hallazgos en v1 (código, verificados con una sonda)

Sonda: snapshot sintético (Garchomp/Incineroar + Amoonguss/Gholdengo contra
Kingambit/Rillaboom + Urshifu/Flutter Mane), laptop, una semilla.

| Medida | v1 |
|---|---|
| Joints en la raíz | **663** (4 objetivos por movimiento, `pass` siempre, mega duplicada) |
| 50 sims | 50 aristas con N=1, todas con *move0* en el hueco 0; profundidad 1,0; 3,6 s |
| 200 sims (default) | 200 aristas con N=1; el hueco 0 nunca prueba *move3*, mega ni cambios; **26 s/decisión** |
| 1500 sims | máx. N de una arista = 3; el árbol casi no pasa de la raíz |
| Acción elegida | `move0→rival a` + `move0→rival a` (id más bajo con \(Q=+1\)) |
| Protect en S2 | quita **12,9 %** de HP al rival |

Errores, por gravedad:

1. **Raíz inflada y explorada en orden de id.** Máscara del snapshot en vez de
   `valid_orders`; la expansión toma "la primera no visitada" (orden ascendente). Con
   \(N_{\rm sim}<663\), las acciones de id alto nunca se miran.
2. **No era open-loop.** El hijo de cada acción se creaba una vez con **una** acción
   rival y **una** tirada; las visitas siguientes reutilizaban ese estado.
3. **Rival del árbol uniforme** sobre ~660 joints (incluye `pass`, golpes al aliado).
4. **S2 irreal** (`simulator.py`): movimientos de estado hacen 8–18 % (fallback),
   objetivos aliado/self forzados al rival b, siempre actuamos primero, sin Protect,
   prioridad, área ni stats reales, y sin reemplazo tras KO.
5. **Datos del estado.** Banca rival = `reserve[:2]` (podía incluir Pokémon no traídos);
   el simulador usaba `atk.moves[mi]` y la orden real `available_moves[mi]` (desalineados
   con Choice/Encore/movimiento deshabilitado).
6. **Fallbacks silenciosos al azar** (excepción → orden por defecto; orden ilegal →
   jugada aleatoria) sin contadores.
7. **Lento**: ~130 ms/sim por `deepcopy` y rollouts greedy completos hasta el turno 40.

Correcto en v1: fórmula PUCT, backup, decisión \(\arg\max N\), RNG reproducible.

## 3. Plan acordado (usuario, 2026-11-19)

1. Este diagnóstico + spec v2.0 con los arreglos baratos: raíz desde `valid_orders` y
   poda, open-loop real, rival del árbol, hoja corta con evaluación, higiene.
2. Medir v2.0 contra `greedy_joint` validado.
3. Después, sustituir S2 por un modelo de turno realista (velocidad, Protect,
   prioridad, área, stats reales). Es el cambio grande.

## 4. v2.0 implementada (paso 1)

Ver spec §4–§7. Misma sonda:

| Medida | v1 | v2.0 |
|---|---|---|
| Joints en la raíz | 663 | **68** |
| 50 sims | 3,6 s, prof. 1,0, máx. N 1 | 0,10 s, prof. 2,2, máx. N 16 |
| 200 sims | 26 s, prof. 1,0, máx. N 1 | **0,40 s**, prof. 3,3, máx. N 113, 20 raíces visitadas |
| 800 sims | — | 1,8 s, prof. 4,5, las 68 raíces visitadas |
| Coste | ~130 ms/sim | **~2 ms/sim** |
| Elegida (200) | move0→a + move0→a | Dragon Claw→Rillaboom + Flare Blitz→Rillaboom |
| Determinista con la misma semilla | sí | sí |

`MCTS_PUCT_VERSION=1` reproduce v1. S2 no cambia por defecto (MCTS–Nash, congelado,
lo comparte); v2 activa `status_chip=False` y una copia de estado más barata.
Tests: `tests/test_mcts_puct_v2.py` (+ los de v1, MCTS–Nash y el barrido).

## 5. Medición (paso 2) — encolada

Cola `scripts/server/mcts_g_v2.sh` (usuario, 2026-11-20: "encolarlo a las ejecuciones
actuales"). Corre en el clon aislado `~/iapsvgc_scratch/spinfo/IAPSVGC2022` tras la
marca de fin del kick-off de DDQN v2 (`KICKOFF_DDQN_V2_DONE` o `PILOT_DONE`; cambiable
con `WAIT_FILE`/`WAIT_PAT`). Todo contra `greedy_joint` validado, `--procs 28`:

| Paso | Config | Semillas × equipos | Propósito |
|---|---|---|---|
| 1 | tests MCTS | — | Se detiene si fallan |
| 2 | smoke v2 | 9001 × 20 | Se detiene si hay errores de serie o excepciones/fallbacks del player (`MCTS_STATS_FILE`) |
| 3 | `v2_n200`, preview greedy | 9101–9117 × 300 (5100 BO3) | Combate aislado (como el factorial de expectimax) |
| 4 | `v2_n200_ownpreview` | 9101–9105 × 300 | Pipeline G real con preview Nash congelado |
| 5 | `v1_n200` (`MCTS_PUCT_VERSION=1`) | 9101 × 100 | Referencia de v1 |
| 6 | `v2_n50/100/400`, preview greedy | 9101–9105 × 300 | Barrido real de \(N_{\rm sim}\) (pareado con `v2_n200`) |

Resumen: `~/iapsvgc_runs/mcts_g_v2/summary.md`
(`python -m showdown.inference_sim.mcts_g_summary`), con anomalías del player en
`anomalies.tsv`. Es exploración (semillas 9101-9117): cualquier afirmación "G > joint"
necesitará un bloque fresco según el protocolo.

Resultados: se añadirán aquí.
