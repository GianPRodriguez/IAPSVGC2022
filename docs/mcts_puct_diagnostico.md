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
marca de fin del kick-off completo de DDQN v2 (`KICKOFF_DDQN_V2_DONE`, tras las semillas 2–4; cambiable
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

### 5.1 Resultados (2026-11-20, commit e2f4c62, `~/iapsvgc_runs/mcts_g_v2/`)

Cola iniciada 05:59 tras el kick-off de DDQN. Tests 33/33; smoke 0 errores de serie y 0
anomalías del player.

| Config | Semillas × equipos | Series G/total | WR G | IC 95 % (Wilson) | Notas |
|---|---|---:|---:|---|---|
| `v2_n200`, preview greedy | 9101–9117 × 300 | 1886/5100 | **37,0 %** | [35,7, 38,3] | Por semilla 34,3–39,3 %; ~1,6 min por semilla; 0 fallbacks |
| `v2_n200_ownpreview` (pipeline G real) | 9101–9105 × 300 | 504/1500 | **33,6 %** | [31,2, 36,0] | s9104 `rc=1` por un fallo de sync en 1 matchup (`invalid=1`); el resto válido |
| `v1_n200` (referencia) | 9101 × 100 | 2 victorias / 66 decididas | ~3 % | — | **Terminada a mano** a las 08:28 (usuario): 93/100 series en 1 h 53 min, 27 errores y 4 omitidas por timeouts de 2400 s con hasta 3 reintentos; servidor ocioso esperando. Sin reporte; datos del log parcial `raw/v1_n200_s9101.txt` |

Fallbacks del adaptador: **0 en v2**. Las 488 líneas de `anomalies.tsv` vienen todas de
v1 (el archivo dejó de crecer a las 06:46, con v1 corriendo; ninguna corrida v2 tiene
avisos `mcts_puct:` en su log): v1 juega casi siempre el joint `(0,0)`, a menudo ilegal.

Lectura (exploración, no afirmación confirmatoria):

- v2 corrige el colapso de v1 (de ~3 % a 37 %), pero **G sigue claramente por debajo de
  greedy_joint**; como referencia, greedy_solo ≈ 48 % y expectimax ≈ 50–53 % contra joint.
- El preview Nash congelado le resta ~3,4 puntos frente al preview greedy, coherente con
  su HP `score 0.5` (revisar en una fase posterior).
- Con la búsqueda ya sana, la hipótesis es que el techo lo pone **S2** (sin velocidad,
  Protect, área ni stats reales): la búsqueda optimiza un modelo equivocado. El barrido
  de \(N_{\rm sim}\) (paso 6) lo pone a prueba: si más sims no ayudan, el límite es el
  modelo, no el presupuesto.

### 5.2 Barrido de \(N_{\rm sim}\) (terminado 08:50, `MCTS_G_V2_DONE`)

Preview greedy, semillas 9101–9105 × 300 (1500 BO3 por punto). Δ pareado por semilla
frente a `v2_n200` en las mismas 5 semillas (ahí `v2_n200` = 565/1500 = 37,7 %);
± = semiancho IC 95 % aprox. Tiempo = media por semilla de 300 BO3 con 28 procesos.

| \(N_{\rm sim}\) | Series G/1500 | WR G | IC 95 % | Δ vs 200 (pareado) | Tiempo/semilla |
|---:|---:|---:|---|---|---:|
| 50 | 529 | 35,3 % | [32,9, 37,7] | −2,4 ± 2,0 pts | 44 s |
| 100 | 541 | 36,1 % | [33,7, 38,5] | −1,6 ± 1,4 pts | 57 s |
| 200 | 565 | 37,7 % | — | — | 81 s |
| 400 | 559 | 37,3 % | [34,9, 39,7] | −0,4 ± 1,4 pts | 118 s |

Lectura: de 50 a 200 sims se ganan ~2 puntos y de 200 a 400 nada (Δ −0,4,
indistinguible de 0) con ~1,5× más tiempo. **Meseta en 200 sims**, que se mantiene como
presupuesto por defecto. El techo de G v2 (~37–38 % contra joint) **no lo pone el
presupuesto sino el modelo**: la búsqueda converge a lo que S2 considera mejor, y S2 se
equivoca. Esto confirma el paso 3 del plan: sustituir S2 por un modelo de turno realista
(velocidad, Protect, prioridad, área, stats reales).

Reproducir: `bash scripts/server/mcts_g_v2.sh <out>` (resumen con
`python -m showdown.inference_sim.mcts_g_summary <out>`).

## 6. Fase 3a: modelo S3 (v3.0)

Plan (usuario, 2026-11-20): reemplazar S2 por fases y medir cada una; empezar por 3a.
Implementación: `showdown/rl/mcts_puct/s3.py`, spec §4.2. Resumen:

- Tabla de daño real por decisión (un lote a la calculadora con los pedidos de P5) para
  los 4 Pokémon de cada lado, en variantes individual, área y fuego amigo.
- Stats del rival por prior meta leave-one-out. Orden por prioridad y velocidad efectiva
  (Tailwind, Trick Room). Redirección, área ×0,75 y precisión por movimiento.
- Prior, rival del árbol y rollout puntúan con la tabla.

Hallazgo de la implementación: el dex de S2 (`API/JSONs/moves.json`) no tiene **21 de
los 212 movimientos** de los equipos del meta (Make It Rain, Kowtow Cleave, Glaive Rush,
Electro Shot, Population Bomb, Aqua Cutter…). Con v2 esos ataques no hacían nada (la v2
quitó el chip de 8–18 %). S3 usa el dex de Champions de P5, que los tiene todos.

Coste medido (laptop, 200 sims): ~0,5 s por decisión (v2: 0,4 s); tabla de ~107 pedidos
en 19 ms sin caché y ~6 ms con caché; arranque de Node ~10 s una vez por proceso.

Medición: cola `scripts/server/mcts_g_v3.sh`, mismas semillas que v2 para comparar en
pareado: tests, smoke con la calculadora real (se detiene ante cualquier anomalía),
`v3_n200` 9101–9117, `v3_n200_ownpreview` 9101–9105, barrido 100/400 en 9101–9105.

Resultados: se añadirán aquí.
