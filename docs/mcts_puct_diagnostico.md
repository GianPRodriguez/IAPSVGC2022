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
| `v2_n200_ownpreview` (preview "propio") | 9101–9105 × 300 | 504/1500 | **33,6 %** | [31,2, 36,0] | **En realidad orden fijo `/team 1234`** (ver nota abajo). s9104 `rc=1` por un fallo de sync en 1 matchup (`invalid=1`); el resto válido |
| `v1_n200` (referencia) | 9101 × 100 | 2 victorias / 66 decididas | ~3 % | — | **Terminada a mano** a las 08:28 (usuario): 93/100 series en 1 h 53 min, 27 errores y 4 omitidas por timeouts de 2400 s con hasta 3 reintentos; servidor ocioso esperando. Sin reporte; datos del log parcial `raw/v1_n200_s9101.txt` |

Fallbacks del adaptador: **0 en v2**. Las 488 líneas de `anomalies.tsv` vienen todas de
v1 (el archivo dejó de crecer a las 06:46, con v1 corriendo; ninguna corrida v2 tiene
avisos `mcts_puct:` en su log): v1 juega casi siempre el joint `(0,0)`, a menudo ilegal.

Lectura (exploración, no afirmación confirmatoria):

- v2 corrige el colapso de v1 (de ~3 % a 37 %), pero **G sigue claramente por debajo de
  greedy_joint**; como referencia, greedy_solo ≈ 48 % y expectimax ≈ 50–53 % contra joint.
- ~~El preview Nash congelado le resta ~3,4 puntos~~ **Corrección (20-11-2026):** no
  existe un preview Nash entrenado. El único `final.pt` del servidor es el de un smoke de
  tests (1 actualización, `provenance: mock`, en otro clon) y el clon de medición no lo
  tiene; `frozen_preview` cae entonces al orden fijo `/team 1234` (los 4 primeros de la
  hoja, leads 1–2). Los −4,0 ± 2,0 puntos son del **orden fijo**, no de un preview Nash.
  El tune de `team_preview_nash` (`score 0.5`) también era `dry_run_fake_score`, y su CLI
  de entrenamiento solo genera etiquetas *mock* (no hay pipeline BO1 live).
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

### 6.1 Resultados de exploración (2026-11-20, commit 6f85aa9, terminada 09:59)

Tests 40/40 en el servidor; smoke con la calculadora real sin anomalías; **0 anomalías**
del player en toda la cola (sin fallos de la calculadora, pedidos sin respuesta,
fallbacks ni rivales sin movimientos). Rival: greedy_joint validado. Δ pareado por semilla
frente a `v2_n200` en las semillas comunes.

| Config | Semillas × equipos | Series G/total | WR G | IC 95 % (Wilson) | Δ vs v2_n200 (pareado) | Tiempo/semilla |
|---|---|---:|---:|---|---|---:|
| `v2_n200` (referencia, §5.1) | 9101–9117 × 300 | 1886/5100 | 37,0 % | [35,7, 38,3] | — | 81 s |
| **`v3_n200`**, preview greedy | 9101–9117 × 300 | **2935/5100** | **57,5 %** | [56,2, 58,9] | **+20,6 ± 1,5 pts** (17/17 semillas suben) | 83 s |
| `v3_n100` | 9101–9105 × 300 | 853/1500 | 56,9 % | [54,3, 59,4] | +19,2 ± 2,5 | 57 s |
| `v3_n400` | 9101–9105 × 300 | 868/1500 | 57,9 % | [55,4, 60,3] | +20,2 ± 2,5 | 133 s |
| `v3_n200_ownpreview` (**orden fijo `/team 1234`**, sin preview Nash) | 9101–9105 × 300 | 801/1500 | 53,4 % | [50,9, 55,9] | +15,8 ± 2,4 | 83 s |

Lectura (exploración):

- **El modelo era el cuello de botella**: con la misma búsqueda, cambiar S2 por S3 sube
  a G de 37 % a 57,5 % contra joint, en todas las semillas.
- El presupuesto sigue en meseta: 100 → 200 → 400 sims dan 56,9 / 57,5 / 57,9 % (pasos
  dentro del ruido); se mantiene 200 por defecto.
- El orden fijo de la hoja cuesta ~4 puntos frente al preview greedy (53,4 % frente a
  57,5–57,9 % en las mismas semillas), igual que en v2. Por eso G v3.1 usa el preview
  greedy por defecto (decisión del usuario; `team_preview_nash` nunca se entrenó).
- Confirmación pre-registrada en `protocolo_evaluacion.md` §4.1 (G1–G3 + control G4,
  bloques C3/C4, solo baselines): resultados en §6.2.

### 6.2 Confirmación (bloques nuevos, 2026-11-20, código congelado `bb9ef00`)

Detalle y estadística completa en `protocolo_evaluacion.md` §5.2. Resumen (binomial
exacta, IC 95 % por *cluster bootstrap*, Holm m = 3; C3 = 9401–9417, C4 = 9501–9517):

| Hipótesis | Tasa C3+C4 | IC 95 % | Semillas > 50 % | Veredicto |
|---|---:|---|---|---|
| G1 G > greedy_joint | 56,28 % | [55,44; 57,11] | 34/34 | Superioridad práctica (por poco) |
| G2 G > expectimax v1.9 | 56,25 % | [55,42; 57,10] | 34/34 | Superioridad práctica (por poco) |
| G3 G > greedy_solo | 59,60 % | [58,81; 60,41] | 34/34 | Superioridad práctica |
| G4 G > random (control) | 99,86 % | [99,76; 99,94] | 17/17 | Correcto |

La exploración (57,5 % contra joint) se replica en bloques nuevos con ~1 punto menos
(56,3 %). G v3.1 es ahora el agente sin aprendizaje más fuerte de la tesis. Incidente: el
disco del servidor se llenó a las 11:36 por la caché de la calculadora (sin límite, clave
con el HP actual); las corridas afectadas se repitieron y desde `65f271e` la caché es por
corrida. Desde v3.1.1 S3 no escribe en la caché en disco (`persist=False`).

Siguiente: fase 3b (Protect, Fake Out, redirección, Helping Hand), medida en pareado
contra v3.1 en exploración y, si mejora, confirmada en un bloque nuevo.

### 6.3 Fase 3b (v4.0): exploración (2026-11-20, commit 20f41c7, terminada 15:10)

Cola `scripts/server/mcts_g_v4.sh`: tests 50/50 en el servidor, smoke sin anomalías, 0
anomalías en toda la cola. G v4 (`MCTS_PUCT_VERSION=4`, 200 sims, preview greedy) contra
greedy_joint validado, mismas semillas que v3 (9101–9117 × 300).

| Config | Series G/5100 | WR G | IC 95 % (Wilson) | Δ vs v3 por semilla | McNemar por serie | Tiempo/semilla |
|---|---:|---:|---|---|---|---:|
| v3_n200 (§6.1) | 2935 | 57,5 % | [56,2, 58,9] | — | — | 83 s |
| **v4_n200** | **3073** | **60,3 %** | [58,9, 61,6] | **+2,7 ± 1,4 pts** (13 suben, 2 igual, 2 bajan) | 745 frente a 606 series discordantes, **p = 0,0002** | 113 s |

Lectura (exploración): modelar Protect, Wide Guard, Fake Out, redirección y Helping Hand
mejora a G ~2,7 puntos contra joint, con significación en el diseño pareado. El costo es
~35 % más de tiempo por decisión. Pendiente: confirmación en bloques nuevos antes de
pasar v4 a ser el valor por defecto.

### 6.4 Confirmación de v4 (bloques C5 + C6, 2026-11-21)

Detalle en `protocolo_evaluacion.md` §5.3 (código congelado `0df75cb`, preview greedy,
0 anomalías, 34/34 semillas en cada hipótesis):

| Comparación | v3.1 (§6.2) | **v4** |
|---|---:|---:|
| contra greedy_joint | 56,3 % (C3+C4) | **62,6 %** [61,8; 63,4] |
| contra expectimax v1.9 | 56,3 % (C3+C4) | **61,2 %** [60,3; 62,1] |
| contra greedy_solo | 59,6 % (C3+C4) | **64,3 %** [63,5; 65,1] |
| v4 frente a v3.1 en las mismas series (McNemar) | — | **+5,09 pts**, p = 10⁻²³ (C5 +5,10, C6 +5,09) |

La fase 3b se confirma con una ganancia mayor que en exploración (+5,1 frente a +2,7 puntos)
y v4 pasa a ser el valor por defecto. Evolución contra greedy_joint: v1 ~3 % → v2 37 % →
v3.1 56,3 % → **v4 62,6 %** (confirmadas v3.1 y v4).

### 6.5 Fase 3c (v5.0): exploración (2026-11-21, commit 6ee616c, terminada 06:47)

Cola `scripts/server/mcts_g_v5.sh`: tests 64/64, smoke sin anomalías, 0 anomalías. G v5
(`MCTS_PUCT_VERSION=5`, 200 sims, preview greedy) contra greedy_joint, mismas series que v4
(9101–9117 × 300).

| Config | Series G/5100 | WR G | IC 95 % (Wilson) | Δ vs v4 por semilla | McNemar por serie | Tiempo/semilla |
|---|---:|---:|---|---|---|---:|
| v4_n200 (§6.3) | 3073 | 60,3 % | [58,9, 61,6] | — | — | 113 s |
| **v5_n200** | **3276** | **64,2 %** | [62,9, 65,5] | **+4,0 ± 1,2 pts** (16 suben, 1 igual, 0 bajan) | 737 frente a 533, **p = 1·10⁻⁸** | 140 s |

Lectura (exploración): velocidad dinámica, Tailwind / Trick Room, estados, boosts en el
árbol, Intimidate, clima y pivotes suman ~4 puntos sobre v4, con el costo de ~25 % más de
tiempo. Pendiente: comparar con v6 (terreno) y confirmar la mejor en bloques nuevos.

### 6.6 Fase 3d (v6.0, terreno): exploración (2026-11-21, commit d9abb31, terminada 07:33)

G v6 contra greedy_joint, mismas series que v5 (9101–9117 × 300). 0 anomalías.

| Config | Series G/5100 | WR G | Δ vs v5 por semilla | McNemar por serie |
|---|---:|---:|---|---|
| v5_n200 (§6.5) | 3276 | 64,2 % | — | — |
| v6_n200 | 3289 | 64,5 % | +0,2 ± 1,1 pts | 350 frente a 340, p = 0,73 |

Por subgrupos (`scripts/analysis/terrain_subgroup.py`; "con terreno" = algún equipo con
Grassy / Psychic / Electric / Misty Surge o Hadron Engine; 225 de los 300 equipos lo llevan):

| Subgrupo | Series emparejadas | Gana solo v6 | Gana solo v5 | Diferencia | p |
|---|---:|---:|---:|---:|---|
| Con terreno | 3865 | 281 | 271 | +0,26 pts | 0,70 |
| Sin terreno | 130 | 0 | 0 | 0 | — |
| Ambiguo (nombre truncado en el log) | 1075 | 69 | 69 | 0 | 1,0 |

Lectura: sin creadores de terreno v6 juega **exactamente igual** que v5 (control de
coherencia). Con terreno, modelarlo cambia el resultado de ~14 % de las series, pero en las
dos direcciones por igual: **sin efecto neto**. La tabla de daño ya incluía el terreno de la
raíz; lo que el modelo de 3d gana en algunas situaciones (Grassy Glide, bloqueo de prioridad de
Psychic Terrain) se compensa con lo que pierde en otras (p. ej. la duración supuesta del
terreno sin dato de la raíz). Como referencia, v5 frente a v4 sí mejora en todos los
subgrupos (con terreno +3,6 pts, p = 7·10⁻⁶).

**Decisión:** se confirma **v5** (fase 3c), más simple y equivalente; v6 queda como
"probada, sin efecto neto" (`MCTS_PUCT_VERSION=6`).

### 6.7 Confirmación de v5 (bloques C7 + C8, 2026-11-21)

Detalle en `protocolo_evaluacion.md` §5.4 (código congelado `cce1356`, preview greedy,
0 anomalías, 34/34 semillas en cada hipótesis):

| Comparación | v4 (§6.4) | **v5** |
|---|---:|---:|
| contra greedy_joint | 62,6 % (C5+C6) | **65,5 %** [64,7; 66,3] |
| contra expectimax v1.9 | 61,2 % (C5+C6) | **63,6 %** [62,7; 64,4] |
| contra greedy_solo | 64,3 % (C5+C6) | **66,8 %** [66,0; 67,6] |
| v5 frente a v4 en las mismas series (McNemar) | — | **+4,48 pts**, p = 6·10⁻¹⁹ (C7 +3,46, C8 +5,50) |
| contra random (control) | 99,8 % | 99,8 % |

La fase 3c se confirma con una ganancia similar a la de exploración (+4,5 frente a +4,0
puntos) y v5 pasa a ser el valor por defecto. Evolución contra greedy_joint: v1 ~3 % → v2 37 %
→ v3.1 56,3 % → v4 62,6 % → **v5 65,5 %** (confirmadas v3.1, v4 y v5). Siguiente paso
acordado: preview por búsqueda (spec §4.6) sobre v5, en pareado contra el preview greedy y con
regla fijada de antemano.

## 7. Preview Nash para G (entrenamiento y medición, fijado el 20-11-2026 antes de medir)

Decisión del usuario: entrenar por fin `team_preview_nash` (nunca se había entrenado; G
usa el preview greedy desde v3.1) y medir si mejora al preview greedy. Solo baselines.

**Entrenamiento:** `team_preview_nash.train --backend live`, piloto greedy_joint congelado
(como pide su spec), 200 actualizaciones × 2048 BO1, equipos `fit` (`PS_TEAM_SPLIT=fit`,
como el preview v2.1), semillas 0 y 1. ~50 min por semilla.

**Medición (exploración):** combate G v3.1 (el confirmado, `MCTS_PUCT_VERSION=3`) contra
greedy_joint validado, con equipos **`eval`** (no vistos por el preview:
`--team-split eval`, 100 equipos, 100 BO3 por semilla), semillas 9101–9134 (3400 BO3 por
configuración). Configuraciones en pareado (mismas series):

| Id | Preview |
|---|---|
| P0 | greedy (tipado + BST, el actual) |
| P1 | Nash, semilla 0 |
| P2 | Nash, semilla 1 |

**Regla de decisión:** McNemar exacta P1 frente a P0 y P2 frente a P0, Holm con m = 2. Se
adopta el preview Nash solo si alguna semilla mejora a P0 con p Holm < 0,05 (si las dos,
la de mayor diferencia); entonces se confirma en un bloque nuevo con el commit congelado.
Si no, G sigue con el preview greedy.

**Secundaria (memorización, descriptiva):** P1 y P2 con los 300 equipos (`all`, incluye
los de entrenamiento), semillas 9101–9117, en pareado con `v3_n200` (preview greedy, §6.1).

### 7.1 Resultados (2026-11-20, commit 9761b03, `~/iapsvgc_runs/mcts_g_nash/`)

Entrenamiento live (etiquetas de BO1 reales con greedy_joint): semilla 0 en 36 min (loss
BCE final 0,619), semilla 1 en 36 min (0,648); la caché compartida de la calculadora
creció ~95 GB durante el entrenamiento (bajo el límite de 250 GB). Smoke con preview Nash
sin anomalías (el checkpoint cargó; no cayó al orden fijo). 0 anomalías en la medición.

G v3.1 contra greedy_joint validado, equipos `eval` (no vistos), 9101–9134:

| Id | Preview | Victorias / N | Tasa | IC 95 % (cluster) | Semillas > 50 % |
|---|---|---|---|---|---|
| P0 | greedy (tipado + BST) | 2000 / 3394 | **58,93 %** | [57,51; 60,40] | 33/34 |
| P1 | Nash, semilla 0 | 1783 / 3395 | 52,52 % | [50,91; 54,11] | 21/34 |
| P2 | Nash, semilla 1 | 1753 / 3397 | 51,60 % | [49,97; 53,16] | 18/34 |

| Comparación (McNemar) | Series donde gana solo cada una | Diferencia | p | p Holm |
|---|---|---|---|---|
| P1 frente a P0 | 599 frente a 816 | **−6,42 pts** | 8,8·10⁻⁹ | 8,8·10⁻⁹ |
| P2 frente a P0 | 584 frente a 835 | **−7,42 pts** | 2,9·10⁻¹¹ | 5,7·10⁻¹¹ |

**Decisión (regla §7): no se adopta.** Las dos semillas del preview Nash son
significativamente **peores** que el preview greedy (−6,4 y −7,4 puntos); la regla exige
mejora. G sigue con el preview greedy, y la confirmación de v4 (protocolo §4.2) lo usa
(`nash_decision` devuelve `greedy` porque las diferencias son negativas).

Lectura: el preview Nash aprende la matriz de pagos con partidas de greedy_joint y la
resuelve como un juego minimax de horizonte uno; con 200 actualizaciones y muestreo
uniforme de pares (360 × 360 formaciones por contexto) la matriz es ruidosa (loss ≈ 0,62–0,65
frente a 0,69 de etiquetas al azar), y las formaciones que elige rinden peor que la regla
de tipado + BST. Por referencia, G con preview greedy saca 58,9 % con equipos no vistos,
coherente con el 57,5 % con los 300 equipos (§6.1).

Secundaria (memorización, 300 equipos = incluye los `fit` con los que entrenó el preview;
9101–9117, en pareado con `v3_n200`, preview greedy):

| Preview | Victorias / N | Tasa | IC 95 % (cluster) | McNemar frente al greedy |
|---|---|---|---|---|
| Greedy (`v3_n200`, §6.1) | 2935 / 5100 | 57,5 % | — | — |
| Nash s0 | 2737 / 5083 | 53,85 % | [52,43; 55,16] | 1011 frente a 1200, **−3,73 pts**, p = 6·10⁻⁵ |
| Nash s1 | 2671 / 5083 | 52,55 % | [51,23; 53,83] | 975 frente a 1224, **−4,92 pts**, p = 1·10⁻⁷ |

Incluso con los equipos que vio al entrenar, el preview Nash rinde peor que el greedy;
la desventaja es menor que con equipos no vistos (−3,7/−4,9 frente a −6,4/−7,4 puntos),
lo que sugiere algo de memorización, pero no la suficiente para compensar. Cola terminada
19:11 (`MCTS_G_NASH_DONE`), 0 anomalías. La confirmación de v4 arrancó a las 19:12 con
`preview: greedy (decisión §7: greedy)`.

## 8. Preview por búsqueda para G v5 (fijado el 21-11-2026 antes de medir)

Decisión del usuario: como método de árbol, el preview debe mirar los primeros turnos
(spec §4.6). Hallazgo previo: el preview greedy elige bien los 4 Pokémon pero no los leads.
Criterio y diseño idénticos a `protocolo_evaluacion.md` §4.4:

**Versión congelada:** commit `3269711` (G v5, 200 sims, `MCTS_SEED=0`; preview por búsqueda
con memo por proceso, `MCTS_PREVIEW_DEADLINE=90` bajo carga de 28 procesos). Solo **baselines**
como rivales. La cola aborta si cambia el código de `mcts_puct` o de los baselines.

**Exploración** (9101–9117 × 300 BO3, contra greedy_joint, mismas series):

| Config | Preview |
|---|---|
| S0 | greedy (`--preview-a greedy`, el actual) |
| S1 | búsqueda, decisión `mean` |
| S2 | búsqueda, decisión `maximin` |

Una variante es **elegible** si tiene 0 errores de serie y fallbacks del preview (deadline o
fallo → greedy) en ≤ 5 % de las series. Se elige la elegible con mayor diferencia McNemar
frente a S0 y pasa a confirmación **solo si** la diferencia es > 0 y p < 0,05. Si no, G se
queda con el preview greedy, se documenta y la cola termina.

**Confirmación** (bloques nuevos C9 = 10001–10017 y C10 = 10101–10117; variante elegida):

| Id | Hipótesis | Prueba |
|---|---|---|
| X1 | preview por búsqueda > greedy, ambos con G v5 contra greedy_joint, mismas series | McNemar exacta, C9 + C10 |
| X2 | preview por búsqueda > greedy, ambos con G v5 contra expectimax v1.9, mismas series | McNemar exacta, C9 + C10 |

Holm con m = 2 y consistencia de signo en C9 y C10.

**Smoke del 21-11-2026 (14:32):** la cola se detuvo sola por 112 anomalías `preview_foe_no_moves`: el preview leía al rival de `teampreview_opponent_team` (solo especie). Corregido en `d247fab` (lee la hoja abierta de `opponent_team`, como `frozen_preview`); se relanza con ese commit congelado y el mismo diseño.

**Bloqueo del bucle (21-11-2026, 15:20–17:40):** con la búsqueda síncrona, la primera semilla de S1 se colgó dos veces (9 errores `timed out during opening handshake`; series con 3 timeouts de 1800 s): la búsqueda (~7 s, más bajo carga) bloqueaba el bucle asyncio del proceso y las conexiones nuevas de otros jugadores caían por el `open_timeout` de 10 s de poke-env. Corregido en `3269711` (búsqueda en un hilo, `teampreview` asíncrono). S0 (preview greedy, 3276/5100) se conserva: el cambio solo toca la ruta `MCTS_PREVIEW=search`. Logs fallidos en `raw_failed/`; se repiten el smoke y S1 desde 9101. **Cambio de valor por defecto:** el
preview por búsqueda pasa a ser el de G solo si X1 se confirma y X2 no es significativamente
negativa (p Holm < 0,05 con diferencia < 0). Se reportan además las tasas binomiales de cada
brazo y los fallbacks del preview. Cola: `scripts/server/mcts_g_search_preview.sh`.

## 9. ¿Aporta el motor de inferencia de SP a G? (fijado el 21-11-2026 antes de medir)

**Motivo.** G corre con `sp_info = inference` (clase validated por defecto), mientras joint / solo
usan `meta` y expectimax `none`. Con greedy, la inferencia no estrechó ningún stat en partida
(0/10 140, 14-11-2026) y la Velocidad exacta del rival no aportó (`oracle_spe` ≈ línea base);
el prior meta LOO ya acierta la Spe en ~92 %. Pero G sí usa la Velocidad en el árbol (orden
dinámico, Tailwind, Trick Room), así que hay que medirlo en G. Decisión del usuario: si la
inferencia no aporta, desactivarla.

**Diseño** (`scripts/server/mcts_g_spinfo.sh`, código congelado `3269711`, tras la cola del
preview por búsqueda; G v5, 200 sims, preview greedy fijo, contra greedy_joint, 9101–9117 × 300):

| Config | `--sp-info-a` | Qué es |
|---|---|---|
| conteo | inference + `PS_SP_INFO_DEBUG=1` | 56 series: cuántos stats estrecha la inferencia en partidas de G |
| I0 | inference | default actual |
| I1 | oracle_spe | techo: Velocidad real del rival (diagnóstico) |
| I2 | meta | sin inferencia; 6 stats del rival al reparto más frecuente del prior meta LOO |

**Regla:** si I2 no es significativamente peor que I0 (no: diferencia < 0 y p < 0,05) y además
|diferencia| ≤ 1 pt o I2 es significativamente mejor, el default de G pasa a `meta` (inferencia
apagada). Si I2 es significativamente peor, se mantiene la inferencia. I1 frente a I0 mide el
valor máximo de conocer la Velocidad; si es significativo, queda como motivo para mejorar la
inferencia de Velocidad (no cambia el default). Se reporta también el tiempo por semilla y la
reproducibilidad I0 frente a S0 (misma config en la cola del preview).
