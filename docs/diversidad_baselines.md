# Diversidad de juego de los baselines heurísticos

Los agentes RL que se entrenen necesitan rivales que jueguen **bien y de forma variada**.
Si el rival es previsible, el agente aprende a explotarlo en vez de aprender a jugar. Este
documento mide cuánto difieren entre sí los heurísticos y describe la variante estocástica
que se agregó para darles variedad.

## 1. Diversidad entre agentes (medida, 15-11-2026)

Expectimax registra en cada decisión qué habrían jugado joint (su fase 1) y solo en el
mismo estado (`expectimax_divergence`, pie de cada log de `run_robust_e2e`). Se compara el
par completo de acciones (movimiento, objetivo y mega de los dos lados).

| Corrida (commit a6d706c) | Decisiones | Expectimax distinto de joint | Expectimax distinto de solo |
|---|---|---|---|
| H1_C1, contra joint | 53 382 | 41,4 % | 69,5 % |
| H1_C2, contra joint | 53 700 | 41,2 % | 69,6 % |
| H3_C1, contra solo | 50 292 | 42,1 % | 70,4 % |
| S1_C1 (v1.8, sin cambios voluntarios) | 53 194 | 40,4 % | 69,5 % |
| S3_C1, contra random | 56 354 | 43,8 % | 73,1 % |
| techo BASE (9101–9117) | 53 530 | 41,2 % | 70,8 % |
| techo SPOR (oráculo de SP) | 53 288 | 43,3 % | 69,9 % |

**Lectura:** expectimax juega distinto de joint en ~41 % de las decisiones y de solo en
~70 %, pero los tres están a 1–3 puntos de fuerza entre sí (protocolo §5.1). Son estilos
distintos con un nivel parecido, lo que conviene para un pool de rivales.

Advertencias:

- "Lo que haría joint" y "lo que haría solo" se calculan **con las creencias de
  expectimax** (modo `none`). El joint real usa la opción C, así que la cifra aproxima el
  desacuerdo real.
- La comparación con solo se calcula solo si queda presupuesto en la fase 1. Si no, se
  toma la elección de joint, lo que puede subestimar algo el desacuerdo con solo.
- **Joint frente a solo no se registraba.** Desde este cambio se mide con
  `PS_DIVERSITY_LOG` (sección 3). Lo único que había era el perfil de jugadas de joint
  (`greedy_joint_pair_diag`): ~76 % de sus turnos son doble ataque, 11–27 % atacan con un
  solo lado y ~2–4 % son Protect, Helping Hand o redirección.

## 2. Diversidad dentro de cada agente: variante estocástica

Los tres agentes son **deterministas**: ante el mismo estado juegan siempre lo mismo, y la
variedad entre partidas sale solo de los equipos y los estados. La Fase 1 de expectimax
mostró muchos **casi-empates**: con un peso de amenaza de solo 0,05 cambia el 12 % de las
decisiones (`expectimax_diagnostico.md`). Elegir al azar entre jugadas casi equivalentes
debería dar variedad con un costo pequeño en fuerza.

| Agente | Parámetro (0 = determinista, por defecto) | Unidades | Qué hace |
|---|---|---|---|
| greedy_joint | `PS_JOINT_TIE_MARGIN` (`tie_margin` de `best_joint_double_action`) | puntaje de par de joint (% de daño + bonos) | muestreo uniforme entre los pares con puntaje ≥ mejor − margen |
| expectimax (alternativa) | `phase1_tie_margin` (`EXPECTIMAX_OVERRIDES`) | puntaje de par de joint | sorteo en la fase 1 (joint) de expectimax; cambia lo jugado solo cuando la puerta se cierra |
| expectimax | `tie_sample_margin` (`EXPECTIMAX_OVERRIDES`) | V del par (ofensa + utilidad + tempo) | **después de la puerta**, muestreo uniforme entre los pares con V en [V(jugado) − margen, V(jugado)] |

En expectimax no se consideran los pares con V mayor que el jugado: son los que la puerta
rechazó a propósito, y meterlos equivaldría a aflojarla. La semilla del muestreo es
aleatoria por proceso; `PS_TIE_SEED` la fija.

Código: `showdown/battle_bots/diversity_log.py` (muestreo y traza),
`greedy_joint/joint.py` (`LAST_TIE`), `expectimax_bayesian/expectimax.py`
(`tie_sample_margin`, `meta.tie_cands`, `meta.tie_changed`). Tests: `tests/test_diversity.py`.

## 3. Medición

`PS_DIVERSITY_LOG=<ruta>` escribe una línea por decisión de joint (¿igual a solo?) y de
expectimax (¿igual a joint?, ¿igual a solo?), con los candidatos del muestreo y si cambió la
jugada. `python -m showdown.inference_sim.analyze_diversity <ruta>` lo resume.

Plan (semillas de exploración 9101–9117, exploratorio):

| Corrida | Para qué |
|---|---|
| joint contra solo, determinista, con traza | desacuerdo joint frente a solo |
| joint (margen 3 y 8) contra solo | variedad ganada y costo en tasa de victoria (pareado contra la determinista) |
| expectimax (margen 10 y 25) contra joint | ídem, contra `techo/BASE` |

Una variante sirve como rival de entrenamiento si gana variedad (% de decisiones con jugada
distinta de la determinista) sin perder fuerza de forma apreciable (por ejemplo, menos de
1 punto, con TOST si hace falta).

## Resultados (15-11-2026, semillas 9101–9117, exploratorio)

Commit 8de1b3b. Joint contra solo y expectimax contra joint v1.8 (opción C).

| Config | Tasa | Frente a la determinista (McNemar) | Decisiones con > 1 candidato | Lectura |
|---|---|---|---|---|
| DJ0: joint determinista | 52,29 % | — | 0 % | joint juega **lo mismo que solo en 35,6 %** de las decisiones (distinto en 64 %) |
| DJ3: joint, margen 3 | 52,39 % | +0,18 pts, p = 0,57 | 13,9 % (2,3 cand.) | **sin costo** |
| DJ8: joint, margen 8 | 52,23 % | −0,10 pts, p = 0,83 | 24,6 % (2,6 cand.) | **sin costo** |
| DE2: expectimax, margen 2 (5 semillas) | 46,43 % | **−7,14 pts**, p = 2·10⁻⁹ | 61,1 % (3,4 cand.) | jugada distinta en 38 %; muy caro |
| DE10: expectimax, margen 10 | 45,52 % | **−6,65 pts**, p = 10⁻²⁵ | 64,2 % (3,9 cand.) | jugada distinta en 41 %; muy caro |

DE25 se canceló (DE10 ya era −7 pts); DE2 se cortó en 5 semillas y DE5 se canceló porque
el resultado ya era claro.

El % de "jugada distinta" de DJ3/DJ8 en esta corrida está mal (83 %): el contador de joint
comparaba por identidad de objeto. Se corrigió en d9798b2 (el juego no cambia) y se repite
la medición (DJ3b/DJ8b).

**Desacuerdo entre los tres (deterministas):** joint/solo 64 %, expectimax/joint ~41 %,
expectimax/solo ~70 %.

**Por qué el sorteo en expectimax es tan caro:** con margen 2 ya hay varios candidatos en el
61 % de las decisiones, es decir, la V de expectimax tiene muchos **empates casi exactos**.
El desempate determinista (el orden de los pares, que viene del ranking de la fase 1 tipo
joint) lleva información real, y sortear la destruye. Cuando la puerta se cierra (~60 % de
los turnos) se jugaba la elección de joint, y el sorteo la cambiaba por otro par con V igual
pero peor según el puntaje de joint.

**Alternativa (`phase1_tie_margin`, 0 = off):** sortear en la fase 1 de expectimax con el
puntaje de joint (el mismo `tie_margin` que no cuesta nada en joint). Solo cambia lo jugado
cuando se juega la fase 1.

### Medición final (contadores corregidos: d9798b2 y 0298e26)

| Config | Rival | Tasa | Frente a la determinista (McNemar) | Casi-empates | Jugada distinta de la determinista |
|---|---|---|---|---|---|
| DJ0: joint determinista | solo | 52,29 % | — | 0 % | 0 % |
| DJ3b: joint, `PS_JOINT_TIE_MARGIN=3` | solo | 51,99 % | −0,22 pts (82 frente a 93), p = 0,45 | 13,5 % (2,3 cand.) | **7,3 %** |
| DJ8b: joint, `PS_JOINT_TIE_MARGIN=8` | solo | 52,03 % | −0,20 pts (174 frente a 184), p = 0,63 | 24,2 % (2,6 cand.) | **13,8 %** |
| BASE: expectimax determinista | joint | 52,24 % | — | — | — |
| DEP3: expectimax, `phase1_tie_margin=3` | joint | 52,10 % | −0,06 pts (63 frente a 66), p = 0,86 | 14,6 % (2,3 cand.) | **4,2 %** |
| DEP8: expectimax, `phase1_tie_margin=8` | joint | 51,69 % | −0,54 pts (90 frente a 117), p = 0,07 | 25,6 % (2,7 cand.) | **7,4 %** |
| DE2/DE10: expectimax, `tie_sample_margin` | joint | 46,4 / 45,5 % | −7,1 / −6,7 pts | 61 / 64 % | 38 / 41 % |

En DEP, "jugada distinta" cuenta solo los turnos en que se juega la fase 1 (puerta cerrada).

## Recomendación para los rivales del entrenamiento RL

| Rival | Configuración | Variedad (jugada distinta) | Costo medido |
|---|---|---|---|
| greedy_joint | `PS_JOINT_TIE_MARGIN=8` | ~14 % de las decisiones | −0,2 pts, no significativo |
| expectimax | `EXPECTIMAX_OVERRIDES=phase1_tie_margin=3` | ~4 % | −0,06 pts, nulo |
| greedy_solo | determinista (sin opción de sorteo) | — | — |

Además, usar **los tres en el pool**: deterministas difieren entre sí en 41–70 % de las
decisiones con fuerza parecida (1–3 puntos), y esa es la mayor fuente de variedad. El sorteo
agrega variedad *dentro* de cada rival, para que ante el mismo estado no responda siempre
igual. **No usar `tie_sample_margin`** en expectimax.

Pendientes antes de entrenar: llevar estos parámetros a la configuración de rivales de
`rl_env_spec` (hoy por defecto son deterministas) y, si se quiere afirmar "sin costo" de
forma formal, una prueba TOST (protocolo §3b) con δ = 1 punto.
