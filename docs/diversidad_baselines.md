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

## Resultados

Pendiente.
