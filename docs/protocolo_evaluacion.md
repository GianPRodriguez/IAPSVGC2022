# Protocolo de evaluación: cómo se afirma que un método es mejor que otro

Este documento fija cómo la tesis afirma "A > B" entre agentes de combate. Se aplica a la
Fase 3 de expectimax y a la escalera de heurísticos.

La herramienta de los análisis es `python -m showdown.inference_sim.confirm_stats`.

## 1. Exploración frente a confirmación

| Etapa | Semillas | Uso | Cómo se reporta |
|---|---|---|---|
| Exploración | 8101–8117, 9101–9117 | Diagnóstico, ajuste de parámetros, elección de variantes | Como **exploratoria**. No respalda afirmaciones de superioridad |
| **Confirmación** | **Bloques nuevos** (C1 = 9201–9217, C2 = 9301–9317) que no se usaron para ajustar nada | **Una** medición por hipótesis, con la versión congelada (commit fijo, configuración por defecto) | **Este es el resultado de la tesis** |

Elegir la mejor variante y medirla con las mismas semillas infla el resultado por
selección. Medir con semillas nuevas da una estimación de la ventaja sin ese sesgo.

## 2. Condiciones fijas

- `run_robust_e2e --teams 300 --procs 28 --restart-ps`: 300 BO3 por semilla, cruzadas.
  Cada par de equipos se juega con los equipos intercambiados y con pools A/B disjuntos.
  17 semillas por bloque = 5100 BO3.
- **Información simétrica:** B validado (por defecto), prior leave-one-out (por defecto) y
  el modo de SP por defecto de cada agente (A en solo, joint y expectimax).
- **Preview:** el propio de cada agente. Desde expectimax v1.7, solo, joint y expectimax
  comparten el preview de tipado + BST, así que las diferencias entre ellos son de combate.
- **Trazabilidad:** commit, comando, semillas y logs por semilla archivados. El reporte de
  `run_robust_e2e` registra `sp_info`, `b_validated` y `preview`.

## 3. Pruebas estadísticas

| Pregunta | Diseño | Prueba | Herramienta |
|---|---|---|---|
| ¿A > B en enfrentamiento directo? | A contra B | **Binomial exacta bilateral** contra 50 %, con **IC 95 % por *cluster bootstrap***. El clúster es (semilla, par de equipos), porque las dos BO3 cruzadas de un par no son independientes | `confirm_stats binom` |
| ¿La versión A' mejora a la A? (ambas contra el mismo rival) | Pareado: mismas semillas y series | **McNemar exacta** sobre las series en que una gana y la otra pierde | `confirm_stats mcnemar` |
| Varias afirmaciones a la vez | Familia de hipótesis | **Holm-Bonferroni**, α familiar = 0,05 | `confirm_stats holm` |

**Regla de decisión para afirmar "A > B"**:

1. el p ajustado por Holm es ≤ 0,05;
2. el IC 95 % por *cluster bootstrap* excluye el 50 % y queda del lado de A;
3. la dirección se repite en cada bloque de confirmación medido (consistencia).

Si falla cualquiera de las tres, se reporta "sin diferencia demostrada", aunque la tasa
observada esté por encima de 50 %.

**Potencia.** Con 5100 BO3, el margen de error a ciertas tasas (cerca de 50 %) es de
±1,37 puntos al 95 %. Para efectos de alrededor de 1,5 puntos conviene usar dos bloques
(10 200 BO3). En el diseño pareado (McNemar) se detectan efectos menores, de ~0,5 puntos,
porque la mayoría de las series salen iguales.

## 4. Hipótesis pre-registradas (fijadas el 15-11-2026, antes de medir)

**Versión congelada de expectimax para la Fase 3:** v1.8 + los cambios que pasen la
exploración (cambios voluntarios al 25 % y lo que salga de los bloques 1–3). Se congela como
**v1.9 por defecto** en un commit antes de la confirmación.

**Familia principal** (escalera; Holm con m = 3):

| Id | Hipótesis | Comparación | Bloques |
|---|---|---|---|
| H1 | expectimax v1.9 > greedy_joint | A = expectimax, B = greedy_joint | C1 + C2 |
| H2 | greedy_joint > greedy_solo | A = greedy_joint, B = greedy_solo | C1 |
| H3 | expectimax v1.9 > greedy_solo | A = expectimax, B = greedy_solo | C1 |

**Hipótesis secundarias** (no entran en la familia; se reportan con su p sin ajustar y se
etiquetan como secundarias):

| Id | Hipótesis | Prueba |
|---|---|---|
| S1 | v1.9 mejora a v1.8 contra joint (efecto de la Fase 2) | McNemar en C1 |
| S2 | el oráculo de SP no mejora a v1.9 (valor de la información) | McNemar en C1 |
| S3 | v1.9 > random (control de cordura) | binomial en C1 |

## 5. Plantilla de reporte

| Hipótesis | Victorias / N | Tasa | IC 95 % (cluster) | p | p Holm | Semillas > 50 % | Veredicto |
|---|---|---|---|---|---|---|---|
| H1 | … | … | … | … | … | …/34 | … |

Cada afirmación lleva su alcance: "en Reg M-C, con el pool MetaVGC de 300 equipos e
información simétrica".

## 6. Limitaciones

- Un solo formato (Reg M-C) y un solo pool de equipos.
- La relación entre agentes puede no ser transitiva, por eso cada par se mide directamente.
- Expectimax trabaja con un tope de tiempo de pared por decisión. Los resultados dependen
  de la carga del servidor; se usan siempre `--procs 28` y 2 workers.
