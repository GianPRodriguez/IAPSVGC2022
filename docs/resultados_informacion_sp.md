# Valor de la información de Stat Points (SP) del rival

Resultados experimentales del 14 y 15 de noviembre de 2026 (servidor phantom). La pregunta es
cuánto rinde conocer los SP del rival, comparado con solo conocer sus stats base más el
motor de inferencia. Agentes: greedy_solo, greedy_joint y expectimax_bayesian.

Expectimax se analiza a fondo en [`expectimax_diagnostico.md`](expectimax_diagnostico.md).

## Protocolo

- `run_robust_e2e --teams 300 --procs 28 --seed S --restart-ps`, un BO3 cruzado por cada
  par de equipos de pools A/B disjuntos. Son 300 BO3 por semilla y 17 semillas por
  configuración, es decir **5100 BO3**.
- **Información simétrica:** B validado (`--b-validated`; es el valor por defecto desde
  `7b61168`). Las corridas de las secciones 2 y 3 son anteriores al prior leave-one-out
  (§4), pero su efecto es pequeño.
- IC 95 % binomial: ±1,4 puntos con 5100 BO3. Los agentes greedy son deterministas y las
  semillas de batalla dependen del contenido, así que dos configuraciones con las mismas
  semillas forman un **diseño pareado**.
- Modos de información (`--sp-info-a/b`):
  - `inference` es el motor de inferencia, es decir **B**;
  - `none` deja las creencias sin estrechar, es decir **A**;
  - `oracle` da los 6 SP verdaderos;
  - `oracle_spe` da solo la Velocidad verdadera;
  - `spe_min` / `spe_max` fijan la Velocidad rival en SP 0 / SP 32.

## 1. Qué hace hoy el agente sin información exacta

Con un rango de SP abierto, `apply_belief_stats` aplica dos reglas:

- **HP/Atk/Def/SpA/SpD:** usa SP 32 en cada stat por separado. Es una estimación puntual
  pesimista, análoga al criterio de Wald, pero aplicada stat a stat y sin respetar el
  presupuesto de 66 SP.
- **Velocidad:** usa el SP del reparto más frecuente del meta para esa especie y naturaleza.

El motor de inferencia **no estrecha ningún stat durante la partida**: 0 de 10 140 stats en
2813 observaciones (30 BO3). Sus salvaguardas (margen de 24 en Velocidad, descarte de podas
grandes, mesetas) descartan casi toda poda.

## 2. A frente a B frente a oráculo (greedy), semillas 8101–8117

| Prueba | Victorias | Tasa | IC 95 % | Lectura |
|---|---|---|---|---|
| Joint: B contra B (3 semillas) | 452/900 | 50,2 % | — | |
| Joint: B contra A (3 semillas) | 452/900 | 50,2 % | — | **Juegos idénticos** en cada semilla (154, 151, 147) |
| Joint: oráculo completo contra B | 2564/5100 | 50,3 % | 48,9–51,6 | Sin efecto significativo |
| Joint: oráculo de Velocidad contra B | 2511/5100 | 49,2 % | 47,9–50,6 | Igual a la referencia de juego idéntico |
| Solo: oráculo completo contra B | 2598/5100 | 50,9 % | 49,6–52,3 | Sin efecto significativo |
| Solo: oráculo de Velocidad contra B | 2533/5100 | 49,7 % | 48,3–51,0 | Sin efecto |
| **Joint: Velocidad rival en SP 0 contra SP 32** | 2511/5100 | 49,2 % | 47,9–50,6 | Casi los mismos juegos por semilla |

**Referencia de juego idéntico.** En las semillas 8101–8117, dos agentes que juegan
prácticamente igual dan **49,2 %**, no 50 %. Frente a esa referencia, el oráculo completo de
joint suma **+1,0 punto**, con un IC pareado aproximado de −0,1 a +2,2: en el borde, sin
confirmar.

**Costo.** Con B en ambos lados, cada semilla tarda 55–64 s; con A en un lado, 31–35 s. La
inferencia hace llamadas a la calculadora por cada candidato de SP.

## 3. Comparaciones con información simétrica (semillas 8101–8117)

| Prueba | Victorias | Tasa | IC 95 % | Lectura |
|---|---|---|---|---|
| Joint contra solo, los dos validados con B | 2664/5100 | **52,2 %** | 50,9–53,6 | Joint es mejor (+2 puntos), significativo |
| Joint validado contra joint con B crudo | 2668/5100 | **52,3 %** | 50,9–53,7 | El B crudo pierde ~2 puntos solo por información |

El **B crudo** (sin wrapper) toma las stats que poke-env da del rival: las bases de la
especie como si fueran stats finales y el HP en porcentaje. Los resultados anteriores
medidos contra un B crudo **favorecían al A en ~2 puntos**. Por ejemplo, expectimax daba
50,3 % contra joint crudo, pero 47,7 % contra joint validado.

## 4. Filtración del prior y leave-one-out

El prior de SP (`meta_sp_prior`) y los mundos de expectimax (`belief.py`) se construían con
`metavgc_regmc_teams.json`, que contiene **los mismos 300 equipos** de la evaluación. Con
leave-one-out (desde `24818b6`, por defecto) se descuentan los repartos del equipo rival.
Esta es la precisión del prior sobre los 1800 repartos:

| Medida | Con filtración | Leave-one-out |
|---|---|---|
| Velocidad exacta | 95,3 % | **91,7 %** (error medio de 1,25 SP) |
| Reparto completo exacto | 80,1 % | 74,2 % |
| Reparto real entre los 4 más frecuentes de la especie | 97,2 % | ~92 % |

El meta está muy concentrado. Un prior honesto ya acierta la Velocidad en ~92 % de los
casos, por lo que la filtración era pequeña y las conclusiones de §2 se mantienen.

## 5. Expectimax (semillas 9101–9117, B validado y leave-one-out)

| Prueba (contra joint validado) | Victorias | Tasa | Diferencia con A |
|---|---|---|---|
| Referencia joint contra joint | 2550/5100 | 50,0 % | — |
| A: sin inferencia | 2536/5100 | 49,7 % | — |
| B: con inferencia | 2536/5100 | 49,7 % | 0, **juegos idénticos**, en el doble de tiempo |
| Oráculo completo | 2558/5100 | 50,2 % | +0,4, no significativo |
| Oráculo de Velocidad | 2549/5100 | 50,0 % | +0,25, no significativo; cambia algunas decisiones |

Este resultado es coherente con la estructura actual de expectimax:

- su V depende casi solo de la ofensa propia, y el daño recibido tiene peso 0;
- la búsqueda está anclada a joint: Top-K y una puerta con margen de 40;
- sus mundos promedian repartos del meta que ya se parecen a la verdad.

## Conclusiones

1. **"La información exacta impacta, pero poco": se confirma.** Con los SP verdaderos, la
   mejora es como mucho de ~1 punto, y no es significativa en ninguno de los tres agentes.
2. **"El motor de inferencia ayuda notablemente": se refuta.** El motor actual no acota
   nada en partida. A y B dan juegos idénticos, y B cuesta el doble de tiempo.
3. **"Un motor perfecto no suma": se confirma** para los tres agentes heurísticos.
4. **Velocidad.** Es el stat más fácil de deducir, pero el prior del meta ya la acierta en
   ~92 % de los casos.
   - greedy_solo no la usa.
   - greedy_joint no cambia su resultado ni fijándola en los extremos.
   - En expectimax cambia algunas decisiones sin mejorar el resultado.
5. **Decisiones tomadas:**
   - A por defecto en solo y joint (specs v1.6).
   - Prior leave-one-out (v1.7).
   - B validado por defecto en `run_robust_e2e`.
   - Rivales validados en el entrenamiento RL (`rl_env_spec` v1.19).
   - Pendiente: A por defecto en expectimax.
6. **Limitación de diseño.** El supuesto de SP 32 en 5 stats describe un rival imposible
   (160 SP con un presupuesto de 66). Los oráculos muestran que eso no cuesta partidas
   medibles a ninguno de los tres agentes.

## Reproducir

```bash
# Ejemplo: oráculo completo de joint frente a B (una semilla; repetir para 8101..8117)
python -m showdown.inference_sim.run_robust_e2e --teams 300 --procs 28 --seed 8101 --restart-ps \
  --player-a greedy_joint --player-b greedy_joint --sp-info-a oracle --sp-info-b inference
```

Los registros por semilla del servidor están en `~/iapsvgc_scratch/spinfo/runs/` y
`~/iapsvgc_scratch/expx/runs/`, con un `summary.txt` en cada carpeta.
