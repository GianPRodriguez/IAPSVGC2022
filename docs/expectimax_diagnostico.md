# Expectimax-Bayesian: diagnóstico y plan de mejora

Documento de trabajo, actualizado en cada fase. El contrato del agente está en
[`expectimax_bayesian_spec.md`](expectimax_bayesian_spec.md), y el valor de la información
de SP en [`resultados_informacion_sp.md`](resultados_informacion_sp.md).

## Condiciones comunes

- `run_robust_e2e --teams 300 --procs 28 --seed S --restart-ps`, con semillas
  **9101–9117**: 17 semillas × 300 BO3 = **5100 BO3** por configuración. IC 95 %: ±1,4 puntos.
- B validado y prior leave-one-out (los valores por defecto). Rival: greedy_joint validado.
- La **referencia de juego idéntico** (joint contra joint) en estas semillas es **50,0 %**.
- Expectimax corre en su versión estable C235 (spec v1.5) salvo que se indique otra cosa.
- Alcance acordado: mejoras **dentro** de expectimax bayesiano. Se excluyen la profundidad 2
  y Nash, porque ya los cubren los agentes MCTS/Nash.

## Fase 0: punto de partida (15-11-2026)

### 0a y 0b: línea base e información de SP

| Configuración (contra joint validado) | Victorias | Tasa | Lectura |
|---|---|---|---|
| Joint contra joint | 2550/5100 | 50,0 % | referencia |
| Expectimax C235 (B: con inferencia) | 2536/5100 | 49,7 % | empata con joint |
| Expectimax en A (sin inferencia) | 2536/5100 | 49,7 % | **juegos idénticos** a B; ~11,5 min frente a ~21 min |
| Expectimax con oráculo completo | 2558/5100 | 50,2 % | +0,4, no significativo |
| Expectimax con oráculo de Velocidad | 2549/5100 | 50,0 % | +0,25, no significativo |

El valor de la información de SP está acotado por la estructura actual de expectimax:

- V = daño esperado + P(KO)·250 + Fake Out/Knock Off + crédito de tempo, y la amenaza recibida
  tiene peso 0;
- la búsqueda está anclada a joint (Top-K 7 y una puerta con margen de 40);
- los mundos promedian repartos del meta que ya se parecen a la verdad.

### 0c: diseño factorial de preview × combate (`--preview-a`)

| | Combate joint | Combate expectimax |
|---|---|---|
| **Preview greedy (tipos + BST)** | 50,0 % | **52,8 %** (2695; IC 51,5–54,2; sube en 13/17 semillas) |
| **Preview expectimax (roles + leads)** | **46,2 %** (2355; IC 44,8–47,5; baja en 16/17 semillas) | 49,7 % |

- **El combate de expectimax supera al de joint por +2,8 puntos**, de forma significativa.
- **Su preview resta −3,8 puntos**, también significativo.
- La interacción es de +0,7, no significativa.
- El empate de expectimax completo con joint era un efecto del preview. El diagnóstico
  anterior ("techo del diseño alcanzado") confundía preview y combate.

## Diagnóstico del preview de expectimax

### Offline: 3000 enfrentamientos al azar del pool

| Medida | Resultado |
|---|---|
| Mismos 4 elegidos que greedy | 42,5 % |
| Mismos leads que greedy | 13,9 % |
| Leads = los 2 primeros de la hoja | greedy 100 % (empates en su puntuación) · expectimax 18,4 % |
| Peso de cada término en expectimax (media ± desviación) | roles **40,7 ± 24,9** · ofensa 22,1 ± 4,7 · defensa −13,6 ± 5,6 · BST 17,0 ± 0,6 · calidad de lead 9,2 ± 3,7 |

Los bonos de rol (Tailwind 32, redirección 28, Trick Room 26, control de Velocidad 18,
Fake Out 14, diversidad 12) varían 4–5 veces más que los términos de tipos. En la práctica,
deciden qué 4 Pokémon se llevan.

### En combate: preview variado, combate de joint en ambos lados

| Preview del lado A | Victorias | Tasa | Lectura |
|---|---|---|---|
| greedy | 2550/5100 | 50,0 % | referencia |
| expectimax completo | 2355/5100 | 46,2 % | −3,8 |
| expectimax sin roles | 2438/5100 | 47,8 % | los roles cuestan ~1,6 |
| expectimax sin calidad de lead (leads en el orden de la hoja) | 2370/5100 | 46,5 % | la calidad de lead cuesta ~0,3, no significativo |
| los 4 primeros de la hoja | 2273/5100 | 44,6 % | el orden del diseñador por sí solo es malo |
| solo los pesos de tipos de expectimax (1,05 / 1,25 / 0,008; sin roles ni lead) | 2505/5100 | 49,1 % | los pesos cuestan ~0,9, no significativo |
| greedy + roles al 25 % | 2436/5100 | 47,8 % | **incluso los roles al 25 % cuestan ~2,2** |

**Atribución de los −3,8 puntos:**

| Componente | Costo estimado | Base de la estimación |
|---|---|---|
| Bonos de rol | 1,6–2,2 | sin roles: +1,6; roles al 25 % sobre greedy: −2,2 |
| Calidad de lead | 0,3–1,3 | sin calidad de lead: +0,3 (con roles); sin roles con lead frente a solo pesos de tipos: −1,3 |
| Pesos de tipos y BST | ~0,9, no significativo | solo pesos de expectimax frente a greedy |

Ningún componente del preview de expectimax mejora sobre tipos + BST. Probablemente los
roles fallan porque se asignan **por especie**, con listas fijas, y no por los movimientos
del team sheet. Un Pokémon de una especie "de rol" entra aunque no lleve ese movimiento, o
desplaza a otro con mejor encaje de tipos.

### Opción 2: roles leídos de los movimientos del team sheet (sobre greedy)

El bono solo cuenta si el Pokémon lleva de verdad el movimiento (Follow Me/Rage Powder,
Tailwind, Trick Room, Icy Wind/Electroweb/Thunder Wave…, Fake Out).

| Preview (combate joint en ambos lados) | Victorias | Tasa | Diferencia con greedy |
|---|---|---|---|
| greedy + roles por movimiento al 10 % | 2472/5100 | 48,5 % | −1,5 (IC 47,1–49,8) |
| greedy + roles por movimiento al 25 % | 2430/5100 | 47,6 % | −2,4 |
| greedy + roles por especie al 25 % | 2436/5100 | 47,8 % | −2,2 |

**Los roles restan aunque se lean de los movimientos reales**, y más cuanto mayor es su
peso. El problema no era la lista por especie. El bono desplaza a Pokémon con mejor encaje
de tipos, y ni joint ni expectimax explotan esos roles lo suficiente para compensarlo: en
la V de expectimax, las utilidades de soporte tienen peso 0 o casi 0. **La opción 2 queda
descartada.**

### Corrección aplicada (v1.7)

El preview de expectimax ahora es **tipado + BST de Greedy** (α=β=1, γ=0,01, sin roles ni
calidad de lead). En 1200 enfrentamientos al azar del pool elige exactamente lo mismo que
greedy_solo/greedy_joint (1200/1200: mismos 4 y mismos leads). Por lo tanto **B, C y D
comparten el mismo preview**, y cualquier diferencia entre ellos viene solo del combate.

| Confirmación (semillas 9101–9117, contra joint validado) | Victorias | Tasa |
|---|---|---|
| Expectimax v1.7 completo (preview propio = tipado + BST) | 2695/5100 | **52,8 %** (IC 51,5–54,2) |
| Referencia: combate expectimax con `--preview-a greedy` (Fase 0c) | 2695/5100 | 52,8 % |

Las victorias por semilla de v1.7 son **idénticas** a las de la Fase 0c con `--preview-a greedy`
(164, 142, 154, …), lo que confirma que el preview es el mismo. **Expectimax v1.7 supera a
joint validado por +2,8 puntos** (antes 49,7 %), una diferencia significativa.

## Información de SP con v1.7 (semillas 9101–9117, contra joint validado)

| Configuración | Victorias | Tasa | Lectura |
|---|---|---|---|
| v1.7 con inferencia (B) | 2695/5100 | 52,8 % | |
| v1.7 sin inferencia (A) | 2695/5100 | 52,8 % | **juegos idénticos** en las 17 semillas, la mitad de tiempo |
| v1.7 con oráculo completo | 2722/5100 | 53,4 % | +0,5 (IC pareado aprox. −0,3..+1,3), no significativo |

**Decisión (spec v1.8): A por defecto**, como en los dos greedy.

## Fase 1: diagnóstico sin cambiar el agente (v1.7, semillas 9101–9103)

Son 9390 decisiones de expectimax contra joint validado, medidas con `PS_VTERMS_LOG`,
`PS_OPPMODEL_LOG` y `expectimax_divergence`.

### Modelo del rival (5953 movimientos observados, emparejados con la predicción de cada rival)

| Medida | Valor |
|---|---|
| Acierto top-1 del movimiento | 30,4 % (ataques 31,6 %) |
| Acierto top-1 de movimiento y objetivo | 21,6 % |
| p media asignada al movimiento real | 0,262 (uniforme: 0,194) |
| Movimientos de control reales (Protect, etc.) | 4,0 % |
| **Masa predicha para control y cambios** | **0** |

El modelo es mejor que el azar, pero débil. Sus probabilidades están razonablemente
calibradas en las franjas altas (p ≥ 0,5: acierto ≥ 94 %). **Nunca predice Protect ni
movimientos de control**, aunque son el 4 % de las jugadas reales.

### Términos de V (9390 decisiones, hasta 49 pares cada una)

| Término | Media en todos los pares | Rango medio **dentro** de cada decisión (lo que decide) | Peso actual |
|---|---|---|---|
| Ofensa (daño esperado + P(KO)·250) | 201 | **356** | 1 |
| Amenaza recibida | 139 | **172** | **0** |
| Tempo | 31 | 109 | 1 |
| Sinergia | 32 | 76 | 0 |
| Utilidad VGC (Fake Out/Knock Off) | 7 | 28 | 1 |

**Contrafactual de la amenaza** (% de decisiones en que cambia el mejor par si V resta
w·amenaza):

| w | 0,05 | 0,1 | 0,2 | 0,3 | 0,5 | 1,0 |
|---|---|---|---|---|---|---|
| Decisiones que cambian | 12,0 % | 14,0 % | 16,8 % | 19,2 % | 22,5 % | 29,4 % |

La amenaza tiene una escala comparable a la ofensa: varía la mitad entre pares. Con
w = 0,05 ya cambia el 12 % de las decisiones, lo que indica **muchos casi-empates de V**
entre los mejores pares. Con w = 0,5 y w = 1 (−3,4 y −5 puntos medidos antes) cambiaba
demasiado.

### Puerta (phase2 frente a joint)

| Medida | Valor |
|---|---|
| El mejor par de phase2 coincide con el de joint | 35,3 % |
| Phase2 aceptado (siempre distinto de joint) | 41,1 % |
| Phase2 prefiere otro par pero la puerta lo rechaza (margen 40) | ~23,6 % |
| V(mejor) − V(joint) | media 100, sd 157 |
| Tipos de acción en el mejor par | daño 16055, estado 2433, speed 100, Helping Hand 88, **Protect 78**, carga 14, redirección 12 |

### Qué implica para la Fase 2

1. **Protect condicional en el modelo del rival:** justificado, porque hoy la masa de
   control es 0.
2. **Calibración de la amenaza:** probar pesos pequeños (0,05–0,2), que cambian entre el
   12 % y el 17 % de las decisiones, frente a los que ya se midieron como dañinos (≥ 0,5).
3. **Puerta:** rechaza ~24 % de las preferencias de phase2. Se reajusta en la Fase 3,
   después de las mejoras.
4. **Expectimax casi nunca elige Protect** (78 de 18 780 acciones): es coherente con que
   la amenaza tenga peso 0. Se revisa junto con la calibración.

- Los roles son la principal causa identificada.
- Los leads no son el problema. Tampoco lo es que greedy herede los leads del diseñador:
  la hoja sola da 44,6 %.
- Lo que aporta greedy es **su elección de 4 por tipos + BST** (+5,4 sobre la hoja).

## Fase 2: mejoras de combate (v1.8, semillas 9101–9117, contra joint validado)

Línea base: expectimax v1.8 (preview tipado + BST, modo A) = **2695/5100 = 52,8 %**.

### 2a: amenaza con pesos pequeños (`v_iso_threat_weight`)

| Peso | Victorias | Tasa | Diferencia con la base |
|---|---|---|---|
| 0 (base) | 2695/5100 | 52,8 % | — |
| 0,05 | 2694/5100 | 52,8 % | −1 |
| 0,1 | 2697/5100 | 52,9 % | +2 |
| 0,2 | 2675/5100 | 52,5 % | −20 (−0,4) |
| 0,5 (medido con v1.6) | | | −3,4 |
| 1,0 (medido con v1.6) | | | −5,0 |

**Calibrar la amenaza no aporta.** Con pesos chicos cambia entre el 12 % y el 17 % de las
decisiones, pero sin mejora neta. Desde 0,2 empieza a empeorar, y lo hace de forma
monótona. Se mantiene en **0**.


### 2b: Protect condicional del rival en la ofensa

Al implementarlo apareció un hueco: **la ofensa no descontaba el Protect del rival.** Se
agregó la opción de multiplicar la ofensa contra cada rival por (1 − p), con p > 0 solo si
el rival lleva un movimiento de protección y no lo usó el turno anterior, y p más alta si
nuestro par le da P(KO) ≥ 0,5 (`opp_protect_base` / `opp_protect_threat`).

| p base / p con amenaza de KO | Victorias | Tasa | Diferencia con la base |
|---|---|---|---|
| 0 / 0 (base) | 2695/5100 | 52,8 % | — |
| 0,08 / 0,25 | 2648/5100 | 51,9 % | −47 (−0,9) |
| 0,12 / 0,40 | 2625/5100 | 51,5 % | −70 (−1,4) |

**Empeora, y más cuanto mayor es p.** Descontar la ofensa por un posible Protect lleva a
evitar al rival más peligroso o a repartir el daño. Coincide con la Fase 1: los rivales
usan control solo en el 4 % de sus jugadas. **Descartado.** El código queda apagado por
defecto (p = 0).

### 2c: cambios voluntarios

**Causa de que nunca se eligieran:** en el modo de valoración por defecto
(`v_solo_isomorphic`), el valor de cambiar (`switch_v`: preservación, posición, habilidad de
entrada, castigo al entrante, cambio libre tras KO) se calculaba pero no se sumaba a V. Un
par con cambio solo aportaba el daño del compañero y nunca podía ganar. Por eso, en
experimentos anteriores, variar `switch_v_scale` daba juegos idénticos. Corregido con
`v_iso_switch_weight` (0 por defecto). Se mide con `enable_voluntary_switch=1` y
`max_switch_hybrids=8`.

**Segundo problema, encontrado al medir:** con `switch_v` activo, cada semilla tardaba
~370 s en vez de ~31 s. Se cortó la corrida y se agregaron trazas (`PS_SWITCH_LOG` +
`analyze_switches`):

| Semilla 9101 | Base v1.8 | `switch_v` 25 % | `switch_v` 50 % |
|---|---|---|---|
| Turnos medios por juego | 5,2 | 5,2 | 5,4 |
| Juegos con algún cambio voluntario | 0 | — | 12 de 600 |
| Decisiones de cambio registradas | 0 | — | 12 395 |
| Rebotes (vuelve quien salió hace ≤ 2 turnos) | — | 0,2 % | 8,3 % |

No era un bucle de cambios en el juego, sino un **bucle de órdenes rechazadas**. Miles de
decisiones se acumulaban en el mismo turno de unos pocos juegos. Con
`PS_PLAYER_LOG_LEVEL=25`, Showdown respondía 6069 veces `[Invalid choice] Can't switch:
The active Pokémon is trapped`.

**Causa:** las reservas se armaban uniendo las listas de cambio de los dos lados, y ambos
Pokémon recibían opciones de cambio aunque uno estuviera atrapado (poke-env vacía la lista
solo del lado atrapado). **Arreglo (`4a8c012`):** `switch_sides` limita los cambios al lado
con cambios disponibles y sin `trapped`/`maybe_trapped`, y el player valida el objetivo.
Comprobación (semilla 9101, `switch_v` 50 %): 34,7 s, 0 rechazos, 168/300 victorias (base
164), 116 cambios voluntarios.

También quedaron disponibles `switch_cost` y `switch_recent_penalty` (costo de tempo y
anti-rebote, en 0 por defecto) por si los cambios resultan excesivos.

**Medición con el arreglo** (17 semillas, contra joint validado; prueba t pareada contra la
base v1.8, que da 2695):

| `v_iso_switch_weight` (con `enable_voluntary_switch=1`, `max_switch_hybrids=8`) | Victorias | Tasa | Diferencia | IC 95 % pareado | t |
|---|---|---|---|---|---|
| 0 (base, sin cambios voluntarios) | 2695/5100 | 52,8 % | — | — | — |
| **0,25** | 2728/5100 | **53,5 %** | **+33** | +0,17..+1,12 pts | 2,89 |
| **0,5** | 2728/5100 | **53,5 %** | **+33** | +0,19..+1,10 pts | 3,01 |
| 1,0 | 2692/5100 | 52,8 % | −3 | — | −0,12 |

**Los cambios voluntarios mejoran (+0,65 puntos, significativo) con un peso moderado**; con
peso 1 se pierde la mejora porque cambia de más. Es la primera mejora de la Fase 2 que pasa
la prueba. **Elegido para los bloques siguientes: 0,25** (empata con 0,5 y cambia menos).
Debe confirmarse con semillas nuevas en la Fase 3.

### Bloques 1–3: reajuste sobre v1.8 + cambios voluntarios (0,25)

La referencia es v1.8 con cambios voluntarios al 25 %: **2728/5100 = 53,5 %**. Se aplica la
prueba t pareada por semilla (16 gl; |t| ≥ 2,12 es significativo).

| Bloque | Cambio | Victorias | t | Lectura |
|---|---|---|---|---|
| 1 | margen de la puerta 0 | 2655 | −2,43 | peor |
| 1 | margen 20 | 2719 | −0,60 | igual |
| 1 | margen 80 | 2679 | −3,74 | peor |
| 1 | puerta apagada | 2655 | −2,43 | peor (equivale a margen 0) |
| 1 | `soft_ko_bonus` 150 | 2737 | +0,65 | igual |
| 1 | `soft_ko_bonus` 400 | 2713 | −1,11 | igual |
| 1 | tempo 0,5 | 2692 | −3,77 | peor |
| 1 | tempo 1,5 | 2725 | −0,35 | igual |
| 2 | tope de 3 s (misma cobertura) | 2728 | 0 | **idéntico en las 17 semillas**: la búsqueda termina antes de 1,6 s |
| 3 | `switch_cost` 40 | 2717 | −2,18 | levemente peor |
| 3 | `switch_recent_penalty` 150 | 2720 | −1,93 | igual / levemente peor |

**Conclusiones:**
- Los valores de C235 (margen 40, `soft_ko_bonus` 250, tempo 1,0) siguen siendo los
  adecuados después de corregir el preview.
- Más tiempo de cálculo no cambia ninguna decisión.
- Encarecer los cambios voluntarios no ayuda, porque no había bucle de cambios.

**Candidata a v1.9:** v1.8 + `enable_voluntary_switch=True`, `max_switch_hybrids=8`,
`v_iso_switch_weight=0.25`, con todo lo demás sin cambios.

## Plan

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Línea base, A frente a B frente a oráculo, factorial preview × combate | **hecha** |
| Preview | Diagnóstico y corrección del preview de roles | **hecha.** Opción 1 aplicada (v1.7): preview = tipado + BST, idéntico al de greedy. Opción 2 (roles por movimientos) descartada: −1,5 / −2,4 |
| 1 | Diagnóstico sin cambiar el agente: acierto del modelo del rival por cada rival, escala de la amenaza frente a la ofensa, aporte de cada término de V, uso y resultado de la puerta | **hecha** (ver arriba). Además: A por defecto (v1.8) |
| 2 | Una mejora por corrida, con el preview fijo (desde v1.7 es el preview por defecto de expectimax, así que no hace falta forzarlo): Protect condicional en el modelo del rival, calibración de la amenaza, cambios voluntarios que nunca se eligen. La actualización bayesiana de los mundos tiene prioridad baja, porque el oráculo no aporta | pendiente |
| 3 | Combinar lo que pase y reajustar la puerta; confirmar con semillas nuevas (contra joint, solo y random); **volver a medir A contra los oráculos** con la versión final (si el oráculo pasa a aportar, se reactiva la actualización bayesiana); documentar | pendiente |


## Reproducir

```bash
# Factorial: combate de expectimax con preview greedy (repetir semillas 9101..9117)
python -m showdown.inference_sim.run_robust_e2e --teams 300 --procs 28 --seed 9101 --restart-ps \
  --player-a expectimax_bayesian --player-b greedy_joint --preview-a greedy
# Diagnóstico de preview con combate joint
python -m showdown.inference_sim.run_robust_e2e --teams 300 --procs 28 --seed 9101 --restart-ps \
  --player-a greedy_joint --player-b greedy_joint --preview-a expectimax_noroles
```

Los registros del servidor están en `~/iapsvgc_scratch/expx/runs/` (con su `summary.txt`).
