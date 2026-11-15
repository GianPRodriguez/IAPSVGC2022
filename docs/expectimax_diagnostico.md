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

- Los roles son la principal causa identificada.
- Los leads no son el problema. Tampoco lo es que greedy herede los leads del diseñador:
  la hoja sola da 44,6 %.
- Lo que aporta greedy es **su elección de 4 por tipos + BST** (+5,4 sobre la hoja).

## Plan

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Línea base, A frente a B frente a oráculo, factorial preview × combate | **hecha** |
| Preview | Diagnóstico y corrección del preview de roles | en curso |
| 1 | Diagnóstico sin cambiar el agente: acierto del modelo del rival por cada rival, escala de la amenaza frente a la ofensa, aporte de cada término de V, uso y resultado de la puerta | pendiente |
| 2 | Una mejora por corrida, **con el preview fijo** (greedy en ambos lados): Protect condicional en el modelo del rival, calibración de la amenaza, cambios voluntarios que nunca se eligen. La actualización bayesiana de los mundos tiene prioridad baja, porque el oráculo no aporta | pendiente |
| 3 | Combinar lo que pase y reajustar la puerta; confirmar con semillas nuevas (contra joint, solo y random); **volver a medir A contra los oráculos** con la versión final (si el oráculo pasa a aportar, se reactiva la actualización bayesiana); documentar | pendiente |

Decisión pendiente: **A por defecto en expectimax**. Da el mismo resultado en la mitad del
tiempo.

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
