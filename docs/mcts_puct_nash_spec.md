# MCTS–PUCT/Nash Agent Spec — SM-MCTS B1 (VGC Gen 9)

| Campo | Valor |
|---|---|
| **Versión** | **1.3** (alineada con [`rl_env_spec.md`](rl_env_spec.md) v1.15) |
| **Estado** | FROZEN; `showdown/rl/mcts_puct_nash/` |
| **Training** | **No**: planificación en tiempo de decisión, sin gradientes |
| **CLI** | `python -m showdown.rl.mcts_puct_nash --seed 0 --sims 8 --smoke` |
| **Dependencias** | NumPy; reutiliza el simulador S2 de MCTS estándar |

## Por qué existe

VGC es simultáneo: ambos jugadores eligen sin conocer la orden actual del otro.
El MCTS open-loop aproxima al rival con una muestra uniforme. Esta variante
mantiene una matriz entre acciones propias y rivales, explora celdas con PUCT y
resume cada nodo mediante un equilibrio cero-suma aproximado. Así pregunta “¿qué
mezcla resiste una buena respuesta?” y no solo “¿qué gana contra azar?”.

## Glosario

| Término | Significado |
|---|---|
| Celda \((i,j)\) | Acción propia Top-K contra acción rival Top-K |
| \(\hat A\) | Matriz empírica de retornos medios |
| Top-K | Recorte greedy de acciones legales por lado |
| Fictitious Play | Aproximación iterativa de estrategias mixtas |
| \(\sigma,\tau\) | Mezclas propia y rival |
| Valor Nash | \(\sigma^\top\hat A\tau\) usado en backup |
| SM-MCTS | Árbol para movimientos simultáneos |

## Algoritmo paso a paso

1. Desde el snapshot se ordenan joints legales y se retienen hasta 16 por lado;
   para el rival se usa el snapshot volteado.
2. El nodo crea una matriz de celdas y prior uniforme \(1/(|I||J|)\).
3. PUCT selecciona una celda \((i,j)\), priorizando celdas sin visitar.
4. S2 aplica simultáneamente ambas acciones.
5. En terminal se toma Candidate 1; en hoja nueva se hace rollout greedy joint.
6. La celda profunda recibe la muestra \(z\).
7. Fictitious Play resuelve la matriz estimada; su valor sube a los ancestros.
8. En la raíz se resuelve con más iteraciones y se juega
   \(\arg\max_i\sigma_i\).

El Top-K limita una matriz que, sin recorte, podría combinar cientos de acciones
por lado. Empates del ranking se resuelven por id joint menor.

### Intuición matricial

En un nodo simultáneo no basta preguntar qué acción propia tiene mayor media:
el rival puede responder. La matriz \(\hat A_{ij}=W_{ij}/n_{ij}\) estima el
retorno de cada pareja. Fictitious Play aproxima mezclas
\((\sigma,\tau)\) y el valor
\[
v=\sigma^\top\hat A\tau.
\]
Una mezcla es útil cuando ninguna acción pura cubre todas las respuestas. El
backup no propaga ciegamente la última celda: tras actualizarla, vuelve a
resolver la matriz del nodo y entrega su valor al ancestro.

Las celdas no visitadas tienen prioridad de expansión. Después PUCT combina su
media con prior uniforme por celda. Top-K reduce el producto cartesiano de hasta
576×576, a costa de excluir respuestas cuyo score greedy previo sea bajo.

### Ciclo de presupuesto

1. Se crean \(I(s)\) y \(J(s)\), ordenados por score y joint id.
2. La raíz contiene hasta \(K^2=256\) celdas.
3. Cada simulación elige una celda no probada o usa PUCT.
4. S2 ejecuta ambas joints simultáneamente y crea un hijo.
5. Una hoja no terminal se completa con rollout greedy.
6. La celda recibe \(z\); cada ancestro resuelve FP interno de 200 iteraciones.
7. Tras 50 simulaciones, la raíz ejecuta FP de 500 iteraciones.
8. Se juega el joint de máxima masa en \(\sigma\), con desempate por id menor.

El coste combina transiciones S2 y solves repetidos. Por eso comparar solo
`n_sims` con MCTS open-loop no iguala necesariamente tiempo de pared.

## Observación, acción y máscaras

La interacción real sigue [`rl_env_spec.md`](rl_env_spec.md); el árbol consume
`BattleSnapshot`, no Obs 1024D. Solo entran joints legales. Mega se conserva y
Tera id 22 permanece enmascarada. La recompensa de hoja es Candidate 1 para que
la matriz mantenga interpretación cero-suma.

## Búsqueda e inferencia

No hay entrenamiento ni checkpoint neural. La CLI implementada es un smoke; el
presupuesto real se configura en `MCTSPuctNashConfig`. S2, ranking Top-K, rollout
y solver permanecen fijos durante evaluación.

```bat
python -m showdown.rl.mcts_puct_nash --seed 0 --sims 8 --smoke
```

Subir simulaciones mejora el muestreo de celdas, pero cada nodo también requiere
resolver Fictitious Play. Para comparar con MCTS open-loop conviene reportar
tanto defaults nativos como costo equivalente en llamadas a `step_sim`.

### Mini-ejemplo

Para \(\hat A=\begin{bmatrix}1&-1\\-1&1\end{bmatrix}\), escoger siempre una
fila permite al rival elegir la columna perdedora. La mezcla aproximada
\(\sigma=(0.5,0.5)\) hace indiferente al rival y produce valor cercano a cero.
Con pocas muestras, una celda ruidosa puede deformar la mezcla; más sims llenan
la matriz, mientras más iteraciones FP estabilizan el solve dado ese contenido.

## Hiperparámetros

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| \(N_{\rm sim}\) | `50` (`8` smoke) | Presupuesto por decisión | Más llena mejor la matriz y cuesta más |
| \(c\) PUCT | `1.5` | Explorar celdas | Mayor dispersa visitas |
| \(K\) | `16` | Acotar matriz | Mayor cubre más y multiplica celdas |
| FP interno | `200` | Valor de nodos | Más iteraciones estabilizan backup |
| FP raíz | `500` | Mezcla jugada | Reduce error de la decisión final |
| Ventana FP | `50` | Promedio final | Suaviza oscilaciones |
| \(\varepsilon_{\rm opt}\) | `1e-6` | Empates óptimos | Controla selección max-entropía |
| Prior | uniforme por celda | Inicio neutral | No usa red previa |
| Samples/visita | `1` | Actualización online | Más reduciría ruido con mayor costo |
| Max turns | `40` | Acotar simulación | Limita profundidad |
| Descuento | `1` | Backup Candidate 1 | Conserva escala cero-suma |
| RNG | seed+turn+sim | Reproducibilidad | Controla azar de S2 y rollout |
| Acción final | `argmax σ` | Convertir mezcla en orden | Desempata por joint id menor |

## Ilegales, máscaras y fallos

- Top-K se calcula después de filtrar legalidad; nunca rehabilita una ilegal.
- Tera 22 permanece fuera y Mega depende del snapshot.
- Si \(I\) queda vacío, se retorna `(pass, pass)`; matrices vacías valen cero.
- L1 del adaptador live puede aplicar fallback seguro y observable.
- L2 invalida la decisión por desync; no se trata como muestra de Nash.
- Una excepción de simulación propaga cero por la ruta y suma `failures`.
- FP aproximado no garantiza equilibrio exacto; su error es error de búsqueda.
- L3/crash aborta de acuerdo con `rl_env_spec`.

## Training frente a inferencia

Todo ocurre en inferencia: construir árbol, simular, resolver matrices y elegir
la mezcla. “Presupuesto” significa simulaciones y solver por turno, no épocas.
No se debe presentar el smoke como entrenamiento ni como checkpoint aprendido.

En desarrollo se evalúan exploitabilidad aproximada, estabilidad de
\(\sigma\), cobertura de celdas, fidelidad S2↔S1 y latencia. La **Conclusion
eval** congela K, simulaciones, FP, ranking, seed policy y rollout; agrega
resultados con el protocolo común y no aprende de partidas previas.

## Team Preview y Conclusión

**No hay Team Preview en el MDP de combate.** El pipeline familiar es
**`team_preview_nash` → MCTS–PUCT/Nash de combate**.
Aunque ambos dicen Nash, resuelven problemas distintos: uno selecciona el equipo
en preview; el otro resuelve matrices locales dentro del árbol de combate.
Detalles del round-robin: [`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

## Mapa de código

| Archivo | Responsabilidad |
|---|---|
| `showdown/rl/mcts_puct_nash/agent.py` | configuración y acción |
| `showdown/rl/mcts_puct_nash/search.py` | SM-MCTS y backup Nash |
| `showdown/rl/mcts_puct_nash/tree.py` | matrices, celdas y PUCT |
| `showdown/rl/mcts_puct_nash/nash.py` | Fictitious Play, ventana y epsilon |
| `showdown/rl/mcts_puct_nash/topk.py` | ranking Top-K estable |
| `showdown/rl/mcts_puct/simulator.py` | transición S2 compartida |
| `showdown/rl/mcts_puct_nash/player.py` | adaptador validado |
| `showdown/battle_bots/mcts_puct_nash/pokeenv_player.py` | integración live |

## Limitaciones

- Top-K puede excluir una respuesta decisiva con score greedy bajo.
- Resolver muchas matrices reduce el número asequible de simulaciones.
- Fictitious Play es aproximado; su error entra en cada backup.
- S2 y rollout greedy conservan sesgo de modelo.
- No realiza belief search de información imperfecta estilo póker.
- Tera está fuera del formato.

## Lectura de telemetría

Además de sims, nodos, fallos y profundidad, conviene inspeccionar la entropía
de \(\sigma\) y cuántas celdas de la raíz recibieron visitas. Una mezcla casi
uniforme puede representar simetría real o falta de presupuesto; la matriz
\(\hat A\) y sus conteos permiten distinguir ambos casos.

El valor Nash estimado no es una probabilidad de victoria calibrada. Es retorno
Candidate 1 bajo S2, Top-K y FP aproximado. Comparar configuraciones requiere
congelar K y contabilizar tanto `step_sim` como iteraciones del solver. Un
`failures>0` se reporta por separado y nunca se celebra como regularización.

También se conserva la seed y el turno para reproducir el RNG de cada sim.
La masa de \(\sigma\) se interpreta junto a los ids Top-K efectivamente
incluidos, no contra el espacio joint completo.

## Changelog

| Versión | Fecha | Notas |
|---|---|---|
| **1.3** | 2026-10-11 | Matemática matricial, presupuesto, fallos, evaluación, code map y ejemplo. |
| **1.2** | 2026-10-11 | Guía dual-audience, search-time budget y composición Nash-preview. |
| **1.1** | 2026-10-10 | Lectura pedagógica. |
| **1.0** | 2026-10-10 | Contrato SM-MCTS B1. |
