# MCTS + PUCT Agent Spec — planificación open-loop (VGC Gen 9)

| Campo | Valor |
|---|---|
| **Versión** | **1.3** (alineada con [`rl_env_spec.md`](rl_env_spec.md) v1.15) |
| **Estado** | FROZEN; `showdown/rl/mcts_puct/` y `battle_bots/mcts_puct/` |
| **Training** | **No**: búsqueda pura en cada decisión; no hay gradientes ni checkpoint de combate |
| **CLI** | `python -m showdown.rl.mcts_puct --seed 0 --sims 16 --smoke` |
| **Dependencias** | NumPy; no requiere Torch |

## Por qué existe

MCTS cambia costo de entrenamiento por costo de decisión. Desde el estado actual
simula futuros con un modelo local S2, equilibra explorar acciones nuevas y
explotar acciones prometedoras mediante PUCT, y juega la raíz más visitada. Es
un baseline de planificación interpretable frente a agentes que comprimen la
experiencia en pesos.

## Glosario

| Término | Significado |
|---|---|
| Simulación | Trayectoria imaginada dentro de S2 |
| Open-loop | El rival se samplea; no tiene un nodo estratégico propio |
| PUCT | Score \(Q\) más bono de prior/visitas |
| Prior \(P\) | Distribución inicial; aquí uniforme sobre legales |
| Rollout | Política `greedy_joint_snapshot` desde la hoja |
| Backup | Acumular retorno en aristas visitadas |
| Presupuesto | Número de simulaciones por decisión, no pasos de training |

## Algoritmo paso a paso

1. Se convierte la batalla sincronizada en `BattleSnapshot`.
2. La raíz expande todas las acciones joint legales.
3. PUCT selecciona
   \[
   \arg\max_a Q(s,a)+c_{\rm puct}P(s,a)\frac{\sqrt{N(s)}}{1+N(s,a)}.
   \]
4. Se samplea la acción rival uniformemente entre sus legales.
5. `step_sim` avanza S2 con azar Monte Carlo.
6. En una hoja nueva, ambos lados usan `greedy_joint_snapshot` hasta terminal o
   cap; el resultado es Candidate 1 \(z\in\{-1,0,1\}\).
7. Se propaga \(z\) sin invertir signo: \(N{+}{=}1,W{+}{=}z,Q=W/N\).
8. Tras agotar `n_sims`, se juega `argmax` de visitas de la raíz.

No aprende pesos entre partidas. El árbol se vuelve a construir para la
decisión actual.

### Intuición de PUCT

\(Q=W/N\) resume la evidencia observada para una arista. El bono
\[
U(s,a)=c_{\rm puct}P(s,a)\frac{\sqrt{N(s)}}{1+N(s,a)}
\]
es grande para aristas poco visitadas. Al principio fuerza cobertura; después
\(Q\) domina. Con prior uniforme, el conocimiento proviene del rollout greedy
y de S2, no de una red previa.

Open-loop no conserva una rama estratégica rival separada. Al crear un hijo se
samplea una joint rival y ese estado queda asociado a la arista propia. Es
barato, pero mezcla la incertidumbre rival dentro del retorno.

### Ciclo de vida del presupuesto

1. Cada turno real crea una raíz y expande sus joints legales.
2. La simulación `k` recibe RNG derivado de `seed|turn|k`.
3. Se baja con PUCT; las aristas no probadas se consumen primero.
4. Al crear un hijo, S2 samplea rival, accuracy y daño.
5. Desde la hoja, ambos bandos ejecutan rollout greedy.
6. Candidate 1 terminal/cap se propaga por el camino.
7. Tras \(N_{\rm sim}\), la raíz elige por visitas y luego Q.
8. El árbol se descarta; el próximo turno usa un snapshot sincronizado nuevo.

La telemetría cuenta sims, nodos, fallos y profundidad media. Una excepción de
simulación aporta cero y se registra; no produce directamente una orden ilegal.

## Observación, acción y máscaras

La batalla real usa el contrato de [`rl_env_spec.md`](rl_env_spec.md). La
búsqueda opera sobre snapshots, no sobre el vector neural 1024D. Sus aristas son
acciones joint \(a=24a_1+a_2\) filtradas por legalidad. Mega puede ser legal;
Tera (id 22) está siempre apagada bajo Reg M-C. Candidate 1 conserva el juego
cero-suma y asigna cero al empate/cap.

## Búsqueda e inferencia

No existe un comando de entrenamiento: la CLI actual solo acepta `--smoke`.
“Mejorar” MCTS significa aumentar o ajustar el presupuesto de búsqueda y medir
latencia/fidelidad, no hacer backpropagation.

```bat
python -m showdown.rl.mcts_puct --seed 0 --sims 16 --smoke
```

En una ejecución final, el player usa el presupuesto configurado en cada turno.
El prior, el modelo rival y la política de rollout quedan congelados; el único
estado efímero son estadísticas del árbol. S1/Showdown live sirve para evaluar
fidelidad, no para entrenar S2.

### Ejemplo de selección

Con \(Q_A=0.30,N_A=20\) y \(Q_B=0.10,N_B=2\), B puede ganar temporalmente por
su bono de exploración. Al recibir visitas, ese bono cae. La decisión final usa
el mayor \(N\), no el Q aislado, y resiste mejor una muestra afortunada.

## Hiperparámetros

| Parámetro | Valor | Propósito | Cómo influye |
|---|---:|---|---|
| \(N_{\rm sim}\) | `200` (`16` smoke) | Presupuesto por decisión | Más reduce varianza y aumenta latencia |
| \(c_{\rm puct}\) | `1.5` | Exploración | Mayor visita acciones menos probadas |
| Prior | uniforme legal | Iniciar aristas | No aporta conocimiento neural |
| Rival en árbol | uniforme legal | Resolver transición | Simplifica, pero no anticipa estrategia |
| Rollout | greedy joint ambos | Valorar hojas | Su sesgo define gran parte del valor |
| Max turns | `40` | Acotar rollout | Caps devuelven señal no ganadora |
| Descuento | `1` | Backup terminal | Mantiene escala Candidate 1 |
| Dirichlet | off | Ruido en raíz | Inferencia determinista dado RNG |
| RNG | seed+turn+sim | Reproducibilidad | Controla rival, accuracy y rolls |
| Acción final | máximo \(N\), desempate \(Q\) | Decidir raíz | Favorece evidencia acumulada |
| Expansión | primera no probada | Cobertura estable | Prueba legales antes de reusar |

## Ilegales, máscaras y fallos

- Cada nodo genera joints legales desde su snapshot; Tera 22 no aparece.
- Mega solo entra cuando el estado simulado la permite.
- Sin aristas, se devuelve `(pass, pass)` como defensa.
- L1 permite el fallback seguro y observable del player live.
- L2 marca desincronización no reparable; la decisión no cuenta como válida.
- Un fallo interno de simulación suma retorno cero y telemetría.
- Más simulaciones no corrigen un sesgo sistemático de S2.
- L3/crash aborta según el contrato común.

## Training frente a inferencia

No hay pesos, dataset ni artefacto de training. Toda computación útil ocurre en
inferencia. Cambiar `--sims` cambia costo y calidad en el momento; no produce un
modelo reutilizable. La reproducibilidad exige congelar seed, simulador, rollout
y presupuesto.

Durante desarrollo se evalúan Candidate 1, fidelidad S2↔S1, latencia,
profundidad y tasa de fallos. No existen loss ni checkpoint. La **Conclusion
eval** congela presupuesto, simulador, rollout y seed policy, usa el protocolo
común y no adapta el árbol con partidas anteriores.

## Team Preview y Conclusión

**No hay Team Preview en el MDP de combate.** El pipeline familiar es
**`team_preview_nash` → MCTS+PUCT de combate**. El Nash
de preview resuelve la selección de cuatro Pokémon; después MCTS planifica cada
turno. No comparten árbol ni parámetros. El presupuesto debe quedar congelado
antes del round-robin. Véase
[`conclusion_eval_protocol.md`](conclusion_eval_protocol.md).

## Mapa de código

| Archivo | Responsabilidad |
|---|---|
| `showdown/rl/mcts_puct/agent.py` | configuración y API |
| `showdown/rl/mcts_puct/search.py` | select, expand, rollout y backup |
| `showdown/rl/mcts_puct/tree.py` | nodos, aristas y PUCT |
| `showdown/rl/mcts_puct/simulator.py` | S2, rival y Candidate 1 |
| `showdown/rl/mcts_puct/greedy_snapshot.py` | política de rollout |
| `showdown/rl/mcts_puct/player.py` | adaptador validado |
| `showdown/battle_bots/mcts_puct/pokeenv_player.py` | integración live |
| `showdown/rl/snapshot.py` | estado compacto |

## Limitaciones

- El rival uniforme open-loop subestima respuestas estratégicas.
- S2 es aproximado: optimizarlo puede no optimizar Showdown real.
- El costo crece aproximadamente con simulaciones y longitud de rollout.
- El prior uniforme desperdicia conocimiento previo.
- La política greedy de hoja introduce sesgo.
- No modela Tera porque el formato la desactiva.

## Lectura de telemetría

`sims` debe coincidir con el presupuesto incluso cuando alguna trayectoria
falla. `nodes` aproxima cuánta expansión efectiva produjo el presupuesto;
`avg_depth` distingue búsqueda superficial con muchas ramas de trayectorias más
largas. `failures>0` exige investigar S2: no es evidencia de una posición
equilibrada. Para comparar configuraciones se reportan también tiempo por
decisión y llamadas a `step_sim`.

La acción más visitada puede tener Q menor que otra poco probada. Eso no es una
contradicción: visitas reflejan el proceso acumulado de selección, y Q de una
muestra escasa tiene alta varianza. La inspección conjunta de \(N,Q,P\) hace
el baseline interpretable sin convertir telemetría en información del agente.

El reporte debe conservar también la seed global y el turno de la decisión.
Así una trayectoria anómala puede reproducirse sin reutilizar el árbol.

## Changelog

| Versión | Fecha | Notas |
|---|---|---|
| **1.3** | 2026-10-11 | PUCT, lifecycle, fallos, evaluación, code map y ejemplo ampliados. |
| **1.2** | 2026-10-11 | Explicación dual-audience, presupuesto search-time y pairing Nash-preview. |
| **1.1** | 2026-10-10 | Lectura pedagógica. |
| **1.0** | 2026-10-10 | Contrato open-loop PUCT. |
