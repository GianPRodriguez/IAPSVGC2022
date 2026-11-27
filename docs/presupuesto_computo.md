# Presupuesto de cómputo por agente (E–K)

**Objetivo (decisión del usuario, 27-11-2026):** que las **horas de cola totales** de cada agente sean similares, para
que las comparaciones de la tesis no favorezcan al que recibió más cómputo. Este documento registra cuánto lleva cada
uno y cuánto le falta para equilibrarse. Se actualiza cuando termina una cola.

## Cómo se mide

Desde `~/iapsvgc_runs` en phantom (medido el 27-11-2026; no incluye `~/iapsvgc_scratch` ni lo hecho en la laptop
antes del 16-11-2026):

- **Entrenamiento puro:** suma de `elapsed` de los `reporte.txt`, clasificados por algoritmo.
- **Horas de cola:** de la primera a la última marca de tiempo de cada `queue.log` (entrenamiento, evaluaciones y
  reanudaciones). Es tiempo de reloj por cola; algunas colas corrieron a la vez, así que no es tiempo exclusivo
  de la máquina.
- **Horas compartidas:** las colas que sirven a varios agentes a la vez se reparten **a partes iguales** entre ellos.
  Son los pilotos y ajustes de E y F (DDQN y PPO mezclados), P5/P1b, el preview v2.1b (lo usan E y F) y las colas
  de la familia QD (I, J y K).

## Tabla (27-11-2026)

| Agente | Entrenamiento puro | Horas de cola propias | Parte de las compartidas | **Total de cola** |
|---|---|---|---|---|
| **E** (preview PPO + DDQN) | 43,0 h (66 entrenamientos DDQN; kick-off DDQN v1: 4,7 h) | 9,2 h (kick-off DDQN v1) | 79,0 h (½ de 157,9 h) | **88,2 h** |
| **F** (preview PPO + PPO) | 36,2 h (40 entrenamientos PPO; kick-off PPO v2: 7,0 h) | 22,2 h (PPO v1 5,3 h, candidatos de PPO v2 8,2 h, kick-off PPO v2 8,7 h) | 79,0 h (½ de 157,9 h) | **101,2 h** |
| **G** (MCTS + PUCT) | — (no entrena: busca en cada decisión) | 61,0 h (v2–v7, confirmaciones v3–v5, preview por búsqueda 13,7 h, Nash 4,5 h) | — | **61,0 h** |
| **H** (preview Nash + MCTS-Nash) | — (no entrena) | 78,8 h (H 2.0 16,2 h, H 2.1 23,8 h, paso A 13,0 h, H 2.2 12,6 h y su confirmación 12,9 h) | — | **78,8 h** |
| **I** (preview QD-PGA + QD-PPO) | exploración I 2.0: 3 h 04 min | — | 18,8 h (⅓ de 56,5 h) | **18,8 h** |
| **J** (preview QD-PGA + QD-PGA) | exploración J 2.0: 2 h 46 min | — | 18,8 h (⅓ de 56,5 h) | **18,8 h** |
| **K** (preview QD-PGA + QD-PPGA) | exploración K 2.0: 1 h 33 min | — | 18,8 h (⅓ de 56,5 h) | **18,8 h** |

Horas compartidas de E y F (157,9 h): pilotos y ajustes `kickoff`…`kickoff22` 141,1 h, P5/P1b (ablación, velocidad,
`p5full`, `p1_div`, `p1b`) 13,7 h y preview v2.1b 3,1 h. Horas compartidas de I/J/K (56,5 h): I 2.0 16,1 h (incluye
el control), J/K 2.0 19,6 h, F1 0,8 h, F2 0,9 h, F3 8,8 h y F4 10,3 h (**F4 y F2b seguían en curso**: su total crecerá).

## Desequilibrio actual y lo que falta

Tomando como referencia al que más lleva (F, 101,2 h):

| Agente | Total | Falta para igualar a F |
|---|---|---|
| F | 101,2 h | — |
| E | 88,2 h | 13,0 h |
| H | 78,8 h | 22,4 h |
| G | 61,0 h | 40,2 h |
| I | 18,8 h | 82,4 h |
| J | 18,8 h | 82,4 h |
| K | 18,8 h | 82,4 h |

**Consecuencias para lo que viene:**

- La cola v3 (PPO v3 / DDQN v2, protocolo §4.16; ~27–41 h estimadas) suma solo a E y F, porque P1, S1 y D1/D2
  también son de E y F. **Agranda el desequilibrio**: E y F quedarían en ~105–120 h cada uno.
- La familia QD (I, J, K) es la más rezagada aunque comparta colas; las fases F2b–F4 en curso la acercan un poco.
- G y H no entrenan: su cómputo es evaluación. Equilibrarlos significa darles más versiones o evaluaciones, no
  entrenamientos.

Este documento solo mide; qué hacer con el desequilibrio (dar más cómputo a los rezagados, limitar a E y F o
normalizar al reportar) lo decide el usuario.
