# DDQN: checklist antes del entrenamiento oficial

Plan surgido de la revisión del agente E (14–15 de noviembre de 2026). Cada punto se
implementa en el laptop, se prueba en el servidor y se registra aquí con su commit. El
entrenamiento oficial (pipeline E: preview PPO + DDQN) empieza cuando todo lo obligatorio
está cerrado.

## A. Errores y riesgos

| # | Punto | Estado | Commit / evidencia |
|---|---|---|---|
| 1 | El backend mock congelaba la observación dentro del episodio (la caché de la obs compara por identidad y el mock mutaba el mismo snapshot). Solo afectaba a smoke tests y corridas `--mock` | **hecho**: el mock devuelve un snapshot nuevo por paso | 1627cc0; `tests/test_rl_prekickoff.py` |
| 6 | Orden inestable del resto del rival: revelar un Pokémon (orden "vistos primero") o descartar uno movía los demás bloques a mitad del combate | **hecho**: orden fijo por especie; descartados en cero en su lugar. Cambia la semántica de la obs (rl_env_spec v1.22) | 1627cc0 |
| 3 | `DDQNPlayer.choose_move` se tragaba las excepciones y jugaba la orden por defecto sin avisar | **hecho**: `decision_stats` (decisiones, excepciones por tipo, fallbacks de orden), log de las primeras 5, `PS_RL_STRICT=1` propaga | 1627cc0 |
| 2 | Paridad entre entrenamiento e inferencia: el entorno arma el snapshot sin `stat_beliefs` y el jugador con ellas | pendiente (dentro de B) | — |

## B. Validación en vivo

Corrida corta contra Showdown real, con `PS_RL_STRICT=1`, que mida: tasa de fallback,
historial frente al protocolo, alineación movimiento i ↔ acción i, máscara nunca vacía,
paridad del punto 2, simetrías con batallas reales y costo en ms de obs y codificador.

Estado: pendiente.

## C. Aprendizaje

| # | Punto | Estado |
|---|---|---|
| 7 | LayerNorm después del codificador (ablación corta) | pendiente |
| 9 | Re-tunear hiperparámetros, incluido `sym_coef` ∈ {0; 0,05; 0,1; 0,3} (los `*_best_hp.json` actuales son corridas de prueba) | pendiente |
| — | Calibrar el paralelismo según el servidor (phantom: 32 hilos, 3 × RTX A5500 de 24 GB, 251 GB de RAM): `--num-envs` 8/16/24/28/32/40/48 y `--envs-per-server` 2/4/6, midiendo pasos/s, uso de CPU, reinicios y timeouts. Se prueba también por encima de 28 (tope automático = núcleos − 4) porque los entornos pasan tiempo esperando a Showdown y a rivales lentos (expectimax, hasta 1,6 s por decisión), así que más batallas simultáneas pueden subir el rendimiento aunque haya más entornos que hilos. La GPU no es el cuello (< 1 % de uso): el aprendiz usa 1 GPU y las otras 2 sirven para trials paralelos del tuning | pendiente |

## D. Metodología

| # | Punto | Estado |
|---|---|---|
| 11 | ¿Equipos de entrenamiento = equipos de evaluación? Si sí, se mediría memorización; reservar equipos solo para evaluar | sin verificar; decisión del usuario |
| 12 | Ablación codificador × simetría (≥ 3 semillas, evaluación pareada) | pendiente |
| — | Suite completa en el servidor con los commits desde dbb188b | **hecho**: 651 pasan, 1 falla, 1 omitido, 6 xfail en 1627cc0. La falla era un test con el orden anterior del rival; actualizado al contrato v1.22 en 925b47a y verificado en el servidor |

## Prerrequisitos del pipeline E

- Preview: puntos 4 (90 formaciones en vez de 360) y 5 (aumentación por permutación),
  `run_preview_value_probe` y entrenamiento del preview PPO. Sin checkpoint de preview,
  DDQN usa `/team 1234`.

## Ya resuelto en otras etapas

- Evaluación con ε = 0 y `DDQN_REQUIRED` (791e5b2, ddqn_spec 1.8).
- Rivales de entrenamiento validados (rl_env_spec v1.19) y con sorteo entre casi-empates
  (v1.21; evaluación determinista). Ver `docs/diversidad_baselines.md`.
