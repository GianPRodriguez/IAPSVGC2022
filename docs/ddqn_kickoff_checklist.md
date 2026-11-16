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
| 2 | Paridad entre entrenamiento e inferencia: el entorno arma el snapshot sin `stat_beliefs` y el jugador con ellas | **hecho**: obs idéntica en 902 de 902 pasos reales y máscara de entrenamiento = máscara de inferencia. Con `OBS_STAT_INFERENCE` apagado (por defecto) las creencias no entran en la obs; si se enciende, hay que hacerlo en entrenamiento e inferencia a la vez | live_validation, 9b29322 |

## B. Validación en vivo

Corrida corta contra Showdown real, con `PS_RL_STRICT=1`, que mida: tasa de fallback,
historial frente al protocolo, alineación movimiento i ↔ acción i, máscara nunca vacía,
paridad del punto 2, simetrías con batallas reales y costo en ms de obs y codificador.

Herramienta: `python -m showdown.rl.live_validation` (mismo camino que el entrenamiento:
`VGCEnv` + `PokeEnvBattleBackend` + `_joint_masks`, acciones legales al azar contra random,
solo, joint y expectimax) + `run_robust_e2e --player-a ddqn` en modo estricto (camino de
inferencia validado).

**Estado: hecho (15-11-2026).**

| Chequeo | 1.ª corrida (5e414e0: 48 episodios, 403 pasos) | 2.ª corrida, con el arreglo (9b29322: 100 episodios, 902 pasos) |
|---|---|---|
| Máscara vacía | 0 | 0 |
| Máscara de entrenamiento ≠ inferencia | 0 | 0 |
| Paridad de la obs (punto 2) | 0 diferencias | 0 diferencias |
| Alineación movimiento k ↔ acción k | 1995/1995 | 4483/4483 |
| Historial frente al protocolo | 960/960 | 2268/2268 |
| Simetrías con combates reales (7 elementos) | 2821/2821 | 6314/6314 |
| **Fallbacks de orden** | **11/403 (2,7 %)** | **0/902** |
| Caídas (truncado por excepción) | 0 | 0 |
| Costo por paso (mediana) | snapshot + obs 1,7 ms; forward 0,4 ms (GPU) | 1,7 ms; 0,5 ms |
| Inferencia estricta: DDQN validado vs joint y vs expectimax (120 BO3) | 0 excepciones; sync e inferencia OK | 0 excepciones; sync e inferencia OK |

**Error encontrado y corregido (rl_env_spec v1.23):** los 11 fallbacks eran cambios forzados
de los dos huecos con un solo reemplazo. La máscara permitía `(pass, pass)` (Showdown exige
que entre el reemplazo) y el adaptador convertía un `pass` legal en un cambio al mismo
Pokémon que el otro hueco; en ambos casos se jugaba una orden al azar y el replay guardaba
una acción no ejecutada. Ahora la máscara sigue la regla de Showdown (`sim/side.ts`,
`forcedPasses = forzados − reemplazos`) y el `pass` legal se respeta. Un test antiguo
(`test_forced_switch_orders`) esperaba que un hueco forzado con reemplazo pudiera pasar; se
corrigió a la regla real (4ab4609).

Suite completa en el servidor tras el arreglo: 663 pasan, 1 omitido, 6 xfail, más el test
corregido.

## C. Aprendizaje

| # | Punto | Estado |
|---|---|---|
| 7 | LayerNorm después del codificador (ablación corta) | pendiente |
| 9 | Re-tunear hiperparámetros, incluido `sym_coef` ∈ {0; 0,05; 0,1; 0,3} (los `*_best_hp.json` actuales son corridas de prueba) | pendiente |
| — | Calibrar el paralelismo según el servidor (phantom: 32 hilos, 3 × RTX A5500 de 24 GB, 251 GB de RAM) | **hecho** (ver "Calibración del paralelismo" abajo). Recomendación: `--num-envs 32 --envs-per-server 4`. Hallazgo: el cuello de botella es el aprendiz, no las batallas |
| — | El aprendiz limitaba el entrenamiento a ~130–155 pasos/s frente a ~450 de las batallas | **hecho** (5ff2507, ddqn_spec 1.13): replay y aumentación por simetría en GPU (mismos lotes, verificado bit a bit) y fracciones por rival incrementales. Paso de gradiente 34,2 → 4,7 ms. Prueba A/B en entrenamiento real (32 × 4, 4 rivales): replay en CPU 153 pasos/s, **en GPU 410 pasos/s** (2,7×). Un currículo de 2M pasos baja de ~4 h a **~1,4 h** |

### Calibración del paralelismo (15-11-2026, commit db6d4de)

DDQN real con `--device cuda`, Ω con los 4 rivales (random, solo, joint, expectimax con
sorteo; pesos 1/1/1/0,35), 8k–18k pasos por configuración. "Sin aprender" = pasos 1k–5k
(antes del warm-up de 5000 pasos: solo batallas y llenado del replay); "con aprendizaje" =
desde el paso 6500 (un paso de gradiente cada 4 pasos de entorno, lote 256).

| `--num-envs` × `--envs-per-server` | Solo batallas (pasos/s) | Con aprendizaje (pasos/s) | CPU media | Reinicios |
|---|---|---|---|---|
| 8 × 4 | 193 | 99 | 21 % | 0 |
| 16 × 4 | 331 | 122 | 27 % | 0 |
| 24 × 4 | 425 | 124 | 29 % | 0 |
| 28 × 4 | 457 | 138 | 33 % | 0 |
| **32 × 4** | **462** | 127 | 31 % | 0 |
| 40 × 4 | 454 | 154 | 36 % | 0 |
| 48 × 4 | 452 | 155 | 37 % | 0 |
| 32 × 2 | 434 | 101 | 67 % | 0 |
| 32 × 6 | 249 | 100 | 96 % | 0 |

Lectura:

- **Las batallas escalan hasta ~28–32 entornos** (~450–460 pasos/s) y luego se
  estabilizan. Más de 4 entornos por servidor Showdown satura la CPU (32 × 6: 96 %) y rinde
  menos; 2 por servidor también rinde algo menos.
- **Con aprendizaje, el rendimiento queda en ~100–155 pasos/s sin importar los entornos:**
  el aprendiz (un paso de gradiente cada 4 pasos, ~30–40 por segundo) es el cuello de
  botella. La CPU total queda baja (≤ 37 %), así que el límite es el bucle principal, no el
  servidor.
- Duración estimada de un currículo de 2M pasos: ~3,6–4,5 h con el aprendiz actual; ~1,2 h
  si el aprendiz alcanzara el ritmo de las batallas.
- 0 reinicios y 0 errores de `|showteam|` en las 9 configuraciones (con el arreglo de
  nombres de especie, db6d4de). Un primer intento con el código anterior (53889fa) tenía
  7 errores `|showteam|` en 1 minuto.

## D. Metodología

| # | Punto | Estado |
|---|---|---|
| 11 | ¿Equipos de entrenamiento = equipos de evaluación? Si sí, se mediría memorización; reservar equipos solo para evaluar | **hecho** (decisión del usuario: 200 para entrenar, 100 para evaluar; rl_env_spec v1.24). Verificado: el entrenamiento usaba los mismos 300 equipos que la evaluación. Separación por grupos de clones (mismas 6 especies) y estratificada; RL y previews en `train`, Conclusión en `eval` (78 equipos), prior y mundos desde `train`. Ver `protocolo_evaluacion.md` §2b |
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
