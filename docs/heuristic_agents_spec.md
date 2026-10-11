# Agentes heurísticos de referencia — índice de la escalera

| Campo | Valor |
|-------|--------|
| **Documento** | **Índice / umbrella** de baselines no aprendidos |
| **Versión** | **1.1** (alineado con `docs/rl_env_spec.md` **v1.9** §6.h) |
| **Estado** | **FROZEN** — los contratos **exclusivos** por agente son las fuentes de verdad |
| **Rol** | Glosario + escalera experimental + punteros; **no** duplica números congelados |

> **Para quien lee sin formación en IA.** Antes de entrenar redes o planificadores
> hace falta una **escalera de rivales fijos**: programas con reglas a mano y
> calculadora de daño. Cada peldaño tiene su propio contrato (como PPO o DDQN).
> Este archivo solo orienta; el detalle auditable está en los cuatro specs exclusivos.

---

## Specs exclusivos (fuente de verdad)

| Id | Spec | Code root |
|----|------|-----------|
| `random` | [`random_spec.md`](random_spec.md) **v1.1** | `showdown/battle_bots/random/` |
| `greedy_solo` | [`greedy_solo_spec.md`](greedy_solo_spec.md) **v1.1** | `showdown/battle_bots/greedy_solo/` |
| `greedy_joint` | [`greedy_joint_spec.md`](greedy_joint_spec.md) **v1.1** | `showdown/battle_bots/greedy_joint/` |
| `expectimax_bayesian` | [`expectimax_bayesian_spec.md`](expectimax_bayesian_spec.md) **v1.1** | `showdown/battle_bots/expectimax_bayesian/` |

**Inmutabilidad:** cambiar semántica de un agente ⇒ bump del **spec exclusivo**
correspondiente **y** entrada en `rl_env_spec.md` §6.h. Este índice solo se
bumpa si cambia la escalera o el glosario.

---

## 0. Glosario mínimo

| Término | Significado en este trabajo |
|---------|-----------------------------|
| **VGC (dobles)** | Formato 2 vs 2: una acción por activo cada turno |
| **Team Preview** | Elegir 4 de 6 y el orden de salida |
| **Acción legal** | Lo que el request Showdown permite (PP, disabled, trapped…) |
| **Heurística** | Regla diseñada a mano; no red entrenada |
| **Greedy** | Maximiza un puntaje del **turno actual** (\(T=1\)) |
| **Factored / Solo** | Cada activo elige sin producto \(A_1\times A_2\) |
| **Joint** | Evalúa pares \((a_1,a_2)\) (sinergia de turno) |
| **Expectimax** | Promedia utilidades bajo incertidumbre |
| **Bayesiano (aquí)** | Promedio sobre hipótesis (SP / mundos) del set rival |
| **Baseline** | Rival fijo para reportar WR |
| **1-ply** | Una capa de decisión; no MCTS multi-turno |
| **KO garantizado** | El calc implica KO incluso en roll mínimo (según motor) |
| **OTS** | Open Team Sheet; harness validado sincroniza Battle ↔ protocolo |

**Analogía.** Random = dado legal. Solo = cada pieza captura lo mejor local.
Joint = mejor acorde del compás. Expectimax = re-evalúa candidatos bajo incertidumbre, anclado a Joint.

---

## 1. Escalera experimental

```
random  <  greedy_solo  <  greedy_joint  ≲  expectimax_bayesian
```

| Uso | Contrato (detalle en specs RL / exclusivos) |
|-----|-----------------------------------------------|
| Curriculum DDQN / PPO | `random` → `greedy_solo` → `greedy_joint` (+ expectimax opcional) |
| Suite offline WR | ≥ `random`, `greedy_joint`, `expectimax_bayesian`; **200 BO1** |
| Mosaic / QD | Mismos ids como foes fijos |
| Robust E2E | `--player-a` / `--player-b` |
| BC | Replays `expectimax_bayesian` / `greedy_joint` |

Compartido por los cuatro: Gen 9 VGC; sin `torch`; no optimizan Cand. 1/2;
acciones ilegales prohibidas; servidor local con `--no-security` en harnesses.

---

## 2. Validación

Wrappers en `showdown/inference_sim/validated_greedy_player.py`
(+ player Expectimax): sync Battle y soundness OTS cuando actúan como player A.

---

## Changelog

| Versión | Fecha | Notas |
|---------|-------|-------|
| **1.1** | 2026-10-10 | Umbrella: specs exclusivos por agente; este doc deja de ser fuente de números congelados. |
| **1.0** | 2026-10-10 | Contrato monolítico inicial (superseded como SoT por 1.1 + exclusivos). |

---

*Fin del índice heurístico v1.1.*
