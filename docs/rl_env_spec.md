# VGC Gen 9 RL Benchmark: Architecture Spec Sheet

| Field | Value |
|-------|--------|
| **Version** | **v1.1** |
| **Status** | **FROZEN / IMMUTABLE** |
| **Target stack** | PyTorch + Gymnasium |
| **Format** | Gen 9 VGC (Regulation M-C, Open Team Sheets) |
| **Scope of this doc** | Contract for `VGCEnv` / agents / training. |
| **Implementation** | `showdown/rl/` (`VGCEnv`, reward, obs, action, BD) — Spec v1.1 |

> **Immutability rule:** Do not silently change Reward / Action / Obs / BD semantics mid-training.
> Any change requires a new version (`v1.2+`) and an explicit changelog entry at the bottom.

---

## 1. Core Architecture

| Item | Contract |
|------|----------|
| Environment | `VGCEnv(gymnasium.Env)` — async wrapper over poke-env |
| Execution | Optimized for multi-env workers; **local heuristic caching**; **no blocking Node.js / `@smogon/calc` API calls inside `step()`** |
| Episodic MDP | **1 Episode = 1 Game (BO1)** |
| Cross-game memory | **None** (BO3 is evaluation/ladder only, not the training MDP) |

### Terminal vs Truncated (Gymnasium)

| Event | `terminated` | `truncated` | \(R_t\) |
|-------|----------------|-------------|-----------|
| Win / Loss / Tie (normal end) | `True` | `False` | Per RewardSpec candidate |
| Crash / harness timeout / L3 desync | `False` | `True` | `0.0` (PPO may bootstrap \(V(s')\)) |

---

## 2. ActionSpec & Masking

### Structure

- **Multi-agent auto-regressive double slot:** \(\text{Slot 1} \rightarrow \text{Slot 2}\)
- Slot 1 choice updates the legal mask for Slot 2 (e.g. switch conflicts, shared targets).

### Masking protocol

1. Network outputs raw logits \(\mathbf{z}\).
2. Invalid-action mask \(M \in \{0,1\}^{|A|}\) (1 = legal).
3. Softmax uses:

\[
\mathrm{Softmax}\big(\mathbf{z} + (1 - M)\cdot(-10^{9})\big)
\]

Source of truth for legality: poke-env / PS `valid_orders` (or equivalent request-derived legal set).

### Fallback protocol (cascaded)

| Level | Trigger | Behavior |
|-------|---------|----------|
| **L1 Mathematical** | Illegal logit before sample | Masking; expected coverage \(>99\%\) |
| **L2 Desync / engine mismatch** | \(a_t \notin \texttt{valid\_orders}\) | Execute `valid_orders[0]`; \(R_t = 0.0\); set `info['skip_buffer'] = True`; increment `illegal_action_fallbacks` |
| **L3 Crash / timeout** | Env/server failure | `truncated=True`, `terminated=False`, \(R_t = 0.0\) |

**L2 buffer rule (mandatory):** Rollout collectors for PPO / BDQN **must drop** transitions with `skip_buffer=True`. Do not train on L2 fallbacks.

---

## 3. ObsSpec (1024D Dense Vector)

### Canonical party ordering (strict)

```
[Self_Active_1, Self_Active_2, Self_Bench_1, Self_Bench_2,
 Foe_Active_1,  Foe_Active_2,  Foe_Bench_1,  Foe_Bench_2]
```

### Segments

| Segment | Dims | Encoding |
|---------|------|----------|
| Embeddings & species | 128D | Species IDs / types. **Illusion:** observation uses **OTS ground-truth species**; **action masking / targeting** follows Showdown rules until Illusion is busted |
| Belief state | 256D | Summarized EV/IV / Spe-tier belief; item/ability as `unknown \| suspected \| confirmed` bits where applicable. **Foe bench (OTS):** team-preview priors for HP/status until combat reveal; unrevealed fields → zeros / priors as specified by encoder |
| Active field & physics | 128D | One-hot weather/terrain; Trick Room, Tailwind, screens, gravity; turn counters normalized |
| Active stats & volatiles | 128D | Stat stages in \([-1.0, 1.0]\); Yawn countdowns; Taunt; Disable; Substitute flags |
| Move matrix & damage | 384D | Priority, terrain modifiers, grounded flags. **Damage:** local Python heuristic cache only — \(\mathrm{BasePower} \times \mathrm{STAB} \times \mathrm{Effectiveness} \times \mathrm{Weather}\). Failsafe `0.0`. **No Node.js in `step()`** |

**Total:** \(128 + 256 + 128 + 128 + 384 = 1024\).

---

## 4. RewardSpec (Dual-Candidate Architecture)

### Candidate 1 — Sparse (MCTS / Self-Play evaluation / QD fitness)

\[
R_t =
\begin{cases}
+1.0 & \text{terminal win} \\
-1.0 & \text{terminal loss} \\
0.0 & \text{tie or non-terminal turn}
\end{cases}
\]

- **Tie penalty:** strictly \(0.0\) (preserve zero-sum bounds for Nash / MCTS).

### Candidate 2 — Dense soft (PPO / BDQN / bootstrap)

\[
R_t =
1.0\,\Delta\mathrm{Win}
- 1.0\,\Delta\mathrm{Loss}
+ 0.15\,\Delta\mathrm{Faints}_{\mathrm{opp}}
- 0.15\,\Delta\mathrm{Faints}_{\mathrm{self}}
+ 0.05\,\Delta\mathrm{HP\%}_{\mathrm{opp\_total}}
\]

| Rule | Contract |
|------|----------|
| \(\Delta\mathrm{HP\%}_{\mathrm{opp\_total}}\) | Entire **4-mon** opposing team |
| Residual damage | Sandstorm, Poison, Burn, Life Orb on foe, etc. **DO** count toward HP% / faint deltas |
| Substitute | Damage to Substitute **does not** count toward HP%; no magic bonus for breaking Sub |
| Tie (terminal) | \(R_t = -0.25\) (anti-stall heuristic for Candidate 2 only) |
| Training clip | \(R_t \in [-1.30, +1.40]\) — clipping threshold derived from formula extremes; not a physical law |

### Episodic framing

- Train / credit assignment: **one game**.
- BO3: out of scope for this MDP (may be used later for ladder eval only).

---

## 5. BDSpec (Behavioral Descriptors for QD)

| Item | Contract |
|------|----------|
| Archive entry filter | Episode length \(\ge 4\) turns |
| BD₁ | **Aggressiveness vs setup ratio** (style proxy; **not** quality): \(\dfrac{\text{Offensive actions}}{\text{Offensive} + \text{Status/Protect/Setup}}\) |
| BD₂ | **Hard switch rate:** \(\dfrac{\text{Hard switches}}{\text{Total turns}}\) |

**Hard switch (BD₂) includes only:** explicit neural `SWITCH` actions.

**Hard switch excludes:**

- Faint replacements
- Forced switches (Roar / Whirlwind / etc.)
- Pivot moves (U-turn, Volt Switch, …) — these count under **BD₁ offensive** when applicable

BD values clipped / normalized to \([0, 1]\).

---

## 6. Algorithmic Matrix

| Algorithm | Active RewardSpec | Architecture / Masking | BDSpec |
|-----------|-------------------|------------------------|--------|
| 1. Masked PPO | Cand. 2 → Cand. 1 | Auto-regressive Slot1→Slot2. **Main path** | Off (telemetry only) |
| 2. BDQN | Cand. 2 | Branching DQN (**experimental**); \(Q(s,a_{\mathrm{invalid}})=-\infty\) | Off |
| 3. MCTS + PUCT | Cand. 1 | Tree expansion over legal actions only | Off |
| 4. MCTS + Nash | Cand. 1 | Legal payoff matrix \(N\times M\); payoff = **sparse terminal** | Off |
| 5. QD-PPO | Cand. 2 (PPO) / Cand. 1 (fitness) | Auto-regressive logit masking | **ACTIVE** |
| 6. QD-PGA-MAP-Elites | Cand. 2 (actor) / Cand. 1 (fitness) | Auto-regressive logit masking | **ACTIVE** |
| 7. QD-PPGA | Cand. 2 (gradients) / Cand. 1 (fitness) | Auto-regressive logit masking | **ACTIVE** |

**Note:** Independent Q-Learning (IQL) is an **ablation only** (breaks Slot1/Slot2 coordination). Prefer centralized BDQN or auto-regressive PPO.

---

## 7. Curriculum & Telemetry

### Curriculum pipeline

1. **Behavioral Cloning (BC):** Replays from internal `expectimax_bayesian` / `greedy_joint` (and optionally filtered valid human matches). **No Smogon ELO dependency.** Prefer games that completed cleanly (`sync_ok`, no timeout/reject) when sourced from the robust harness.
2. **PPO bootstrap:** Train vs heuristic bots with **Reward Candidate 2**.
3. **Self-play:** Progressive decay toward **Candidate 1**.

### Self-play symmetry

- \(R_t\) is always computed from the **`player_id` of the current step**.
- Opponent perspective: \(R_{P2} = -R_{P1}\) under the same sparse/dense definition for that training phase.

### Anti-hack telemetry (TensorBoard / W&B)

| Metric | Meaning |
|--------|---------|
| `useless_switches` | A→B→A ping-pong hard switches |
| `protect_spam` | \>2 consecutive Protect attempts |
| `illegal_action_fallbacks` | Rate of L2 fallback hits |
| `mean_game_length` | Early stall / collapse warning |

These metrics **must not** enter \(R_t\).

### Evaluation suite (mandatory)

Fixed opponents for reported WR (not only self-play): at least `random`, `greedy_joint`, `expectimax_bayesian` (or frozen checkpoints). QD fitness uses Candidate 1 against a fixed eval suite / archive protocol.

---

## 8. Non-goals (explicit)

- Expectimax live \(V(s)\) as Ng potential \(\Phi(s)\) inside \(R_t\) — **forbidden** (stochastic / anytime / \(\pi_{\mathrm{opp}}\)-dependent).
- Node `@smogon/calc` inside env `step()` — **forbidden**.
- Magical per-move bonuses (Tailwind, Fake Out, Sub break, etc.) inside \(R_t\) — **forbidden**.
- Treating harness truncate as Loss — **forbidden**.

---

## Changelog

| Version | Date | Notes |
|---------|------|-------|
| **v1.1** | 2026-09-15 | Frozen unified sheet: Action/Obs/Reward/BD/Matrix + curriculum, L2 `skip_buffer`, local damage heuristic, hard-switch BD₂, BC from internal bots, Gymnasium terminated/truncated |
| v1.0 | — | Drafts (Gemini + Cursor triangulation); superseded by v1.1 |

---

*End of frozen contract v1.1.*

## Implementation map

| Spec section | Module |
|--------------|--------|
| ActionSpec / masking | `showdown/rl/action_space.py` |
| ObsSpec 1024D | `showdown/rl/observation.py` |
| RewardSpec C1/C2 | `showdown/rl/reward.py` |
| BDSpec | `showdown/rl/behavioral.py` |
| Damage heuristic (no Node) | `showdown/rl/damage_heuristic.py` |
| Gymnasium env | `showdown/rl/env.py` (`VGCEnv`, `MockBattleBackend`) |
| Live poke-env backend | `showdown/rl/pokeenv_backend.py` (`PokeEnvBattleBackend`) + `pe_adapter.py` |
| Tests | `tests/test_rl_env_spec.py`, `tests/test_rl_pe_adapter.py` |

`MockBattleBackend` covers offline CI. Live: `VGCEnv(backend=PokeEnvBattleBackend())` with local PS (`node pokemon-showdown start --no-security`).
