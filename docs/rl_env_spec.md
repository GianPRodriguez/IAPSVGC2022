# VGC Gen 9 RL Benchmark: Architecture Spec Sheet

| Field | Value |
|-------|--------|
| **Version** | **v1.2** |
| **Status** | **FROZEN / IMMUTABLE** |
| **Target stack** | PyTorch + Gymnasium |
| **Format** | Gen 9 VGC (Regulation M-C, Open Team Sheets) |
| **Scope of this doc** | Contract for `VGCEnv` / agents / training. |
| **Implementation** | `showdown/rl/` (`VGCEnv`, reward, obs, action, BD, DDQN) — Spec v1.2 |

> **Immutability rule:** Do not silently change Reward / Action / Obs / BD / DDQN semantics mid-training.
> Any change requires a new version (`v1.3+`) and an explicit changelog entry at the bottom.

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
| Crash / harness timeout / L3 desync | `False` | `True` | `0.0` (PPO may bootstrap \(V(s')\); **DDQN must not** store or bootstrap these transitions) |

---

## 2. ActionSpec & Masking

### Structure

- **Multi-agent auto-regressive double slot:** \(\text{Slot 1} \rightarrow \text{Slot 2}\)
- Slot 1 choice updates the legal mask for Slot 2 (e.g. switch conflicts, shared targets).
- **DDQN joint encoding** (see §6.b): \(a = 24\cdot a_1 + a_2 \in \{0,\ldots,575\}\).

### Masking protocol

1. Network outputs raw logits \(\mathbf{z}\) (PPO) or Q-values (DDQN).
2. Invalid-action mask \(M \in \{0,1\}^{|A|}\) (1 = legal).
3. Softmax (PPO) uses:

\[
\mathrm{Softmax}\big(\mathbf{z} + (1 - M)\cdot(-10^{9})\big)
\]

4. DDQN: before \(\arg\max\), set \(Q(s,a)=-\infty\) (or add \((1-M)\cdot(-10^{9})\)) for illegal \(a\).

Source of truth for legality: poke-env / PS `valid_orders` (or equivalent request-derived legal set).

Canonical joint helper (DDQN): `legal_joint_mask(battle) -> (576,)` with

\[
M_{\mathrm{joint}}(a_1, a_2) = M_1(a_1)\cdot M_2(a_2\mid a_1).
\]

### Fallback protocol (cascaded)

| Level | Trigger | Behavior |
|-------|---------|----------|
| **L1 Mathematical** | Illegal logit / Q before sample | Masking; expected coverage \(>99\%\) |
| **L2 Desync / engine mismatch** | \(a_t \notin \texttt{valid\_orders}\) | Execute `valid_orders[0]`; \(R_t = 0.0\); set `info['skip_buffer'] = True`; increment `illegal_action_fallbacks` |
| **L3 Crash / timeout** | Env/server failure | `truncated=True`, `terminated=False`, \(R_t = 0.0\) |

**L2 buffer rule (mandatory):** Rollout collectors for PPO / DDQN / BDQN **must drop** transitions with `skip_buffer=True`. Do not train on L2 fallbacks. Truncated (L3) transitions **must not** enter the DDQN replay.

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
| Belief state | 256D | Summarized EV/IV / Spe-tier belief; item/ability as `unknown \| suspected \| confirmed` bits where applicable. **Foe bench (OTS):** team-preview priors for HP/status until combat reveal; unrevealed fields → zeros / priors as specified by encoder. **DDQN v1.2:** belief segment is **zeros / flat priors** (no Expectimax-style hidden-stat inference module) |
| Active field & physics | 128D | One-hot weather/terrain; Trick Room, Tailwind, screens, gravity; turn counters normalized |
| Active stats & volatiles | 128D | Stat stages in \([-1.0, 1.0]\); Yawn countdowns; Taunt; Disable; Substitute flags |
| Move matrix & damage | 384D | Priority, terrain modifiers, grounded flags. **Damage:** local Python heuristic cache only — \(\mathrm{BasePower} \times \mathrm{STAB} \times \mathrm{Effectiveness} \times \mathrm{Weather}\). Failsafe `0.0`. **No Node.js in `step()`** |

**Total:** \(128 + 256 + 128 + 128 + 384 = 1024\). Obs dtype for RL buffers: **`float32`**.

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

### Candidate 2 — Dense soft (PPO / DDQN / BDQN / bootstrap)

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
| 1. Masked PPO | Cand. 2 → Cand. 1 | Auto-regressive Slot1→Slot2. **Main policy path** | Off (telemetry only) |
| **2. DDQN (joint)** | Cand. 2 | Joint \(Q(s,(a_1,a_2))\), 576 outs; Double target; soft \(\tau\); \(Q_{\mathrm{illegal}}=-\infty\). **Main value-based path** | Off |
| 2b. BDQN | Cand. 2 | Branching DQN (**ablation / experimental**); \(Q(s,a_{\mathrm{invalid}})=-\infty\) | Off |
| 2c. IQL | Cand. 2 | Two independent \(Q_i(s,a_i)\) — **ablation only** (breaks coordination) | Off |
| 3. MCTS + PUCT | Cand. 1 | Tree expansion over legal actions only | Off |
| 4. MCTS + Nash | Cand. 1 | Legal payoff matrix \(N\times M\); payoff = **sparse terminal** | Off |
| 5. QD-PPO | Cand. 2 (PPO) / Cand. 1 (fitness) | Auto-regressive logit masking | **ACTIVE** |
| 6. QD-PGA-MAP-Elites | Cand. 2 (actor) / Cand. 1 (fitness) | Auto-regressive logit masking | **ACTIVE** |
| 7. QD-PPGA | Cand. 2 (gradients) / Cand. 1 (fitness) | Auto-regressive logit masking | **ACTIVE** |

**Note:** Prefer joint DDQN (§6.b) or auto-regressive PPO for Slot1/Slot2 coordination. IQL / BDQN are ablations only.

---

## 6.b DDQNSpec (Joint Double DQN) — v1.2

Contract for the **main value-based** agent. Implementation target: `showdown/rl/ddqn/`. Do not start mid-run changes without a version bump.

> **Exclusive coding spec:** [`docs/ddqn_spec.md`](ddqn_spec.md) **v1.0** is the DDQN-only
> document used for implementation. Keep it in sync with this section.

### MDP delimitation

| Item | Contract |
|------|----------|
| Interaction loop | **Combat only** (moves / switches on the active matrix). One env `step` = one Showdown turn with joint action \((a_1,a_2)\). |
| Team Preview | **Outside** the DDQN MDP. Deterministic **Fixed Export Order**: Showdown `/team 1234` (first four mons of the export, that order). Train and eval use the **same** rule. Reported WR is **conditional** on this preview policy. |
| OTS / harness | `ValidatedDDQNPlayer` (poke-env + `ValidatedBattleMixin`): protocol sync + `accept_open_team_sheet=True`. OTS does **not** imply a refined belief encoder in v1.2. |
| Belief in obs | 256D belief segment = **zeros / flat priors** (no Expectimax SP inference). |

### Action encoding & masking

| Item | Contract |
|------|----------|
| Joint id | \(a = 24\cdot a_1 + a_2 \in \{0,\ldots,575\}\), \(a_1,a_2\in\{0,\ldots,23\}\) |
| Slot layout | Unchanged ActionSpec: 0–15 move×target, 16–19 switch, 20 pass, **21 mega**, **22 tera**, 23 reserved |
| Mega / Tera | Ids 21 / 22 participate in the joint space. \(M_{\mathrm{joint}}\) enables them **only** when `valid_orders` allows them in \(s_t\). Decode follows `pe_adapter` semantics (compact mega/tera with move0/target0). |
| Mask API | `legal_joint_mask(battle) -> shape (576,) dtype float32/bool`, from **`valid_orders`**. Also expose conditioned `action_masks(slot1_action_id=None) -> (M1, M2)`. |
| Illegal Q | Before \(\arg\max\) / \(\varepsilon\)-greedy: \(Q(s,a_{\mathrm{illegal}})=-\infty\) (equiv. \(-10^{9}\) addend). |
| Exploration | \(\varepsilon\)-greedy **uniform over legal** actions only. Never sample \(M=0\). |

### Double DQN target

Online parameters \(\theta\), target \(\theta^-\), legal set \(L(s')\):

\[
y =
\begin{cases}
r_t & \text{if } \texttt{terminated} \\
r_t + \gamma\, Q_{\theta^-}\big(s',\, \arg\max_{a \in L(s')} Q_{\theta}(s', a)\big) & \text{otherwise}
\end{cases}
\]

- \(r_t\): reward returned by `VGCEnv.step` for transition \((s_t,a_t)\to s'\).
- **Must not** use \(\max_{a'} Q_{\theta^-}(s',a')\) (classic DQN; overestimation).
- Transitions with `truncated=True` or `skip_buffer=True` **never** enter replay and are not bootstrapped.

### Network

| Item | Contract |
|------|----------|
| Framework | PyTorch (`torch >= 2.0.0`), dependency file **`requirements-rl.txt`** (or install extra `[rl]`). Heuristic bots must run without this extra. |
| Topology | MLP: \(1024 \xrightarrow{\mathrm{ReLU}} 512 \xrightarrow{\mathrm{ReLU}} 512 \xrightarrow{\mathrm{Linear}} 576\) |
| Dueling / Noisy / Rainbow | **Off** in v1.2 |
| Device default | **CPU** (GPU optional; document in train CLI) |
| Seed | Collector / env seed **must** be CLI-documented and logged |

### Frozen hyperparameters

| Parameter | Value |
|-----------|--------|
| \(\gamma\) | `0.95` |
| Optimizer | Adam |
| Learning rate | `3e-4` |
| Soft update \(\tau\) | `0.005` **every gradient step**: \(\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-\) |
| Batch size | `256` |
| Replay capacity | `200_000` |
| Warmup | `5_000` env steps; **uniform over legal** actions only |
| Train frequency | `1` gradient step per `4` env steps |
| Loss | Huber \(\delta = 1.0\) |
| Grad clip | `max_norm = 10` |
| \(\varepsilon\) schedule | `1.0 → 0.05` linear over first `500_000` env steps; eval / curriculum gate: \(\varepsilon = 0\) |
| Replay tuple | `(s_t, a_joint, r_t, s', terminated, m_next)` with obs `float32`; `a_joint ∈ {0..575}`; `m_next` legal mask at \(s'` (or recomputed equivalently) |
| Max turns / ep | `40` (same as `VGCEnv` default); truncate ≠ loss |

### Integration, curriculum, evaluation

| Item | Contract |
|------|----------|
| Code layout | `showdown/rl/ddqn/{network,replay,agent,train,player}.py` |
| CLI | `python -m showdown.rl.ddqn.train` |
| Player class | `ValidatedDDQNPlayer` as **player A** in validated harness (sync + OTS) |
| Curriculum opponents | `random` → `greedy_solo` → `greedy_joint` → **stop**. **No self-play in v1.2.** |
| Phase gate | Every **10_000** env steps: run **100** BO1 eval games vs **current-phase opponent**, \(\varepsilon=0\). Advance if \(WR \ge 0.60\). Ties count as non-wins: \(WR = wins / n\). |
| Force advance | If gate not met, advance at **500_000** env steps in the phase anyway. |
| Checkpoint | **Best** = highest WR vs **`greedy_joint`** (dedicated eval, \(\varepsilon=0\)). |
| Offline report suite | Vs `random`, `greedy_joint`, `expectimax_bayesian`; **200 BO1** each; \(\varepsilon=0\); same `/team 1234` preview. |

### Acceptance tests (mandatory)

1. **Double update:** mock \(Q_\theta\) / \(Q_{\theta^-}\) proves \(a^*\) is chosen by the **online** net.
2. **`skip_buffer=True`** never increases `len(replay)`.
3. **Encode/decode** joint \(a \leftrightarrow (a_1,a_2)\) is idempotent on \(\{0,\ldots,575\}\).

### DDQN non-goals (v1.2)

- PER, n-step returns, full Rainbow / C51 / QR-DQN.
- Team Preview inside \(s_t\) / joint \(Q\).
- Training on L2 fallbacks or truncated transitions.
- DDQN \(Q(s)\) as Ng potential \(\Phi(s)\) inside \(R_t\).
- Self-play / IQL as the reported main agent.

### References (DDQNSpec)

- Huang, S., & Ontañón, S. (2022). A closer look at invalid action masking in policy gradient algorithms. *Proceedings of the International Florida Artificial Intelligence Research Society Conference, 35*. https://doi.org/10.32473/flairs.v35i.130584
- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529–533. https://doi.org/10.1038/nature14236
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with Double Q-learning. *Proceedings of the AAAI Conference on Artificial Intelligence, 30*(1). https://doi.org/10.1609/aaai.v30i1.10295

---

## 7. Curriculum & Telemetry

### Curriculum pipeline

1. **Behavioral Cloning (BC):** Replays from internal `expectimax_bayesian` / `greedy_joint` (and optionally filtered valid human matches). **No Smogon ELO dependency.** Prefer games that completed cleanly (`sync_ok`, no timeout/reject) when sourced from the robust harness. (Primary for PPO; optional for DDQN.)
2. **PPO / DDQN bootstrap:** Train vs heuristic bots with **Reward Candidate 2**. DDQN phase order and gates: see §6.b.
3. **Self-play:** Progressive decay toward **Candidate 1** (**PPO path**; **out of scope for DDQN v1.2**).

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
| `mean_q` / `mean_target` / `td_error` | DDQN value stability (telemetry only) |
| `eps` | Current exploration rate (DDQN) |

These metrics **must not** enter \(R_t\).

### Evaluation suite (mandatory)

Fixed opponents for reported WR (not only self-play): at least `random`, `greedy_joint`, `expectimax_bayesian` (or frozen checkpoints). QD fitness uses Candidate 1 against a fixed eval suite / archive protocol. DDQN offline volume: **200 BO1** per opponent (§6.b).

---

## 8. Non-goals (explicit)

- Expectimax live \(V(s)\) as Ng potential \(\Phi(s)\) inside \(R_t\) — **forbidden** (stochastic / anytime / \(\pi_{\mathrm{opp}}\)-dependent).
- DDQN \(Q(s,\cdot)\) as Ng potential \(\Phi(s)\) inside \(R_t\) — **forbidden**.
- Node `@smogon/calc` inside env `step()` — **forbidden**.
- Magical per-move bonuses (Tailwind, Fake Out, Sub break, etc.) inside \(R_t\) — **forbidden**.
- Treating harness truncate as Loss — **forbidden**.
- DDQN v1.2: PER / n-step / Rainbow / preview-in-MDP / self-play / train-on-L2 — **forbidden** (see §6.b).

---

## Changelog

| Version | Date | Notes |
|---------|------|-------|
| **v1.2** | 2026-10-10 | **DDQNSpec** (joint Double DQN): 576 actions, `valid_orders` joint mask, soft \(\tau=0.005\), \(\gamma=0.95\), frozen hyperparams, `/team 1234` preview outside MDP, `ValidatedDDQNPlayer`, hybrid curriculum, eval suite 200 BO1. DDQN = main value path; BDQN/IQL = ablations. Obs/Reward/Action structural semantics unchanged. |
| **v1.1** | 2026-09-15 | Frozen unified sheet: Action/Obs/Reward/BD/Matrix + curriculum, L2 `skip_buffer`, local damage heuristic, hard-switch BD₂, BC from internal bots, Gymnasium terminated/truncated |
| v1.0 | — | Drafts (Gemini + Cursor triangulation); superseded by v1.1 |

---

*End of frozen contract v1.2.*

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
| **DDQNSpec (v1.2)** | Coding contract: [`docs/ddqn_spec.md`](ddqn_spec.md) v1.0 → `showdown/rl/ddqn/` + `battle_bots/ddqn` |
| DDQN deps | `requirements-rl.txt` (`torch>=2.0.0`) |
| Tests | `tests/test_rl_env_spec.py`, `tests/test_rl_pe_adapter.py`, `tests/test_rl_ddqn.py` |

`MockBattleBackend` covers offline CI. Live: `VGCEnv(backend=PokeEnvBattleBackend())` with local PS (`node pokemon-showdown start --no-security`).
