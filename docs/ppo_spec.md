# Masked PPO Agent Spec — Auto-regressive Slot1→Slot2 (VGC Gen 9)

| Field | Value |
|-------|--------|
| **Document** | **Masked PPO — exclusive implementation contract** |
| **Version** | **1.0** (aligned with `docs/rl_env_spec.md` **v1.3** §6.a) |
| **Status** | **FROZEN — implemented in `showdown/rl/ppo/` + `battle_bots/ppo`** |
| **Algorithm** | Proximal Policy Optimization (Schulman et al., 2017) + invalid-action masking (Huang & Ontañón, 2022) |
| **Role** | Main **policy-gradient** RL agent (complement to joint DDQN) |
| **Code root (target)** | `showdown/rl/ppo/` |
| **CLI (target)** | `python -m showdown.rl.ppo.train` |
| **Deps** | `requirements-rl.txt` — `torch >= 2.0.0` (heuristics must run without this) |

> **Scope:** This file is the single source of truth for **implementing Masked PPO**.
> Shared environment contracts (Obs 1024D, Reward Candidate 2/1, `VGCEnv` L1/L2/L3,
> ActionSpec slot ids 0–23) live in [`rl_env_spec.md`](rl_env_spec.md) v1.3 and are
> **imported by reference** — do not fork their formulas here.
>
> **Contrast with DDQN:** DDQN uses **joint** \(Q(s,(a_1,a_2))\) over 576 actions
> ([`ddqn_spec.md`](ddqn_spec.md)). PPO uses **auto-regressive**
> \(\pi(a_1\mid s)\,\, \pi(a_2\mid s,a_1)\) with per-slot masks. The factorization
> difference is part of the algorithmic contrast and does **not** invalidate the
> benchmark comparison when both share the same Obs / Reward / Preview / foes.
>
> **Immutability:** Changing any frozen number or semantic below requires bumping
> **this** doc to 1.1+ **and** updating `rl_env_spec.md` §6.a in the same change.
>
---

## 1. Inherited environment (do not reimplement)

| Contract | Source | PPO usage |
|----------|--------|-----------|
| `VGCEnv` Gymnasium API | `rl_env_spec` §1 | Rollout `reset` / `step` |
| Obs \(s \in \mathbb{R}^{1024}\), `float32` | §3 | Shared actor–critic torso |
| Reward Cand. 2 (bootstrap) → Cand. 1 (self-play / fitness) | §4 | See §8 curriculum |
| Slot action ids 0–23 | §2 | Factorized policy heads |
| L2 `skip_buffer` | §2 | **Drop** from rollout buffer |
| Truncation reasons | §1 / this §4 | Discriminate turn-cap vs crash via `info['trunc_reason']` |
| No `@smogon/calc` in `step()` | §1 / §8 | Local damage heuristic only |

**PPO-specific obs rule:** the 256D belief segment is **zeros / flat priors**
(no Expectimax SP / MetaVGC inference). OTS is accepted at the harness for
protocol sync; it does not refine the belief encoder in this version.

---

## 2. MDP delimitation

| Item | Contract |
|------|----------|
| Episode | 1 game (BO1). Max turns = **40** (`VGCEnv` default). Truncate ≠ Loss. |
| Decision unit | One Showdown **combat turn** → ordered pair \((a_1, a_2)\). |
| Env action space | `MultiDiscrete([24, 24])` — Slot1 then Slot2 (matches `VGCEnv`). |
| Team Preview | **Out of MDP.** Fixed Export Order: `/team 1234`. Train = eval. Reported WR is **conditional** on this preview. |
| Self-play | **In scope** after heuristic bootstrap — see §8. |
| Player class (target) | `ValidatedPPOPlayer` (poke-env + `ValidatedBattleMixin`): sync + `accept_open_team_sheet=True`. Used as **player A** in validated harnesses. |

---

## 3. Auto-regressive action space & masking

### Factorization

\[
\pi_\theta(a_1, a_2 \mid s)
= \pi_\theta^{(1)}(a_1 \mid s)\,\,
  \pi_\theta^{(2)}(a_2 \mid s, a_1)
\]

with \(a_1, a_2 \in \{0,\ldots,23\}\).

| Slot id | Meaning |
|---------|---------|
| 0–15 | move \(i\) × target \(t\) |
| 16–19 | hard switch bench |
| 20 | pass |
| 21 | mega (compact; decode via `pe_adapter`) |
| 22 | tera (compact; decode via `pe_adapter`) |
| 23 | reserved |

**Mega / Tera:** ids 21 and 22 participate. \(M_1\) / \(M_2(\cdot\mid a_1)\) enable them
**only** when Showdown `valid_orders` allows them at \(s_t\).

### Invalid Action Masking (IAM)

Source of truth: poke-env / PS **`valid_orders`** via
`action_masks(slot1_action_id=...)`.

1. Actor emits raw logits \(\mathbf{z}^{(1)} \in \mathbb{R}^{24}\), then
   \(\mathbf{z}^{(2)} \in \mathbb{R}^{24}\) conditioned on chosen \(a_1\).
2. Masks \(M_1(s),\; M_2(s\mid a_1) \in \{0,1\}^{24}\) (1 = legal).
3. Sampling and log-prob use **renormalization over the legal subset**:

\[
\mathrm{Softmax}\big(\mathbf{z} + (1-M)\cdot(-10^{9})\big)
\]

4. The **same** mask application must run at interactive sampling **and** when
   recomputing \(\pi_\theta\) during PPO epochs (Huang & Ontañón IAM — not
   reward penalties for illegal actions).

Helpers (existing): `apply_logit_mask` / `softmax_masked` in
`showdown/rl/action_space.py`.

| API | Contract |
|-----|----------|
| `action_masks(slot1_action_id=None) -> (M1, M2)` | \(M_2\) conditioned when `slot1_action_id` set |
| `legal_joint_mask` | Optional telemetry / parity vs DDQN; **not** the PPO sampling path |
| Sampling | Never sample index with \(M=0\). If \(\sum M=0\), engine L2 applies. |
| Log-prob | \(\log\pi = \log\pi^{(1)}(a_1\mid s) + \log\pi^{(2)}(a_2\mid s,a_1)\) under masked categoricals |
| Entropy | See §4 — sum of masked slot entropies conditioned on trajectory \(a_1\) |

L1 coverage target: \(>99\%\) of chosen pairs accepted without L2.

---

## 4. PPO objective (clipped) + GAE

Actor \(\theta\), critic \(\phi\) (shared torso; see §5).

### Truncation semantics (`info['trunc_reason']`)

`VGCEnv` **must** expose `info['trunc_reason']` ∈
`{None, 'turn_cap', 'crash'}` (names frozen).

| Event | `terminated` | `truncated` | Buffer | Bootstrap \(V_\phi(s')\) |
|-------|----------------|-------------|---------|---------------------------|
| Win / Loss / Tie | `True` | `False` | Store | **No** (\(d_t=1\)) |
| Turn-cap (max turns = 40) | `False` | `True` | Store | **Yes** (\(d_t=0\)) |
| Crash / harness timeout / L3 desync | `False` | `True` | **Drop** (`skip_buffer=True`) | **No** |
| L2 illegal fallback | — | — | **Drop** (`skip_buffer=True`) | **No** |

### Advantage (GAE-\(\lambda\))

\[
\delta_t = r_t + \gamma\, V_\phi(s_{t+1})\,(1-d_t) - V_\phi(s_t)
\]

\[
\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l\, \delta_{t+l}
\]

- \(d_t = 1\) iff `terminated`.
- Turn-cap: \(d_t = 0\), use \(V_\phi(s')\).
- Crash / L2: transition absent from buffer.
- Normalize advantages per-rollout (mean/std) by default.
- Returns target: \(\hat{R}_t = \hat{A}_t + V_{\phi_{\mathrm{old}}}(s_t)\).

### Surrogate

\[
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
\quad\text{with } a_t=(a_1,a_2),\ \pi=\pi^{(1)}\pi^{(2)}
\]

\[
L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(
r_t(\theta)\hat{A}_t,\;
\mathrm{clip}(r_t(\theta), 1-\varepsilon_{\mathrm{clip}}, 1+\varepsilon_{\mathrm{clip}})\hat{A}_t
\big)\Big]
\]

\[
L^V(\phi) = \mathbb{E}_t\big[(V_\phi(s_t) - \hat{R}_t)^2\big]
\]

**Value clip:** **OFF** in v1.0 (KL / trust region only via \(L^{\mathrm{CLIP}}\)).

### Entropy regularization

\[
H_t = H\big[\pi_\theta^{(1)}(\cdot\mid s_t)\big]
+ H\big[\pi_\theta^{(2)}(\cdot\mid s_t, a_{1,t})\big]
\]

where \(a_{1,t}\) is the **empirical** Slot1 action from the rollout tuple
(or re-sampled under current \(\theta\) consistently with the ratio path).
Do **not** use an unconditioned marginal \(H[\pi^{(2)}]\) that ignores trajectory \(a_1\).

\[
L(\theta,\phi) = -L^{\mathrm{CLIP}} + c_v L^V - c_e\, \mathbb{E}_t[H_t]
\]

| Rule | Contract |
|------|----------|
| On-policy | Updates only on current rollout; clear buffer after \(K\) epochs |
| Ratio | Must use **masked** \(\pi\) at both old and new |
| Grad clip | Global `max_norm = 0.5` |

---

## 5. Network

| Item | Contract |
|------|----------|
| Input | `float32[1024]` |
| Shared torso | `Linear(1024,512) → ReLU → Linear(512,512) → ReLU` → features \(\mathrm{h}\) |
| Policy head Slot1 | `Linear(512,24)` → logits \(\mathbf{z}^{(1)}\) |
| Policy head Slot2 | Concat \([\mathrm{h};\; \mathrm{onehot}(a_1)]\) with \(\mathrm{onehot}(a_1)\in\mathbb{R}^{24}\) → `Linear(536,24)` → \(\mathbf{z}^{(2)}\) |
| Value head | `Linear(512,1)` → \(V(s)\) (state-value; **not** \(Q\)) |
| Off in v1.0 | Recurrent (LSTM), attention over party, non-shared dual actors, FiLM |

Device default: **CPU**; GPU optional via CLI.

---

## 6. Frozen hyperparameters

| Parameter | Value |
|-----------|--------|
| \(\gamma\) | `0.95` |
| \(\lambda\) (GAE) | `0.95` |
| Clip \(\varepsilon_{\mathrm{clip}}\) | `0.2` |
| Optimizer | Adam |
| LR (PPO from scratch / after BC warmup complete) | `3e-4` |
| LR **post-BC fine-tune** | \(\alpha = 10^{-4}\) (full fine-tune; **no** layer freeze) |
| Rollout length \(T\) | `2048` env steps (summed across envs) |
| Num envs | `8` (mock and/or local PS) |
| Minibatch size | `256` |
| Epochs \(K\) | `4` |
| \(c_v\) | `0.5` |
| \(c_e\) | `0.01` |
| Grad clip | `max_norm = 0.5` |
| Max turns / ep | `40` |
| Value clip | **Off** |
| Seed | Required CLI arg; logged |

**No experience replay** for the PPO path (on-policy). BC is a separate offline stage (§8).

---

## 7. Modules & CLI

| Module | Responsibility |
|--------|----------------|
| `showdown/rl/ppo/network.py` | Shared torso + AR heads + \(V\) |
| `showdown/rl/ppo/buffer.py` | Rollout store; reject `skip_buffer` / `trunc_reason=='crash'` |
| `showdown/rl/ppo/agent.py` | Masked sample + greedy (argmax legal) |
| `showdown/rl/ppo/train.py` | Collect → GAE → PPO epochs → curriculum / self-play |
| `showdown/rl/ppo/player.py` | `PPOPlayer` + `ValidatedPPOPlayer` |
| `showdown/rl/ppo/bc.py` | BC pretrain on expert trajectories |
| `showdown/rl/ppo/__init__.py` | Public exports |
| `showdown/rl/ppo/__main__.py` | Entrypoint for `python -m showdown.rl.ppo.train` |
| Env support | `info['trunc_reason']` on `VGCEnv` (turn_cap vs crash) |
| Tests | `tests/test_rl_ppo.py` (acceptance §11) |

Suggested train flags:

| Flag | Meaning |
|------|---------|
| `--seed` | Required |
| `--device` | Default `cpu` |
| `--opponent` | `mock` \| `random` \| `greedy_solo` \| `greedy_joint` \| `expectimax_bayesian` \| `curriculum` \| `self` |
| `--phases` | Unlock markers + steps (heuristic \(\Omega\)) |
| `--expand-omega` / `--no-expand-omega` | Cumulative \(\Omega\) (default on) vs exclusive phases |
| `--opponent-weights` | Heuristic foe weights (Expectimax down-weight allowed) |
| `--reward` | `candidate2` \| `candidate1` \| `anneal` |
| `--bc-checkpoint` | Warm-start actor from BC; then LR \(10^{-4}\) |
| `--start-ps` | Start local PS (`--no-security`) |
| `--checkpoint-dir` | Default `logs/ppo_checkpoints` |

Examples (target CLI; not runnable until implemented):

```bat
python -m showdown.rl.ppo.train --seed 0 --opponent mock --steps 20000
python -m showdown.rl.ppo.train --seed 0 --opponent curriculum --start-ps --expand-omega
python -m showdown.rl.ppo.train --seed 0 --opponent self --reward anneal --bc-checkpoint logs/ppo_bc.pt --start-ps
```

---

## 8. Curriculum

Aligned with `rl_env_spec` §7; richer than DDQN (BC + self-play in scope).

### Stage 0 — Behavioral Cloning (primary)

| Item | Contract |
|------|----------|
| Loss | Masked cross-entropy on expert \((a_1^*, a_2^*)\) with \(M_1\), \(M_2(\cdot\mid a_1^*)\) |
| Data | Internal `expectimax_bayesian` / `greedy_joint` first; no Smogon ELO scrape dependency |
| Filter | Prefer `sync_ok`, no timeout/reject |
| Transfer | Load actor into PPO; **full fine-tune** (torso + heads), **no freeze**; initial on-policy LR \(\alpha=10^{-4}\) |

### Stage 1 — Heuristic bootstrap (Candidate 2)

Monotonic \(\Omega\) (same defaults as DDQN unless ablated):

| Phase | Unlock | \(\Omega_t\) | Force advance |
|-------|--------|--------------|---------------|
| 1 | `random` | `{random}` | `500_000` steps |
| 2 | `greedy_solo` | `{random, greedy_solo}` | same |
| 3 | `greedy_joint` | `{random, solo, joint}` | same |
| 4 | `expectimax_bayesian` | \(\Omega_3\cup\{\mathrm{expectimax}\}\) | same (optional omit) |

Episode sampling over \(\Omega_t\) with weights `random=solo=joint=1.0`,
`expectimax_bayesian=0.35` when present. `--no-expand-omega` = exclusive phases.

**Gate:** every **10_000** env steps, **100** BO1 vs current focus foe, greedy policy;
advance if \(WR \ge 0.60\); else force advance at phase step budget.

### Stage 2 — Self-play pool (Candidate 1 anneal)

**No** exclusive matchup vs the immediate previous iteration (mitigate catastrophic forgetting).

**Pool definition:**

\[
\mathcal{P} = \{\text{last } K \text{ iterative checkpoints}\}
\cup \{\texttt{best\_vs\_greedy\_joint}\}
\quad K = 10
\]

- Deduplicate by identity: if best is already among the last \(K\),
  \(|\mathcal{P}| \le 11\).
- **`best_vs_greedy_joint` is never evicted** by the \(K\)-FIFO rotation.
- **Ingress:** push a checkpoint into the recent-\(K\) queue at every curriculum
  **gate** evaluation (and whenever a new best vs `greedy_joint` is recorded).

**Matchmaking (v1.0):** strict uniform over the set:

\[
P(o_i) = \frac{1}{|\mathcal{P}|}
\]

Softmax-by-difficulty / full PFSP: **out of scope** for v1.0 (possible v1.1).

**Reward:** anneal Candidate 2 → Candidate 1 during self-play (`--reward anneal`).

### Checkpoints

| Tag | Criterion |
|-----|-----------|
| `best_vs_greedy_joint` | Highest WR vs `greedy_joint` (greedy policy) |
| Recent pool entries | Periodic / gate snapshots for self-play |
| Cand. 1 fitness | Tagged separately; do not silently overwrite Cand. 2 best |

---

## 9. Offline evaluation suite (reported WR)

| Opponent | Games | Mode | Policy | Preview |
|----------|-------|------|--------|---------|
| `random` | 200 | BO1 | greedy (argmax masked) | `/team 1234` |
| `greedy_joint` | 200 | BO1 | greedy | `/team 1234` |
| `expectimax_bayesian` | 200 | BO1 | greedy | `/team 1234` |

Report `illegal_action_fallbacks` alongside WR. Self-play pool Elo is optional
telemetry, not a substitute for the fixed suite.

---

## 10. Telemetry (must not enter \(R_t\))

| Metric | Meaning |
|--------|---------|
| `illegal_action_fallbacks` | L2 rate |
| `approx_kl` / `clip_fraction` | PPO stability |
| `entropy_slot1` / `entropy_slot2` | Collapse detector |
| `explained_variance` | Critic fit |
| `curriculum_phase` / `reward_mode` | Stage |
| `eval_wr_<opponent>` | Gates / suite |
| `bc_acc_legal` | BC stage |
| `selfplay_pool_size` | \(|\mathcal{P}|\) |

---

## 11. Acceptance tests (mandatory before merge)

1. **Mask:** sampled \((a_1,a_2)\) always satisfy \(M_1(a_1)=1\) and
   \(M_2(a_2\mid a_1)=1\) when any legal action exists.
2. **Log-prob:** \(\log\pi(a\mid s)=\log\pi^{(1)}+\log\pi^{(2)}\) under mask; ratio finite.
3. **`skip_buffer=True`:** buffer length unchanged.
4. **Turn-cap GAE:** `trunc_reason=='turn_cap'` contributes \(V(s')\) in a unit mock.
5. **Crash drop:** `trunc_reason=='crash'` never enters the rollout buffer.
6. **Clip identity:** if \(\theta=\theta_{\mathrm{old}}\), \(L^{\mathrm{CLIP}}=\mathbb{E}[\hat{A}]\)
   (up to advantage normalization).
7. **Smoke:** ≥100 env steps on `MockBattleBackend` without crash (CPU).

---

## 12. Non-goals (v1.0)

- Joint 576-way softmax as the **primary** PPO policy (ablation only)
- Full PFSP / Softmax-by-WR matchmaking (uniform pool only)
- Value function clipping
- Training on L2 fallbacks or crash truncations
- Team Preview inside the MDP
- BDSpec inside \(R_t\) (QD-PPO is a later matrix row; stub only)
- Replacing DDQN; both remain first-class
- Recurrent / attention actor
---

## 13. Parity note (PPO vs DDQN)

Empirical comparison of auto-regressive Masked PPO vs joint Double DQN is a
valid methodological contrast under a shared env contract (Obs, Reward phase,
`/team 1234`, foe suite). Differences in action selection topology are inherent
to each algorithm’s handling of large discrete spaces and must be stated in the
thesis write-up; they do not require forcing identical heads.

---

## 14. References

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. *arXiv:1506.02438*.
- Huang, S., & Ontañón, S. (2022). A closer look at invalid action masking in policy gradient algorithms. *Proceedings of the International Florida Artificial Intelligence Research Society Conference, 35*(1). https://doi.org/10.32473/flairs.v35i1.130642
- (Env contract) Project [`rl_env_spec.md`](rl_env_spec.md) v1.3 §1–4, §6.a, §7.
- (Sibling agent) Project [`ddqn_spec.md`](ddqn_spec.md) v1.0 — joint Double DQN.

---

## Changelog

| Version | Date | Notes |
|---------|------|-------|
| **1.0** | 2026-10-10 | Frozen Masked PPO contract + implementation under `showdown/rl/ppo/` / `battle_bots/ppo`: AR IAM, GAE, BC, uniform self-play pool, `trunc_reason` on `VGCEnv`. |

---

*End of Masked PPO agent spec 1.0 — use this file when coding `showdown/rl/ppo/`.*
