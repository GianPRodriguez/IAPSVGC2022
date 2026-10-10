# DDQN Agent Spec — Joint Double Deep Q-Network (VGC Gen 9)

| Field | Value |
|-------|--------|
| **Document** | **DDQN-only implementation contract** |
| **Version** | **1.0** (aligned with `docs/rl_env_spec.md` **v1.2** §6.b) |
| **Status** | **FROZEN — implemented in `showdown/rl/ddqn/` + `battle_bots/ddqn`** |
| **Algorithm** | Double DQN (van Hasselt et al., 2016) + invalid-action masking |
| **Role** | Main **value-based** RL agent (not the PPO policy path) |
| **Code root** | `showdown/rl/ddqn/` |
| **CLI** | `python -m showdown.rl.ddqn.train` |
| **Deps** | `requirements-rl.txt` — `torch >= 2.0.0` (heuristics must run without this) |

> **Scope:** This file is the single source of truth for **implementing DDQN**.
> Shared environment contracts (Obs 1024D, Reward Candidate 2, `VGCEnv` L1/L2/L3,
> ActionSpec slot ids 0–23) live in [`rl_env_spec.md`](rl_env_spec.md) v1.2 and are
> **imported by reference** — do not fork their formulas here.
>
> **Immutability:** Changing any frozen number or semantic below requires bumping
> **this** doc to 1.1+ **and** updating `rl_env_spec.md` §6.b in the same change.

---

## 1. Inherited environment (do not reimplement)

| Contract | Source | DDQN usage |
|----------|--------|------------|
| `VGCEnv` Gymnasium API | `rl_env_spec` §1 | Collector `reset` / `step` |
| Obs \(s \in \mathbb{R}^{1024}\), `float32` | §3 | Network input |
| Reward Candidate 2 + clip \([-1.30,+1.40]\) | §4 | \(r_t\) in replay |
| Slot action ids 0–23 | §2 | Factors of joint id |
| L2 `skip_buffer`, L3 `truncated` | §2 | **Never** enter replay |
| No `@smogon/calc` in `step()` | §1 / §8 | Local damage heuristic only in obs |

**DDQN-specific obs rule:** the 256D belief segment is **zeros / flat priors**
(no Expectimax SP / MetaVGC inference). OTS is still accepted at the harness for
protocol sync; it does not refine the belief encoder in this version.

---

## 2. MDP delimitation

| Item | Contract |
|------|----------|
| Episode | 1 game (BO1). Max turns = **40** (`VGCEnv` default). Truncate ≠ Loss. |
| Decision unit | One Showdown **combat turn** → one joint action \((a_1, a_2)\). |
| Team Preview | **Out of MDP.** Fixed Export Order: `/team 1234`. Same rule in train and eval. Reported WR is **conditional** on this preview. |
| Self-play | **Forbidden** in this version. |
| Player class | `ValidatedDDQNPlayer` (poke-env + `ValidatedBattleMixin`): sync + `accept_open_team_sheet=True`. Used as **player A** in validated harnesses. |

---

## 3. Joint action space & masking

### Encoding

\[
a = 24\cdot a_1 + a_2 \in \{0,\ldots,575\}, \quad a_1,a_2 \in \{0,\ldots,23\}
\]

| Slot id | Meaning |
|---------|---------|
| 0–15 | move \(i\) × target \(t\) |
| 16–19 | hard switch bench |
| 20 | pass |
| 21 | mega (compact; decode via `pe_adapter`) |
| 22 | tera (compact; decode via `pe_adapter`) |
| 23 | reserved |

**Mega / Tera:** ids 21 and 22 appear in the joint space. \(M_{\mathrm{joint}}\) may
enable them **only** when Showdown `valid_orders` allows them at \(s_t\).

### Mask

\[
M_{\mathrm{joint}}(a_1,a_2) = M_1(a_1)\cdot M_2(a_2\mid a_1) \in \{0,1\}^{576}
\]

| API | Contract |
|-----|----------|
| `legal_joint_mask(battle) -> (576,)` | Canonical; dtype `float32` or `bool`; source = **`valid_orders`** |
| `action_masks(slot1_action_id=None) -> (M1, M2)` | \(M_2\) conditioned on slot1 when id is provided |
| Illegal Q | Before \(\arg\max\) / \(\varepsilon\)-greedy: \(Q \leftarrow Q + (1-M)\cdot(-10^{9})\) (equiv. \(Q=-\infty\)) |
| Sampling | Never sample index with \(M=0\). Warmup and \(\varepsilon\)-greedy = **uniform over legal only**. |

L1 coverage target: \(>99\%\) of chosen actions accepted without L2.

---

## 4. Double DQN update

Online \(\theta\), target \(\theta^-\), legal set \(L(s') = \{a : M(s',a)=1\}\):

\[
y =
\begin{cases}
r_t & \text{if } \texttt{terminated} \\
r_t + \gamma\, Q_{\theta^-}\big(s',\, \arg\max_{a \in L(s')} Q_{\theta}(s', a)\big) & \text{otherwise}
\end{cases}
\]

\[
L(\theta) = \mathrm{Huber}\big(Q_\theta(s_t, a_t),\; y;\; \delta=1.0\big)
\]

| Rule | Contract |
|------|----------|
| Double | \(a^*\) from **online**; evaluate with **target**. Forbidden: \(\max_{a'} Q_{\theta^-}(s',a')\). |
| Soft sync | Every gradient step: \(\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-\) with \(\tau=0.005\). |
| Drop | `skip_buffer=True` or `truncated=True` → **no** replay insert, **no** bootstrap. |
| Grad clip | `max_norm = 10` |

---

## 5. Network

| Item | Contract |
|------|----------|
| Input | `float32[1024]` |
| Body | `Linear(1024,512) → ReLU → Linear(512,512) → ReLU → Linear(512,576)` |
| Output | Q-values for all joint actions (mask applied outside the net) |
| Off | Dueling, NoisyNets, PER, n-step, Rainbow, C51, QR-DQN |
| Device default | **CPU** (GPU optional via CLI flag) |

---

## 6. Frozen hyperparameters

| Parameter | Value |
|-----------|--------|
| \(\gamma\) | `0.95` |
| Optimizer | Adam |
| Learning rate | `3e-4` |
| \(\tau\) (soft) | `0.005` / gradient step |
| Batch size | `256` |
| Replay capacity | `200_000` |
| Warmup env steps | `5_000` (legal-uniform) |
| Train frequency | 1 grad step / 4 env steps |
| Loss | Huber \(\delta=1.0\) |
| Grad clip | `max_norm=10` |
| \(\varepsilon\) | `1.0 → 0.05` linear over first `500_000` env steps |
| Eval / gate \(\varepsilon\) | `0` |
| Replay tuple | `(s_t, a_joint, r_t, s', terminated, m_next)` |
| Seed | Required CLI arg; logged |

---

## 7. Modules & CLI (implementation checklist)

| Module | Responsibility |
|--------|----------------|
| `showdown/rl/ddqn/network.py` | Online + target MLP; soft update helper |
| `showdown/rl/ddqn/replay.py` | Capacity 200k; reject `skip_buffer` / truncated at insert API |
| `showdown/rl/ddqn/agent.py` | Masked \(\varepsilon\)-greedy / greedy act; encode/decode joint |
| `showdown/rl/ddqn/train.py` | Collector, curriculum, checkpoint, logging |
| `showdown/rl/ddqn/player.py` | `DDQNPlayer` + `ValidatedDDQNPlayer` |
| `showdown/rl/ddqn/__init__.py` | Public exports |
| `showdown/rl/ddqn/__main__.py` | Entrypoint for `python -m showdown.rl.ddqn.train` |
| Mask helper | Prefer `showdown/rl/action_space.py` or `pe_adapter.py`: `legal_joint_mask` |
| Tests | `tests/test_rl_ddqn.py` (acceptance below) |

Suggested train flags:

| Flag | Meaning |
|------|---------|
| `--seed` | Required |
| `--device` | Default `cpu` |
| `--opponent` | `mock` \| `random` \| `greedy_solo` \| `greedy_joint` \| `expectimax_bayesian` \| `curriculum` |
| `--steps` | Steps for a **single** `--opponent` (ignored if `--phases` set) |
| `--phases` | Unlock markers + steps: `opp:steps,...` |
| `--expand-omega` / `--no-expand-omega` | **Default on:** cumulative \(\Omega\); sample foe each episode. Off = exclusive phases |
| `--opponent-weights` | e.g. `expectimax_bayesian:0.35,greedy_joint:1.5` |
| `--no-expectimax` | Drop Expectimax from `--opponent curriculum` |
| `--curriculum-phase-steps` | Steps per unlock (default `500000`) |
| `--start-ps` | Start local PS for live foes |
| `--checkpoint-dir` | Default `logs/ddqn_checkpoints` |

Examples:

```bat
python -m showdown.rl.ddqn.train --seed 0 --opponent mock --steps 20000
python -m showdown.rl.ddqn.train --seed 0 --opponent curriculum --start-ps
python -m showdown.rl.ddqn.train --seed 0 --start-ps --expand-omega --phases random:100k,greedy_solo:200k,greedy_joint:300k,expectimax_bayesian:50k
python -m showdown.rl.ddqn.train --seed 0 --start-ps --no-expand-omega --phases random:100k,greedy_joint:200k
```

---

## 8. Curriculum (monotonic \(\Omega\))

**Default (`--expand-omega`):** opponent set expands accumulatively. Replay is **never cleared** between phases.

| Phase | Unlock | \(\Omega_t\) | Force advance |
|-------|--------|--------------|---------------|
| 0 Warmup | — | agent legal-uniform | 5k before learn |
| 1 | `random` | `{random}` | `500_000` steps |
| 2 | `greedy_solo` | `{random, greedy_solo}` | same |
| 3 | `greedy_joint` | `{random, solo, joint}` | same |
| 4 | `expectimax_bayesian` | \(\Omega_3\cup\{\mathrm{expectimax}\}\) | same (`--no-expectimax` omits) |

**Episode sampling:** \(o\sim\pi(\Omega_t)\) with weights `random=solo=joint=1.0`, `expectimax_bayesian=0.35` (~10% when all four present).

**Exclusive (`--no-expand-omega`):** one foe per phase; replay still shared.

**Replay tuple:** \(e_t=(s_t,a_t,r_t,s',d_t,M_{s'},\mathrm{opp\_code})\). Telemetry: `buf_frac_*`, `batch_frac_*`.

**Checkpoint:** best WR vs **`greedy_joint`** (\(\varepsilon=0\)).

---

## 9. Offline evaluation suite (reported WR)

| Opponent | Games | Mode | \(\varepsilon\) | Preview |
|----------|-------|------|-----------------|---------|
| `random` | 200 | BO1 | 0 | `/team 1234` |
| `greedy_joint` | 200 | BO1 | 0 | `/team 1234` |
| `expectimax_bayesian` | 200 | BO1 | 0 | `/team 1234` |

---

## 10. Telemetry (must not enter \(R_t\))

| Metric | Meaning |
|--------|---------|
| `illegal_action_fallbacks` | L2 rate |
| `mean_game_length` | Stall / collapse |
| `mean_q` / `mean_target` / `td_error` | Value stability |
| `eps` | Exploration |
| `curriculum_phase` | Current unlock phase |
| `buf_frac_<opp>` / `batch_frac_<opp>` | Replay / minibatch origin mix |
| `eval_wr_<opponent>` | Gate / suite WR |

---

## 11. Acceptance tests (mandatory before merge)

1. **Double:** mock online/target nets → \(a^*\) from online only.
2. **Replay:** `skip_buffer=True` does not increase `len(replay)`.
3. **Codec:** encode/decode joint idempotent on \(\{0,\ldots,575\}\).
4. **Mask:** with \(\varepsilon=0\), sampled/argmax index always has \(M=1\) when any legal exists.
5. **Smoke:** ≥100 env steps on `MockBattleBackend` without crash (CPU).

---

## 12. Non-goals

- PER, n-step, Rainbow, C51, QR-DQN, Dueling, NoisyNets
- Team Preview inside \(s_t\) or \(Q\)
- Training on L2 / truncated transitions
- \(Q\) as reward potential \(\Phi(s)\)
- Self-play, IQL, or BDQN as the reported main agent
- BC pretrain requirement (optional later; not in this contract)

---

## 13. References

- Huang, S., & Ontañón, S. (2022). A closer look at invalid action masking in policy gradient algorithms. *FLAIRS, 35*. https://doi.org/10.32473/flairs.v35i.130584
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529–533. https://doi.org/10.1038/nature14236
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with Double Q-learning. *AAAI, 30*(1). https://doi.org/10.1609/aaai.v30i1.10295

---

## Changelog

| Version | Date | Notes |
|---------|------|-------|
| **1.0** | 2026-10-10 | DDQN contract + code; monotonic \(\Omega\) curriculum (`--expand-omega`), shared replay with `opp_code` telemetry |

---

*End of DDQN agent spec 1.0 — use this file for coding `showdown/rl/ddqn/`.*
