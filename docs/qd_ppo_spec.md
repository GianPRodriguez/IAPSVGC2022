# QD-PPO Agent Spec — Quality–Diversity Proximal Policy Optimization (VGC Gen 9)

| Field | Value |
|-------|--------|
| **Document** | **QD-PPO — exclusive implementation contract** |
| **Version** | **1.0** (aligned with `docs/rl_env_spec.md` **v1.6** §6.e) |
| **Status** | **FROZEN — implemented in `showdown/rl/qd_ppo/`** |
| **Algorithm** | Grid MAP-Elites over BDSpec + Masked PPO (IAM) as sole variation / improvement operator |
| **Role** | Quality–Diversity **policy** path: illuminate behavior space; characterize multiple elites |
| **Code root** | `showdown/rl/qd_ppo/` |
| **CLI** | `python -m showdown.rl.qd_ppo.train` |
| **Deps** | `requirements-rl.txt` — `torch >= 2.0.0` (reuses `showdown/rl/ppo/`) |

> **Scope:** Single source of truth for **implementing QD-PPO**.
> Shared contracts (Obs 1024D, ActionSpec AR + IAM, Reward Cand. 2/1, BDSpec,
> `VGCEnv`) live in [`rl_env_spec.md`](rl_env_spec.md). Masked PPO math lives in
> [`ppo_spec.md`](ppo_spec.md) v1.0 — **reuse; do not fork**.
>
> **Contrast with Masked PPO:** one policy (+ self-play pool) vs **archive of elites**.
> Archive fitness \(U=f\) = **Candidate 1**; inner PPO \(R_t^{(2)}\) = **Candidate 2**.
>
> **Contrast with matrix rows 6–7:** QD-PGA-MAP-Elites / QD-PPGA out of scope.
>
> **Immutability:** bump this doc to 1.1+ **and** `rl_env_spec.md` §6.e together.

---

## 0. OPEN → FROZEN (v1.0)

| ID | Decision |
|----|----------|
| **O1** | Regular 2D Grid MAP-Elites |
| **O2** | \(N_1\times N_2=10\times 10\) (100 cells) |
| **O3** | Full actor–critic `state_dict` per cell; **no** shared critic |
| **O4** | Uniform over **filled** cells |
| **O5** | PPO gradients only; **no** ME-ES weight noise |
| **O6** | \(f=\) mean Cand. 1 over \(G=20\) BO1 |
| **O7** | BD = mean over eligible games; insert iff \(g\ge g_{\min}=10\) |
| **O8** | Inner PPO = Cand. 2 only; Cand. 1 = archive utility only |
| **O9** | Bootstrap \(B_{\mathrm{boot}}=100\,000\) Cand. 2 on mosaic (basal competence); no random-only seed |
| **O10** | Fitness foes = mosaic 25/25/50 |
| **O11** | Serial emitter (methodological control) |
| **O12** | Report Coverage, QD-score, WR(\(\pi^\star\)) |

**Operational freeze:**

| Item | Contract |
|------|----------|
| Improve / Stage 1 foes | Same mosaic i.i.d.; **no** expand-\(\Omega\); **no** self-play |
| BD filter | \(g_{\min}=10\); \(f\) over all \(G=20\); BD over eligible \(g\) |
| Replace | Strict \(f_{\mathrm{new}}>f_{\mathrm{old}}\) |
| Deploy | \(\pi^\star=\arg\max f\) |
| Budget | \(T_{\mathrm{total}}=10^6\) **incl.** \(B_{\mathrm{boot}}\); \(N_{\mathrm{outer}}=\lfloor(T-B_{\mathrm{boot}})/(n_{\mathrm{emit}}\cdot 2048)\rfloor=109\) |
| Fitness / suite steps | Eval overhead; **not** counted in \(T_{\mathrm{total}}\) |

---

## 1. Inherited environment

| Contract | Source | QD-PPO usage |
|----------|--------|--------------|
| `VGCEnv` | `rl_env_spec` §1 | Rollouts; BD in `info` |
| Obs 1024D | §3 | Masked-PPO torso |
| AR + **IAM (mandatory)** | §2 / `ppo_spec` §3 | Sample + PPO recompute |
| Cand. 2 | §4 | \(R_t^{(2)}\) inner PPO + bootstrap |
| Cand. 1 | §4 | \(U(\pi)=f(\pi)\) archive only |
| BDSpec | §5 | Archive coordinates |
| Preview | `/team 1234` | Out of MDP |
| Belief | Flat priors | Parity with PPO v1 |

Helper: `BehavioralTracker` (`archive_eligible` iff turns ≥ 4).

### 1.1 Taxonomy \(R\) vs \(U\)

| Signal | Symbol | Classical role | When |
|--------|--------|----------------|------|
| Cand. 2 | \(R_t^{(2)}\) | Reward (policy optimization) | Improve + Stage 1 |
| Cand. 1 | \(U(\pi)=f(\pi)\) | Utility / fitness | Archive insert + suite |
| BD | \(b\in[0,1]^2\) | Behavior coordinate | Cell index; **not** in \(R_t\) or \(U\) |

---

## 2. Archive (Grid MAP-Elites)

| Item | Contract |
|------|----------|
| Grid | \(10\times 10\) on \([0,1]^2\) |
| Bin | \(i=\min(9,\lfloor\mathrm{BD}_1\cdot 10\rfloor)\); same for \(j\) |
| Payload | Full AC `state_dict` + \(f\) + BD + metadata |
| Insert | Empty → insert; else replace iff \(f'>f\) |
| Ties | Keep incumbent |

\[
\mathrm{Coverage}=\frac{|\mathrm{filled}|}{100},\quad
\mathrm{QD\text{-}score}=\sum_{c\in\mathrm{filled}}\max(0,f(e_c))
\]

---

## 3. Mosaic foes (train + fitness)

\[
P(\texttt{random})=0.25,\ 
P(\texttt{greedy\_solo})=0.25,\ 
P(\texttt{greedy\_joint})=0.50
\]

i.i.d. per episode. Mock backend: mosaic is a no-op (single sim). Live: `set_opponent`.

---

## 4. Outer loop & budget

```
bootstrap Stage0–2 (≤ B_boot train steps, Cand2, mosaic)
N_outer ← floor((T_total - steps_used) / (n_emit * 2048))
for t in 1..N_outer (stop if global_steps >= T_total):
    parents ← Uniform(filled(A), n_emit)   # or seed policy if empty
    for each parent:
        π' ← deepcopy(π)
        π' ← PPO_improve(π'; 2048 steps, R=Cand2, mosaic)  # hard-stop at T_total
        (f, b, g) ← evaluate(π')   # G=20 Cand1; overhead
        if g >= 10: try_insert(A, π', f, b)
deploy π* = argmax f
```

| Parameter | Value |
|-----------|--------|
| \(T_{\mathrm{total}}\) | `1_000_000` |
| \(B_{\mathrm{boot}}\) | `100_000` |
| \(T_{\mathrm{QD}}\) | `900_000` (nominal) |
| \(n_{\mathrm{emit}}\) | `4` |
| Steps / emit | `2048` |
| \(N_{\mathrm{outer}}\) | `109` (derived) |
| Workers | `1` (serial) |
| ME-ES noise | Off |

### 4.1 IAM under variation

\[
\pi^{(k)}=\mathrm{Softmax}\big(\mathbf{z}^{(k)}+(1-M^{(k)})(-10^{9})\big)
\]

Same mask at collect and PPO epochs (Huang & Ontañón, 2022).

---

## 5. PPO improvement (inner)

Import [`ppo_spec.md`](ppo_spec.md) §§3–6 (GAE, clip, AR IAM, buffer rules).  
Reward: **Cand. 2 only**. Foes: mosaic §3. No self-play pool.

---

## 6. Fitness & BD (asymmetric)

| Item | Contract |
|------|----------|
| \(G\) | 20 BO1, Cand. 1 |
| Policy | Greedy masked |
| Foes | Mosaic §3 |
| \(g\) | Count `archive_eligible` |
| Discard | \(g < 10\) → no archive write |
| BD | Mean over \(g\) eligible |
| \(f\) | Mean Cand. 1 over **all** \(G=20\) |

### Offline suite (on \(\pi^\star\))

200 BO1 each vs `random`, `greedy_joint`, `expectimax_bayesian` (greedy masked, `/team 1234`).  
Primary: WR(\(\pi^\star\)); secondary: Coverage, QD-score.

---

## 7. Bootstrap (O9)

| Stage | Contract |
|-------|----------|
| 0 | Optional BC → init AC |
| 1 | PPO Cand. 2 on mosaic, ≤ `100_000` train steps (counts toward \(T_{\mathrm{total}}\)) |
| 2 | Evaluate + insert; ≤ 4 aux emits **from Stage-1 checkpoint** (not random-init) |
| 3 | Outer loop |

**Forbidden:** seeding \(\mathcal{A}\) only with max-entropy / random-init nets.

---

## 8. Modules & CLI

| Module | Role |
|--------|------|
| `archive.py` | Grid, insert, coverage, QD-score, serialize |
| `mosaic.py` | Foe sampling 25/25/50 |
| `evaluate.py` | Cand. 1 fitness + BD filter |
| `improve.py` | PPO collect/update Cand. 2 |
| `train.py` | Budgeted outer loop |
| `agent.py` / `player.py` | \(\pi^\star\); `ValidatedQDPPOPlayer` |
| Reuse | `ppo/network`, `ppo/buffer`, `ppo/agent` |
| Tests | `tests/test_rl_qd_ppo.py` |

```bat
python -m showdown.rl.qd_ppo.train --seed 0 --smoke
python -m showdown.rl.qd_ppo.train --seed 0 --device cpu
```

---

## 9. Telemetry (not in \(R_t\) / \(U\))

`qd_coverage`, `qd_score`, `archive_best_f`, `emit_accept_rate`, PPO health, `illegal_action_fallbacks`.

---

## 10. Acceptance tests

1. IAM legal samples.  
2. BD §5 formulas.  
3. \(g<10\) never inserts.  
4. Binning \((0,0)\to(0,0)\), \((1,1)\to(9,9)\).  
5. Strict replace.  
6. Fitness Cand. 1 / improve Cand. 2.  
7. Independent elite `state_dict`s.  
8. \(N_{\mathrm{outer}}=\lfloor 900000/8192\rfloor=109\).  
9. Smoke ≥1 outer iter on mock; coverage ≥ 1.

---

## 11. Non-goals (v1.0)

- QD-PGA / QD-PPGA; ME-ES; CVT; shared critic  
- Expand-\(\Omega\) / self-play as primary diversity  
- BD / novelty inside \(R_t\); confuse \(R^{(2)}\) with \(U\)  
- Random-only archive seed; multi-emitter  
- Team Preview in MDP; joint-576 head; train on L2/crash  

---

## 12. References

- Mouret & Clune (2015). Illuminating search spaces by mapping elites. *arXiv:1504.04909*.  
- Schulman et al. (2017). PPO. *arXiv:1707.06347*.  
- Huang & Ontañón (2022). Invalid action masking. https://doi.org/10.32473/flairs.v35i1.130642  
- Yannakakis & Togelius (2018). *Artificial intelligence and games*. Springer.  
- [`ppo_spec.md`](ppo_spec.md), [`rl_env_spec.md`](rl_env_spec.md) §4–5.

---

## Changelog

| Version | Date | Notes |
|---------|------|-------|
| **1.0** | 2026-10-10 | Frozen: grid 10×10; full AC elites; uniform parents; PPO+IAM only; mosaic 25/25/50; \(g_{\min}=10\); \(f\) over \(G=20\) / BD over \(g\); \(T=10^6\) incl. \(B_{\mathrm{boot}}\); \(N_{\mathrm{outer}}=109\); serial; deploy \(\pi^\star\). |

---

*End of QD-PPO agent spec 1.0.*
