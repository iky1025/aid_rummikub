# Semantic code audit (2026-07-17)

A full read-through of the decision/label/observation pipeline looking for
*semantic* defects — places where the code is internally consistent but encodes
the wrong meaning, or where a statistic does not measure what its name claims.
Every claim below was verified against real data (`data/dagger1`, 120 pairs /
6,366 decisions) rather than reasoned about abstractly.

**Headline: none of these require regenerating data.** The raw material needed
for every fix is already in the npz shards (`cand_scores`/`cand_votes` stored
*separately*, `events` with raw `-1` padding, `outcome`, `mask`).

---

## Finding 1 — 45% of candidate slots are byte-identical duplicates 🔴

`solve_many` Phase 2 dedupes by `selected_indices` (the set *selection*), but
`tiles_to_vector` encodes tiles as **counts**, dropping the partition. Two
solutions that play the same hand tiles and re-arrange the table differently
therefore produce **byte-identical `cand_feats`**.

Measured (dagger1, 6,366 decisions):

| | |
|---|---|
| decisions containing a duplicate | **58.0%** |
| candidate slots that are duplicates | **44.9%** |
| distinct moves per decision | **mean 6.8, median 2** (20 slots offered) |
| decisions where all 20 slots are one move | **500 (7.9%)** |
| decisions that hit the cap (`n == 20`) | **50.3%** |
| …of those, containing duplicates | **57.2%** (avg **14.4/20** slots wasted) |
| irreducible CE floor from ties | **0.84 nats** |

Root cause is stronger than "redundancy": Phase 2's own comment says it excludes
solutions "to find different **tile-selection** variants", but excluding
`selected_indices` lets the solver return the *same tile selection arranged
differently*. **Phase 2 does not do what it claims.** Phase 1 (one candidate per
tile count) supplies the ~5.6 genuinely distinct moves; Phase 2 fills the rest
with arrangement noise.

**Consequences, graded honestly:**

- ⚠️ **`val_ce` carries a tie floor (~0.84 nats) that depends on the pool's
  duplicate structure → comparing `val_ce` across different data pools is
  invalid.** (Observed live: a 1-epoch dagger1 run reports `val_ce=1.886`, i.e.
  ~45% of it is irreducible.)
- ⚠️ **Crowding-out**: at the 20-cap, duplicates displace real moves. Those moves
  were never stored, so load-time dedupe cannot recover them. Independently
  measured elsewhere: `enumerate_moves` finds **~16% more moves** than
  `solve_many` (272 vs 234 over 163 positions).
- Wasted solver + network compute (identical inputs encoded up to 20×).
- ✅ **Not** a bias in the soft-CE optimum: the duplicate multiplicity `m` enters
  target and model identically and cancels. Deduping is hygiene/speed/
  interpretability — **not** an obvious win-rate gain.

**Fix.** Load-time dedupe by `cand_feats` + remap `action` (free, no
regeneration); `eval_mirror`'s student path must dedupe identically. Fixing
coverage (Phase 2 excluding by `remaining_hand`, or `--exhaustive`) is a
*separate, optional* experiment that **does** need regeneration.

---

## Finding 2 — `--value-coef` is not a value function; the critic is dead 🔴

- `--value-coef` regresses the **actor's logits** to `teacher_score / 50`
  (`distill.py`). It is a per-action Q-like score calibration, **not** a state
  value `V(s)`. (This is why `eval_mirror --student-margin` works — its comment
  notes "logits in score units".)
- `ActorCritic.critic` / `forward_value` exist but are **never called during
  distillation** → dead parameters saved in every checkpoint.
- Semantic tension: CE is shift-invariant (only logit *differences* matter) while
  the regression pins their *absolute* values. Both pull on the same logits.

**What distillation actually optimizes today** (`--soft-temp 0.3 --value-coef 0.5`):
soft CE against `softmax(teacher_scores/50/0.3)` on rows with ≥2 evaluated
options; one-hot CE elsewhere; plus `0.5 × MSE(logits, scores/50)`. The student
learns `Q(s,a) ≈ teacher's rollout score` and plays `argmax Q`.

**Fix.** Rename to `--score-coef`. For Expert Iteration, wire `forward_value`
to the stored `outcome` field — **the value head ①/② need already exists, it is
merely unconnected.** Caveat: `outcome` is the result of *whoever acted*, so
train the critic on **dagger\* (actor=student) shards only**, or it mixes
`V^teacher` and `V^student`.

---

## Finding 3 — checkpoint selection metric ≠ training objective 🟠

Training minimises soft CE + score regression (+ dev weight). `evaluate()`
computes **plain one-hot CE only**, and the save rule is `if val_ce < best_val`.
We train a decathlete and pick the day by the 100 m time.

One-hot CE rewards *copying the teacher's exact index* — the very behaviour that
capped Stage 1 at the "greedy copier" ceiling — while soft CE + calibration is
what let the student surpass its teacher (70.3% > 56.9%). `val_ce` is also
polluted by Finding 1's tie floor. Precedent: R7 selected checkpoints on
`avg_reward` and the choice was effectively random.

**Fix — diagnose before changing.** Log `val_soft_ce` alongside and check whether
`argmin(val_ce)` and `argmin(val_soft_ce)` pick the same epoch. If yes, nothing
to fix. If no, align the metric or simply save the last epoch (10 epochs; the
selection is noise-prone anyway).

---

## Finding 4 — `merged` fuses two incompatible scales 🟠

```python
merged = where(isnan(cand_scores), cand_votes * 50, cand_scores)
```

- `cand_scores`: 1-ply rollout leaf value, ~[-50, +50], **0 = even position**
- `cand_votes * 50`: endgame win-forcing **vote fraction** × 50, [0, 50],
  **0 = no forced win found**

The same number `0` means two different things. The docstring's claim of "one
calibrated scale" is false. Soft CE is unharmed (monotone in votes → ordering
preserved); only the **absolute** regression breaks.

**Fix.** Exclude vote-derived entries from the `--value-coef` regression only
(`is_vote = isnan(cand_scores) & ~isnan(cand_votes)`), keeping them for soft CE.
The two channels are stored separately, so this is a train-time change.

---

## Finding 5 — opponent-event history is mis-encoded 🟡

`_event_history` left-aligns and pads at the **end**, so the slot holding "the
most recent turn" **drifts with game phase** (index 2 early, index 5 once ≥6
events exist). The MLP must learn "count the non-zero rows" from scratch, and
early- vs mid-game history are encoded incompatibly. There is also **no validity
bit** — padding is zeroed, yet `drew=0` is a legitimate value.

`EVENT_HISTORY_LEN = 6` has **no recorded rationale**: `git log -S` puts it in the
first distillation commit, before `--history` existed as a training option, and
its own docstring calls it "Compact history for *future* sequence encoders" — a
schema placeholder, never tuned.

This matters because `--history` was written to close a *real* imitation gap: the
`--consistent` teacher's rejection sampler reads exactly this history, so without
it part of the teacher's deviations is unlearnable from the observation. It was
judged "+3~4%p, noise range" — but measured (a) in the collapsing off-policy
regime and (b) with this handicapped encoding. **The idea has never had a fair test.**

**Fix.** Right-align, add a validity column (24 → 6×5 = 30 dims) in
`event_feats` — the raw `-1`-padded `events` are stored, so no regeneration —
then re-evaluate `--history` in the DAgger regime.

**Asymmetry worth remembering:** a *blind* (`--consistent`) teacher's extra
information **is** the event history, so `--history` can close that gap. An
*oracle* teacher's extra information is the true hand, which the student can
never receive — `--history` cannot help there.

---

## Finding 6 — a load-bearing convention that holds by coincidence 🟡

The teacher's action is always the **lowest index of its duplicate group**
(`rollout_agent` keeps the first occurrence per `remaining_hand` in `chosen`, and
candidate 0 heads its own group). `torch.argmax` independently breaks ties toward
the **lowest index**. These two facts happen to agree, which is the only reason
`val_acc` and the `action != 0` deviation statistic mean anything.

Verified: **0 violations** across stage0 / stage1 / stage1s / dagger1. Also
verified: of 571 `action != 0` rows, **0** are byte-identical to candidate 0 —
so the deviation and student–teacher mismatch statistics reported throughout
CLAUDE.md are **clean**, not inflated by duplicates.

But the alignment is undocumented and would break **silently** if candidate
ordering changed (e.g. `enumerate_moves` sorts by `-used_hand_tile_count`).

**Fix (done).** `check_label_convention()` in `distill.py` fails loudly at load.

---

## Minor

| Item | Problem | Fix |
|---|---|---|
| hard-coded `21` | missing optional fields padded to `(n, 21)`; silently wrong if `max_candidates != 20` | derive `n_actions` from `mask.shape[1]` — correct by construction, no check needed **(done)** |
| `STATE_DIM` comment | said `opp_hand`; it is the opponent's hand **count** | comment fixed **(done)** |
| candidate regeneration | `last_candidates == []` conflates "not computed" with "computed, zero candidates" → the solver is re-run every forced-draw turn | sentinel: `None` = not computed, `[]` = none exist. **Deferred** — touches `ppo_env`, shared by the live generation run and the pending gate eval; behaviour-neutral, buys only speed |

---

## Verified clean (audit dead-ends, recorded so they are not re-litigated)

Two alarms raised during the audit were **disproved by measurement**:

1. *"deviation statistics are corrupted by duplicates"* → **No.** 0 of 571
   `action != 0` rows are identical to candidate 0. The mismatch rates
   (32.1% / 36.5% / 48.0%) stand.
2. *"val_acc is capped at 0.643"* → **Wrong calculation.** argmax's
   lowest-index tie-break coincides with the label convention, so there is no
   cap (see Finding 6 — but it is a coincidence).

Also confirmed correct:

- **Scaling is consistent across all three paths**: selfplay stores `×2` (raw
  counts) → `distill` divides by 2 → `eval_mirror` consumes the env's already
  `/2`-scaled obs directly. No train/eval skew.
- **`event["drew"] = used == 0` cannot mislabel**: `solve_many` only returns
  candidates with `used_hand_tile_count ≥ 1`, so `used == 0` ⟺ a real draw.
- **train/val split is leak-free**: split by pair seed keeps both mirror seats of
  a game together; training seeds (100000+…230000+) never collide with the
  evaluation seeds (2000–2159).

---

## Efficiency analysis (profiled 2026-07-17)

Measured, not reasoned: `cProfile` over 4 teacher decisions in a real mid-game
position (oracle+endgame teacher, det 8 / turns 24 / cap 4), **13.5 s per decision**.

| | time | share |
|---|---|---|
| **`rummikub_dp.solve`** (3,802 calls, **12.8 ms each**) | **45.1 s** tottime | **84%** |
| `dict.setdefault` — **17,495,092 calls** | 2.0 s | DP inner loop |
| `list.append` — **36,888,234 calls** | 1.0 s | DP inner loop |
| `highspy._core.run` (ILP) — 57 calls | 0.63 s | **1.2%** |

~950 DP solves per decision, matching the arithmetic: det 8 × cap 4 × turns 24 =
768 solves for the 1-ply part, plus the endgame DFS.

### This closes the "server is 10× slower" mystery

CLAUDE.md listed it as unresolved, with the hypothesis that our bottleneck is a
**pure-Python interpreter loop** that gets no numpy/BLAS acceleration. **The
profile confirms it**: 84% of runtime is DP `solve`, whose interior is ~54M
dict/list operations; HiGHS accounts for 0.63 s. The workload is 100%
interpreter-bound, which fully explains M4 vs a 2.0 GHz virtualised x86. The
open TODO can be closed.

### Free win available: 54% of solve calls are exact repeats

There is **no cache anywhere in the solver** (`_my_turn_wins`'s `memo` caches
search *outcomes*, not solves). Keying on
`(hand, table, min_play_value, ignore_table, require_use_at_least_one, exact_k)`
over 6 real decisions:

```
solve calls   : 446
distinct keys : 205
redundant     : 241 (54.0%)      most-repeated keys: 19, 16, 10, 8, 8, 8 calls
```

The 8 determinizations keep re-solving the same (hand, table) positions.

### Levers, by benefit / cost

| # | Lever | Gain | Cost | Risk |
|---|---|---|---|---|
| **B-0** | **solve memo cache** | **2.3× measured** (13.52 → 5.85 s/decision) | done | verified byte-identical |
| **C-0** | **numba/Cython the DP** | 5–10× on the 84% → **~4–7× overall** | 1–2 d | tie-break drift (below) |
| B-0+C-0 | cache then compile | **~8–15×** | | turns a 2-day regeneration into 3–6 h |
| — | more workers / CPU-optimised droplet | ~3× | billing | low |
| — | shrink det / turns / cap | linear | 0 | ❌ **changes the teacher** — not a free speedup |

### B-0 done (2026-07-19) — the cache belongs in `RummikubDP.solve`, not the wrapper

First implemented at the `RummikubILPSolver._solve_via_dp` layer → only **1.2×**,
because `enumerate_moves` and `feasible()` call `self.dp` **directly**, bypassing
the wrapper. Moving the memo into `RummikubDP.solve` itself (keyed on
`(hand_counter, table_counter, min_play_value)`, sentinel-guarded so a cached
`None` is distinguished from a miss) covers every path → **2.3× measured**,
uncached DP calls 3802 → 1670 (−56%, matching the 54% redundancy prediction).

Safety, all passing:
- `solver_regression.py` — byte-level fingerprint of solve / solve_many /
  enumerate over 500 positions + 20 full deterministic greedy games. **MATCH.**
  (The harness itself had to be de-randomised first: a `{...}` string-set in the
  position generator made it hash-order-dependent across processes.)
- The paranoid DP/ILP crosscheck coexists with the cache (it re-validates cached
  results) — no error over a greedy-sanity run.
- Greedy mirror sanity = **exactly 50.0%** — end-to-end behaviour unchanged.

The cached `(used, sets)` is shared, not copied: every consumer reads via
`list(s)` / `Counter(s)` / `flatten` (audited), so there is no aliasing. Any
future caller that mutates a returned arrangement must copy first.

### Both are branch-independent, but neither is gate-independent

Branch A (ExIt: search-improved labels) and branch B (regenerate with a stronger
blind teacher at det 16 / cap 6 / nodes 400 ≈ 3–4× cost) are **both** multiples of
solve cost, so B-0/C-0 pay off either way and belong in the queue regardless.
They must nevertheless wait for the verdict:

- `eval_mirror` uses the solver. Changing it mid-gate would measure the 160-pair
  verdict on a different solver than the 70.3% baseline was measured on.
- **Precedent**: the `upBound=2` fix strengthened the greedy opponent and
  invalidated the entire baseline suite (sanity 48.8 / fair 50.9 / semi 55.9 …).
- **The specific hazard is tie-breaking.** The DP backtracks to reconstruct an
  arrangement; a re-implementation that visits states in a different order returns
  a *different but equally optimal* arrangement → different `selected_indices` →
  different candidate ordering → the greedy opponent's exact move can change, and
  Finding 6's label convention can break. **The paranoid crosscheck will not catch
  this**: `dp_crosscheck_every` compares the *objective* (DP vs ILP tile count),
  not which arrangement was returned. A byte-level regression harness (same seeds
  → identical candidate lists) is needed, plus a greedy-sanity re-measure.

### Scale does not help where it is being asked to

- **More training data is already a rejected hypothesis**: "동종 데이터 증량은 무효
  (500페어 학습과 지표 동일) — 병목은 표본 수가 아니라 라벨 노이즈/분포".
- **DAgger returns are diminishing**, measured here: correction volume −44% vs
  dagger1 (5,134 → 2,860 per 200 pairs).
- **Variance actually lives in evaluation and in training seeds**: 160 pairs give
  ±0.27–0.37 pair net, and init noise is ±3–4%p (the aux sweep invented a monotone
  63→70 trend out of exactly this).
- **Therefore: spend any speed gains on more seeds and more evaluation pairs, not
  on more training data.** Regeneration only becomes worth its cost when the
  *distribution* changes (branch B's stronger blind teacher).

## Phased plan

**Phase A — immediate, zero behavioural change (done 2026-07-17)**
- `check_label_convention()` guard + documented convention (Finding 6)
- `n_actions` derived from the data (hard-coded 21)
- `STATE_DIM` comment
- **Procedural: `val_ce` must not be compared across data pools** (Finding 1) —
  the pending gate judges the additive vs swap pools by **win rate only**.

**Phase B — after the gate verdict; no regeneration; measure each with `--seed 0/1/2`**
- Finding 3 diagnosis → fix only if the metrics disagree
- Finding 4 (exclude vote rows from the score regression)
- Finding 5 (right-align + validity bit) → re-evaluate `--history`
- Finding 2 rename; the deferred `ppo_env` sentinel

**Phase C — with Expert Iteration, or as dedicated experiments**
- Finding 1 dedupe as an **A/B** (uncertain win-rate benefit — soft-CE optimum
  is unchanged; crowding-out is the part worth testing)
- Finding 2: wire the critic to `outcome`, **dagger shards only**
- Candidate coverage (Phase 2 exclusion by `remaining_hand`, or `--exhaustive`)
  — the **only** item that needs regeneration (~2 days); measured upside ~14–16%

## Interaction with the running dagger2 gate

- **Do not rsync code to the server while it runs.** The wrapper is a
  self-restart loop; new code would be picked up on any restart, so the shards
  would split across two code versions mid-dataset.
- **Finding 5 is a prerequisite for gate branch 1.** If dagger2 saturates
  (imitation gap confirmed → switch to a stronger *blind* teacher), that teacher
  reads the event history, so its gap is exactly the one `--history` can close —
  and the encoding must be fixed first.
- **Finding 6's guard doubles as a gate sanity check**: it verifies the
  oracle teacher obeys the same label convention when dagger2 lands. If it did
  not, every deviation/mismatch statistic in the verdict would be wrong.
