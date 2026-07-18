# R10 gate ③ — execution plan

Live plan for the semi-oracle teacher gate and everything queued behind it.
Written 2026-07-17. Companion docs: `docs/code_audit.md` (the audit this plan
schedules), `ROADMAP.md` (R10 strategy), `CLAUDE.md` (round log).

## Where we are

```
VERDICT (2026-07-19): oracle teacher REJECTED. Branch B is now active.
  additive pool (stage1s+dagger1+dagger2), 160 pairs × seeds 0/1/2:
    56.2% / 65.0% / 70.3%  → mean 63.8% / +0.25  vs baseline 70.3% / +0.99
  No seed beats baseline; swap pool (no dagger1) collapsed at 32.5%.
  Imitation gap confirmed: mismatch still 36% yet no lift → π*(a|O,h)
  is unlearnable from O. See CLAUDE.md "③ 게이트 판정".

Baseline to beat : 70.3%  (distill_s1s_dagger1.pt, 160 pairs, seeds 2000–2159)
Server           : data/dagger2 COMPLETE (500 pairs, retrieved). Droplet KEPT
                   (user) for the Branch B regeneration below.
Phase A          : DONE (label-convention guard, hard-coded 21 removed,
                   STATE_DIM comment, docs/code_audit.md + efficiency section).
Uncommitted      : Phase A + audit docs + verdict record in the working tree.
Local assets     : data/dagger2, symlink pools data/s1s_d1_d2 (additive) &
                   data/s1s_d2 (swap), distill_dagger2_add_s{0,1,2}.pt.
```

**Next: execute Branch B (below).** First fix Finding 5 (event-history encoding),
then regenerate with a strong *blind* teacher. This is now a round boundary, so
pipeline changes are permitted — but the solver must not be touched until the
Branch B baseline is itself established (same freeze logic re-applies).

**The gate question.** dagger1 was labelled by the *fair combo* teacher
(`--consistent`, 56.9%, blind). dagger2 is labelled by a *semi-oracle* teacher
(`--oracle hand`, 60%, sees the true opponent hand — legal because it only
labels, offline; the student never receives the hand). Everything else is
identical. **Does a +3%p stronger teacher lift the student past 70.3%?**

---

## STEP 1 — completion & retrieval (~20 min once the server finishes)

1. `ssh rummi ls /root/aid_rummikub/data/dagger2 | wc -l` → expect **500**
2. `rsync -az rummi:/root/aid_rummikub/data/dagger2/ data/dagger2/`
3. **🔴 Destroy the DO Droplet.** It bills hourly and has no further use.
4. **Guard check (free sanity):** run `distill.check_label_convention` over
   dagger2. If the oracle teacher violates the convention, every deviation /
   mismatch statistic in the verdict is meaningless — check *before* judging.
5. **Health check** vs dagger1: decisions, teacher-deviation rate, student–teacher
   mismatch. Smoke predicted ≈19,600 decisions, ~16% deviation, ~36% mismatch.

## STEP 2 — re-distillation (~10 min)

Two pools × three seeds = six runs, ~40 s each. Build pools as symlink dirs
(seed ranges do not collide: stage1s 210000+, dagger1 220000+, dagger2 230000+).

| Pool | Contents | Question it answers |
|---|---|---|
| **additive** | stage1s(1000) + dagger1(500) + dagger2(500) | does *adding* oracle data push further? |
| **swap** | stage1s(1000) + dagger2(500) | at equal scale, is the **oracle teacher > fair teacher**? (isolates teacher strength) |

```
distill.py --data <pool> --soft-temp 0.3 --value-coef 0.5 --epochs 10 --seed {0,1,2}
```

The recipe must stay identical to the 70.3% run — data is the only variable, or
the attribution is not clean.

## STEP 3 — verdict (~1.5 h, local; the server is 10× slower)

1. 40-pair smoke (seed 2000) on each pool's seed-0 model → pick the promising pool
2. 160-pair confirm (seeds 2000–2159) vs **70.3%**
3. **Judge by win rate only.** `val_ce` must NOT be compared across pools —
   ~45% of candidate slots are byte-identical duplicates, so `val_ce` carries a
   tie floor (~0.84 nats) whose size depends on the pool (see `code_audit.md`
   Finding 1).
4. If borderline, run all three seeds at 160 pairs (n=1 drowns in ±3–4%p init noise).

---

## STEP 4 — the branch

### 🟢 Branch A — dagger2 clearly > 70.3%
Hand information *does* transfer through distillation; the oracle line is live.

- **Start ① (Expert Iteration).** Wire the critic — **the value head already
  exists and is simply unconnected** (`ActorCritic.forward_value`); regress it on
  the stored `outcome`. Use **dagger\* shards only**: `outcome` is the result of
  whoever acted, so mixing teacher-acted shards would blend `V^teacher` and
  `V^student`.
- Strengthen the teacher further (semi-oracle + deeper search: `--search-nodes`↑,
  `--determinizations`↑) for the next DAgger round.
- Phase B order: Finding 3 → 4 → 2. **Finding 5 is not needed** — an oracle
  teacher's extra information is the true hand, which `--history` can never supply.

### 🔴 Branch B — dagger2 ≈ or < 70.3% (saturation)
Imitation gap confirmed. The reasoning is already set up: mismatch is still 36%,
so a flat result is *not* "nothing left to learn" — it is "these corrections are
unlearnable from the observation", i.e. the student cannot represent
`π*(a|O,h)` when it only sees `O`.

- **Finding 5 is promoted to first priority.** A stronger *blind* teacher reads
  the opponent-event history to build its determinization, so its gap is exactly
  the one `--history` *can* close — but the encoding is broken (left-aligned, no
  validity bit) and has never had a fair test.
- Then regenerate with a **strong blind teacher**: `--consistent` kept, det 16 /
  cap 6 / `--search-nodes` 400 (~2 days). Pure quality gain, zero imitation gap.
- ① is still worth starting (critic wiring is branch-independent).

**Branch B is not unlikely.** The hand is only ~5% predictable and the
semi-oracle ceiling is ~56%, so the oracle teacher's edge is thin to begin with;
imitation-gap theory predicts the labels may not transfer at all.

---

## STEP 5 — Phase B (after the verdict; no regeneration; each measured with `--seed 0/1/2`)

1. **Finding 3 — diagnose first (5 min).** Log `val_soft_ce` next to `val_ce` and
   check whether they select the same epoch. If yes, change nothing. If no,
   align the metric or just save the last epoch (10 epochs; selection is noisy).
2. **Finding 4.** Exclude vote-derived entries from the score regression only
   (`is_vote = isnan(cand_scores) & ~isnan(cand_votes)`); soft CE keeps them.
3. **Finding 5.** Right-align + validity bit (24 → 6×5 = 30 d) in `event_feats`,
   then re-evaluate `--history` in the DAgger regime. *(1st priority under branch B.)*
4. **Finding 2.** Rename `--value-coef` → `--score-coef`.
5. **Deferred from Phase A.** `ppo_env` candidate sentinel (`None` = not computed,
   `[]` = none exist) — held back because `ppo_env` is shared by the live
   generation run and the pending gate eval.

## STEP 6 — Phase C (with ①, or as dedicated experiments)

1. **Critic wiring** → the leaf evaluator ① and ② both need.
2. **Finding 1 dedupe as an A/B**, not a "fix": the soft-CE optimum is unchanged
   by duplicates, so the win-rate benefit is uncertain. The part worth testing is
   **crowding-out** (at the 20-cap, duplicates displace real moves).
3. **Candidate coverage** — `solve_many` Phase 2 excludes by `selected_indices`
   and therefore mostly emits arrangement variants, contradicting its own comment.
   Fix by excluding on `remaining_hand`, or use `--exhaustive`. **The only item
   that needs regeneration** (~2 days); measured upside ~14–16% more moves found.

## Side experiments (promised; run when the verdict frees up time)

- **`aux_diag` on a dagger2-trained model** — does hand predictability exceed the
  5% variance-explained ceiling in this endgame-heavier distribution? (~1 min)
- **Window-stitching feasibility** — consecutive decisions store overlapping
  6-turn windows, so the full opponent-event history may be reconstructible
  offline. If so, K=12 / cumulative features become testable **without
  regeneration**. (~30 min; risk: gaps >6 turns between recorded decisions.)

---

## Standing rules

| Rule | Why |
|---|---|
| **Freeze the pipeline during an experiment** | changes invalidate baselines; make them at round boundaries |
| **Multi-seed (`--seed 0/1/2`) always** | the aux lesson: n=1 sweeps invent monotone trends out of ±3–4%p init noise |
| **Never compare `val_ce` across pools** | tie floor differs per pool (audit Finding 1) |
| **Never rsync code to a live server run** | the wrapper self-restarts and would re-exec new code, splitting shards across versions |
| **Mirror-pair evaluation only** | no single-game win-rate comparisons |

## Known risks

- **DAgger diminishing returns are already visible**: absolute correction volume
  is down 44% vs dagger1 (5,134 → 2,860 per 200 pairs), from fewer decisions
  (games are 8 turns shorter because the actor is much stronger: 66.7% vs 33.3%
  in-run win rate) *and* a lower mismatch rate (48.0% → 36.5%). Even under branch
  A, the next round's gain should be smaller.
- **The budget map says the remaining headroom is search depth, not information**:
  hand knowledge is capped at ~+6%p (semi-oracle 56%), deck foresight (~89% full
  oracle) is unpredictable in principle, and the autopsy proved the real ceiling
  vs greedy is ~100% with the gap being *search depth*.
