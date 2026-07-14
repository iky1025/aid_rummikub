# Chapter Plan + INSIGHT Collection
**Working title:** *Breaking the Imitation Ceiling: Distilling a Search-Based Teacher into a Forward-Only Policy for Two-Player Rummikub*

Produced via `academic-paper` **plan mode** (Socratic). Feeds `full` mode drafting.

- **Paper type:** ML workshop / conference short paper (IMRaD), ~6 pages (~4,000 words)
- **Main text language:** English · **Citations:** ML numeric `[n]` recommended (finalize at full-mode intake)
- **Lead framing (locked):** "pure network reaches **teacher-level** play" (robust); "empirically **exceeds** the noisy teacher (70.3% vs 56.9%)" presented as a striking observation via the ExIt lens.

---

## INSIGHT Collection

**[INSIGHT: thesis_statement]** Distilling a search-based teacher's move decisions (information-consistent PIMC rollout + endgame win-forcing DFS) into a **forward-only network** yields teacher-level (empirically teacher-exceeding) play in a two-player hidden-information game, and — proven by ablation — the resulting edge comes from the **network's move selection**, not the solver's candidate generation. Secondary: first learned agent to significantly beat the optimization baseline (greedy ILP) in this game.

**[INSIGHT: framing_decision]** Lead with "matches teacher-level" (defensible); present "exceeds" (70.3% vs 56.9%) as a surprising observation explained by the ExIt effect (distillation averages out the noisy 1-ply teacher's per-move variance).

**[INSIGHT: intro_gap]** Primary = *method gap*, framed **empirically, not as absolute novelty** (components — DAgger, ExIt, PIMC, game distillation — are individually known; the *combination + the collapse→breakthrough contrast in this class of hidden-info tile game* is undemonstrated). Secondary = *domain gap* (no significant learned edge over the optimization baseline in 2-player Rummikub). **TODO:** verify gap against literature (lit-review) before full draft.

**[INSIGHT: contribution_claim]** (user's own words)
1. Establishes that in hidden-info games like Rummikub, DAgger can reach teacher-level play.
2. What disappears if removed: a *method to solve hidden-info games — a model that solves the problem instead of RL*.
3. Who decides differently: **game designers** — to add axes that make such models harder to solve.
> Q1(ⓐ) scoping (locked): claim #2 scoped to "demonstrated in a representative hidden-info tile game"; general method offered as an explicit **conjecture** in future work.

---

## Chapter Plan

### Abstract (~180 words)
Key sentence: "A forward-only network beats the optimal one-turn baseline 70.3% (mirror 160 pairs, ~7σ), and ablations attribute this edge entirely to the network's move selection, not the solver's candidate generation."

### 1. Introduction (~700 words)
- **Core argument:** Search is strong but slow in hidden-info games; off-policy distillation of that intelligence collapses under covariate shift — so can a pure reactive network inherit a search teacher's skill?
- **Evidence/promise:** headline contribution = the DAgger distillation pipeline (teacher → student-trajectory relabel → forward-only policy); results promise (collapse→breakthrough, ablation, generalization).
- **Gap:** empirically-scoped method gap (see INSIGHT). Contribution bullets: (1) pipeline [method], (2) 28.7%→70.3% collapse-vs-breakthrough [empirical], (3) ablation attributes edge to selection [empirical], (4) first significant learned edge in this game [secondary].

### 2. Background / Related Work (~600 words)
- **Core argument:** each component is known; combining them so a pure network reaches teacher-level in a hidden-info game is open.
- **4 threads (evidence):** ⓐ PIMC / hidden-info (Long et al. 2010 — strategy fusion, non-locality; Maven/Suphx/DouZero); ⓑ DAgger / covariate shift (Ross et al. 2011; **On-Policy Distillation for Noisy Experts, arXiv 2606.30923**); ⓒ ExIt / distillation (Anthony et al. 2017; AlphaZero); ⓓ Rummikub optimization (van Rijn & Takes 2016).
- **Positioning (locked):** *consistent with* known PIMC limits — uniform determinization is worthless here; our contribution is the information-consistent + endgame-search recovery and its distillability. (NOT a counter-example to a naive belief.)

### 3. Method (~900 words)
- **M1 Teacher:** information-consistent PIMC rollout (det=8) + endgame win-forcing DFS + greedy-margin guard. Hyperparameters → appendix (mirror-pair tuned); emphasize the *student* recipe is nearly tuning-free.
- **M2 DAgger collection:** student plays (visits its own state distribution); teacher relabels every decision (+ per-candidate rollout scores / endgame votes).
- **M3 Distillation:** state encoder + permutation-invariant candidate encoder + score head; loss = soft CE (teacher rollout margins) + value regression; mirror-pair evaluation.
- **Defenses (woven):** circularity → random-opponent generalization + disjoint train/eval seeds; "why not RL?" → R1–R7 PPO plateaus ~50% (2–3 sentences, scoped to *our setting*); "pure network?" → candidate enumeration = **rules engine** (legal-move generator), all *selection* is the network (ablation), full end-to-end = future work.

### 4. Experiments (~1,000 words) — argument spine
| # | Experiment | Answers | Result |
|---|---|---|---|
| E1 | **Main:** student vs greedy (160 pairs) | Does it win? | **70.3%** / pair net +0.99 / ~7σ |
| E2 | Control: same recipe, no DAgger data | Is DAgger the cause? | **28.7%** collapse (draw-spamming) |
| E3 | **Ablation:** identical candidate set, 3 selection functions | Source of the edge? | random 40.6 / greedy(max-tiles) 46.2 / **network 70.3** |
| E4 | Deviation instrumentation + **derived: deviation quality** | Not a copier + teacher-aligned? | 34.2% override, 19.5% strategic holding; won/lost parity (34.4 vs 33.7); **deviation quality (30 pairs, 1,191 dec, fresh seeds 4000+): overall student↔teacher agreement 75.1%; when teacher deviates → student deviates 77.4% (exact-move 48.7%); when teacher greedy → student stays greedy 82.5%. Honest nuance: student deviates more than teacher (34.2% vs 21.9%) — possible source of teacher-exceeding.** |
| E5 | Generalization: student vs random | Opponent-specific? | **70.3%** (vs greedy's 52.5% vs random) |
| E6 | Large-scale confirmation (1,000 pairs, fresh seeds) | Tighten significance | **[server run: PENDING, SE~1.4%]** |
- **Setup note:** mirror-pair (duplicate-deal) evaluation for variance reduction; meld=30; seeds disjoint across train/tune/eval.
- **Headline = E1; rigor spine = E3.** Two surprises reported: exceeds teacher; random < greedy.

### 5. Discussion (~500 words)
- **Dialogue (ExIt foregrounded):** distilled policy matching/exceeding a search teacher — extended from perfect-info tree search to **hidden-info + a noisy PIMC teacher**. DAgger/noisy-expert = supporting; PIMC limits = brief scoping.
- **"What makes it win" subsection (faces the uncomfortable data):** ablation (selection is the lever) → instrumentation (active, not a copier) → won/lost parity (quality > quantity) → deviation quality (teacher-aligned) → honest boundary (luck-heavy game; mirror eval cancels luck).
- **Take-home:** "In hidden-info games, the intelligence of expensive search can be compiled into a reactive network that needs no search at inference."
- **Implications:** practical (cheap inference → search-level play for real-time / low-resource) primary; methodological transferability = explicit conjecture (future work).

### 6. Conclusion (~200 words)
- Heart paragraph: off-policy distillation collapses under covariate shift in this hidden-info game; DAgger distilling the search teacher into a forward-only network breaks the copier ceiling and reaches teacher-level (empirically exceeding) play, with the edge coming from the network's selection.
- **Future work (primary):** fully end-to-end network (candidate generation too) → complete "pure network."

### Limitations (~150 words, mandatory)
Solver-dependent candidate generation (scoped as rules engine); teacher-bounded strategy discovery (cannot invent strategies the teacher never demonstrates — ExIt limit); single game / single rule (meld=30) / single teacher config → cross-game generalization unproven; luck-heavy game → absolute win-rate ceiling is game-specific.

---

## Open TODOs before / during full draft
1. **Literature verification** (lit-review or deep-research) of the method-gap and domain-gap claims — the user's standing caveat.
2. Fill **E4 devquality** number (deviation-quality: student↔teacher alignment) — analysis running.
3. Fill **E6 large-scale** (server 1,000-pair suite: student vs greedy / random, greedy references) — running.
4. Decide citation style + venue at full-mode intake.
5. Figures (visualization_agent): E3 ablation bar (3 selection functions) as the key figure; E1/E5 win-rate with CIs.
