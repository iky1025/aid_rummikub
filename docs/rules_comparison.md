# Implemented rules vs. official Rummikub

This project plays a **simplified two-player** variant of Rummikub. This document
records exactly what we implemented and how it differs from the official rules,
so results are interpreted against the right game.

**Sources for the official rules:** [Rummikub official site](https://www.rummikub.org/rules) ·
[Wikipedia](https://en.wikipedia.org/wiki/Rummikub) ·
[Pagat](https://www.pagat.com/rummy/rummikub.html) ·
[Rules PDF](https://fgbradleys.com/wp-content/uploads/rules/Rummikub_Rules.pdf).
Our implementation: `rummikub_env.py`, `ppo_env.py`, `rummikub_solver.py`.

## Side-by-side

| Aspect | Official Rummikub | This project | Why we changed it |
|---|---|---|---|
| Tiles | **106** = 104 numbered (1–13 × 4 colors × 2) + **2 jokers** | **104**, **no jokers** | Jokers add wildcard combinatorics that blow up the action space; removing them keeps the learning signal clean. |
| Players | 2–4 | **2 only** | Head-to-head is the cleanest setting to measure skill vs. a baseline. |
| Deal | 14 tiles each | 14 tiles each | same |
| Valid sets | **Run** (≥3 consecutive, same color) · **Group** (3–4 same number, different colors) | same | same |
| Initial meld | First play must total **≥ 30 points**, from your own rack only; no table manipulation until you've melded | same (`initial_meld_value=30`, `ignore_table` until melded) | matches the official rule (our standard config) |
| Table manipulation | Allowed **after** your initial meld (rearrange any table tiles) | Allowed after meld; the solver may fully re-partition the table | matches; the ILP/DP solver performs the rearrangement |
| Turn: can't/won't play | Draw **1** tile from the pool; turn ends | Draw 1 tile; turn ends | matches |
| Pool (deck) empties | Play continues; you simply **can't draw**. Game ends when the pool is empty **and no one can play** | Drawing becomes a **no-op** (`draw_tile()` returns `None`, hand unchanged); the game does **not** end on pool-empty | approximated — see §"Deck exhaustion" |
| Going out ("Rummikub!") | Play your last tile → you win | Hand reaches 0 → immediate win/loss (`ppo_env.py:224,252`) | matches |
| End / tiebreak when no one goes out | Pool empty & stuck → player with the **lowest sum of tile *values*** wins | **100-turn cap** → outcome `"timeout"`, margin by remaining tile **count** (`ppo_env.py:267,274–280`) | engineering safeguard + count-based signal — see §"Turn cap" and §"Scoring" |
| Scoring | Winner gains the sum of opponents' remaining tile **values**; losers lose their own; joker = **30** penalty | Win/loss is binary (who empties first); a count-based margin is used only for reward / `pair_net` and only at timeout | simplified to a per-tile signal, not value arithmetic |
| Per-turn time limit | Some editions: ~1–2 min sand timer **per turn** | none | not modeled |
| First player | Chosen at setup | Optional alternation (`alternate_first_player`) for balanced training/eval | variance control, not a rule change |

## Notable differences, explained

### No jokers
The biggest rule change. Jokers are wildcards that can stand for any tile and be
retrieved/replaced during manipulation. They multiply the number of legal sets
and make the one-turn optimization harder. We dropped them so the solver and the
policy operate on a clean, fully-determined tile space.

### Deck exhaustion (`draw_tile` returns `None`)
Official Rummikub ends the game when the **pool is empty and no player can make a
move**, then compares remaining tile values. We do **not** implement that end
condition. Instead, once the deck is empty a "draw" action does nothing (the hand
stays the same, with a small penalty) and play continues until either someone
empties their hand or the turn cap fires. This means a stalled, deck-empty
position rides out to the 100-turn cap rather than ending immediately.

**Measured:** across 80 test games (student vs. greedy, meld=30) the deck reached
0 in **0 games**, and 0 games hit the turn cap — games end (someone goes out)
with a median of ~29 tiles still in the deck (minimum observed: 5). So in
practice neither the deck-empty nor the timeout path is exercised; losses are
"going-out races" the opponent wins first, not tile-count endings.

### Turn cap = 100 (the "timeout")
This is an **RL engineering safeguard, not a Rummikub rule.** A Gymnasium episode
must terminate; without a cap, a deck-empty position where both players are stuck
would loop forever. `turn_count` increments once per agent turn (own move +
opponent move), so 100 is a generous bound. In practice it almost never triggers
— across our large evaluations the timeout rate was **0%** (games end by someone
going out first). On truncation the episode is labeled `"timeout"` (not a clean
win/loss), and a count-based margin is recorded.

### Scoring by tile *count*, not tile *value*
Official scoring sums the **face values** of the tiles left on the losers' racks
(and a joker is a 30-point penalty). We use a binary win/loss (who empties their
hand first) plus, at timeout only, a margin equal to the **difference in the
number of tiles** held. So "holding a 13 and a 12" and "holding a 1 and a 2" are
equivalent in our margin, but very different under official scoring. Our reward
shaping (`+0.1 × tiles played`, endgame margin) reflects this count-based view.

## Why this matters for the results

- The agent optimizes **"empty your hand / hold fewer tiles,"** not tile-value
  arithmetic. Strategies that hoard high-value tiles are not penalized the way
  official scoring would penalize them.
- Because the game ends on going-out (not on pool-empty value comparison), the
  late-game dynamics we analyze (drawing / holding, the 100-turn tail) are a
  property of *this* implementation, not of official Rummikub.
- Any claim in this repo is a claim about **this simplified two-player, jokerless,
  count-scored variant** — not tournament Rummikub.

## Faithful-mode ideas (future work)

To narrow the gap without changing the core research: add jokers to the tile set
and solver; end the game on pool-empty-and-stuck; score by summed tile values
(joker = 30); replace the fixed turn cap with a "both players passed/stuck"
terminal check.

## Roadmap: when and in what order to restore these rules

**This is a milestone, not a backlog.** The decision of *when* to re-introduce
each simplified rule is itself a research checkpoint — recorded here so we don't
move the goalposts mid-development.

### Guiding principle — freeze the rules until the *method* is proven

The current ruleset is a **fixed testbed** for developing the learning method,
not the object of study. The object of study is the milestone *"a learned agent
beats the greedy ILP baseline"* (수준 3, pure forward-only network). Every rule
change **invalidates all baselines** — greedy-sanity win rate, the oracle
ceilings (semi ~56% / full ~89%), the autopsy "~100% winnable" result, all the
46–70% numbers. We have already paid this cost twice (the `meld=0 → 30`
re-measurement and the HiGHS-presolve-bug re-measurement); each cost a full
re-run of the evaluation suite. So: **do not change the rules while the method is
still being developed.** Changing them now = moving the goalposts *and* discarding
the ledger.

### Trigger

Restore rules **only after 수준 3 is achieved and stable on the current ruleset**
— i.e., once the pipeline (solver candidates → DAgger/ExIt distillation → mirror
evaluation) is trusted and reproducible. At that point each added rule becomes a
clean *"does the method generalize?"* experiment rather than a moving target.

### Order of incorporation — cheapest / action-space-preserving first

| Rule | Cost | Nature of the change | When |
|---|---|---|---|
| **Value-based scoring** (sum of tile *values*, joker = 30) | Low | **Reward only** — action space and solver unchanged; `net` margin is already tracked | Can come relatively early. Changes optimal *risk* behavior (when behind, gamble on high tiles / hoard low ones) and finally penalizes high-tile hoarding, which the count-based signal does not. |
| **Deck-exhaustion & turn rules** (end on pool-empty-and-stuck; drop the 100-turn cap for a "both stuck" terminal) | Medium | Game-engine change; end-condition and reward tail | Middle. In practice the current sims rarely reach deck-empty, so impact is small until value-scoring makes tile-count endings matter. |
| **Jokers** | **High** | Solver **DP state must carry wildcards** + a **new strategic dimension** (hold a joker as insurance, retrieve/replace table jokers, deny the opponent a joker) + engine | **Last, as its own round (R11+).** The flagship "full Rummikub" extension. |

Each addition **re-opens the baseline ledger** — re-measure greedy-sanity and the
oracle ceilings on the new ruleset *before* comparing anything. Budget for that
re-run up front.

### Why jokers are saved for last — and why they matter most

Jokers are deferred not because they are unimportant but because they are the
**biggest** change: wildcards explode the one-turn solver's combinatorics, and
they add genuine strategy the current variant lacks.

Crucially, jokers **revive the opponent-modeling / hand-knowledge channel that is
nearly dead in the current ruleset.** In the jokerless, rearranging-opponent
variant we established (R8–R10) that hidden-hand information is worth little
(~5% predictable, semi-oracle ceiling only ~56%) because there is no durable
blocking channel — the only lever is "how many tiles I leave in hand." Jokers
change this: holding a joker as insurance, and denying the opponent a joker sitting
on the table, are real defensive/holding moves whose value **depends on knowing
the opponent's hand.** So:

- Our conclusion *"hidden-hand info is weak, opponent modeling (the aux head)
  doesn't help"* is **ruleset-specific.** Under jokers it may reverse — the
  opponent-hand prediction line (DouZero+/Suphx-style aux, currently shelved as
  net-zero) could become worth revisiting.
- Jokers likely make the game **less luck-saturated and more skill-rich**, so
  learning gains may be *largest* there. That is a strong reason to eventually do
  them — but only once the method is trusted, so we are not chasing a moving
  target.

### Summary

Freeze now. After 수준 3 is proven on the current rules: **value-scoring first
(reward-only, low cost) → deck/turn rules → jokers last (own round).** Re-measure
baselines at every step. Keep jokers in mind as the point where the shelved
hand-prediction research may come back to life.
