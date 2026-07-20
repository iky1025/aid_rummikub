# Official Rummikub Rules — Authoritative Reference

A complete, audited reference for the **official** Rummikub game, compiled for the
2-player AI research project so our simplified variant can be checked against the
real rules. Where sources disagree or a rule is ambiguous, it is flagged
explicitly with a ⚠ marker.

**Primary sources used**

- Official Rummikub rulebook (Pressman/Goliath / M&M Ventures, doc `D-2600-1236-0041`,
  "The Original Rummikub"), the sheet packed with retail sets —
  [PDF](https://rummikub.com/wp-content/uploads/2019/12/2600-English-1.pdf).
  Quotations below labelled **[Official rulebook]** are from this document.
- **NZRC 2024 Official Playing Rules** (New Zealand Rummikub Club tournament rules,
  representative of Rummikub International Federation / Sabra tournament play) —
  [PDF](https://rummikub.co.nz/wp-content/uploads/2024/05/NZRC-Offical-Playing-Rules.pdf).
  Labelled **[NZRC tournament]**.
- [Wikipedia — Rummikub](https://en.wikipedia.org/wiki/Rummikub).
- [Pagat — Rummikub](https://www.pagat.com/rummy/rummikub.html).
- [Official Rummikub site rules](https://www.rummikub.org/rules).

The de-facto standard modern ruleset is the **Sabra** version; that is what the
official retail rulebook and tournaments use, and what this document describes
unless noted.

---

## 1. Equipment & Setup

| Item | Value |
|---|---|
| Total tiles | **106** |
| Numbered tiles | **104** = numbers **1–13**, in **4 colors**, **2 copies** of each (8 "sets" of 1–13 × 4 colors) |
| Colors | black, blue (sometimes shown as a lighter blue/orange), red, orange (blue, red, orange, black) |
| Jokers | **2** |
| Racks | one per player |
| Players | 2–4 (Rummikub XP / Mini XP uses 160 tiles + 4 jokers for 5–6 players — not standard) |
| Starting hand | **14 tiles** per player |
| Pool ("stock") | all remaining tiles after the deal, face down |

**Setup [Official rulebook]:** all tiles are placed face down and mixed. Each
player draws one tile; **highest number goes first**. Tiles are returned and
mixed; it is recommended to stack them in piles of 7. Each player takes 14 tiles.
The rest form the **pool**.

> ⚠ **Common misconception — 106 vs 104.** People often say "104 tiles." The full
> set is **106 including the 2 jokers**; 104 is only the numbered tiles. Our project
> uses 104 (jokers removed).

---

## 2. Initial Meld ("First Move") — the 30-point rule

**[Official rulebook], verbatim intent:**

> "In order to make an initial meld, each player must place tiles on the table in
> one or more sets that total at least 30 points. These points must come from the
> tiles on each player's rack; for their initial meld, players may not use tiles
> already played on the table. A joker used in the initial meld scores the value of
> the tile it represents."

Precise breakdown:

- **Threshold:** the first tiles a player lays down must total **≥ 30 points**.
- **Value = sum of the face numbers** of the tiles laid down (a red 10 + blue 10 +
  orange 10 = 30). The count is the **numbers in the sets you place**, not a count
  of tiles.
- **Multiple sets allowed:** the 30 may be **spread across one or more sets** laid
  in the same turn (e.g. a run 3-4-5 = 12 plus a group of four 9s = 36 → 48; or two
  runs). It does **not** have to be a single set. [NZRC tournament #4: "the first
  set **or sets** … must add up to a minimum of 30 points."]
- **From your own rack only:** you may **not** use tiles already on the table, and
  you may **not** manipulate/rearrange existing table sets, until after you have
  melded. [NZRC #4: "A player may not add or manipulate any sets on the table prior
  to, or during, their initial meld."]
- **Jokers in the initial meld:** **permitted.** A joker counts as **the value of
  the tile it represents** (a joker standing in for a 10 contributes 10 to the 30).
- **Delaying the meld:** a player who cannot (or chooses not to) reach 30 simply
  **draws a tile and passes**, and may keep delaying entry turn after turn. [NZRC #5]

> ⚠ **Common misconceptions / edge cases**
> - "The 30 must be one set." **False** — sets can be combined to reach 30.
> - "You can borrow a table tile to reach 30." **False** — initial meld is rack-only.
> - **Tournament penalty for never melding** [NZRC #12]: if the game ends and you
>   never made your initial meld, you are charged **100 points** if you genuinely
>   could not have melded 30 face value — but **200 points** if you *could* have
>   melded and chose not to. (This penalty is tournament-specific; the retail
>   rulebook has no such rule.)

---

## 3. Valid Sets

Two and only two kinds of sets [Official rulebook]:

### Group
- Same **number**, **different colors**.
- Size **3 or 4** tiles.
- Because there are only 4 colors, a group is at most 4 tiles.
- ⚠ **A group may NOT contain two tiles of the same color.** All colors in a group
  must be distinct. (This matters because the deck has 2 copies of each colored
  number — you cannot make `red9 red9 blue9` a legal group.)

### Run (sequence)
- **Consecutive numbers**, all the **same color**.
- Size **3 or more** (up to 13: a full 1–13 same-color run).
- ⚠ **Runs do NOT wrap around.** "The number 1 is always played as the lowest
  number, **it cannot follow the number 13**." So `12-13-1` and `13-1-2` are
  **illegal**. 1 may only precede 2. [Official rulebook; NZRC #8; Wikipedia]

> ⚠ **Duplicate-tile subtlety (relevant to solvers).** Since each colored number
> exists **twice**, the table can legitimately hold **two identical sets** at once
> (e.g. two separate `red9 blue9 black9` groups, or the same run twice). A solver
> that models set-selection as binary (each distinct set chosen 0/1 times) is
> **wrong** — it can miss optimal plays that need the same set twice. (This exact
> bug was found and fixed in this project's ILP; see CLAUDE.md "ILP 원죄 버그".)

---

## 4. Manipulation / Rearranging the Table

After a player has made their initial meld, on any later turn they may **rearrange
existing table sets** — adding to them, splitting them, and recombining them —
freely, provided the constraints below hold. [Official rulebook: "Manipulation"]

**The single hard constraint:**

> "…as long as at the end of each [turn] only legitimate sets remain and no loose
> tiles are left over." [Official rulebook]

So **mid-manipulation the table may be temporarily invalid**, but when the player
declares the turn finished, **every tile on the table must belong to a valid group
or run** — no leftovers, no illegal sets.

**You must also play at least one tile from your own rack.** Manipulation is not an
end in itself: a turn in which you touch the table must result in **at least one
tile leaving your rack onto the table** (otherwise you have not "played" — you
would draw instead). This is implied by the win condition (you win by emptying
your rack) and made explicit for jokers (§5) and in tournament rules.

**Canonical manipulation moves** [Official rulebook examples]:

1. **Add tiles to a set** — e.g. run `blue 4-5-6` on the table, add `blue 3` from
   rack to make `blue 3-4-5-6`; add a 4th color to a group of three.
2. **Remove the 4th tile from a group** and reuse it — take `blue 4` out of a
   group of four 4s to complete `blue 3-4-5-6`, leaving a still-legal group of
   three 4s.
3. **Add-and-remove** — add `blue 11` to a run `8-9-10-11`, freeing the `8`s to
   form a new group.
4. **Split a run** — `4-5-6-7-8` + a rack `6` → `4-5-6` and `6-7-8`.
5. **Combined / multiple split** — break several sets and reassemble them all into
   new legal sets in one turn.

There is **no explicit limit on how much you may rearrange** in a single turn, as
long as the end state is fully legal — this is the strategic heart of the game.

**Time limit on manipulation** — see §8.

> Note on this project: our solver performs a **full re-partition** of the table
> each turn (ILP/DP), which is the maximal form of legal manipulation. Because the
> opponent can always re-optimize the arrangement, **only the multiset of tiles
> matters, not their arrangement** — so "arrangement-based blocking" does not exist
> against a re-arranging opponent (see CLAUDE.md "수 공간 전수조사").

---

## 5. Jokers

**Two jokers.** [Official rulebook: "The Joker"]

### As wildcards
- A joker "can be used as **any tile** in a set, and its number and color are that
  of the tile needed to complete the set."
- It can fill a slot in either a group or a run.

### Point value
- **In a meld / initial meld:** counts as the value of the tile it represents (a
  joker as a 12 counts 12). [Official rulebook, §2]
- **As an end-of-game penalty on your rack:**
  - **Retail / home rules: 30 points.** [Official rulebook: "the penalty for
    having a joker on a rack is 30 points."]
  - ⚠ **Tournament (NZRC): 100 points.** [NZRC #11: "Each joker remaining on the
    rack at the end of the game counts as 100 points."] **This is a real
    discrepancy** — home = 30, this tournament ruleset = 100. Some other tournament
    circuits also use 30; confirm with the organizer.

### Joker Retrieval / Replacement — the crucial rule
This is the most misunderstood rule. **[Official rulebook], verbatim:**

> "On future turns, a joker can be retrieved from a set on the table by a player who
> can replace it during his/her turn with any tiles that can keep the set
> legitimate. This tile can come from the table or from a player's rack. In the case
> of a group of three tiles, the joker can be replaced by a tile of either of the
> missing colors. When a player retrieves a joker, the joker will once again have
> any value or color. **However, a player who retrieves a joker must play the joker
> on his/her current turn to make a new set, and must also use at least one tile
> from his/her rack on that turn** (just as on any other turn). **A player cannot
> retrieve a joker before s/he has played his/her initial meld.**"

Precise breakdown of retrieval:

1. **You replace the joker** with the actual tile it currently represents. The
   replacement tile may come **from your rack OR from elsewhere on the table**.
   The set it was in must remain legal after the swap.
2. **Group-of-three special case:** a joker in a 3-tile group can be replaced by a
   tile of **either** missing color (you don't need to supply both missing colors).
3. **You must immediately re-use the freed joker THIS SAME TURN**, as part of a
   **new** set. You cannot pick a joker up and hold it for later.
4. **You must also play at least one tile from your own rack that turn** (the joker
   coming off the table doesn't count as "your play").
5. **You cannot retrieve a joker until you have made your own initial meld.**

Four illustrated ways to clear a joker [Official rulebook]:
- Replace with a matching rack tile (or two) directly.
- Split a run so the joker's position is filled by real tiles, freeing it.
- Add a real tile so the run extends and the joker becomes redundant, freeing it.
- Multiple split across groups (move real tiles into groups, freeing the joker).

> ⚠ **Common misconceptions**
> - "You can take a joker and keep it on your rack." **False** — a retrieved joker
>   must be replayed the same turn.
> - "Retrieving a joker is your whole move." **False** — you must also lay at least
>   one rack tile.
> - "You can grab the opponent-blocking joker anytime." **False** — only after your
>   own initial meld, and only if you can legally replace it.
> - [NZRC #9]: "(The joker cannot be returned to one's rack) at least one tile from
>   the player's rack must be used on the same turn." Confirms both constraints.

---

## 6. Drawing

- **When you draw** [Official rulebook]: "When players cannot play any tiles from
  their racks, **or purposely choose not to**, they must draw a tile from the pool.
  After they draw, their turn is over."
- So drawing is **allowed even when you *could* play** — you may voluntarily draw
  (e.g. to hold tiles back strategically) instead of melding. [NZRC #5 confirms:
  a player may keep drawing "although he could meld one or more sets from his
  rack."] Each turn is therefore **either** "lay one or more tiles" **or** "draw
  one tile," never both.
- **You draw exactly ONE tile**, and drawing **ends your turn immediately.**
- **You cannot play a tile you just drew** — "Players cannot lay down a tile they
  just drew; they must wait until their next turn." [Official rulebook]
- Tiles never move from the table back to the rack (except as a penalty). [NZRC #3]

> ⚠ **Common misconception:** "If you can play, you must play." **False** in
> official rules — you may choose to draw instead (a genuine strategic option,
> e.g. holding back the 4th tile of a set).

---

## 7. Winning & Scoring

### Going out
- A player wins by **playing the last tile off their rack** and calling
  **"Rummikub!"** — this ends the game. [Official rulebook]

### Pool exhausted, nobody out
- If the pool empties and no one has gone out, "play continues until **no more
  plays can be made**." The game then ends. [Official rulebook]
- The winner is then the player with the **lowest total tile value** remaining.
  [Official rulebook; NZRC #16 adds each player gets one final turn first.]

### Scoring (per game)
[Official rulebook]:
- Each non-winner sums the **face values** of the tiles left on their rack as a
  **negative** number.
- The **winner receives a positive score equal to the total of all the losers'
  points** (so the winner's + equals the sum of the others' −; they cancel to
  zero across the table — a useful checksum).
- **Joker left on rack = 30-point penalty** (retail) / **100** (NZRC tournament,
  §5 above).

Worked example from the rulebook (4 players, one game):

| | A | B | C | D |
|---|---|---|---|---|
| Tiles left (value) | +24 (winner) | −5 | −16 | −3 |

Here A went out; A scores +24 = 5 + 16 + 3.

### Pool-empty tiebreak scoring
[Official rulebook]: the player with the **lowest tile value** wins the round;
each player subtracts their tile total from the winner's total (negative for each),
and the sum of those negatives is scored to the winner as a positive.

### Tournament aggregation
- A **round is made of multiple games**: with *N* players, a round = *N* games
  (2 players → 2 games per round; 4 players → 4 games). Players agree how many
  rounds to play. [Official rulebook]
- Players track **both** cumulative points **and the number of games each has won.**
- **Overall winner** [Official rulebook, "Winning"]: "the player who has won the
  **most games** in all rounds combined." **Ties broken by highest cumulative
  score.** So the primary tournament metric is **games won**, with total points as
  the tiebreaker.

> ⚠ **Ambiguity / variation.** Different tournament circuits weight this
> differently — some rank purely by cumulative points, some by games won then
> points (as above), some by a fixed number of rounds with cut-offs. The retail
> rulebook uses "most games won, ties by points." Always confirm the circuit's
> scoring.

---

## 8. Time Limits & Tournament-Specific Rules

### Per-turn time limit
- **Retail rulebook: 1 minute per player per turn.** "Players who go over the time
  limit must **draw a tile from the pool, ending their turns**." [Official rulebook,
  "Time Limit"]
- ⚠ **Tournament (NZRC): 40 seconds per turn.** [NZRC #7] **Discrepancy** — retail
  = 60s, this tournament = 40s. (A common sand-timer edition uses ~2 min; times
  vary by edition/circuit.)

### Timeout during a manipulation ("incomplete move")
- **Retail:** "Players who cannot complete a move within the … time limit must
  **replace the tiles that were on the table to their previous positions, take back
  the tiles they played, and draw 3 tiles from the pool as a penalty.** This ends
  the turn." [Official rulebook, "Incomplete Runs"]
- **Tournament (NZRC #7):** same idea — all tiles return to original positions and
  **3 penalty tiles** are drawn. Other players at the table help restore the board;
  any leftover tiles whose original set is forgotten are picked up onto the
  offending player's rack **plus** the 3 penalty tiles.

### Other tournament rules [NZRC]
- **#13:** at game end, no one empties their rack until the referee has counted the
  points remaining on all racks.
- **#14 (fresh start):** if a player is dealt **3 "doubles"** (two tiles identical
  in both number and color) before play starts, they may request a fresh deal for
  the whole table.
- **#12 (no-meld penalty):** never melding → **100 points** (couldn't have) or
  **200 points** (could have but chose not to). See §2.
- **#16:** pool-empty endgame gives every player one more turn before counting.
- **#15:** organizers reserve the right to amend rules.

> ⚠ **Illegal-move penalties** beyond the timeout rule (e.g. playing an illegal set
> and being caught) are **not standardized in the retail rulebook**; tournament
> circuits handle them via the referee. The clearest codified penalty is the
> incomplete-manipulation rule (revert + 3 tiles).

---

## 9. Two-Player Specifics

Rummikub is designed for 2–4 and **the core rules are identical for 2 players.**
Points that are clarified or specific to head-to-head:

- **Round length:** with 2 players a **round = 2 games** [Official rulebook], and
  positions/first-player naturally alternate across the two games — worth noting
  for balanced play (our project uses mirror-pair evaluation for exactly this
  variance reason).
- **Scoring is a zero-sum transfer:** with only one opponent, the winner's positive
  score exactly equals the single loser's remaining tile value (with the joker at
  30/100). This makes 2-player scoring a clean signed margin.
- **First player:** still determined by the highest drawn tile at setup.
- **No dedicated 2-player rule changes** exist in the official rulebook — there is
  no special deal size, no altered meld threshold, no different joker handling for
  head-to-head. All §1–§8 rules apply unchanged.

---

## 10. Quick Reference — Numbers That Matter

| Rule | Retail / official | Tournament (NZRC) |
|---|---|---|
| Total tiles | 106 (104 numbered + 2 jokers) | same |
| Colors × numbers × copies | 4 × 13 × 2 | same |
| Starting hand | 14 | 14 |
| Initial meld minimum | 30 (rack-only, jokers at represented value) | 30 |
| Group | 3–4 tiles, same number, **distinct colors** | same |
| Run | ≥3 tiles, consecutive, same color, **no wrap (1 can't follow 13)** | same |
| Draw per turn | exactly 1, ends turn | same |
| May draw even if able to play | **yes** | yes |
| Joker penalty on rack | **30** | **100** |
| Never-melded penalty | none stated | 100 (couldn't) / 200 (could, chose not to) |
| Per-turn time limit | **60 s** → over-time = draw + end turn | **40 s** |
| Incomplete-manipulation penalty | revert board + draw **3** | revert board + draw **3** |
| Retrieved joker | must be replayed same turn + play ≥1 rack tile; only after own meld | same |
| End | someone empties rack ("Rummikub!") OR pool empty & no plays possible | same (+1 final turn each) |
| Winner (per game) | most tiles-value from losers | same |
| Overall tournament winner | most **games won**, ties by total points | varies by circuit |

---

## 11. Points Most Often Gotten Wrong (Summary Flags)

1. **106 not 104** tiles (the 2 jokers).
2. **Initial meld is rack-only** and **can be split across several sets**; jokers
   count at their represented value.
3. **Runs never wrap** — `13-1-2` is illegal.
4. **Groups can't repeat a color**; but the **table can hold two identical sets**
   (2 copies of each tile exist) — a real trap for set-selection solvers.
5. **You may voluntarily draw even when you can play.**
6. **Joker retrieval:** must replace it legally, **replay it the same turn**, also
   play **≥1 rack tile**, and **only after your own initial meld** — you can't pocket
   a joker.
7. **Joker penalty differs: 30 (home) vs 100 (this tournament ruleset).**
8. **Per-turn time limit differs: 60 s (home) vs 40 s (tournament);** over-time and
   failed-manipulation penalties (draw 1, or revert + draw 3) are real rules people
   ignore in casual play.
9. **Manipulation may pass through illegal intermediate states**, but the table
   must be **fully legal at end of turn with no loose tiles.**
10. **Tournament ranking is by games won, not raw points** (in the official
    rulebook) — a frequently-missed aggregation detail.
