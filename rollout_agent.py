"""R8: Determinized-rollout policy (Maven/PIMC style).

For each ILP candidate move, sample the hidden information (opponent hand +
deck order) from the unseen-tile pool several times, play the game forward
greedy-vs-greedy for a bounded number of turns, and score the outcome. Pick
the candidate with the best average score.

No learning involved — this is the "quantum jump" baseline to beat greedy ILP.
"""

import random
from collections import Counter

from rummikub_solver import (
    COLORS,
    NUMBERS,
    COPIES_PER_TILE,
    Tile,
    RummikubILPSolver,
    flatten,
)


FULL_DECK = [
    Tile(color, number)
    for _ in range(COPIES_PER_TILE)
    for color in COLORS
    for number in NUMBERS
]

WIN_SCORE = 50.0

# A joker held mid-game is a strategic ASSET (fills any gap later / completes a
# winning set), not a liability. The tile-count leaf would otherwise score a
# held joker as +1 tile in hand = worse, actively penalising holding and
# fighting the draw / joker-aware playout. Credit it instead: in the leaf a held
# joker counts as this many tiles of ADVANTAGE. (Crude proxy; Layer 2's learned
# value replaces it.) Terminal win/loss is unchanged — to go out you must play
# the joker, so at terminal it is genuinely gone.
JOKER_LEAF_BONUS = 1.5


class RolloutPolicy:
    def __init__(
        self,
        n_determinizations=8,
        max_rollout_turns=12,
        candidate_cap=6,
        greedy_margin=0.0,
        oracle=False,
        consistent=False,
        max_consistency_events=6,
        max_sample_retries=12,
        endgame_search=False,
        search_nodes=120,
        search_deck_trigger=16,
        search_hand_trigger=4,
        search_prune_hand=6,
        joker_fix=False,
        seed=0,
    ):
        self.n_determinizations = n_determinizations
        self.max_rollout_turns = max_rollout_turns
        self.candidate_cap = candidate_cap
        # v2: stick with the greedy play (candidate 0) unless another
        # candidate beats it by more than this per-rollout margin. Guards
        # against the winner's curse on noisy rollout estimates — worst case
        # degrades to greedy instead of below it.
        self.greedy_margin = greedy_margin
        # Oracle mode: read the TRUE opponent hand and deck order instead of
        # sampling determinizations. Cheating — for measuring the ceiling of
        # rollout-over-greedy, never for a fair agent.
        # oracle="full": true hand + true deck order (total ceiling).
        # oracle="hand": true hand, deck order sampled — ceiling of a perfect
        #                opponent-hand predictor (deck order is pure luck).
        self.oracle = oracle
        # R9: information-consistent determinization. Against the greedy
        # opponent every draw is a hard constraint: "that hand had no playable
        # move vs that table" (pre-meld: "couldn't reach the 30 threshold").
        # Sample opponent hands by rejection against the trailing draw streak
        # since their last play (after a play their remaining hand is
        # uninformed again, so the streak is the whole usable signal).
        # NOTE: sound only vs the deterministic greedy opponent — a human
        # draw may mean "chose not to play".
        self.consistent = consistent
        self.max_consistency_events = max_consistency_events
        self.max_sample_retries = max_sample_retries
        # R9-3: endgame win-forcing search. When the game is nearly decided
        # (deck low / a hand small), run a bounded DFS per determinization —
        # the opponent is deterministic, so "can this move force a win?" is
        # exactly decidable within the node budget. A found forced win
        # overrides the 1-ply rollout score.
        self.endgame_search = endgame_search
        self.search_nodes = search_nodes
        self.search_deck_trigger = search_deck_trigger
        self.search_hand_trigger = search_hand_trigger
        self.search_prune_hand = search_prune_hand
        # R11 joker experiments (default OFF = the plain teacher that generated
        # fair2/dagger2). When on: voluntary draw-to-hold, joker-aware playout,
        # guaranteed joker-hold candidate, and a leaf that credits held jokers.
        self.joker_fix = joker_fix
        self.rng = random.Random(seed)
        self.solver = RummikubILPSolver()
        # R10: per-decision evaluation record for distillation targets.
        # After select_action returns, holds None (nothing was evaluated) or
        # {"scores": {cand_idx: avg rollout score}, "votes": {action: frac},
        #  "n_det": k}. "votes" uses env.max_candidates as the draw action.
        self.last_eval = None

    def select_action(self, env):
        """Pick an action for RummikubPPOEnv `env` on the PPO player's turn.

        Uses only public information: own hand, table, opponent hand COUNT,
        deck COUNT. Hidden tiles are sampled, never read from env.env.deck.
        """
        self.last_eval = None
        # Bound solver memory: the determinization rollouts + endgame DFS below
        # do thousands of solves per decision, and their DP cache accumulates
        # across turns otherwise (jokered fat entries -> tens of GB over a long
        # game, OOM). Clear at each decision boundary; the speedup comes from
        # within-decision re-solves, so no throughput is lost. No game-length
        # cap needed -> long collapse games (the DAgger correction signal) are
        # preserved.
        self.solver.clear_cache()
        candidates = env.last_candidates
        if not candidates:
            return env.max_candidates  # forced draw

        # Immediate win available — take it.
        for i, result in enumerate(candidates):
            if len(result.remaining_hand) == 0:
                return i

        # Candidates whose remaining hand is identical are equivalent for the
        # race; keep the first of each (solve_many orders max-k first).
        chosen = []
        seen_hands = set()
        for i, result in enumerate(candidates):
            key = tuple(sorted((t.color, t.number) for t in result.remaining_hand))
            if key in seen_hands:
                continue
            seen_hands.add(key)
            chosen.append(i)
            if len(chosen) >= self.candidate_cap:
                break
        # H1 (joker_fix): guarantee a joker-HOLDING candidate (plays tiles but
        # keeps the joker) is evaluated. Such moves play fewer tiles, so the
        # tile-ordered cap drops them first — the strategic option we must not
        # lose. Already in the env set (student's obs shows it); we only ensure
        # the teacher rolls it out. One is enough. Scan the FULL list, not cap.
        if self.joker_fix:
            for i, result in enumerate(candidates):
                if any(t.is_joker for t in result.remaining_hand):
                    key = tuple(sorted((t.color, t.number) for t in result.remaining_hand))
                    if key not in seen_hands:
                        seen_hands.add(key)
                        chosen.append(i)
                    break
        if len(chosen) == 1:
            return chosen[0]

        my_hand = list(env.hands[env.ppo_player])
        table_sets = [list(s) for s in env.env.table_sets]
        opp_count = len(env.hands[env.ilp_player])
        ppo_pre_meld = not env.first_meld_done[env.ppo_player]
        opp_meld_done = env.first_meld_done[env.ilp_player]
        initial_meld_value = env.initial_meld_value

        unseen = Counter(FULL_DECK)
        unseen.subtract(Counter(my_hand))
        unseen.subtract(Counter(flatten(table_sets)))
        unseen_list = []
        for tile, count in unseen.items():
            unseen_list.extend([tile] * count)

        if self.oracle == "full" or self.oracle is True:
            # True hidden state; playouts are then deterministic, so one
            # "determinization" is exact.
            determinizations = [
                (list(env.hands[env.ilp_player]), list(env.env.deck))
            ]
        elif self.oracle == "hand":
            true_opp = list(env.hands[env.ilp_player])
            deck_pool = Counter(unseen_list)
            deck_pool.subtract(Counter(true_opp))
            deck_base = []
            for tile, count in deck_pool.items():
                deck_base.extend([tile] * count)
            determinizations = []
            for _ in range(self.n_determinizations):
                deck = list(deck_base)
                self.rng.shuffle(deck)
                determinizations.append((list(true_opp), deck))
        elif self.consistent:
            events = getattr(env, "opponent_events", [])
            determinizations = [
                self._sample_consistent(
                    unseen_list, opp_count, events, initial_meld_value
                )
                for _ in range(self.n_determinizations)
            ]
        else:
            determinizations = []
            for _ in range(self.n_determinizations):
                sample = list(unseen_list)
                self.rng.shuffle(sample)
                determinizations.append((sample[:opp_count], sample[opp_count:]))

        if self.endgame_search:
            # Endgame = a hand is small (deck-independent: a game can be nearly
            # decided while the pool still has tiles). The forward search stays
            # cheap not by requiring an empty deck but by PRUNING any branch
            # whose hand grows past search_prune_hand (see _my_turn_wins): a
            # forced win keeps the hand shrinking, so a branch that accumulates
            # tiles isn't a quick win and is dropped. Hence all explored nodes
            # stay small -> the uncapped exhaustive enumeration is cheap.
            near_end = min(len(my_hand), opp_count) <= self.search_hand_trigger
            if near_end:
                choice = self._endgame_choice(
                    env, candidates, chosen, my_hand, table_sets,
                    determinizations, opp_meld_done, initial_meld_value,
                )
                if choice is not None:
                    return choice

        scores = [0.0] * len(chosen)
        draw_score = 0.0
        draw_meld_done = not ppo_pre_meld   # drawing completes no meld
        # Only weigh a voluntary draw when there is a reason to hold: a joker in
        # hand. Elsewhere declining to play only loses tempo, so skipping the
        # draw rollout there costs nothing and keeps the teacher fast (Layer 1).
        my_has_joker = self.joker_fix and any(t.is_joker for t in my_hand)
        for opp_hand, deck in determinizations:

            for j, ci in enumerate(chosen):
                result = candidates[ci]
                # Mirror ppo_env._meld_params: candidates only leave the table
                # untouched (and thus must be appended) in initial-meld mode.
                if ppo_pre_meld and initial_meld_value > 0:
                    new_table = table_sets + [
                        list(s.completed_tiles) for s in result.selected_sets
                    ]
                else:
                    new_table = [
                        list(s.completed_tiles) for s in result.selected_sets
                    ]
                scores[j] += self._rollout(
                    my_hand=list(result.remaining_hand),
                    opp_hand=list(opp_hand),
                    table_sets=[list(s) for s in new_table],
                    deck=list(deck),
                    my_meld_done=True,
                    opp_meld_done=opp_meld_done,
                    initial_meld_value=initial_meld_value,
                )

            # Change 2: voluntary draw — decline to play, draw the next tile,
            # keep the whole hand (incl. the joker), opponent to move. Lets
            # "draw to save the joker for later" compete with spending it now.
            if my_has_joker:
                d = list(deck)
                drawn = [d.pop()] if d else []
                draw_score += self._rollout(
                    my_hand=list(my_hand) + drawn,
                    opp_hand=list(opp_hand),
                    table_sets=[list(s) for s in table_sets],
                    deck=d,
                    my_meld_done=draw_meld_done,
                    opp_meld_done=opp_meld_done,
                    initial_meld_value=initial_meld_value,
                )

        # Record avg scores for distillation targets, propagated to candidates
        # that were deduped away (same remaining hand => same score).
        n_det = len(determinizations)
        chosen_score = {}
        for j, ci in enumerate(chosen):
            key = self._mkey(candidates[ci].remaining_hand)
            chosen_score[key] = scores[j] / n_det
        score_map = {}
        for i, result in enumerate(candidates):
            key = self._mkey(result.remaining_hand)
            if key in chosen_score:
                score_map[i] = chosen_score[key]
        # Voluntary-draw score in the draw slot (index == max_candidates) so
        # selfplay records it into cand_scores[draw] and distill's soft target /
        # deviation weighting see it (a draw-to-hold is a deviation from greedy).
        if my_has_joker:
            score_map[env.max_candidates] = draw_score / n_det
        self.last_eval = {"scores": score_map, "votes": None, "n_det": n_det}

        best_j = max(range(len(chosen)), key=lambda j: scores[j])
        # chosen[0] is always the greedy max-play; scores are paired (same
        # determinizations per candidate), so the difference is low-variance.
        # Options: best play vs voluntary draw (only weighed with a joker in
        # hand); deviate from the greedy baseline (chosen[0]) only by the margin.
        threshold = self.greedy_margin * len(determinizations)
        if (my_has_joker and draw_score > scores[best_j]
                and draw_score - scores[0] > threshold):
            return env.max_candidates            # draw to hold the joker
        if scores[best_j] - scores[0] <= threshold:
            return chosen[0]
        return chosen[best_j]

    @staticmethod
    def _mkey(tiles):
        return tuple(sorted((t.color, t.number) for t in tiles))

    def _meld_mode(self, meld_done, imv):
        if meld_done or imv <= 0:
            return 0, False
        return imv, True

    def _opp_reply(self, opp, table, deck, om, imv):
        """Deterministic greedy reply. Returns (opp, table, deck, om, won)."""
        min_val, ignore = self._meld_mode(om, imv)
        r = self.solver.solve(
            hand_tiles=opp, table_sets=table,
            require_use_at_least_one_hand_tile=False,
            min_play_value=min_val, ignore_table=ignore,
        )
        if r.status == "Optimal" and r.used_hand_tile_count > 0:
            new_sets = [list(s.completed_tiles) for s in r.selected_sets]
            table = table + new_sets if ignore else new_sets
            opp = list(r.remaining_hand)
            return opp, table, deck, True, len(opp) == 0
        deck = list(deck)
        if deck:
            opp = opp + [deck.pop()]
        return opp, table, deck, om, False

    def _enum_my_moves(self, hand, table, mm, imv, cap=8):
        min_val, ignore = self._meld_mode(mm, imv)
        # Exhaustive enumeration only for small (endgame) hands: 2^|hand| subset
        # feasibility checks (dp.feasible, which handles jokers), so the win-
        # forcing search is COMPLETE exactly where it matters — the endgame —
        # while staying bounded. Larger hands fall back to a capped diverse set
        # (exhaustive there costs hundreds of ms per node).
        # NOTE (2026-07-23): R11 b36829b re-routed this through the generating DP
        # (dp.generate_moves) for speed. That was a double regression: it (a)
        # capped the endgame at 8 too (losing the completeness this small-hand
        # branch gives), and (b) enumerates the joker move space combinatorially
        # -> a single joker-heavy endgame node ran for MINUTES (py-spy: suffixes
        # recursion), which also filled the solve cache to OOM. Reverted.
        moves = None
        if len(hand) <= 6:
            moves = self.solver.enumerate_moves(
                hand_tiles=hand, table_sets=table,
                min_play_value=min_val, ignore_table=ignore,
                subset_limit=512,
            )
        if moves is None:
            moves = self.solver.solve_many(
                hand_tiles=hand, table_sets=table, max_solutions=4,
                require_use_at_least_one_hand_tile=True,
                min_play_value=min_val, ignore_table=ignore,
            )
        # No cap: in the endgame WHICH tiles you play decides the win, so the
        # search must branch on ALL enumerated moves. Feasibility comes from
        # keeping the exhaustive path to genuinely small hands (few subsets ->
        # few moves), not from truncating the candidate list.
        out = []
        for r in moves:
            new_sets = [list(s.completed_tiles) for s in r.selected_sets]
            new_table = table + new_sets if ignore else new_sets
            out.append((list(r.remaining_hand), new_table))
        return out

    def _win_after_my_move(self, my, opp, table, deck, mm, om, imv, budget, memo):
        if not my:
            return True
        opp2, table2, deck2, om2, opp_won = self._opp_reply(
            list(opp), [list(s) for s in table], list(deck), om, imv,
        )
        if opp_won:
            return False
        return self._my_turn_wins(my, opp2, table2, deck2, mm, om2, imv, budget, memo)

    def _my_turn_wins(self, my, opp, table, deck, mm, om, imv, budget, memo):
        budget[0] -= 1
        if budget[0] <= 0:
            return False
        # Prune: a forced win keeps the hand shrinking (you go out). A branch
        # that has drawn the hand past the endgame bound is not a quick win, so
        # drop it — this also keeps every explored node small enough for the
        # exhaustive per-node enumeration to stay cheap.
        if len(my) > self.search_prune_hand:
            return False
        key = (self._mkey(my), self._mkey(opp),
               self._mkey(flatten(table)), len(deck), mm, om)
        if key in memo:
            return memo[key]
        memo[key] = False  # cycles are not wins
        for new_hand, new_table in self._enum_my_moves(my, table, mm, imv):
            if self._win_after_my_move(
                new_hand, opp, new_table, deck, True, om, imv, budget, memo,
            ):
                memo[key] = True
                return True
        d = list(deck)
        h = list(my)
        if d:
            h.append(d.pop())
        if self._win_after_my_move(h, opp, table, d, mm, om, imv, budget, memo):
            memo[key] = True
            return True
        return False

    def _endgame_choice(
        self, env, candidates, chosen, my_hand, table_sets,
        determinizations, opp_meld_done, initial_meld_value,
    ):
        """Vote per root option (chosen candidates + draw) on whether it
        forces a win. Returns an action, or None to fall back to 1-ply."""
        ppo_pre = not env.first_meld_done[env.ppo_player]
        votes = [0] * (len(chosen) + 1)
        for opp_hand, deck in determinizations:
            for j, ci in enumerate(chosen):
                r = candidates[ci]
                if ppo_pre and initial_meld_value > 0:
                    new_table = table_sets + [
                        list(s.completed_tiles) for s in r.selected_sets
                    ]
                else:
                    new_table = [
                        list(s.completed_tiles) for s in r.selected_sets
                    ]
                if self._win_after_my_move(
                    list(r.remaining_hand), list(opp_hand),
                    [list(s) for s in new_table], list(deck),
                    True, opp_meld_done, initial_meld_value,
                    [self.search_nodes], {},
                ):
                    votes[j] += 1
            d = list(deck)
            h = list(my_hand)
            if d:
                h.append(d.pop())
            if self._win_after_my_move(
                h, list(opp_hand), [list(s) for s in table_sets], d,
                not ppo_pre, opp_meld_done,
                initial_meld_value, [self.search_nodes], {},
            ):
                votes[-1] += 1

        # Record win-forcing vote fractions for distillation targets (draw is
        # keyed as env.max_candidates), propagated to deduped-away candidates.
        n_det = len(determinizations)
        chosen_vote = {
            self._mkey(candidates[ci].remaining_hand): votes[j] / n_det
            for j, ci in enumerate(chosen)
        }
        vote_map = {
            i: chosen_vote[self._mkey(r.remaining_hand)]
            for i, r in enumerate(candidates)
            if self._mkey(r.remaining_hand) in chosen_vote
        }
        vote_map[env.max_candidates] = votes[-1] / n_det
        self.last_eval = {"scores": None, "votes": vote_map, "n_det": n_det}

        best = max(votes)
        if best == 0:
            return None
        j = votes.index(best)  # earliest max: prefers max-play, draw last
        if j == len(chosen):
            return env.max_candidates
        return chosen[j]

    def _trailing_draw_streak(self, events):
        """Opponent draw events since their last play, oldest first."""
        streak = []
        for event in reversed(events):
            if not event["drew"]:
                break
            streak.append(event)
        streak.reverse()
        return streak[-self.max_consistency_events:]

    def _sample_consistent(self, unseen_list, opp_count, events, initial_meld_value):
        """Sample (opp_hand, deck) consistent with the opponent's draw streak.

        Constructive: sample the streak-start hand, verify it had no playable
        move at each streak turn (adding one random tile per successful draw),
        retry on violation. Falls back to an unconstrained sample."""
        streak = self._trailing_draw_streak(events)

        fallback = None
        for _ in range(self.max_sample_retries):
            pool = list(unseen_list)
            self.rng.shuffle(pool)
            if not streak:
                return pool[:opp_count], pool[opp_count:]

            hand = pool[: streak[0]["hand_before"]]
            next_tile = len(hand)
            ok = True
            for event in streak:
                if event["pre_meld"] and initial_meld_value > 0:
                    min_val, ignore_tbl = initial_meld_value, True
                else:
                    min_val, ignore_tbl = 0, False
                r = self.solver.solve(
                    hand_tiles=hand,
                    table_sets=event["table"],
                    require_use_at_least_one_hand_tile=True,
                    min_play_value=min_val,
                    ignore_table=ignore_tbl,
                )
                if r.status == "Optimal" and r.used_hand_tile_count > 0:
                    ok = False
                    break
                if event["hand_after"] > event["hand_before"]:
                    hand.append(pool[next_tile])
                    next_tile += 1

            sample = (list(hand), pool[next_tile:])
            if ok and len(hand) == opp_count:
                return sample
            fallback = sample if len(hand) == opp_count else fallback

        if fallback is not None:
            return fallback
        pool = list(unseen_list)
        self.rng.shuffle(pool)
        return pool[:opp_count], pool[opp_count:]

    def _play_turn(self, hand, table_sets, min_val, ignore_tbl, joker_aware):
        """One greedy move. Returns (played, new_table_sets, new_hand).

        H2 (joker-aware, my turn only): a joker is a high-value strategic
        resource, not a +1-tile filler. So keep hand jokers UNLESS the move
        goes out (winning trumps holding) or there is no non-joker move (must
        spend it or draw). The opponent stays plain-greedy (matches the real
        greedy baseline, which dumps jokers) so the rollout models it faithfully.
        """
        r = self.solver.solve(
            hand_tiles=hand, table_sets=table_sets,
            require_use_at_least_one_hand_tile=True,
            min_play_value=min_val, ignore_table=ignore_tbl,
        )
        if r.status != "Optimal" or r.used_hand_tile_count == 0:
            return False, table_sets, hand
        apply, kept = r, []
        if joker_aware and len(r.remaining_hand) > 0:
            jokers = [t for t in hand if t.is_joker]
            if jokers:
                hand_nj = [t for t in hand if not t.is_joker]
                r2 = self.solver.solve(
                    hand_tiles=hand_nj, table_sets=table_sets,
                    require_use_at_least_one_hand_tile=True,
                    min_play_value=min_val, ignore_table=ignore_tbl,
                )
                if r2.status == "Optimal" and r2.used_hand_tile_count > 0:
                    apply, kept = r2, jokers   # hold the joker(s)
        new_sets = [list(s.completed_tiles) for s in apply.selected_sets]
        new_table = table_sets + new_sets if ignore_tbl else new_sets
        return True, new_table, list(apply.remaining_hand) + kept

    def _rollout(
        self,
        my_hand,
        opp_hand,
        table_sets,
        deck,
        my_meld_done,
        opp_meld_done,
        initial_meld_value,
    ):
        """Greedy-vs-greedy playout, opponent to move. Score from my perspective:
        win/loss = +-(WIN_SCORE + loser's tile count), leaf = tile-count diff.
        My turn plays joker-aware (holds jokers); opponent plays plain greedy."""
        hands = {"me": my_hand, "opp": opp_hand}
        meld_done = {"me": my_meld_done, "opp": opp_meld_done}
        turn = "opp"

        for _ in range(self.max_rollout_turns):
            other = "me" if turn == "opp" else "opp"
            hand = hands[turn]

            if meld_done[turn] or initial_meld_value <= 0:
                min_val, ignore_tbl = 0, False
            else:
                min_val, ignore_tbl = initial_meld_value, True

            played, table_sets, hands[turn] = self._play_turn(
                hand, table_sets, min_val, ignore_tbl,
                joker_aware=(self.joker_fix and turn == "me"),
            )
            if played:
                meld_done[turn] = True
                if len(hands[turn]) == 0:
                    loser_tiles = len(hands[other])
                    if turn == "me":
                        return WIN_SCORE + loser_tiles
                    return -(WIN_SCORE + loser_tiles)
            elif deck:
                hands[turn].append(deck.pop())

            turn = other

        if self.joker_fix:
            def leaf_tiles(hand):   # held jokers are assets, not liabilities
                jok = sum(1 for t in hand if t.is_joker)
                return len(hand) - (1.0 + JOKER_LEAF_BONUS) * jok
        else:
            def leaf_tiles(hand):
                return len(hand)
        return leaf_tiles(hands["opp"]) - leaf_tiles(hands["me"])
