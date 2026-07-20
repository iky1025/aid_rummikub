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
        search_deck_trigger=0,
        search_hand_trigger=5,
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
            near_end = (
                (self.search_deck_trigger > 0
                 and len(env.env.deck) <= self.search_deck_trigger)
                or min(len(my_hand), opp_count) <= self.search_hand_trigger
            )
            if near_end:
                choice = self._endgame_choice(
                    env, candidates, chosen, my_hand, table_sets,
                    determinizations, opp_meld_done, initial_meld_value,
                )
                if choice is not None:
                    return choice

        scores = [0.0] * len(chosen)
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
        self.last_eval = {"scores": score_map, "votes": None, "n_det": n_det}

        best_j = max(range(len(chosen)), key=lambda j: scores[j])
        # chosen[0] is always the greedy max-play; scores are paired (same
        # determinizations per candidate), so the difference is low-variance.
        threshold = self.greedy_margin * len(determinizations)
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
        # R11: the generating DP gives the COMPLETE distinct-move list fast (no
        # ILP, no sub-multiset blow-up), jokerless and jokered alike — the old
        # enumerate/solve_many path fell back to the ILP on jokered nodes
        # (~30% of jokered fair-combo time; this is ~27% faster end to end).
        moves = self.solver.generate_candidates(
            hand_tiles=hand, table_sets=table, max_candidates=cap,
            min_play_value=min_val, ignore_table=ignore,
        )
        out = []
        for r in moves[:cap]:
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
        win/loss = +-(WIN_SCORE + loser's tile count), leaf = tile-count diff."""
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

            result = self.solver.solve(
                hand_tiles=hand,
                table_sets=table_sets,
                require_use_at_least_one_hand_tile=True,
                min_play_value=min_val,
                ignore_table=ignore_tbl,
            )
            if result.status == "Optimal" and result.used_hand_tile_count > 0:
                new_sets = [
                    list(s.completed_tiles) for s in result.selected_sets
                ]
                if ignore_tbl:
                    table_sets = table_sets + new_sets
                else:
                    table_sets = new_sets
                hands[turn] = list(result.remaining_hand)
                meld_done[turn] = True
                if len(hands[turn]) == 0:
                    loser_tiles = len(hands[other])
                    if turn == "me":
                        return WIN_SCORE + loser_tiles
                    return -(WIN_SCORE + loser_tiles)
            elif deck:
                hands[turn].append(deck.pop())

            turn = other

        return len(hands["opp"]) - len(hands["me"])
