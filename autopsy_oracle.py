"""R9: autopsy of full-oracle losses.

The greedy-ILP opponent is deterministic, so "was this deal winnable?" is a
single-agent search: DFS over OUR moves only (candidates + chosen draw), with
the opponent's reply computed deterministically. For every game the 1-ply
full oracle lost, this proves one of:
  - WIN FOUND    -> the 1-ply oracle missed a winning line (depth limitation)
  - NO WIN       -> unwinnable vs greedy within the current move space
  - BUDGET       -> search budget exhausted (undecided)

Usage: python autopsy_oracle.py [--pairs 40] [--seed 2000] [--meld 30]
"""

import argparse
import sys
import time
from collections import Counter

from ppo_env import RummikubPPOEnv
from rollout_agent import RolloutPolicy
from rummikub_solver import RummikubILPSolver, flatten


def meld_params(meld_done, initial_meld_value):
    if meld_done or initial_meld_value <= 0:
        return 0, False
    return initial_meld_value, True


class GameSim:
    """Deterministic simulation vs greedy opponent, from a true state."""

    def __init__(self, solver, initial_meld_value, max_turns=100):
        self.solver = solver
        self.imv = initial_meld_value
        self.max_turns = max_turns

    def opponent_reply(self, opp_hand, table, deck, opp_meld):
        """Greedy opponent move. Returns new (opp_hand, table, deck, opp_meld, opp_won)."""
        min_val, ignore = meld_params(opp_meld, self.imv)
        r = self.solver.solve(
            hand_tiles=opp_hand, table_sets=table,
            require_use_at_least_one_hand_tile=False,
            min_play_value=min_val, ignore_table=ignore,
        )
        if r.status == "Optimal" and r.used_hand_tile_count > 0:
            new_sets = [list(s.completed_tiles) for s in r.selected_sets]
            table = table + new_sets if ignore else new_sets
            opp_hand = list(r.remaining_hand)
            return opp_hand, table, deck, True, len(opp_hand) == 0
        deck = list(deck)
        if deck:
            opp_hand = opp_hand + [deck.pop()]
        return opp_hand, table, deck, opp_meld, False

    def my_moves(self, hand, table, my_meld):
        """Deduped candidate list (remaining-hand multisets) for our turn."""
        min_val, ignore = meld_params(my_meld, self.imv)
        results = self.solver.solve_many(
            hand_tiles=hand, table_sets=table, max_solutions=8,
            require_use_at_least_one_hand_tile=True,
            min_play_value=min_val, ignore_table=ignore,
        )
        moves, seen = [], set()
        for r in results:
            key = tuple(sorted((t.color, t.number) for t in r.remaining_hand))
            if key in seen:
                continue
            seen.add(key)
            new_sets = [list(s.completed_tiles) for s in r.selected_sets]
            new_table = table + new_sets if ignore else new_sets
            moves.append((list(r.remaining_hand), new_table))
        return moves

    @staticmethod
    def state_key(hand, opp, table, deck_n, my_meld, opp_meld):
        return (
            tuple(sorted((t.color, t.number) for t in hand)),
            tuple(sorted((t.color, t.number) for t in opp)),
            tuple(sorted((t.color, t.number) for t in flatten(table))),
            deck_n, my_meld, opp_meld,
        )

    def search(self, hand, opp, table, deck, my_meld, opp_meld, budget=200000):
        """DFS: does ANY line of our moves win vs the deterministic opponent?
        Returns (result, nodes): result in {'WIN', 'NO_WIN', 'BUDGET'}."""
        self.nodes = 0
        self.budget = budget
        memo = {}

        def rec(hand, opp, table, deck, my_meld, opp_meld, turns):
            if turns >= self.max_turns:
                return False
            self.nodes += 1
            if self.nodes > self.budget:
                raise TimeoutError
            key = self.state_key(hand, opp, table, len(deck), my_meld, opp_meld)
            if key in memo:
                return memo[key]
            memo[key] = False  # cycle -> not a win

            # options: (new_hand, new_table, new_my_meld, deck_after)
            options = []
            for new_hand, new_table in self.my_moves(hand, table, my_meld):
                if len(new_hand) == 0:
                    memo[key] = True
                    return True
                options.append((new_hand, new_table, True, deck))
            # chosen/forced draw (playing moves tried first: faster wins)
            if deck:
                options.append((hand + [deck[-1]], table, my_meld, deck[:-1]))
            else:
                options.append((hand, table, my_meld, deck))

            for new_hand, new_table, new_meld, deck_after in options:
                o2, t2, d2, om2, opp_won = self.opponent_reply(
                    list(opp), [list(s) for s in new_table],
                    list(deck_after), opp_meld,
                )
                if opp_won:
                    continue
                if rec(new_hand, o2, t2, d2, new_meld, om2, turns + 1):
                    memo[key] = True
                    return True
            return False

        try:
            result = rec(list(hand), list(opp), [list(s) for s in table],
                         list(deck), my_meld, opp_meld, 0)
            return ("WIN" if result else "NO_WIN"), self.nodes
        except TimeoutError:
            return "BUDGET", self.nodes


def replay_oracle_game(seed, seat, imv):
    """Replay one full-oracle game; returns (outcome, env_states_for_search)."""
    env = RummikubPPOEnv(seed=seed, ppo_player=seat, opponent="ilp",
                         initial_meld_value=imv)
    policy = RolloutPolicy(oracle="full", max_rollout_turns=60,
                           candidate_cap=6, seed=seed)
    obs, _ = env.reset(seed=seed)
    start = {
        "hand": list(env.hands[env.ppo_player]),
        "opp": list(env.hands[env.ilp_player]),
        "table": [list(s) for s in env.env.table_sets],
        "deck": list(env.env.deck),
        "my_meld": env.first_meld_done[env.ppo_player],
        "opp_meld": env.first_meld_done[env.ilp_player],
    }
    done = False
    info = {}
    while not done:
        action = policy.select_action(env)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    return info.get("outcome", "timeout"), start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--meld", type=int, default=30)
    ap.add_argument("--budget", type=int, default=200000)
    args = ap.parse_args()

    solver = RummikubILPSolver()
    sim = GameSim(solver, args.meld)

    losses = []
    t0 = time.time()
    for p in range(args.pairs):
        for seat in (0, 1):
            outcome, start = replay_oracle_game(args.seed + p, seat, args.meld)
            if outcome != "win":
                losses.append((args.seed + p, seat, outcome, start))
        print(f"replayed pair {p + 1}/{args.pairs} "
              f"({len(losses)} losses so far, {time.time() - t0:.0f}s)",
              flush=True)

    print(f"\n{len(losses)} lost/timeout games to autopsy")
    verdicts = Counter()
    for seed, seat, outcome, s in losses:
        t1 = time.time()
        verdict, nodes = sim.search(
            s["hand"], s["opp"], s["table"], s["deck"],
            s["my_meld"], s["opp_meld"], budget=args.budget,
        )
        verdicts[verdict] += 1
        print(f"seed={seed} seat={seat} outcome={outcome} -> {verdict} "
              f"(nodes={nodes}, {time.time() - t1:.0f}s)", flush=True)

    print("\n=== autopsy verdicts ===")
    print(dict(verdicts))
    print("WIN = 1-ply oracle missed a winning line (depth limit)")
    print("NO_WIN = unwinnable vs greedy within current move space")


if __name__ == "__main__":
    main()
