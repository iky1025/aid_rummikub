"""Verify if there's a real first-mover (a)symmetry in our game.

Plays N games where BOTH players use random play (uniform sampling from
valid candidates + draw), starting from the same setup. Reports who wins
from each position.

If P0 and P1 win at roughly the same rate (~50%), the game is symmetric.
If P0 wins << 50% (e.g., 30%), going first is disadvantageous.
If P0 wins >> 50%, going first is advantageous.
"""

import argparse
import random
import time

from rummikub_env import RummikubEnv
from rummikub_solver import RummikubILPSolver


def play_game(seed, max_turns=100, max_candidates=20, initial_meld_value=0):
    """Play one game with both players random. Return (winner, p0_hand_size_at_end, p1_hand_size_at_end, turns).
    winner: 0 if P0 won, 1 if P1 won, None if timeout."""
    solver = RummikubILPSolver()
    env = RummikubEnv(seed=seed, hand_size=14)
    env.reset(shuffle=True)

    # Deal
    p0_hand = list(env.hand)
    p1_hand = []
    for _ in range(env.hand_size):
        t = env.draw_tile()
        if t is None:
            break
        p1_hand.append(t)

    rng = random.Random(seed)
    first_meld_done = [False, False]

    def player_turn(player_id, hand):
        is_initial = not first_meld_done[player_id]
        min_val = initial_meld_value if is_initial else 0
        ignore_tbl = is_initial and initial_meld_value > 0
        env.hand = list(hand)
        candidates = solver.solve_many(
            hand, env.table_sets,
            max_solutions=max_candidates,
            min_play_value=min_val,
            ignore_table=ignore_tbl,
        )
        n_opt = len(candidates)
        choice = rng.randint(0, n_opt)
        if choice < n_opt:
            env.apply_solution(candidates[choice], append_to_table=ignore_tbl)
            first_meld_done[player_id] = True
            return list(env.hand)
        else:
            t = env.draw_tile()
            if t is not None:
                hand = hand + [t]
            return hand

    for turn in range(max_turns):
        p0_hand = player_turn(0, p0_hand)
        if not p0_hand:
            return 0, 0, len(p1_hand), turn
        p1_hand = player_turn(1, p1_hand)
        if not p1_hand:
            return 1, len(p0_hand), 0, turn

    return None, len(p0_hand), len(p1_hand), max_turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--initial-meld-value", type=int, default=0)
    args = parser.parse_args()

    p0_wins = 0
    p1_wins = 0
    timeouts = 0
    total_turns = 0
    p0_loss_margin = []  # P0's tiles when P1 wins
    p1_loss_margin = []  # P1's tiles when P0 wins

    t0 = time.time()
    for i in range(args.games):
        winner, p0_left, p1_left, turns = play_game(
            seed=args.seed + i,
            max_turns=args.max_turns,
            initial_meld_value=args.initial_meld_value,
        )
        total_turns += turns
        if winner == 0:
            p0_wins += 1
            p1_loss_margin.append(p1_left)
        elif winner == 1:
            p1_wins += 1
            p0_loss_margin.append(p0_left)
        else:
            timeouts += 1
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(
                f"  game {i + 1}/{args.games}  "
                f"P0={p0_wins} P1={p1_wins} T={timeouts}  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    n = args.games
    print()
    print("=" * 60)
    print(f"Result over {n} games:")
    print("=" * 60)
    print(f"P0 wins      : {p0_wins} ({p0_wins / n:.1%})")
    print(f"P1 wins      : {p1_wins} ({p1_wins / n:.1%})")
    print(f"timeouts     : {timeouts} ({timeouts / n:.1%})")
    if p1_loss_margin:
        print(f"P0 wins by   : {sum(p1_loss_margin) / len(p1_loss_margin):.2f} tiles avg")
    if p0_loss_margin:
        print(f"P1 wins by   : {sum(p0_loss_margin) / len(p0_loss_margin):.2f} tiles avg")
    print(f"avg turns    : {total_turns / n:.1f}")

    # Statistical interpretation
    print()
    se = (p0_wins / n * (1 - p0_wins / n) / n) ** 0.5
    ci_lo = p0_wins / n - 1.96 * se
    ci_hi = p0_wins / n + 1.96 * se
    print(f"P0 win rate 95% CI: [{ci_lo:.1%}, {ci_hi:.1%}]")
    if ci_hi < 0.5:
        print("→ P0 (first player) is DISADVANTAGED.")
    elif ci_lo > 0.5:
        print("→ P0 (first player) is ADVANTAGED.")
    else:
        print("→ Cannot distinguish from symmetric play.")


if __name__ == "__main__":
    main()
