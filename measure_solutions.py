"""
Measure how many distinct feasible candidate solutions exist per turn.

For each position encountered during a simulated game, enumerate all distinct
"play at least 1 tile" solutions by repeated solve with exclusion constraints.
Reports distribution of counts and enumeration timing.
"""

import argparse
import time
from collections import Counter

import numpy as np

from rummikub_env import RummikubEnv
from rummikub_solver import RummikubILPSolver


def enumerate_all_solutions(solver, hand_tiles, table_sets, max_count=500):
    """Enumerate distinct feasible solutions until exhausted or cap reached.

    Uses iterative ILP solve with exclusion constraints. Since the objective
    is max tile usage, solutions come out in decreasing tile-count order.
    """
    results = []
    excluded = []
    while len(results) < max_count:
        r = solver.solve(
            hand_tiles=hand_tiles,
            table_sets=table_sets,
            require_use_at_least_one_hand_tile=True,
            excluded_solutions=excluded,
        )
        if r.status != "Optimal":
            break
        if r.used_hand_tile_count <= 0:
            break
        if len(r.selected_indices) == 0:
            break
        results.append(r)
        excluded.append(r.selected_indices)
    return results


def measure(n_games=20, max_turns=80, max_enum=500, verbose=False):
    """Simulate 2-player games (PPO vs ILP, both greedy) and measure
    candidate counts at PPO's decision points — matching the real env."""
    solver = RummikubILPSolver()

    counts = []
    enum_times = []
    hand_size_at_position = []
    table_size_at_position = []

    t_start = time.time()
    for seed in range(n_games):
        game_start = time.time()
        positions_before = len(counts)
        # Set up two-handed game using a single RummikubEnv as deck/table holder
        rummikub_env = RummikubEnv(seed=seed, hand_size=14)
        rummikub_env.reset(shuffle=True)

        p0_hand = list(rummikub_env.hand)  # "PPO" player
        p1_hand = []                       # "ILP" opponent
        for _ in range(14):
            t = rummikub_env.draw_tile()
            if t is None:
                break
            p1_hand.append(t)

        for turn in range(max_turns):
            # === PPO's turn — MEASURE here ===
            rummikub_env.hand = list(p0_hand)
            t0 = time.time()
            solutions = enumerate_all_solutions(
                solver, p0_hand, rummikub_env.table_sets, max_count=max_enum,
            )
            elapsed = time.time() - t0

            n = len(solutions)
            counts.append(n)
            enum_times.append(elapsed * 1000)
            hand_size_at_position.append(len(p0_hand))
            table_size_at_position.append(
                sum(len(s) for s in rummikub_env.table_sets)
            )

            if verbose:
                tc = [s.used_hand_tile_count for s in solutions]
                print(
                    f"seed={seed} turn={turn:2d} "
                    f"p0hand={len(p0_hand):2d} table={sum(len(s) for s in rummikub_env.table_sets):2d} "
                    f"sols={n:3d} enum_ms={elapsed * 1000:6.0f} "
                    f"max_tiles={max(tc) if tc else 0}"
                )

            # Apply greedy max-play for p0
            if solutions:
                rummikub_env.apply_solution(solutions[0])
                p0_hand = list(rummikub_env.hand)
            else:
                tile = rummikub_env.draw_tile()
                if tile is None:
                    break
                p0_hand.append(tile)

            if not p0_hand:
                break

            # === ILP opponent's turn (greedy) ===
            rummikub_env.hand = list(p1_hand)
            r = solver.solve(
                hand_tiles=p1_hand,
                table_sets=rummikub_env.table_sets,
                require_use_at_least_one_hand_tile=False,
            )
            if r.status == "Optimal" and r.used_hand_tile_count > 0:
                rummikub_env.apply_solution(r)
                p1_hand = list(rummikub_env.hand)
            else:
                tile = rummikub_env.draw_tile()
                if tile is None:
                    break
                p1_hand.append(tile)

            if not p1_hand:
                break

        # Per-game progress
        n_pos = len(counts) - positions_before
        recent_counts = counts[positions_before:]
        recent_sols = [c for c in recent_counts if c > 0]
        elapsed_game = time.time() - game_start
        elapsed_total = time.time() - t_start
        eta = elapsed_total / (seed + 1) * (n_games - seed - 1)
        max_recent = max(recent_counts) if recent_counts else 0
        avg_nz = sum(recent_sols) / len(recent_sols) if recent_sols else 0
        print(
            f"game {seed + 1:2d}/{n_games} | "
            f"positions={n_pos:3d} | max_sols={max_recent:3d} | "
            f"avg_sols_nonzero={avg_nz:5.1f} | "
            f"game_t={elapsed_game:5.1f}s | total_t={elapsed_total:6.1f}s | "
            f"eta={eta:6.0f}s",
            flush=True,
        )

    counts = np.array(counts)
    enum_times = np.array(enum_times)
    hand_sizes = np.array(hand_size_at_position)

    print("\n" + "=" * 60)
    print(f"Measurement summary over {len(counts)} positions "
          f"({n_games} games, up to {max_turns} turns each)")
    print("=" * 60)

    print("\n--- Solution count distribution ---")
    print(f"  min : {counts.min()}")
    print(f"  p10 : {np.percentile(counts, 10):.0f}")
    print(f"  p25 : {np.percentile(counts, 25):.0f}")
    print(f"  p50 : {np.percentile(counts, 50):.0f}")
    print(f"  p75 : {np.percentile(counts, 75):.0f}")
    print(f"  p90 : {np.percentile(counts, 90):.0f}")
    print(f"  p99 : {np.percentile(counts, 99):.0f}")
    print(f"  max : {counts.max()}")
    print(f"  mean: {counts.mean():.1f}")

    bins = [0, 1, 5, 10, 20, 50, 100, 200, 500, max(501, int(counts.max()) + 1)]
    print("\n--- Histogram ---")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        n = int(((counts >= lo) & (counts < hi)).sum())
        bar = "#" * int(50 * n / len(counts))
        print(f"  [{lo:4d}, {hi:4d}): {n:5d} {bar}")

    print("\n--- Enumeration time (ms per position) ---")
    print(f"  mean: {enum_times.mean():.1f}")
    print(f"  p50 : {np.percentile(enum_times, 50):.1f}")
    print(f"  p90 : {np.percentile(enum_times, 90):.1f}")
    print(f"  p99 : {np.percentile(enum_times, 99):.1f}")
    print(f"  max : {enum_times.max():.1f}")

    print("\n--- Solutions vs hand size ---")
    for hs in sorted(set(hand_sizes.tolist())):
        mask = hand_sizes == hs
        if mask.sum() < 5:
            continue
        cs = counts[mask]
        print(
            f"  hand={hs:2d}: n={mask.sum():3d} "
            f"mean_sols={cs.mean():5.1f} "
            f"p50={np.percentile(cs, 50):.0f} "
            f"p90={np.percentile(cs, 90):.0f} "
            f"max={cs.max()}"
        )

    capped = int((counts >= max_enum).sum())
    if capped > 0:
        print(f"\n[WARN] {capped} positions hit the enumeration cap "
              f"({max_enum}) — true counts may be higher.")

    print("\n--- ML implications ---")
    p90 = np.percentile(counts, 90)
    p99 = np.percentile(counts, 99)
    print(f"  90% of positions have <= {p90:.0f} solutions")
    print(f"  99% of positions have <= {p99:.0f} solutions")
    print(f"  For 'enumerate all', max_candidates should be >= {int(p99)}")

    avg_ilp_ms = enum_times.mean() / max(1, counts.mean())
    print(f"\n  Per-ILP overhead avg: ~{avg_ilp_ms:.1f} ms")
    print(f"  At ~40 turns/game, total enum time per game: "
          f"~{enum_times.mean() * 40 / 1000:.1f}s")
    n_envs_target = 10
    n_steps_target = 128
    total_secs = enum_times.mean() * n_steps_target / 1000
    print(f"  Per-update enum time (n_steps={n_steps_target}, "
          f"n_envs={n_envs_target} parallel): ~{total_secs:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--max-enum", type=int, default=500,
                        help="enumeration cap per position")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    measure(
        n_games=args.games,
        max_turns=args.max_turns,
        max_enum=args.max_enum,
        verbose=args.verbose,
    )
