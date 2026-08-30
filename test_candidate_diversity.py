from collections import Counter

from ppo_env import RummikubPPOEnv
from rummikub_env import RummikubEnv
from rummikub_solver import flatten, parse_tiles


def main():
    env = RummikubEnv(hand_size=14)
    env.reset(
        hand=parse_tiles(
            "R1 R2 R3 R4 R5 R6 R7 R8 B7 Y7 K7 B8 Y8 K8"
        ),
        table_sets=[],
        shuffle=False,
        initial_meld_done=False,
    )

    candidates = env.solve_candidate_moves(max_candidates=10)
    assert len(candidates) >= 8
    assert candidates[0].strategy == "tile_count"

    best_count = candidates[0].used_hand_tile_count
    assert all(
        best_count - 2 <= candidate.used_hand_tile_count <= best_count
        for candidate in candidates
    )

    state_keys = {
        (
            _counter_key(Counter(candidate.remaining_hand)),
            _counter_key(Counter(_next_table_tiles(candidate))),
        )
        for candidate in candidates
    }
    assert len(state_keys) == len(candidates)

    strategies = {candidate.strategy for candidate in candidates}
    assert any(strategy.startswith("preserve_run") for strategy in strategies)
    assert any(strategy.startswith("preserve_group") for strategy in strategies)
    used_tile_counts = {candidate.used_hand_tile_count for candidate in candidates}
    assert best_count - 1 in used_tile_counts
    assert best_count - 2 in used_tile_counts

    stats = env.solver.last_solve_many_stats
    assert stats["strategy_solution_count"] >= len(candidates)
    assert stats["pool_solution_count"] >= len(candidates)

    ppo_env = RummikubPPOEnv(max_candidates=10, max_turns=10, seed=1)
    ppo_env.env.reset(
        hand=list(env.hand),
        table_sets=[],
        shuffle=False,
        initial_meld_done=False,
    )
    ppo_env.hands = [list(env.hand), parse_tiles("B1")]
    ppo_env.initial_meld_done = [False, False]
    ppo_env.current_player = ppo_env.ppo_player
    _, candidate_features, mask = ppo_env.get_policy_inputs()
    feature_count = int(mask[:-1].sum())
    assert candidate_features.shape == (
        ppo_env.max_candidates,
        ppo_env.CAND_FEAT_DIM,
    )
    assert len(
        {
            candidate_features[index].tobytes()
            for index in range(feature_count)
        }
    ) == feature_count

    print("candidate_count:", len(candidates))
    print("strategies:", sorted(strategies))
    print("used_tile_counts:", [c.used_hand_tile_count for c in candidates])
    print("stats:", stats)


def _next_table_tiles(candidate):
    return flatten(
        selected_set.completed_tiles
        for selected_set in candidate.selected_sets
    )


def _counter_key(counter):
    return tuple(
        sorted(
            (tile.color, tile.number, count)
            for tile, count in counter.items()
            if count > 0
        )
    )


if __name__ == "__main__":
    main()
