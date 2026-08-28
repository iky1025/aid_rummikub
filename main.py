from rummikub_solver import parse_tiles
from rummikub_env import RummikubEnv


def input_table_sets():
    table_sets = []

    print("\nEnter table sets one line at a time. Example: R3 R4 R5")
    print("Press Enter on an empty line to finish input.")

    while True:
        line = input("table set: ").strip()

        if line == "":
            break

        table_sets.append(parse_tiles(line))

    return table_sets


def print_candidate_results(results):
    print("\n=== ILP candidates ===")

    if not results:
        print("no candidates")
        return

    for i, result in enumerate(results):
        print(f"\n--- candidate {i} ---")
        print("strategy:", result.strategy)
        print("used hand tiles:", result.used_hand_tile_count)
        print("used hand score:", result.used_hand_tile_score)

        print("selected sets:")
        for j, tile_set in enumerate(result.selected_sets):
            print(f"  {j}: {tile_set}")

        if result.remaining_hand:
            from rummikub_solver import format_tiles
            print("remaining hand:", format_tiles(result.remaining_hand))
        else:
            print("remaining hand: empty")


def main():
    print("=== Rummikub Env + ILP Solver (No Joker) ===")
    print("Input format example: R1 B13 Y7 K10")

    table_sets = input_table_sets()

    env = RummikubEnv(hand_size=14)

    env.reset(
        table_sets=table_sets,
        shuffle=True,
    )

    print("\nInitial state")
    env.render()

    results = env.solve_candidate_moves(max_candidates=10)

    print_candidate_results(results)

    if results:
        answer = input("\napply candidate index? (empty to skip): ").strip()

        if answer != "":
            index = int(answer)

            if 0 <= index < len(results):
                env.apply_solution(results[index])

                print("\nState after apply")
                env.render()
            else:
                print("invalid candidate index")
        else:
            print("no candidate applied")
    else:
        print("\nNo playable candidate from current hand.")


if __name__ == "__main__":
    main()
