from dataclasses import dataclass
from itertools import combinations
from collections import Counter

import pulp


COLORS = ["R", "B", "Y", "K"]
NUMBERS = range(1, 14)

COPIES_PER_TILE = 2


@dataclass(frozen=True)
class Tile:
    color: str
    number: int

    def __repr__(self):
        return f"{self.color}{self.number}"


def parse_tile(label):
    label = label.strip().upper()
    color = label[0]
    number = int(label[1:])

    if color not in COLORS:
        raise ValueError(f"invalid color: {color}")

    if number not in NUMBERS:
        raise ValueError(f"number must be 1..13: {number}")

    return Tile(color, number)


def parse_tiles(line):
    if not line.strip():
        return []
    return [parse_tile(label) for label in line.split()]


def format_tiles(tiles):
    return " ".join(str(tile) for tile in tiles)


def flatten(list_of_lists):
    result = []
    for inner in list_of_lists:
        for x in inner:
            result.append(x)
    return result


def generate_all_valid_sets():
    all_sets = []

    # Run: same color, consecutive, length >= 3
    for color in COLORS:
        for start in range(1, 14):
            for end in range(start + 2, 14):
                run = [Tile(color, n) for n in range(start, end + 1)]
                all_sets.append(run)

    # Group: same number, different colors, size 3 or 4
    for number in NUMBERS:
        for size in [3, 4]:
            for color_comb in combinations(COLORS, size):
                group = [Tile(color, number) for color in color_comb]
                all_sets.append(group)

    return all_sets


ALL_VALID_SETS = generate_all_valid_sets()


def is_valid_set(tile_set):
    if len(tile_set) < 3:
        return False

    colors = [tile.color for tile in tile_set]
    numbers = [tile.number for tile in tile_set]

    # Group
    if len(set(numbers)) == 1:
        return len(tile_set) in [3, 4] and len(set(colors)) == len(colors)

    # Run
    if len(set(colors)) == 1:
        nums = sorted(numbers)
        for i in range(len(nums) - 1):
            if nums[i] + 1 != nums[i + 1]:
                return False
        return True

    return False


def validate_table_sets(table_sets):
    for tile_set in table_sets:
        if not is_valid_set(tile_set):
            return False
    return True


@dataclass
class CandidateSet:
    completed_tiles: list
    real_used: Counter

    @property
    def length(self):
        return len(self.completed_tiles)

    def __repr__(self):
        return "[" + " ".join(str(tile) for tile in self.completed_tiles) + "]"


def make_candidate_info(candidate_set, available_counter):
    need = Counter(candidate_set)
    real_used = Counter()

    for tile, count in need.items():
        have = available_counter[tile]
        use_real = min(have, count)
        if use_real > 0:
            real_used[tile] = use_real

    return CandidateSet(
        completed_tiles=list(candidate_set),
        real_used=real_used,
    )


def filter_available_sets(available_tiles):
    available_counter = Counter(available_tiles)
    candidates = []

    for candidate_set in ALL_VALID_SETS:
        need = Counter(candidate_set)
        feasible = True
        for tile, count in need.items():
            if available_counter[tile] < count:
                feasible = False
                break

        if feasible:
            candidates.append(make_candidate_info(candidate_set, available_counter))

    return candidates


@dataclass
class ILPResult:
    status: str
    selected_sets: list
    selected_indices: list
    candidates: list
    objective_value: float
    table_tile_count: int
    selected_tile_count: int
    used_hand_tile_count: int
    remaining_hand: list


class RummikubILPSolver:
    def solve(
        self,
        hand_tiles,
        table_sets=None,
        require_use_at_least_one_hand_tile=False,
        excluded_solutions=None,
    ):
        if table_sets is None:
            table_sets = []

        if excluded_solutions is None:
            excluded_solutions = []

        if not validate_table_sets(table_sets):
            raise ValueError("table has invalid set(s).")

        hand_tiles = list(hand_tiles)
        table_sets = [list(tile_set) for tile_set in table_sets]

        table_tiles = flatten(table_sets)
        available_tiles = hand_tiles + table_tiles

        table_counter = Counter(table_tiles)
        available_counter = Counter(available_tiles)
        table_tile_count = len(table_tiles)

        candidates = filter_available_sets(available_tiles)

        problem = pulp.LpProblem("Rummikub_ILP", pulp.LpMaximize)

        x = []
        for i in range(len(candidates)):
            x.append(pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary"))

        used_real_expr = {}
        for tile in available_counter:
            used_real_expr[tile] = pulp.lpSum(
                candidates[i].real_used[tile] * x[i]
                for i in range(len(candidates))
            )

        used_total_expr = pulp.lpSum(
            candidates[i].length * x[i]
            for i in range(len(candidates))
        )

        # Cannot use more than available
        for tile in available_counter:
            problem += used_real_expr[tile] <= available_counter[tile]

        # Existing table tiles must remain used
        for tile, count in table_counter.items():
            if count > 0:
                problem += used_real_expr[tile] >= count

        used_hand_tile_expr = used_total_expr - table_tile_count

        if require_use_at_least_one_hand_tile:
            problem += used_hand_tile_expr >= 1

        for excluded in excluded_solutions:
            if len(excluded) > 0:
                problem += pulp.lpSum(x[i] for i in excluded) <= len(excluded) - 1

        problem += used_hand_tile_expr

        result_status = problem.solve(pulp.COIN_CMD(msg=False, threads=1))
        status = pulp.LpStatus[result_status]

        selected_sets = []
        selected_indices = []

        if status == "Optimal":
            for i, candidate in enumerate(candidates):
                value = pulp.value(x[i])
                if value is not None and round(value) == 1:
                    selected_sets.append(candidate)
                    selected_indices.append(i)

        selected_tile_count = sum(tile_set.length for tile_set in selected_sets)
        used_hand_tile_count = selected_tile_count - table_tile_count

        remaining_hand = self._compute_remaining_hand(
            hand_tiles,
            selected_sets,
            table_counter,
        )

        objective_value = pulp.value(problem.objective)
        if objective_value is None:
            objective_value = 0.0

        return ILPResult(
            status=status,
            selected_sets=selected_sets,
            selected_indices=selected_indices,
            candidates=candidates,
            objective_value=float(objective_value),
            table_tile_count=table_tile_count,
            selected_tile_count=selected_tile_count,
            used_hand_tile_count=used_hand_tile_count,
            remaining_hand=remaining_hand,
        )

    def solve_many(
        self,
        hand_tiles,
        table_sets=None,
        max_solutions=10,
        require_use_at_least_one_hand_tile=True,
    ):
        if table_sets is None:
            table_sets = []

        results = []
        excluded_solutions = []

        for _ in range(max_solutions):
            result = self.solve(
                hand_tiles=hand_tiles,
                table_sets=table_sets,
                require_use_at_least_one_hand_tile=require_use_at_least_one_hand_tile,
                excluded_solutions=excluded_solutions,
            )

            if result.status != "Optimal":
                break
            if result.used_hand_tile_count <= 0:
                break
            if len(result.selected_indices) == 0:
                break

            results.append(result)
            excluded_solutions.append(result.selected_indices)

        return results

    def _compute_remaining_hand(self, hand_tiles, selected_sets, table_counter):
        hand_counter = Counter(hand_tiles)

        selected_real_counter = Counter()
        for tile_set in selected_sets:
            selected_real_counter.update(tile_set.real_used)

        used_from_hand = Counter()
        for tile, used_count in selected_real_counter.items():
            table_count = table_counter[tile]
            used_from_hand[tile] = max(0, used_count - table_count)

        remaining_counter = hand_counter - used_from_hand

        remaining_hand = []
        for tile, count in remaining_counter.items():
            remaining_hand.extend([tile] * count)

        return remaining_hand
