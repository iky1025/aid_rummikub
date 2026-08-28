from dataclasses import dataclass
from itertools import combinations
from collections import Counter
import ctypes
import os
import threading

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
    tile_score: int

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
        tile_score=sum(tile.number for tile in candidate_set),
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
    used_hand_tile_score: int
    remaining_hand: list
    table_locked: bool
    strategy: str


@dataclass
class _ILPProblemContext:
    problem: object
    variables: list
    candidates: list
    hand_tiles: list
    table_sets: list
    table_counter: Counter
    table_tile_count: int
    table_locked: bool
    hand_counter: Counter
    used_from_hand_expr: dict
    used_hand_tile_expr: object
    used_hand_score_expr: object


class RummikubILPSolver:
    def __init__(self):
        self._exact_solver = pulp.PULP_CBC_CMD(
            msg=False,
            threads=1,
            timeLimit=30,
        )
        self._candidate_solver = pulp.PULP_CBC_CMD(
            msg=False,
            threads=1,
            timeLimit=5,
        )
        self.last_solve_many_stats = {
            "solve_attempt_count": 0,
            "raw_solution_count": 0,
            "pool_solution_count": 0,
            "unique_solution_count": 0,
            "duplicate_solution_count": 0,
            "strategy_solution_count": 0,
        }

    def solve(
        self,
        hand_tiles,
        table_sets=None,
        require_use_at_least_one_hand_tile=False,
        table_locked=False,
        min_used_hand_tile_score=0,
        excluded_solutions=None,
    ):
        if table_sets is None:
            table_sets = []

        if excluded_solutions is None:
            excluded_solutions = []

        context = self._build_problem(
            hand_tiles=hand_tiles,
            table_sets=table_sets,
            require_use_at_least_one_hand_tile=require_use_at_least_one_hand_tile,
            table_locked=table_locked,
            min_used_hand_tile_score=min_used_hand_tile_score,
        )
        for excluded in excluded_solutions:
            self._add_exclusion_constraint(context, excluded)

        return self._solve_context(context)

    def _build_problem(
        self,
        hand_tiles,
        table_sets,
        require_use_at_least_one_hand_tile,
        table_locked,
        min_used_hand_tile_score,
    ):
        if not validate_table_sets(table_sets):
            raise ValueError("table has invalid set(s).")

        hand_tiles = list(hand_tiles)
        table_sets = [list(tile_set) for tile_set in table_sets]

        table_tiles = flatten(table_sets)
        available_tiles = hand_tiles if table_locked else hand_tiles + table_tiles

        table_counter = Counter() if table_locked else Counter(table_tiles)
        available_counter = Counter(available_tiles)
        table_tile_count = 0 if table_locked else len(table_tiles)

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
        hand_counter = Counter(hand_tiles)
        used_from_hand_expr = {
            tile: used_real_expr[tile] - table_counter[tile]
            for tile in hand_counter
        }
        used_hand_score_expr = pulp.lpSum(
            tile.number * used_from_hand_expr[tile]
            for tile in hand_counter
        )

        if require_use_at_least_one_hand_tile:
            problem += used_hand_tile_expr >= 1

        if min_used_hand_tile_score > 0:
            problem += used_hand_score_expr >= min_used_hand_tile_score

        problem += used_hand_tile_expr

        return _ILPProblemContext(
            problem=problem,
            variables=x,
            candidates=candidates,
            hand_tiles=hand_tiles,
            table_sets=table_sets,
            table_counter=table_counter,
            table_tile_count=table_tile_count,
            table_locked=table_locked,
            hand_counter=hand_counter,
            used_from_hand_expr=used_from_hand_expr,
            used_hand_tile_expr=used_hand_tile_expr,
            used_hand_score_expr=used_hand_score_expr,
        )

    def _solve_context(
        self,
        context,
        solver=None,
        hard_timeout=35,
        strategy="tile_count",
    ):
        if solver is None:
            solver = self._exact_solver
        watchdog = threading.Timer(
            hard_timeout,
            self._terminate_cbc_children,
        )
        watchdog.daemon = True
        watchdog.start()
        try:
            result_status = context.problem.solve(solver)
        except pulp.PulpSolverError:
            return self._empty_result(context, "TimeLimit", strategy)
        finally:
            watchdog.cancel()
        status = pulp.LpStatus[result_status]

        selected_sets = []
        selected_indices = []

        if status == "Optimal":
            for i, candidate in enumerate(context.candidates):
                value = pulp.value(context.variables[i])
                if value is not None and round(value) == 1:
                    selected_sets.append(candidate)
                    selected_indices.append(i)

        selected_tile_count = sum(tile_set.length for tile_set in selected_sets)
        used_hand_tile_count = selected_tile_count - context.table_tile_count
        used_hand_tile_score = self._compute_used_hand_tile_score(
            selected_sets,
            context.table_counter,
        )

        remaining_hand = self._compute_remaining_hand(
            context.hand_tiles,
            selected_sets,
            context.table_counter,
        )

        objective_value = pulp.value(context.problem.objective)
        if objective_value is None:
            objective_value = 0.0

        return ILPResult(
            status=status,
            selected_sets=selected_sets,
            selected_indices=selected_indices,
            candidates=context.candidates,
            objective_value=float(objective_value),
            table_tile_count=context.table_tile_count,
            selected_tile_count=selected_tile_count,
            used_hand_tile_count=used_hand_tile_count,
            used_hand_tile_score=used_hand_tile_score,
            remaining_hand=remaining_hand,
            table_locked=context.table_locked,
            strategy=strategy,
        )

    def _empty_result(self, context, status, strategy="tile_count"):
        return ILPResult(
            status=status,
            selected_sets=[],
            selected_indices=[],
            candidates=context.candidates,
            objective_value=0.0,
            table_tile_count=context.table_tile_count,
            selected_tile_count=0,
            used_hand_tile_count=0,
            used_hand_tile_score=0,
            remaining_hand=list(context.hand_tiles),
            table_locked=context.table_locked,
            strategy=strategy,
        )

    def _terminate_cbc_children(self):
        if os.name != "nt":
            return

        class ProcessEntry32(ctypes.Structure):
            _fields_ = [
                ("dwSize", ctypes.c_ulong),
                ("cntUsage", ctypes.c_ulong),
                ("th32ProcessID", ctypes.c_ulong),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", ctypes.c_ulong),
                ("cntThreads", ctypes.c_ulong),
                ("th32ParentProcessID", ctypes.c_ulong),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", ctypes.c_ulong),
                ("szExeFile", ctypes.c_wchar * 260),
            ]

        kernel32 = ctypes.windll.kernel32
        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            return

        entry = ProcessEntry32()
        entry.dwSize = ctypes.sizeof(ProcessEntry32)
        try:
            has_entry = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
            while has_entry:
                is_child = entry.th32ParentProcessID == os.getpid()
                is_cbc = entry.szExeFile.lower() == "cbc.exe"
                if is_child and is_cbc:
                    process = kernel32.OpenProcess(0x0001, False, entry.th32ProcessID)
                    if process:
                        kernel32.TerminateProcess(process, 1)
                        kernel32.CloseHandle(process)
                has_entry = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            kernel32.CloseHandle(snapshot)

    def _add_exclusion_constraint(self, context, selected_indices):
        if selected_indices:
            context.problem += (
                pulp.lpSum(context.variables[i] for i in selected_indices)
                <= len(selected_indices) - 1
            )

    def solve_many(
        self,
        hand_tiles,
        table_sets=None,
        max_solutions=10,
        require_use_at_least_one_hand_tile=True,
        table_locked=False,
        min_used_hand_tile_score=0,
        max_attempt_multiplier=1,
        max_consecutive_duplicates=3,
    ):
        if table_sets is None:
            table_sets = []

        if max_solutions <= 0:
            return []

        pool = []
        seen_states = set()
        raw_solution_count = 0
        duplicate_solution_count = 0
        context = self._build_problem(
            hand_tiles=hand_tiles,
            table_sets=table_sets,
            require_use_at_least_one_hand_tile=require_use_at_least_one_hand_tile,
            table_locked=table_locked,
            min_used_hand_tile_score=min_used_hand_tile_score,
        )

        objectives = self._candidate_objectives(context)
        if max_solutions == 1:
            objectives = objectives[:1]
        solve_attempt_count = 0
        best_used_count = None
        strategy_solution_count = 0

        for strategy, objective in objectives:
            context.problem.setObjective(objective)
            result = self._solve_context(
                context,
                self._candidate_solver,
                hard_timeout=10,
                strategy=strategy,
            )
            solve_attempt_count += 1

            if result.status != "Optimal":
                if best_used_count is None:
                    break
                continue
            if result.used_hand_tile_count <= 0:
                if best_used_count is None:
                    break
                continue
            if len(result.selected_indices) == 0:
                if best_used_count is None:
                    break
                continue

            if best_used_count is None:
                best_used_count = result.used_hand_tile_count
                minimum_count = max(1, best_used_count - 2)
                context.problem += context.used_hand_tile_expr >= minimum_count

            self._add_exclusion_constraint(context, result.selected_indices)
            raw_solution_count += 1
            strategy_solution_count += 1

            state_key = self._result_state_key(result, table_sets)
            if state_key in seen_states:
                duplicate_solution_count += 1
                continue

            seen_states.add(state_key)
            pool.append(result)

        # Fill any remaining pool slots with additional maximum-tile solutions.
        fallback_attempts = (
            max(
                0,
                (max_solutions - len(pool)) * max_attempt_multiplier + 2,
            )
            if best_used_count is not None and len(pool) < max_solutions
            else 0
        )
        context.problem.setObjective(context.used_hand_tile_expr)
        consecutive_duplicates = 0
        for _ in range(fallback_attempts):
            if len(pool) >= max_solutions + 2:
                break
            result = self._solve_context(
                context,
                self._candidate_solver,
                hard_timeout=10,
                strategy="tile_count_alternative",
            )
            solve_attempt_count += 1
            if (
                result.status != "Optimal"
                or result.used_hand_tile_count <= 0
                or not result.selected_indices
            ):
                break

            self._add_exclusion_constraint(context, result.selected_indices)
            raw_solution_count += 1
            state_key = self._result_state_key(result, table_sets)
            if state_key in seen_states:
                duplicate_solution_count += 1
                consecutive_duplicates += 1
                if consecutive_duplicates >= max_consecutive_duplicates:
                    break
                continue

            consecutive_duplicates = 0
            seen_states.add(state_key)
            pool.append(result)

        results = self._select_diverse_results(
            pool,
            table_sets,
            max_solutions,
        )

        self.last_solve_many_stats = {
            "solve_attempt_count": solve_attempt_count,
            "raw_solution_count": raw_solution_count,
            "pool_solution_count": len(pool),
            "unique_solution_count": len(results),
            "duplicate_solution_count": duplicate_solution_count,
            "strategy_solution_count": strategy_solution_count,
        }

        return results

    def _candidate_objectives(self, context):
        used_count = context.used_hand_tile_expr
        used_score = context.used_hand_score_expr
        hand_counter = context.hand_counter

        run_support = {
            tile: self._run_support(hand_counter, tile)
            for tile in hand_counter
        }
        group_support = {
            tile: self._group_support(hand_counter, tile)
            for tile in hand_counter
        }
        isolated_weight = {
            tile: max(0, 4 - run_support[tile] - group_support[tile])
            for tile in hand_counter
        }
        duplicate_edge_weight = {
            tile: (
                3 * max(0, hand_counter[tile] - 1)
                + (2 if tile.number in (1, 13) else 1 if tile.number in (2, 12) else 0)
            )
            for tile in hand_counter
        }

        def hand_weighted(weights):
            return pulp.lpSum(
                weights[tile] * context.used_from_hand_expr[tile]
                for tile in hand_counter
            )

        preserve_run = 0.25 * used_count - hand_weighted(run_support)
        preserve_group = 0.25 * used_count - hand_weighted(group_support)
        preserve_balanced = (
            0.25 * used_count
            - hand_weighted(
                {
                    tile: run_support[tile] + group_support[tile]
                    for tile in hand_counter
                }
            )
        )

        old_table_keys = Counter(
            self._canonical_set_key(tile_set)
            for tile_set in context.table_sets
        )
        preserve_table = pulp.lpSum(
            (
                1
                if old_table_keys[self._canonical_set_key(candidate.completed_tiles)] > 0
                else 0
            )
            * context.variables[index]
            for index, candidate in enumerate(context.candidates)
        )
        selected_set_count = pulp.lpSum(context.variables)
        run_tiles = pulp.lpSum(
            candidate.length * context.variables[index]
            for index, candidate in enumerate(context.candidates)
            if self._is_run(candidate.completed_tiles)
        )
        group_tiles = pulp.lpSum(
            candidate.length * context.variables[index]
            for index, candidate in enumerate(context.candidates)
            if self._is_group(candidate.completed_tiles)
        )
        long_run_score = pulp.lpSum(
            candidate.length * candidate.length * context.variables[index]
            for index, candidate in enumerate(context.candidates)
            if self._is_run(candidate.completed_tiles)
        )

        return [
            ("tile_count", used_count),
            ("high_score", used_score + 0.01 * used_count),
            ("isolated_tiles", hand_weighted(isolated_weight) + 0.1 * used_count),
            ("duplicate_edge_tiles", hand_weighted(duplicate_edge_weight) + 0.1 * used_count),
            ("preserve_run", preserve_run),
            ("preserve_group", preserve_group),
            ("preserve_balanced", preserve_balanced),
            ("minimal_table_change", 5 * preserve_table + used_count - selected_set_count),
            ("long_runs", long_run_score + 0.1 * used_count),
            ("run_focused", run_tiles + 0.1 * used_count),
            ("group_focused", group_tiles + 0.1 * used_count),
            ("compact_table", used_count - 0.5 * selected_set_count),
            ("fragmented_table", selected_set_count + 0.1 * used_count),
        ]

    def _select_diverse_results(self, pool, table_sets, max_solutions):
        if len(pool) <= max_solutions:
            return list(pool)

        selected = [pool[0]]
        remaining = list(pool[1:])
        best_count = max(result.used_hand_tile_count for result in pool)

        while remaining and len(selected) < max_solutions:
            def selection_score(candidate):
                novelty = min(
                    self._result_distance(candidate, chosen, table_sets)
                    for chosen in selected
                )
                quality = candidate.used_hand_tile_count / max(1, best_count)
                return novelty + 0.15 * quality

            chosen = max(remaining, key=selection_score)
            selected.append(chosen)
            remaining.remove(chosen)

        return selected

    def _result_distance(self, left, right, table_sets):
        left_hand = Counter(left.remaining_hand)
        right_hand = Counter(right.remaining_hand)
        hand_tiles = set(left_hand) | set(right_hand)
        hand_distance = sum(
            abs(left_hand[tile] - right_hand[tile])
            for tile in hand_tiles
        ) / max(1, len(left.remaining_hand) + len(right.remaining_hand))

        left_table = Counter(self._result_table_tiles(left, table_sets))
        right_table = Counter(self._result_table_tiles(right, table_sets))
        table_tiles = set(left_table) | set(right_table)
        table_distance = sum(
            abs(left_table[tile] - right_table[tile])
            for tile in table_tiles
        ) / max(1, sum(left_table.values()) + sum(right_table.values()))
        return hand_distance + table_distance

    def _run_support(self, hand_counter, tile):
        return sum(
            min(1, hand_counter[Tile(tile.color, number)])
            for number in (tile.number - 2, tile.number - 1, tile.number + 1, tile.number + 2)
            if number in NUMBERS
        )

    def _group_support(self, hand_counter, tile):
        return sum(
            min(1, hand_counter[Tile(color, tile.number)])
            for color in COLORS
            if color != tile.color
        )

    def _is_run(self, tile_set):
        return len({tile.color for tile in tile_set}) == 1

    def _is_group(self, tile_set):
        return len({tile.number for tile in tile_set}) == 1

    def _canonical_set_key(self, tile_set):
        return tuple(sorted((tile.color, tile.number) for tile in tile_set))

    def _result_table_set_keys(self, result, table_sets):
        keys = []
        if result.table_locked:
            keys.extend(self._canonical_set_key(tile_set) for tile_set in table_sets)
        keys.extend(
            self._canonical_set_key(selected_set.completed_tiles)
            for selected_set in result.selected_sets
        )
        return tuple(sorted(keys))

    def _result_table_tiles(self, result, table_sets):
        table_tiles = flatten(table_sets) if result.table_locked else []
        table_tiles += flatten(
            selected_set.completed_tiles
            for selected_set in result.selected_sets
        )
        return table_tiles

    def _result_state_key(self, result, table_sets):
        return (
            self._tile_counter_key(Counter(result.remaining_hand)),
            self._tile_counter_key(
                Counter(self._result_table_tiles(result, table_sets))
            ),
        )

    def _tile_counter_key(self, counter):
        return tuple(
            sorted(
                (tile.color, tile.number, count)
                for tile, count in counter.items()
                if count > 0
            )
        )

    def _compute_used_hand_tile_score(self, selected_sets, table_counter):
        selected_real_counter = Counter()
        for tile_set in selected_sets:
            selected_real_counter.update(tile_set.real_used)

        score = 0
        for tile, used_count in selected_real_counter.items():
            table_count = table_counter[tile]
            used_from_hand = max(0, used_count - table_count)
            score += tile.number * used_from_hand

        return score

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
