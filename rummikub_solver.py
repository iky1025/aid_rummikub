from dataclasses import dataclass
from itertools import combinations
from collections import Counter

import pulp


COLORS = ["R", "B", "Y", "K"]
NUMBERS = range(1, 14)

COPIES_PER_TILE = 2
JOKER_COUNT = 2


@dataclass(frozen=True)
class Tile:
    color: str
    number: int
    is_joker: bool = False

    def __repr__(self):
        if self.is_joker:
            return "J"
        return f"{self.color}{self.number}"

    @property
    def score(self):
        if self.is_joker:
            return 30
        return self.number
    # 점수계산은 지우기


JOKER = Tile("J", 0, True)


def parse_tile(label):
    # 문자열을 타일객체로 변환
    label = label.strip().upper()

    if label == "J":
        return JOKER

    color = label[0]
    number = int(label[1:])

    if color not in COLORS:
        raise ValueError(f"잘못된 색상입니다: {color}")

    if number not in NUMBERS:
        raise ValueError(f"숫자는 1부터 13까지 가능합니다: {number}")

    return Tile(color, number)
    


def parse_tiles(line):
    # 문자열 나열된거를 타일 객체로 변환    
    if not line.strip():
        return []
    return [parse_tile(label) for label in line.split()]



def format_tiles(tiles):
    return " ".join(str(tile) for tile in tiles)


def flatten(list_of_lists):
    # 2차원 리스트인 테이블을 1차원으로 변환
    result = []

    for inner in list_of_lists:
        for x in inner:
            result.append(x)

    return result


def generate_all_valid_sets_without_joker():
    # 모든 가능한 세트 생성
    all_sets = []

    # Run: 같은 색, 연속 숫자 3개 이상
    for color in COLORS:
        for start in range(1, 14):
            for end in range(start + 2, 14):
                run = [Tile(color, n) for n in range(start, end + 1)]
                all_sets.append(run)

    # Group: 같은 숫자, 서로 다른 색 3개 또는 4개
    for number in NUMBERS:
        for size in [3, 4]:
            for color_comb in combinations(COLORS, size):
                group = [Tile(color, number) for color in color_comb]
                all_sets.append(group)

    return all_sets


ALL_VALID_SETS_WITHOUT_JOKER = generate_all_valid_sets_without_joker()


def is_valid_set_without_joker(tile_set):
    if len(tile_set) < 3:
        return False

    if any(tile.is_joker for tile in tile_set):
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


def is_valid_set(tile_set):
    if len(tile_set) < 3:
        return False
    # 타일 개수가 3보다 작으면 그냥 안됨

    joker_count = sum(1 for tile in tile_set if tile.is_joker)
    # 조커 개수 분리
    normal_tiles = [tile for tile in tile_set if not tile.is_joker]

    if joker_count == 0:
        return is_valid_set_without_joker(tile_set)

    normal_counter = Counter(normal_tiles)

    for valid_set in ALL_VALID_SETS_WITHOUT_JOKER:
        if len(valid_set) != len(tile_set):
            continue

        valid_counter = Counter(valid_set)

        missing = 0
        extra = 0

        for tile, count in valid_counter.items():
            have = normal_counter[tile]

            if have < count:
                missing += count - have

        for tile, count in normal_counter.items():
            if count > valid_counter[tile]:
                extra += count - valid_counter[tile]

        if missing == joker_count and extra == 0:
            return True

    return False


def validate_table_sets(table_sets):
    # 테이블 전체 세트의 유효성 검사
    for tile_set in table_sets:
        if not is_valid_set(tile_set):
            return False

    return True


@dataclass
class CandidateSet:
    completed_tiles: list
    real_used: Counter
    joker_used: int
    joker_as: list

    @property
    def length(self):
        return len(self.completed_tiles)

    @property
    def score(self):
        return sum(tile.score for tile in self.completed_tiles)

    def display_tiles(self):
        real_used_left = Counter(self.real_used)
        joker_as_left = Counter(self.joker_as)

        result = []

        for tile in self.completed_tiles:
            if real_used_left[tile] > 0:
                result.append(str(tile))
                real_used_left[tile] -= 1
            elif joker_as_left[tile] > 0:
                result.append(f"J({tile})")
                joker_as_left[tile] -= 1
            else:
                result.append(str(tile))

        return result

    def __repr__(self):
        return "[" + " ".join(self.display_tiles()) + "]"


def can_make_set_with_joker(candidate_set, available_counter):
    joker_count = available_counter[JOKER]
    need = Counter(candidate_set)

    missing = 0

    for tile, count in need.items():
        have = available_counter[tile]

        if have < count:
            missing += count - have

    return missing <= joker_count


def make_candidate_info(candidate_set, available_counter):
    need = Counter(candidate_set)

    real_used = Counter()
    joker_used = 0
    joker_as = []

    for tile, count in need.items():
        have = available_counter[tile]

        use_real = min(have, count)
        use_joker = count - use_real

        if use_real > 0:
            real_used[tile] = use_real

        if use_joker > 0:
            joker_used += use_joker
            joker_as.extend([tile] * use_joker)

    return CandidateSet(
        completed_tiles=list(candidate_set),
        real_used=real_used,
        joker_used=joker_used,
        joker_as=joker_as,
    )


def filter_available_sets(available_tiles):
    available_counter = Counter(available_tiles)
    candidates = []

    for candidate_set in ALL_VALID_SETS_WITHOUT_JOKER:
        if can_make_set_with_joker(candidate_set, available_counter):
            candidate = make_candidate_info(candidate_set, available_counter)
            candidates.append(candidate)

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
    used_hand_score: int
    remaining_hand: list

    def print_summary(self):
        print("\n=== ILP 결과 ===")
        print("상태:", self.status)
        print("목적함수 값:", self.objective_value)
        print("기존 테이블 타일 수:", self.table_tile_count)
        print("선택 결과 전체 타일 수:", self.selected_tile_count)
        print("새로 사용한 손패 타일 수:", self.used_hand_tile_count)
        print("새로 사용한 손패 점수:", self.used_hand_score)

        if self.remaining_hand:
            print("남은 손패:", format_tiles(self.remaining_hand))
        else:
            print("남은 손패: 없음")

        print("\n=== 선택된 세트 ===")

        if not self.selected_sets:
            print("없음")
        else:
            for i, tile_set in enumerate(self.selected_sets):
                print(f"{i}: {tile_set}")


class RummikubILPSolver:
    """
    maximize_hand_tiles만 사용하는 ILP Solver.

    solve():
        최적해 1개 반환

    solve_many():
        여러 개의 ILP 해 후보 반환
        PPO에서 action 후보로 사용 가능
    """

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
            raise ValueError("테이블에 유효하지 않은 세트가 있습니다.")

        hand_tiles = list(hand_tiles)
        table_sets = [list(tile_set) for tile_set in table_sets]

        table_tiles = flatten(table_sets)
        available_tiles = hand_tiles + table_tiles

        table_counter = Counter(table_tiles)
        available_counter = Counter(available_tiles)

        table_tile_count = len(table_tiles)
        table_score = sum(tile.score for tile in table_tiles)

        table_joker_count = table_counter[JOKER]
        available_joker_count = available_counter[JOKER]

        candidates = filter_available_sets(available_tiles)

        problem = pulp.LpProblem("Rummikub_ILP", pulp.LpMaximize)

        x = []

        for i in range(len(candidates)):
            variable = pulp.LpVariable(
                f"x_{i}",
                lowBound=0,
                upBound=1,
                cat="Binary",
            )
            x.append(variable)

        normal_tiles = []

        for tile in available_counter:
            if not tile.is_joker:
                normal_tiles.append(tile)

        used_real_expr = {}

        for tile in normal_tiles:
            used_real_expr[tile] = pulp.lpSum(
                candidates[i].real_used[tile] * x[i]
                for i in range(len(candidates))
            )

        used_joker_expr = pulp.lpSum(
            candidates[i].joker_used * x[i]
            for i in range(len(candidates))
        )

        used_total_expr = pulp.lpSum(
            candidates[i].length * x[i]
            for i in range(len(candidates))
        )

        used_score_expr = pulp.lpSum(
            candidates[i].score * x[i]
            for i in range(len(candidates))
        )

        # 1. 가진 일반 타일보다 많이 쓸 수 없다.
        for tile in normal_tiles:
            problem += used_real_expr[tile] <= available_counter[tile]

        # 2. 기존 테이블 일반 타일은 반드시 다시 사용해야 한다.
        for tile, count in table_counter.items():
            if not tile.is_joker and count > 0:
                problem += used_real_expr[tile] >= count

        # 3. 조커는 가진 개수보다 많이 쓸 수 없다.
        problem += used_joker_expr <= available_joker_count

        # 4. 기존 테이블 조커도 반드시 다시 사용해야 한다.
        if table_joker_count > 0:
            problem += used_joker_expr >= table_joker_count

        used_hand_tile_expr = used_total_expr - table_tile_count
        used_hand_score_expr = used_score_expr - table_score

        # 5. 필요하면 손패를 최소 1개 이상 쓰도록 강제
        if require_use_at_least_one_hand_tile:
            problem += used_hand_tile_expr >= 1

        # 6. 이미 찾은 해는 제외한다.
        # selected_indices가 [1, 5, 9]였다면,
        # 다음 solve에서는 x1 + x5 + x9 <= 2 로 만들어 같은 조합을 막는다.
        for excluded in excluded_solutions:
            if len(excluded) > 0:
                problem += pulp.lpSum(x[i] for i in excluded) <= len(excluded) - 1

        # 목적함수: 새로 사용한 손패 타일 개수 최대화
        problem += used_hand_tile_expr

        result_status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
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

        selected_score = sum(tile_set.score for tile_set in selected_sets)
        used_hand_score = selected_score - table_score

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
            used_hand_score=used_hand_score,
            remaining_hand=remaining_hand,
        )

    def solve_many(
        self,
        hand_tiles,
        table_sets=None,
        max_solutions=10,
        require_use_at_least_one_hand_tile=True,
    ):
        """
        여러 개의 ILP 후보를 생성한다.

        동작:
        1. ILP로 최적해를 하나 찾는다.
        2. 그 해를 제외하는 제약을 추가한다.
        3. 다시 ILP를 푼다.
        4. max_solutions개까지 반복한다.

        반환:
        ILPResult 리스트
        """

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
        selected_joker_used = 0

        for tile_set in selected_sets:
            selected_real_counter.update(tile_set.real_used)
            selected_joker_used += tile_set.joker_used

        used_from_hand = Counter()

        for tile, used_count in selected_real_counter.items():
            table_count = table_counter[tile]
            used_from_hand[tile] = max(0, used_count - table_count)

        used_from_hand[JOKER] = max(
            0,
            selected_joker_used - table_counter[JOKER],
        )

        remaining_counter = hand_counter - used_from_hand

        remaining_hand = []

        for tile, count in remaining_counter.items():
            remaining_hand.extend([tile] * count)

        return remaining_hand