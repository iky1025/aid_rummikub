import random
from typing import Optional, Sequence

from rummikub_solver import (
    COLORS,
    NUMBERS,
    COPIES_PER_TILE,
    JOKER_COUNT,
    JOKER,
    Tile,
    RummikubILPSolver,
    flatten,
    format_tiles,
    validate_table_sets,
)


class RummikubEnv:
    """
    PPO 없이 사용하는 순수 루미큐브 환경.

    역할:
    1. 전체 덱 생성
    2. 게임 시작 시 랜덤 손패 14장 지급
    3. 테이블 세트 관리
    4. 타일 뽑기
    5. ILP solver 호출
    6. ILP 결과를 환경에 적용
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        hand_size: int = 14,
    ):
        self.random = random.Random(seed)
        self.hand_size = hand_size
        self.solver = RummikubILPSolver()

        self.deck = []
        self.hand = []
        self.table_sets = []

        self.reset()

    def make_full_deck(self):
        deck = []

        for _ in range(COPIES_PER_TILE):
            for color in COLORS:
                for number in NUMBERS:
                    deck.append(Tile(color, number))

        for _ in range(JOKER_COUNT):
            deck.append(JOKER)

        return deck

    def reset(
        self,
        hand: Optional[Sequence[Tile]] = None,
        table_sets: Optional[Sequence[Sequence[Tile]]] = None,
        shuffle: bool = True,
    ):
        self.deck = self.make_full_deck()

        if shuffle:
            self.random.shuffle(self.deck)

        if table_sets is None:
            self.table_sets = []
        else:
            self.table_sets = [list(tile_set) for tile_set in table_sets]

        if hand is None:
            self.hand = []

            for _ in range(self.hand_size):
                tile = self.deck.pop()
                self.hand.append(tile)
        else:
            self.hand = list(hand)
            self._remove_known_tiles_from_deck(self.hand)

        table_tiles = flatten(self.table_sets)
        self._remove_known_tiles_from_deck(table_tiles)

        if not validate_table_sets(self.table_sets):
            raise ValueError("초기 테이블에 유효하지 않은 세트가 있습니다.")

        return self.get_state()

    def _remove_known_tiles_from_deck(self, tiles):
        for tile in tiles:
            try:
                self.deck.remove(tile)
            except ValueError:
                pass

    def get_state(self):
        return {
            "hand": list(self.hand),
            "table_sets": [list(tile_set) for tile_set in self.table_sets],
            "deck_count": len(self.deck),
        }

    def draw_tile(self):
        if not self.deck:
            return None

        tile = self.deck.pop()
        self.hand.append(tile)

        return tile

    def solve_best_move(self):
        result = self.solver.solve(
            hand_tiles=self.hand,
            table_sets=self.table_sets,
            require_use_at_least_one_hand_tile=False,
        )

        return result

    def solve_candidate_moves(self, max_candidates=10):
        """
        PPO action 후보로 쓸 수 있는 여러 ILP 해를 반환한다.
        """
        results = self.solver.solve_many(
            hand_tiles=self.hand,
            table_sets=self.table_sets,
            max_solutions=max_candidates,
            require_use_at_least_one_hand_tile=True,
        )

        return results

    def apply_solution(self, result):
        if result.status != "Optimal":
            raise RuntimeError(f"Optimal 결과가 아니므로 적용할 수 없습니다: {result.status}")

        self.table_sets = []

        for selected_set in result.selected_sets:
            self.table_sets.append(list(selected_set.completed_tiles))

        self.hand = list(result.remaining_hand)

        return self.get_state()

    def render(self):
        print("\n=== RummikubEnv ===")

        if self.hand:
            print("손패:", format_tiles(self.hand))
        else:
            print("손패: 없음")

        print("테이블:")

        if not self.table_sets:
            print("  없음")
        else:
            for i, tile_set in enumerate(self.table_sets):
                print(f"  {i}: {format_tiles(tile_set)}")

        print("덱 남은 수:", len(self.deck))