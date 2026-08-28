import random
from typing import Optional, Sequence

from rummikub_solver import (
    COLORS,
    NUMBERS,
    COPIES_PER_TILE,
    Tile,
    RummikubILPSolver,
    flatten,
    format_tiles,
    validate_table_sets,
)


class RummikubEnv:
    INITIAL_MELD_MIN_SCORE = 30

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
        self.initial_meld_done = False

        self.reset()

    def make_full_deck(self):
        deck = []

        for _ in range(COPIES_PER_TILE):
            for color in COLORS:
                for number in NUMBERS:
                    deck.append(Tile(color, number))

        return deck

    def reset(
        self,
        hand: Optional[Sequence[Tile]] = None,
        table_sets: Optional[Sequence[Sequence[Tile]]] = None,
        shuffle: bool = True,
        initial_meld_done: bool = False,
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
            raise ValueError("invalid table sets in reset")

        self.initial_meld_done = initial_meld_done
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
            "initial_meld_done": self.initial_meld_done,
        }

    def draw_tile(self):
        if not self.deck:
            return None

        tile = self.deck.pop()
        self.hand.append(tile)
        return tile

    def solve_best_move(self):
        return self.solver.solve(
            hand_tiles=self.hand,
            table_sets=self.table_sets,
            require_use_at_least_one_hand_tile=False,
            table_locked=not self.initial_meld_done,
            min_used_hand_tile_score=self._required_initial_meld_score(),
        )

    def solve_candidate_moves(self, max_candidates=10):
        return self.solver.solve_many(
            hand_tiles=self.hand,
            table_sets=self.table_sets,
            max_solutions=max_candidates,
            require_use_at_least_one_hand_tile=True,
            table_locked=not self.initial_meld_done,
            min_used_hand_tile_score=self._required_initial_meld_score(),
        )

    def apply_solution(self, result):
        if result.status != "Optimal":
            raise RuntimeError(f"cannot apply non-optimal result: {result.status}")

        if result.table_locked:
            new_table_sets = [list(tile_set) for tile_set in self.table_sets]
        else:
            new_table_sets = []

        for selected_set in result.selected_sets:
            new_table_sets.append(list(selected_set.completed_tiles))

        self.table_sets = new_table_sets
        self.hand = list(result.remaining_hand)
        if result.used_hand_tile_score >= self.INITIAL_MELD_MIN_SCORE:
            self.initial_meld_done = True
        return self.get_state()

    def _required_initial_meld_score(self):
        if self.initial_meld_done:
            return 0
        return self.INITIAL_MELD_MIN_SCORE

    def render(self):
        print("\n=== RummikubEnv ===")

        if self.hand:
            print("hand:", format_tiles(self.hand))
        else:
            print("hand: empty")

        print("table:")
        if not self.table_sets:
            print("  empty")
        else:
            for i, tile_set in enumerate(self.table_sets):
                print(f"  {i}: {format_tiles(tile_set)}")

        print("deck:", len(self.deck))
        print("initial meld:", "done" if self.initial_meld_done else "not done")
