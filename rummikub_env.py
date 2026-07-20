import random
from collections import Counter
from typing import Optional, Sequence

from rummikub_solver import (
    COLORS,
    NUMBERS,
    COPIES_PER_TILE,
    JOKER,
    Tile,
    RummikubILPSolver,
    flatten,
    format_tiles,
    validate_table_sets,
)

NUM_JOKERS = 2  # official Rummikub has 2 jokers (106 tiles total)


class RummikubEnv:
    def __init__(
        self,
        seed: Optional[int] = None,
        hand_size: int = 14,
        with_jokers: bool = False,
    ):
        self.random = random.Random(seed)
        self.hand_size = hand_size
        self.with_jokers = with_jokers
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

        if self.with_jokers:
            for _ in range(NUM_JOKERS):
                deck.append(Tile(JOKER.color, JOKER.number))

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
            raise ValueError("invalid table sets in reset")

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

    def solve_best_move(self, min_play_value=0, ignore_table=False):
        return self.solver.solve(
            hand_tiles=self.hand,
            table_sets=self.table_sets,
            require_use_at_least_one_hand_tile=False,
            min_play_value=min_play_value,
            ignore_table=ignore_table,
        )

    def solve_candidate_moves(self, max_candidates=10, min_play_value=0, ignore_table=False):
        return self.solver.solve_many(
            hand_tiles=self.hand,
            table_sets=self.table_sets,
            max_solutions=max_candidates,
            require_use_at_least_one_hand_tile=True,
            min_play_value=min_play_value,
            ignore_table=ignore_table,
        )

    def enumerate_candidate_moves(self, max_candidates=20, min_play_value=0, ignore_table=False):
        """R9: exhaustive move list (all playable hand-multisets). Falls back
        to solve_many when the subset space is too large."""
        moves = self.solver.enumerate_moves(
            hand_tiles=self.hand,
            table_sets=self.table_sets,
            min_play_value=min_play_value,
            ignore_table=ignore_table,
        )
        if moves is None:
            return self.solve_candidate_moves(
                max_candidates=max_candidates,
                min_play_value=min_play_value,
                ignore_table=ignore_table,
            )
        return moves[:max_candidates]

    def generate_candidate_moves(self, max_candidates=20, min_play_value=0,
                                 ignore_table=False):
        """R11: candidate moves via the generating DP (complete + fast, no
        arrangement duplicates)."""
        return self.solver.generate_candidates(
            hand_tiles=self.hand,
            table_sets=self.table_sets,
            max_candidates=max_candidates,
            min_play_value=min_play_value,
            ignore_table=ignore_table,
        )

    def apply_solution(self, result, append_to_table=False):
        """Apply ILP result.

        append_to_table=False (default): replace table_sets entirely with the
        new arrangement (allowed because table tiles can be rearranged).

        append_to_table=True: keep existing table_sets and ADD the new sets.
        Used for initial-meld plays where the solver was run with
        ignore_table=True, so the result's selected_sets are NEW sets only.
        """
        if result.status != "Optimal":
            raise RuntimeError(f"cannot apply non-optimal result: {result.status}")

        if append_to_table:
            for selected_set in result.selected_sets:
                self.table_sets.append(list(selected_set.completed_tiles))
        else:
            self.table_sets = []
            for selected_set in result.selected_sets:
                self.table_sets.append(list(selected_set.completed_tiles))

        self.hand = list(result.remaining_hand)

        # Guard against table-tile duplication (e.g. appending a full
        # rearrangement instead of replacing the table).
        counts = Counter(flatten(self.table_sets)) + Counter(self.hand)
        worst = max(counts.values(), default=0)
        if worst > COPIES_PER_TILE:
            raise RuntimeError(
                f"tile duplicated after apply_solution (count={worst})"
            )
        return self.get_state()

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
