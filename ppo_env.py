from collections import Counter

import numpy as np

from rummikub_env import RummikubEnv
from rummikub_solver import COLORS, flatten


OPPONENT_HAND_NORMALIZATION_CAP = 30.0


class RummikubPPOEnv:
    def __init__(
        self,
        max_candidates=20,
        max_turns=100,
        seed=None,
        ppo_player=0,
    ):
        self.max_candidates = max_candidates
        self.max_turns = max_turns
        self.ppo_player = ppo_player
        self.ilp_player = 1 - ppo_player

        self.env = RummikubEnv(
            seed=seed,
            hand_size=14,
        )

        self.turn_count = 0
        self.current_player = self.ppo_player
        self.hands = [[], []]
        self.last_candidates = []
        self.opened = [False, False]

    def reset(self):
        self.env.reset(
            table_sets=[],
            shuffle=True,
        )

        first_hand = list(self.env.hand)
        second_hand = []
        for _ in range(self.env.hand_size):
            tile = self.env.draw_tile()
            if tile is None:
                break
            second_hand.append(tile)

        self.hands = [[], []]
        self.hands[self.ppo_player] = first_hand
        self.hands[self.ilp_player] = second_hand

        self.turn_count = 0
        self.current_player = self.ppo_player
        self.last_candidates = []
        self.opened = [False, False]

        self._sync_env_hand(self.ppo_player)
        return self.get_observation(self.ppo_player)

    def step(self, action):
        if self.current_player != self.ppo_player:
            raise RuntimeError("step(action) must be called on PPO player's turn.")

        self.turn_count += 1
        reward = 0.0

        self._sync_env_hand(self.ppo_player)
        candidates = self.last_candidates
        if not candidates:
            candidates = self._generate_candidates_for_player(self.ppo_player)
            self.last_candidates = candidates

        if action < len(candidates):
            result = candidates[action]
            if (not self.opened[self.ppo_player]) and self._compute_hand_points_used(self.hands[self.ppo_player], result) < 30:
                raise RuntimeError("invalid PPO move: initial meld must be at least 30 points")
            if self.opened[self.ppo_player]:
                self.env.apply_solution(result)
            else:
                self._apply_initial_meld(result)
            if not self.opened[self.ppo_player]:
                self.opened[self.ppo_player] = True
            reward += result.used_hand_tile_count
            if len(self.env.hand) == 0:
                reward += 50.0
                self.hands[self.ppo_player] = list(self.env.hand)
                self.last_candidates = []
                obs = self.get_observation(self.ppo_player)
                info = self._build_info(len(candidates), 0)
                return obs, reward, True, info
        else:
            drawn_tile = self.env.draw_tile()
            if drawn_tile is None:
                reward -= 2.0
            else:
                reward -= 1.0

        self.hands[self.ppo_player] = list(self.env.hand)
        reward -= 0.1

        self.current_player = self.ilp_player
        self._sync_env_hand(self.ilp_player)

        ilp_used_hand_tiles = 0
        ilp_done = False
        ilp_result = None
        if self.opened[self.ilp_player]:
            ilp_result = self.env.solve_best_move()
        else:
            valid_candidates = self._generate_candidates_for_player(self.ilp_player)
            if valid_candidates:
                ilp_result = max(valid_candidates, key=lambda c: c.used_hand_tile_count)

        if ilp_result is not None and ilp_result.status == "Optimal" and ilp_result.used_hand_tile_count > 0:
            if self.opened[self.ilp_player]:
                self.env.apply_solution(ilp_result)
            else:
                self._apply_initial_meld(ilp_result)
            if not self.opened[self.ilp_player]:
                self.opened[self.ilp_player] = True
            ilp_used_hand_tiles = ilp_result.used_hand_tile_count
            if len(self.env.hand) == 0:
                ilp_done = True
        else:
            self.env.draw_tile()

        self.hands[self.ilp_player] = list(self.env.hand)

        if ilp_used_hand_tiles > 0:
            reward -= 0.2 * ilp_used_hand_tiles

        if ilp_done:
            reward -= 50.0
            self.last_candidates = []
            obs = self.get_observation(self.ppo_player)
            info = self._build_info(len(candidates), ilp_used_hand_tiles)
            return obs, reward, True, info

        self.current_player = self.ppo_player
        self._sync_env_hand(self.ppo_player)
        self.last_candidates = []

        done = self.is_done()
        obs = self.get_observation(self.ppo_player)
        info = self._build_info(len(candidates), ilp_used_hand_tiles)
        return obs, reward, done, info

    def is_done(self):
        if len(self.hands[self.ppo_player]) == 0:
            return True
        if len(self.hands[self.ilp_player]) == 0:
            return True
        if self.turn_count >= self.max_turns:
            return True
        return False

    def get_observation(self, player_id=None):
        if player_id is None:
            player_id = self.ppo_player

        hand_vector = self.tiles_to_vector(self.hands[player_id])
        table_tiles = flatten(self.env.table_sets)
        table_vector = self.tiles_to_vector(table_tiles)
        deck_count = np.array([len(self.env.deck) / 104.0], dtype=np.float32)
        opponent_id = 1 - player_id
        opponent_hand_count = np.array(
            [
                min(
                    len(self.hands[opponent_id]),
                    OPPONENT_HAND_NORMALIZATION_CAP,
                )
                / OPPONENT_HAND_NORMALIZATION_CAP
            ],
            dtype=np.float32,
        )
        opened_state = np.array(
            [float(self.opened[player_id]), float(self.opened[opponent_id])],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [hand_vector, table_vector, deck_count, opponent_hand_count, opened_state]
        )
        return obs.astype(np.float32)

    def get_policy_inputs(self):
        if self.current_player != self.ppo_player:
            raise RuntimeError("get_policy_inputs() must be called on PPO player's turn.")

        self._sync_env_hand(self.ppo_player)
        self.last_candidates = self._generate_candidates_for_player(self.ppo_player)

        obs = self.get_observation(self.ppo_player)
        cand_feats = np.zeros((self.max_candidates, 104), dtype=np.float32)
        mask = np.zeros(self.max_candidates + 1, dtype=np.float32)

        for i, candidate in enumerate(self.last_candidates):
            next_hand = self.tiles_to_vector(candidate.remaining_hand)

            candidate_table_sets = [
                selected_set.completed_tiles
                for selected_set in candidate.selected_sets
            ]
            if self.opened[self.ppo_player]:
                next_table_sets = candidate_table_sets
            else:
                next_table_sets = self.env.table_sets + candidate_table_sets

            next_table = self.tiles_to_vector(flatten(next_table_sets))
            cand_feats[i] = np.concatenate([next_hand, next_table]).astype(np.float32)
            mask[i] = 1.0

        mask[self.max_candidates] = 1.0
        return obs, cand_feats, mask

    def _filter_candidates_for_player(self, player_id, candidates):
        if self.opened[player_id]:
            return candidates

        hand_before = self.hands[player_id]
        filtered = []
        for candidate in candidates:
            used_points = self._compute_hand_points_used(hand_before, candidate)
            if used_points >= 30:
                filtered.append(candidate)
        return filtered

    def _generate_candidates_for_player(self, player_id):
        if self.opened[player_id]:
            raw = self.env.solve_candidate_moves(max_candidates=self.max_candidates)[: self.max_candidates]
            return self._deduplicate_candidates(player_id, raw)

        # First meld: generate candidates from hand only (no table tile usage).
        raw = self.env.solver.solve_many(
            hand_tiles=self.hands[player_id],
            table_sets=[],
            max_solutions=self.max_candidates,
            require_use_at_least_one_hand_tile=True,
        )[: self.max_candidates]
        filtered = self._filter_candidates_for_player(player_id, raw)
        return self._deduplicate_candidates(player_id, filtered)

    def _deduplicate_candidates(self, player_id, candidates):
        unique_candidates = []
        seen_states = set()

        for candidate in candidates:
            state_signature = self._candidate_state_signature(player_id, candidate)
            if state_signature in seen_states:
                continue

            seen_states.add(state_signature)
            unique_candidates.append(candidate)

        return unique_candidates

    def _candidate_state_signature(self, player_id, candidate):
        candidate_table_sets = [
            selected_set.completed_tiles
            for selected_set in candidate.selected_sets
        ]
        if self.opened[player_id]:
            next_table_sets = candidate_table_sets
        else:
            next_table_sets = self.env.table_sets + candidate_table_sets

        return (
            self._tiles_signature(candidate.remaining_hand),
            self._tiles_signature(flatten(next_table_sets)),
        )

    @staticmethod
    def _tiles_signature(tiles):
        counts = Counter(tiles)
        return tuple(
            sorted(
                (tile.color, tile.number, count)
                for tile, count in counts.items()
            )
        )

    def _apply_initial_meld(self, result):
        self._apply_initial_meld_to_env(self.env, result)

    def _apply_initial_meld_to_env(self, target_env, result):
        if result.status != "Optimal":
            raise RuntimeError(f"cannot apply non-optimal initial meld: {result.status}")

        existing_table = [list(tile_set) for tile_set in target_env.table_sets]
        new_sets = [list(selected_set.completed_tiles) for selected_set in result.selected_sets]
        target_env.table_sets = existing_table + new_sets
        target_env.hand = list(result.remaining_hand)

    def _compute_hand_points_used(self, hand_before, result):
        before_counter = Counter(hand_before)
        after_counter = Counter(result.remaining_hand)
        used_counter = before_counter - after_counter
        return sum(tile.number * count for tile, count in used_counter.items())

    def _sync_env_hand(self, player_id):
        self.env.hand = list(self.hands[player_id])

    def _build_info(self, candidate_count, ilp_used_hand_tiles):
        return {
            "ppo_hand_count": len(self.hands[self.ppo_player]),
            "ilp_hand_count": len(self.hands[self.ilp_player]),
            "deck_count": len(self.env.deck),
            "candidate_count": candidate_count,
            "ilp_used_hand_tiles": ilp_used_hand_tiles,
        }

    def tiles_to_vector(self, tiles):
        vector = np.zeros(52, dtype=np.float32)
        for tile in tiles:
            color_index = COLORS.index(tile.color)
            number_index = tile.number - 1
            index = color_index * 13 + number_index
            vector[index] += 1.0
        return vector / 2.0
