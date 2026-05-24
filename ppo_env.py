import numpy as np

from rummikub_env import RummikubEnv
from rummikub_solver import COLORS, flatten


class RummikubPPOEnv:
    def __init__(
        self,
        max_candidates=10,
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
            candidates = self.env.solve_candidate_moves(max_candidates=self.max_candidates)[: self.max_candidates]
            self.last_candidates = candidates

        if action < len(candidates):
            result = candidates[action]
            self.env.apply_solution(result)
            reward += 0.1 * result.used_hand_tile_count
            if len(self.env.hand) == 0:
                reward += 5.0
                self.hands[self.ppo_player] = list(self.env.hand)
                self.last_candidates = []
                obs = self.get_observation(self.ppo_player)
                info = self._build_info(len(candidates), 0)
                return obs, reward, True, info
        else:
            drawn_tile = self.env.draw_tile()
            if drawn_tile is None:
                reward -= 1.0
            else:
                reward -= 0.5

        self.hands[self.ppo_player] = list(self.env.hand)
        reward -= 0.01

        self.current_player = self.ilp_player
        self._sync_env_hand(self.ilp_player)

        ilp_used_hand_tiles = 0
        ilp_done = False
        ilp_result = self.env.solve_best_move()

        if ilp_result.status == "Optimal" and ilp_result.used_hand_tile_count > 0:
            self.env.apply_solution(ilp_result)
            ilp_used_hand_tiles = ilp_result.used_hand_tile_count
            if len(self.env.hand) == 0:
                ilp_done = True
        else:
            self.env.draw_tile()

        self.hands[self.ilp_player] = list(self.env.hand)

        if ilp_used_hand_tiles > 0:
            reward -= 0.02 * ilp_used_hand_tiles

        if ilp_done:
            reward -= 5.0
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

        obs = np.concatenate([hand_vector, table_vector, deck_count])
        return obs.astype(np.float32)

    def get_policy_inputs(self):
        if self.current_player != self.ppo_player:
            raise RuntimeError("get_policy_inputs() must be called on PPO player's turn.")

        self._sync_env_hand(self.ppo_player)
        self.last_candidates = self.env.solve_candidate_moves(max_candidates=self.max_candidates)[: self.max_candidates]

        obs = self.get_observation(self.ppo_player)
        cand_feats = np.zeros((self.max_candidates, 104), dtype=np.float32)
        mask = np.zeros(self.max_candidates + 1, dtype=np.float32)

        for i, result in enumerate(self.last_candidates):
            next_hand = self.tiles_to_vector(result.remaining_hand)
            next_table_tiles = []
            for s in result.selected_sets:
                next_table_tiles.extend(s.completed_tiles)
            next_table = self.tiles_to_vector(next_table_tiles)
            cand_feats[i] = np.concatenate([next_hand, next_table]).astype(np.float32)
            mask[i] = 1.0

        mask[self.max_candidates] = 1.0
        return obs, cand_feats, mask

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
