import numpy as np

from rummikub_env import RummikubEnv
from rummikub_solver import COLORS, flatten


class RummikubPPOEnv:
    REWARD_VERSION = 2
    TILE_VECTOR_SIZE = 52
    OBS_DIM = 109
    CANDIDATE_METRIC_DIM = 12
    CAND_FEAT_DIM = 104 + CANDIDATE_METRIC_DIM

    def __init__(
        self,
        max_candidates=10,
        max_turns=100,
        seed=None,
        ppo_player=0,
        candidate_search_limit=10,
        reward_gamma=0.99,
        shaping_scale=0.1,
        win_reward=20.0,
    ):
        if ppo_player not in (0, 1):
            raise ValueError("ppo_player must be 0 or 1")

        self.max_candidates = max_candidates
        self.candidate_search_limit = min(
            candidate_search_limit, max_candidates
        )
        self.max_turns = max_turns
        self.ppo_player = ppo_player
        self.ilp_player = 1 - ppo_player
        self.reward_gamma = reward_gamma
        self.shaping_scale = shaping_scale
        self.win_reward = win_reward

        self.env = RummikubEnv(seed=seed, hand_size=14)

        self.turn_count = 0
        self.current_player = 0
        self.hands = [[], []]
        self.dealt_hands = [[], []]
        self.initial_meld_done = [False, False]
        self.last_candidates = None
        self.last_candidate_stats = self._empty_candidate_stats()
        self.last_ilp_used_hand_tiles = 0

    def reset(self):
        self.env.reset(table_sets=[], shuffle=True)

        first_hand = list(self.env.hand)
        second_hand = []
        for _ in range(self.env.hand_size):
            tile = self.env.draw_tile()
            if tile is None:
                break
            second_hand.append(tile)

        # Seat 0 always receives the first hand and starts. Swapping ppo_player
        # therefore produces a paired game with the same deal and opposite role.
        self.hands = [first_hand, second_hand]
        self.dealt_hands = [list(first_hand), list(second_hand)]
        self.initial_meld_done = [False, False]
        self.turn_count = 0
        self.current_player = 0
        self.last_candidates = None
        self.last_candidate_stats = self._empty_candidate_stats()
        self.last_ilp_used_hand_tiles = 0

        if self.current_player == self.ilp_player:
            used_tiles, _ = self._play_ilp_turn()
            self.last_ilp_used_hand_tiles = used_tiles

        self.current_player = self.ppo_player
        self._sync_env_hand(self.ppo_player)
        return self.get_observation(self.ppo_player)

    def step(self, action):
        if self.current_player != self.ppo_player:
            raise RuntimeError("step(action) must be called on PPO player's turn.")
        if self.is_done():
            raise RuntimeError("step(action) cannot be called after the episode is done.")

        previous_potential = self._potential()
        self.turn_count += 1
        self._sync_env_hand(self.ppo_player)

        candidates = self._get_or_solve_candidates()
        if action == self.max_candidates:
            self.env.draw_tile()
        elif 0 <= action < len(candidates):
            self.env.apply_solution(candidates[action])
            self.initial_meld_done[self.ppo_player] = self.env.initial_meld_done
        else:
            raise ValueError(f"invalid or masked action: {action}")

        self.hands[self.ppo_player] = list(self.env.hand)
        ppo_won = len(self.hands[self.ppo_player]) == 0

        ilp_used_hand_tiles = 0
        ilp_won = False
        timeout_after_ppo_turn = (
            self.ppo_player == 1 and self.turn_count >= self.max_turns
        )
        if not ppo_won and not timeout_after_ppo_turn:
            self.current_player = self.ilp_player
            ilp_used_hand_tiles, ilp_won = self._play_ilp_turn()

        self.last_ilp_used_hand_tiles = ilp_used_hand_tiles
        self.current_player = self.ppo_player
        self._sync_env_hand(self.ppo_player)
        self.last_candidates = None

        timed_out = (
            self.turn_count >= self.max_turns
            and not ppo_won
            and not ilp_won
        )
        done = ppo_won or ilp_won or timed_out
        terminal_reward = self._terminal_reward(
            ppo_won,
            ilp_won,
            timed_out,
        )

        next_potential = 0.0 if done else self._potential()
        shaping_reward = self._potential_shaping(
            previous_potential, next_potential
        )
        reward = terminal_reward + shaping_reward

        obs = self.get_observation(self.ppo_player)
        info = self._build_info(
            len(candidates),
            ilp_used_hand_tiles,
            terminal_reward,
            shaping_reward,
        )
        return obs, reward, done, info

    def _play_ilp_turn(self):
        self._sync_env_hand(self.ilp_player)
        result = self.env.solve_best_move()
        used_hand_tiles = 0

        if result.status == "Optimal" and result.used_hand_tile_count > 0:
            self.env.apply_solution(result)
            self.initial_meld_done[self.ilp_player] = self.env.initial_meld_done
            used_hand_tiles = result.used_hand_tile_count
        else:
            self.env.draw_tile()

        self.hands[self.ilp_player] = list(self.env.hand)
        return used_hand_tiles, len(self.hands[self.ilp_player]) == 0

    def is_done(self):
        return (
            len(self.hands[self.ppo_player]) == 0
            or len(self.hands[self.ilp_player]) == 0
            or self.turn_count >= self.max_turns
        )

    def get_observation(self, player_id=None):
        if player_id is None:
            player_id = self.ppo_player

        opponent_id = 1 - player_id
        hand_vector = self.tiles_to_vector(self.hands[player_id])
        table_vector = self.tiles_to_vector(flatten(self.env.table_sets))
        state_features = np.array(
            [
                len(self.env.deck) / 104.0,
                len(self.hands[opponent_id]) / 104.0,
                float(self.initial_meld_done[player_id]),
                float(self.initial_meld_done[opponent_id]),
                min(self.turn_count / self.max_turns, 1.0),
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [hand_vector, table_vector, state_features]
        ).astype(np.float32)

    def get_policy_inputs(self):
        if self.current_player != self.ppo_player:
            raise RuntimeError("get_policy_inputs() must be called on PPO player's turn.")
        if self.is_done():
            raise RuntimeError("get_policy_inputs() cannot be called after the episode is done.")

        self._sync_env_hand(self.ppo_player)
        candidates = self._get_or_solve_candidates()

        obs = self.get_observation(self.ppo_player)
        cand_feats = np.zeros(
            (self.max_candidates, self.CAND_FEAT_DIM), dtype=np.float32
        )
        mask = np.zeros(self.max_candidates + 1, dtype=np.float32)

        for i, candidate in enumerate(candidates):
            next_hand = self.tiles_to_vector(candidate.remaining_hand)
            next_table_sets = self._next_table_sets(candidate)
            next_table = self.tiles_to_vector(flatten(next_table_sets))
            metrics = self._candidate_metrics(candidate, next_table_sets)
            cand_feats[i] = np.concatenate([next_hand, next_table, metrics])
            mask[i] = 1.0

        mask[self.max_candidates] = 1.0
        return obs, cand_feats, mask

    def get_info(self):
        return self._build_info(
            len(self.last_candidates or []),
            self.last_ilp_used_hand_tiles,
            0.0,
            0.0,
        )

    def _get_or_solve_candidates(self):
        if self.last_candidates is None:
            self.last_candidates = self.env.solve_candidate_moves(
                max_candidates=self.candidate_search_limit
            )[: self.candidate_search_limit]
            self.last_candidate_stats = dict(
                self.env.solver.last_solve_many_stats
            )
        return self.last_candidates

    def _next_table_sets(self, candidate):
        if candidate.table_locked:
            table_sets = [list(tile_set) for tile_set in self.env.table_sets]
        else:
            table_sets = []

        for selected_set in candidate.selected_sets:
            table_sets.append(list(selected_set.completed_tiles))
        return table_sets

    def _potential(self):
        own_hand = self.hands[self.ppo_player]
        own_counter = self._tile_counter(own_hand)
        hand_advantage = (
            len(self.hands[self.ilp_player]) - len(own_hand)
        )
        meld_advantage = 0.5 * (
            float(self.initial_meld_done[self.ppo_player])
            - float(self.initial_meld_done[self.ilp_player])
        )
        connection_count = (
            self._run_link_count(own_counter)
            + self._group_link_count(own_counter)
        )
        isolated_count = sum(
            count
            for tile, count in own_counter.items()
            if self._run_support(own_counter, tile) == 0
            and self._group_support(own_counter, tile) == 0
        )
        duplicate_excess = sum(
            max(0, count - 1)
            for count in own_counter.values()
        )
        deadwood_score = sum(tile.number for tile in own_hand)

        return float(
            hand_advantage
            + meld_advantage
            + 0.15 * connection_count
            - 0.30 * isolated_count
            - 0.10 * duplicate_excess
            - 0.01 * deadwood_score
        )

    def _terminal_reward(self, ppo_won, ilp_won, timed_out):
        if ppo_won:
            return self.win_reward
        if ilp_won:
            return -self.win_reward
        if timed_out:
            hand_advantage = (
                len(self.hands[self.ilp_player])
                - len(self.hands[self.ppo_player])
            )
            return float(np.clip(2.0 * hand_advantage, -10.0, 10.0))
        return 0.0

    def _potential_shaping(self, previous_potential, next_potential):
        return self.shaping_scale * (
            self.reward_gamma * next_potential - previous_potential
        )

    def _candidate_metrics(self, candidate, next_table_sets):
        current_hand_count = max(1, len(self.hands[self.ppo_player]))
        current_hand_score = max(
            1,
            sum(tile.number for tile in self.hands[self.ppo_player]),
        )
        remaining = list(candidate.remaining_hand)
        remaining_count = len(remaining)
        remaining_counter = self._tile_counter(remaining)

        isolated_count = sum(
            count
            for tile, count in remaining_counter.items()
            if self._run_support(remaining_counter, tile) == 0
            and self._group_support(remaining_counter, tile) == 0
        )
        duplicate_excess = sum(
            max(0, count - 1)
            for count in remaining_counter.values()
        )
        run_links = self._run_link_count(remaining_counter)
        group_links = self._group_link_count(remaining_counter)

        run_set_count = sum(self._is_run(tile_set) for tile_set in next_table_sets)
        group_set_count = sum(
            self._is_group(tile_set) for tile_set in next_table_sets
        )
        table_set_count = len(next_table_sets)
        average_set_length = (
            sum(len(tile_set) for tile_set in next_table_sets) / table_set_count
            if table_set_count
            else 0.0
        )
        current_set_keys = {
            self._canonical_set_key(tile_set)
            for tile_set in self.env.table_sets
        }
        next_set_keys = {
            self._canonical_set_key(tile_set)
            for tile_set in next_table_sets
        }
        preserved_set_ratio = (
            len(current_set_keys & next_set_keys) / len(current_set_keys)
            if current_set_keys
            else 1.0
        )

        return np.array(
            [
                candidate.used_hand_tile_count / current_hand_count,
                candidate.used_hand_tile_score / current_hand_score,
                sum(tile.number for tile in remaining) / current_hand_score,
                isolated_count / max(1, remaining_count),
                duplicate_excess / max(1, remaining_count),
                run_links / max(1, remaining_count * 2),
                group_links / max(1, remaining_count * 2),
                table_set_count / 35.0,
                run_set_count / max(1, table_set_count),
                group_set_count / max(1, table_set_count),
                average_set_length / 13.0,
                preserved_set_ratio,
            ],
            dtype=np.float32,
        )

    def _tile_counter(self, tiles):
        counter = {}
        for tile in tiles:
            counter[tile] = counter.get(tile, 0) + 1
        return counter

    def _run_support(self, counter, tile):
        return sum(
            counter.get(type(tile)(tile.color, number), 0) > 0
            for number in (
                tile.number - 2,
                tile.number - 1,
                tile.number + 1,
                tile.number + 2,
            )
            if 1 <= number <= 13
        )

    def _group_support(self, counter, tile):
        return sum(
            counter.get(type(tile)(color, tile.number), 0) > 0
            for color in COLORS
            if color != tile.color
        )

    def _run_link_count(self, counter):
        tiles = list(counter)
        return sum(
            left.color == right.color
            and 1 <= abs(left.number - right.number) <= 2
            for index, left in enumerate(tiles)
            for right in tiles[index + 1 :]
        )

    def _group_link_count(self, counter):
        tiles = list(counter)
        return sum(
            left.number == right.number and left.color != right.color
            for index, left in enumerate(tiles)
            for right in tiles[index + 1 :]
        )

    def _is_run(self, tile_set):
        return len({tile.color for tile in tile_set}) == 1

    def _is_group(self, tile_set):
        return len({tile.number for tile in tile_set}) == 1

    def _canonical_set_key(self, tile_set):
        return tuple(sorted((tile.color, tile.number) for tile in tile_set))

    def _sync_env_hand(self, player_id):
        self.env.hand = list(self.hands[player_id])
        self.env.initial_meld_done = self.initial_meld_done[player_id]

    def _build_info(
        self,
        candidate_count,
        ilp_used_hand_tiles,
        terminal_reward,
        shaping_reward,
    ):
        ppo_won = len(self.hands[self.ppo_player]) == 0
        ilp_won = len(self.hands[self.ilp_player]) == 0
        timeout = self.turn_count >= self.max_turns and not ppo_won and not ilp_won

        return {
            "ppo_hand_count": len(self.hands[self.ppo_player]),
            "ilp_hand_count": len(self.hands[self.ilp_player]),
            "deck_count": len(self.env.deck),
            "candidate_count": candidate_count,
            "raw_candidate_count": self.last_candidate_stats["raw_solution_count"],
            "pool_candidate_count": self.last_candidate_stats["pool_solution_count"],
            "duplicate_candidate_count": self.last_candidate_stats["duplicate_solution_count"],
            "strategy_candidate_count": self.last_candidate_stats["strategy_solution_count"],
            "candidate_solve_attempt_count": self.last_candidate_stats["solve_attempt_count"],
            "ilp_used_hand_tiles": ilp_used_hand_tiles,
            "ppo_initial_meld_done": self.initial_meld_done[self.ppo_player],
            "ilp_initial_meld_done": self.initial_meld_done[self.ilp_player],
            "terminal_reward": terminal_reward,
            "shaping_reward": shaping_reward,
            "winner": "ppo" if ppo_won else "ilp" if ilp_won else None,
            "timeout": timeout,
        }

    def _empty_candidate_stats(self):
        return {
            "solve_attempt_count": 0,
            "raw_solution_count": 0,
            "pool_solution_count": 0,
            "unique_solution_count": 0,
            "duplicate_solution_count": 0,
            "strategy_solution_count": 0,
        }

    def tiles_to_vector(self, tiles):
        vector = np.zeros(self.TILE_VECTOR_SIZE, dtype=np.float32)
        for tile in tiles:
            color_index = COLORS.index(tile.color)
            number_index = tile.number - 1
            index = color_index * 13 + number_index
            vector[index] += 1.0
        return vector / 2.0
