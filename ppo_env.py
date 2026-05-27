import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rummikub_env import RummikubEnv
from rummikub_solver import COLORS, flatten


STATE_DIM = 52 + 52 + 1 + 1 + 1 + 1  # hand + table + deck + opp_hand + meld_ppo + meld_opp
CAND_FEAT_DIM = 52 + 52


class RummikubPPOEnv(gym.Env):
    """
    Gymnasium-compatible PPO env.

    The opponent type is configurable via `opponent` arg:
      - "ilp"    (default): greedy ILP solver, always plays max tiles when possible
      - "random": uniformly picks among valid candidates and draw

    The opponent's randomness uses a separate RNG seeded by `seed`, so PPO's
    own randomness doesn't perturb opponent decisions.

    Observation is a Dict with:
      - state:      current hand/table/deck encoding, shape (STATE_DIM,)
      - cand_feats: per-candidate next-state features, shape (max_candidates, CAND_FEAT_DIM)
      - mask:       valid action mask, shape (max_candidates + 1,)

    Action space is Discrete(max_candidates + 1): last index is "draw".
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_candidates=20,
        max_turns=100,
        seed=None,
        ppo_player=0,
        opponent="ilp",
        alternate_first_player=False,
        initial_meld_value=0,
    ):
        super().__init__()
        self.max_candidates = max_candidates
        self.max_turns = max_turns
        self.ppo_player = ppo_player
        self.ilp_player = 1 - ppo_player
        self._init_seed = seed
        assert opponent in ("ilp", "random"), f"unknown opponent: {opponent}"
        self.opponent_type = opponent
        # R7: alternate ppo_player between 0 and 1 per reset for symmetric training
        self.alternate_first_player = alternate_first_player
        self._next_ppo_player = ppo_player
        # R7: initial meld threshold (Rummikub house rule). 0 disables.
        self.initial_meld_value = initial_meld_value
        self.first_meld_done = [False, False]
        # separate RNG so opponent randomness doesn't perturb env shuffles
        self.opponent_random = random.Random(
            seed if seed is not None else 0
        )

        self.env = RummikubEnv(
            seed=seed,
            hand_size=14,
        )

        self.observation_space = spaces.Dict({
            "state": spaces.Box(
                low=0.0, high=10.0, shape=(STATE_DIM,), dtype=np.float32,
            ),
            "cand_feats": spaces.Box(
                low=0.0, high=10.0,
                shape=(self.max_candidates, CAND_FEAT_DIM), dtype=np.float32,
            ),
            "mask": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_candidates + 1,), dtype=np.float32,
            ),
        })
        self.action_space = spaces.Discrete(self.max_candidates + 1)

        self.turn_count = 0
        self.current_player = self.ppo_player
        self.hands = [[], []]
        self.last_candidates = []
        self._last_candidate_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.env = RummikubEnv(seed=seed, hand_size=14)
            self.opponent_random = random.Random(seed)

        # R7: alternate ppo_player per reset
        if self.alternate_first_player:
            self.ppo_player = self._next_ppo_player
            self.ilp_player = 1 - self.ppo_player
            self._next_ppo_player = 1 - self._next_ppo_player

        self.env.reset(table_sets=[], shuffle=True)
        self.first_meld_done = [False, False]

        # Deal tiles. Position 0 always gets the first 14 dealt, position 1 the next 14.
        pos0_hand = list(self.env.hand)
        pos1_hand = []
        for _ in range(self.env.hand_size):
            tile = self.env.draw_tile()
            if tile is None:
                break
            pos1_hand.append(tile)

        self.hands = [pos0_hand, pos1_hand]

        self.turn_count = 0
        self.last_candidates = []

        # R7: if PPO is at position 1, the opponent (position 0) acts first.
        # Run one opponent turn before returning the observation.
        if self.ppo_player == 1:
            self._opponent_turn()

        self.current_player = self.ppo_player
        obs = self._compute_obs()
        info = self._build_info(self._last_candidate_count, 0)
        info["ppo_player"] = self.ppo_player
        return obs, info

    def _meld_params(self, player_id):
        """Return (min_play_value, ignore_table) for solver calls."""
        if self.first_meld_done[player_id] or self.initial_meld_value <= 0:
            return 0, False
        return self.initial_meld_value, True

    def _opponent_turn(self):
        """Run one opponent turn. Updates env state and self.hands.
        Returns (used_hand_tile_count, won_this_turn)."""
        self._sync_env_hand(self.ilp_player)
        min_val, ignore_tbl = self._meld_params(self.ilp_player)
        used = 0
        won = False

        if self.opponent_type == "ilp":
            r = self.env.solve_best_move(
                min_play_value=min_val, ignore_table=ignore_tbl,
            )
            if r.status == "Optimal" and r.used_hand_tile_count > 0:
                self.env.apply_solution(r, append_to_table=ignore_tbl)
                used = r.used_hand_tile_count
                self.first_meld_done[self.ilp_player] = True
                if len(self.env.hand) == 0:
                    won = True
            else:
                self.env.draw_tile()
        else:  # random
            opp_cands = self.env.solve_candidate_moves(
                max_candidates=self.max_candidates,
                min_play_value=min_val,
                ignore_table=ignore_tbl,
            )
            n_opt = len(opp_cands)
            choice = self.opponent_random.randint(0, n_opt)
            if choice < n_opt:
                picked = opp_cands[choice]
                self.env.apply_solution(picked, append_to_table=ignore_tbl)
                used = picked.used_hand_tile_count
                self.first_meld_done[self.ilp_player] = True
                if len(self.env.hand) == 0:
                    won = True
            else:
                self.env.draw_tile()

        self.hands[self.ilp_player] = list(self.env.hand)
        return used, won

    def step(self, action):
        if self.current_player != self.ppo_player:
            raise RuntimeError("step(action) must be called on PPO player's turn.")

        self.turn_count += 1
        reward = 0.0

        # Capture meld state at decision time so info reflects pre/post-meld.
        ppo_was_pre_meld = not self.first_meld_done[self.ppo_player]

        self._sync_env_hand(self.ppo_player)
        candidates = self.last_candidates
        if not candidates:
            min_val, ignore_tbl = self._meld_params(self.ppo_player)
            candidates = self.env.solve_candidate_moves(
                max_candidates=self.max_candidates,
                min_play_value=min_val,
                ignore_table=ignore_tbl,
            )[: self.max_candidates]
            self.last_candidates = candidates

        ppo_is_initial = not self.first_meld_done[self.ppo_player]

        if action < len(candidates):
            result = candidates[action]
            self.env.apply_solution(result, append_to_table=ppo_is_initial)
            reward += 0.1 * result.used_hand_tile_count
            self.first_meld_done[self.ppo_player] = True
            if len(self.env.hand) == 0:
                ilp_remaining = len(self.hands[self.ilp_player])
                reward += 5.0 + 0.3 * ilp_remaining
                self.hands[self.ppo_player] = list(self.env.hand)
                self.last_candidates = []
                obs = self._compute_obs()
                info = self._build_info(len(candidates), 0)
                info["outcome"] = "win"
                info["win_margin"] = ilp_remaining
                info["loss_margin"] = 0
                info["ppo_player"] = self.ppo_player
                info["pre_meld"] = ppo_was_pre_meld
                return obs, reward, True, False, info
        else:
            drawn_tile = self.env.draw_tile()
            if drawn_tile is None:
                reward -= 1.0
            else:
                reward -= 0.5

        self.hands[self.ppo_player] = list(self.env.hand)
        reward -= 0.01

        self.current_player = self.ilp_player
        ilp_used_hand_tiles, ilp_done = self._opponent_turn()

        if ilp_used_hand_tiles > 0:
            reward -= 0.02 * ilp_used_hand_tiles

        if ilp_done:
            ppo_remaining = len(self.hands[self.ppo_player])
            reward -= 5.0 + 0.3 * ppo_remaining
            self.last_candidates = []
            obs = self._compute_obs()
            info = self._build_info(len(candidates), ilp_used_hand_tiles)
            info["outcome"] = "loss"
            info["win_margin"] = 0
            info["loss_margin"] = ppo_remaining
            info["ppo_player"] = self.ppo_player
            return obs, reward, True, False, info

        self.current_player = self.ppo_player
        self._sync_env_hand(self.ppo_player)
        self.last_candidates = []

        truncated = self.turn_count >= self.max_turns
        terminated = False

        obs = self._compute_obs()
        info = self._build_info(self._last_candidate_count, ilp_used_hand_tiles)
        info["ppo_player"] = self.ppo_player
        info["pre_meld"] = ppo_was_pre_meld
        if truncated:
            ppo_remaining = len(self.hands[self.ppo_player])
            ilp_remaining = len(self.hands[self.ilp_player])
            reward += 0.3 * (ilp_remaining - ppo_remaining)
            info["outcome"] = "timeout"
            info["win_margin"] = max(0, ilp_remaining - ppo_remaining)
            info["loss_margin"] = max(0, ppo_remaining - ilp_remaining)
        return obs, reward, terminated, truncated, info

    def _compute_obs(self):
        """Compute the full Dict observation. Triggers ILP candidate generation."""
        self._sync_env_hand(self.ppo_player)
        min_val, ignore_tbl = self._meld_params(self.ppo_player)
        self.last_candidates = self.env.solve_candidate_moves(
            max_candidates=self.max_candidates,
            min_play_value=min_val,
            ignore_table=ignore_tbl,
        )[: self.max_candidates]
        self._last_candidate_count = len(self.last_candidates)

        hand_vector = self.tiles_to_vector(self.hands[self.ppo_player])
        table_tiles = flatten(self.env.table_sets)
        table_vector = self.tiles_to_vector(table_tiles)
        deck_count = np.array([len(self.env.deck) / 104.0], dtype=np.float32)
        ilp_hand_norm = np.array(
            [len(self.hands[self.ilp_player]) / 14.0], dtype=np.float32,
        )
        meld_flags = np.array(
            [
                1.0 if self.first_meld_done[self.ppo_player] else 0.0,
                1.0 if self.first_meld_done[self.ilp_player] else 0.0,
            ],
            dtype=np.float32,
        )
        state = np.concatenate(
            [hand_vector, table_vector, deck_count, ilp_hand_norm, meld_flags]
        ).astype(np.float32)

        cand_feats = np.zeros((self.max_candidates, CAND_FEAT_DIM), dtype=np.float32)
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

        return {"state": state, "cand_feats": cand_feats, "mask": mask}

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
