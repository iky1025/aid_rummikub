import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rummikub_env import RummikubEnv
from rummikub_solver import COLORS, JOKER_VALUE, flatten


# Per-tile one-hot dim: 52 (4 colors x 13) jokerless; +1 joker slot (index 52)
# under with_jokers (R11). hand + table + deck_frac + opp_hand_COUNT + 2 meld.
# The opponent slot is their hand *size* (len/14), never their tiles — the
# hidden hand is only ever an aux label / oracle-teacher input, never an obs.
TILE_DIM = 52
TILE_DIM_JOKER = 53


def state_dim_for(with_jokers=False):
    td = TILE_DIM_JOKER if with_jokers else TILE_DIM
    return 2 * td + 4


def cand_feat_dim_for(with_jokers=False):
    td = TILE_DIM_JOKER if with_jokers else TILE_DIM
    return 2 * td


STATE_DIM = state_dim_for(False)       # 108 (jokerless default)
CAND_FEAT_DIM = cand_feat_dim_for(False)  # 104


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
        exhaustive_candidates=False,
        generating_candidates=False,
        opponent_policy=None,
        value_scoring=False,
        end_on_stuck=False,
        with_jokers=False,
        reward_mode="dense",
    ):
        super().__init__()
        self.with_jokers = with_jokers
        self.tile_dim = TILE_DIM_JOKER if with_jokers else TILE_DIM
        self.state_dim = state_dim_for(with_jokers)
        self.cand_feat_dim = cand_feat_dim_for(with_jokers)
        self.max_candidates = max_candidates
        self.max_turns = max_turns
        # Track B (real rules): value_scoring -> margins are the SUM OF TILE
        # VALUES (official scoring) instead of tile count. end_on_stuck -> the
        # game ends when the pool is empty AND both players consecutively can't
        # play (official pool-empty-and-stuck end), with the lower tile-value
        # sum winning; otherwise it rides to the max_turns safety cap as before.
        self.value_scoring = value_scoring
        # R12: reward_mode. "dense" is the R1-R7 shaping (+0.1/played tile,
        # -0.5 draw, -0.01 per turn, ...). Its optimal policy is "play as many
        # tiles as possible every turn" -- i.e. EXACTLY the greedy opponent, so
        # 50% is a fixed point of learning (diagnosis cause #1). "sparse" pays
        # only the game outcome (+1 win / -1 loss / 0 tie), which is the only
        # reward whose optimum is actually "win more games". Use sparse for any
        # fine-tune of the distilled policy; dense only reproduces R1-R7.
        assert reward_mode in ("dense", "sparse"), reward_mode
        self.reward_mode = reward_mode
        self._w_dense = 1.0 if reward_mode == "dense" else 0.0
        self._w_sparse = 0.0 if reward_mode == "dense" else 1.0
        self.end_on_stuck = end_on_stuck
        self._stuck_streak = 0
        self.ppo_player = ppo_player
        self.ilp_player = 1 - ppo_player
        self._init_seed = seed
        assert opponent in ("ilp", "random", "student"), \
            f"unknown opponent: {opponent}"
        self.opponent_type = opponent
        # opponent="student": an injected policy callable obs-dict -> action int
        self.opponent_policy = opponent_policy
        # R7: alternate ppo_player between 0 and 1 per reset for symmetric training
        self.alternate_first_player = alternate_first_player
        self._next_ppo_player = ppo_player
        # R7: initial meld threshold (Rummikub house rule). 0 disables.
        self.initial_meld_value = initial_meld_value
        # R9: exhaustive move enumeration for the PPO player's candidates.
        self.exhaustive_candidates = exhaustive_candidates
        # R11: generating-DP candidate list (complete + fast, no arrangement
        # duplicates). Takes precedence over exhaustive_candidates when set.
        self.generating_candidates = generating_candidates
        self.first_meld_done = [False, False]
        # separate RNG so opponent randomness doesn't perturb env shuffles
        self.opponent_random = random.Random(
            seed if seed is not None else 0
        )

        self.env = RummikubEnv(
            seed=seed,
            hand_size=14,
            with_jokers=with_jokers,
        )

        self.observation_space = spaces.Dict({
            "state": spaces.Box(
                low=0.0, high=10.0, shape=(self.state_dim,), dtype=np.float32,
            ),
            "cand_feats": spaces.Box(
                low=0.0, high=10.0,
                shape=(self.max_candidates, self.cand_feat_dim), dtype=np.float32,
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
        self.opponent_events = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.env = RummikubEnv(seed=seed, hand_size=14,
                                   with_jokers=self.with_jokers)
            self.opponent_random = random.Random(seed)

        # R7: alternate ppo_player per reset
        if self.alternate_first_player:
            self.ppo_player = self._next_ppo_player
            self.ilp_player = 1 - self.ppo_player
            self._next_ppo_player = 1 - self._next_ppo_player

        self.env.reset(table_sets=[], shuffle=True)
        self.first_meld_done = [False, False]
        # R9: public record of opponent turns, for information-consistent
        # determinization (each draw is a constraint on their hidden hand).
        self.opponent_events = []

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
        self._stuck_streak = 0

        # R7: if PPO is at position 1, the opponent (position 0) acts first.
        # Run one opponent turn before returning the observation.
        if self.ppo_player == 1:
            self._opponent_turn()

        self.current_player = self.ppo_player
        obs = self._compute_obs()
        info = self._build_info(self._last_candidate_count, 0)
        info["ppo_player"] = self.ppo_player
        return obs, info

    def _score(self, tiles):
        """Losing-margin for a hand: sum of tile VALUES (official) if
        value_scoring, else tile COUNT (legacy). A joker held in hand is a
        30-point penalty (official)."""
        if self.value_scoring:
            return sum(JOKER_VALUE if t.is_joker else t.number for t in tiles)
        return len(tiles)

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

        # R9: snapshot public state BEFORE the opponent acts.
        event = {
            "table": [list(s) for s in self.env.table_sets],
            "pre_meld": not self.first_meld_done[self.ilp_player],
            "hand_before": len(self.hands[self.ilp_player]),
        }

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
        elif self.opponent_type == "student":
            # Drive the opponent with an injected policy. Swap perspective so the
            # shared _compute_obs (which reads self.ppo_player) encodes from the
            # opponent's side exactly as during training, then restore.
            self.ppo_player, self.ilp_player = self.ilp_player, self.ppo_player
            obs = self._compute_obs()
            cands = self.last_candidates
            a = int(self.opponent_policy(obs))
            self.ppo_player, self.ilp_player = self.ilp_player, self.ppo_player
            if a < len(cands):
                picked = cands[a]
                self.env.apply_solution(picked, append_to_table=ignore_tbl)
                used = picked.used_hand_tile_count
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

        event["drew"] = used == 0
        event["hand_after"] = len(self.hands[self.ilp_player])
        self.opponent_events.append(event)
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
            candidates = self._generate_candidates(min_val, ignore_tbl)
            self.last_candidates = candidates

        # append_to_table must match the ignore_table mode the candidates were
        # generated with. With initial_meld_value=0 candidates always rearrange
        # the whole table (ignore_table=False), so they must REPLACE it —
        # appending would duplicate table tiles.
        _, ppo_ignore_tbl = self._meld_params(self.ppo_player)

        if action < len(candidates):
            result = candidates[action]
            self.env.apply_solution(result, append_to_table=ppo_ignore_tbl)
            reward += self._w_dense * 0.1 * result.used_hand_tile_count
            self.first_meld_done[self.ppo_player] = True
            self._stuck_streak = 0  # a play breaks a stuck streak
            if len(self.env.hand) == 0:
                ilp_remaining = len(self.hands[self.ilp_player])
                reward += self._w_dense * (5.0 + 0.3 * ilp_remaining)
                reward += self._w_sparse * 1.0        # win
                self.hands[self.ppo_player] = list(self.env.hand)
                obs = self._terminal_obs()
                info = self._build_info(len(candidates), 0)
                info["outcome"] = "win"
                info["win_margin"] = self._score(self.hands[self.ilp_player])
                info["loss_margin"] = 0
                info["ppo_player"] = self.ppo_player
                info["pre_meld"] = ppo_was_pre_meld
                return obs, reward, True, False, info
        else:
            drawn_tile = self.env.draw_tile()
            if drawn_tile is None:
                reward -= self._w_dense * 1.0
                self._stuck_streak += 1  # pool-empty pass (can't draw, can't play)
            else:
                reward -= self._w_dense * 0.5
                self._stuck_streak = 0

        self.hands[self.ppo_player] = list(self.env.hand)
        reward -= self._w_dense * 0.01

        if self.end_on_stuck and self._stuck_streak >= 2:
            return self._stuck_terminal(reward, ppo_was_pre_meld, len(candidates))

        self.current_player = self.ilp_player
        opp_before = len(self.hands[self.ilp_player])
        ilp_used_hand_tiles, ilp_done = self._opponent_turn()
        opp_passed = (ilp_used_hand_tiles == 0
                      and len(self.hands[self.ilp_player]) == opp_before)
        if ilp_used_hand_tiles > 0:
            self._stuck_streak = 0
        elif opp_passed:
            self._stuck_streak += 1
        else:
            self._stuck_streak = 0  # drew a real tile

        if ilp_used_hand_tiles > 0:
            reward -= self._w_dense * 0.02 * ilp_used_hand_tiles

        if self.end_on_stuck and not ilp_done and self._stuck_streak >= 2:
            return self._stuck_terminal(reward, ppo_was_pre_meld, len(candidates))

        if ilp_done:
            ppo_remaining = len(self.hands[self.ppo_player])
            reward -= self._w_dense * (5.0 + 0.3 * ppo_remaining)
            reward -= self._w_sparse * 1.0        # loss
            obs = self._terminal_obs()
            info = self._build_info(len(candidates), ilp_used_hand_tiles)
            info["outcome"] = "loss"
            info["win_margin"] = 0
            info["loss_margin"] = self._score(self.hands[self.ppo_player])
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
            ppo_s = self._score(self.hands[self.ppo_player])
            ilp_s = self._score(self.hands[self.ilp_player])
            # sparse: a turn-cap timeout is not a game result -> 0, so the
            # policy is never paid for hoarding a tile lead it never converted.
            reward += self._w_dense * 0.3 * (len(self.hands[self.ilp_player])
                                             - len(self.hands[self.ppo_player]))
            info["outcome"] = "timeout"
            info["win_margin"] = max(0, ilp_s - ppo_s)
            info["loss_margin"] = max(0, ppo_s - ilp_s)
        return obs, reward, terminated, truncated, info

    def _stuck_terminal(self, reward, ppo_was_pre_meld, cand_count):
        """Official pool-empty-and-stuck end: neither player can move and the
        deck is empty. The lower tile-value (or count) sum wins; a tie draws."""
        ppo_s = self._score(self.hands[self.ppo_player])
        ilp_s = self._score(self.hands[self.ilp_player])
        obs = self._terminal_obs()
        info = self._build_info(cand_count, 0)
        info["ppo_player"] = self.ppo_player
        info["pre_meld"] = ppo_was_pre_meld
        if ppo_s < ilp_s:
            info["outcome"] = "win"
            info["win_margin"] = ilp_s - ppo_s
            info["loss_margin"] = 0
            reward += self._w_dense * 5.0 + self._w_sparse * 1.0
        elif ilp_s < ppo_s:
            info["outcome"] = "loss"
            info["win_margin"] = 0
            info["loss_margin"] = ppo_s - ilp_s
            reward -= self._w_dense * 5.0 + self._w_sparse * 1.0
        else:
            info["outcome"] = "timeout"  # exact tie
            info["win_margin"] = 0
            info["loss_margin"] = 0
        return obs, reward, True, False, info

    def _terminal_obs(self):
        """Observation for a terminated episode. The episode is over, so the
        candidate list is never used — skip the expensive ILP calls entirely."""
        self.last_candidates = []
        self._last_candidate_count = 0
        state = self._state_vector()
        cand_feats = np.zeros((self.max_candidates, self.cand_feat_dim), dtype=np.float32)
        mask = np.zeros(self.max_candidates + 1, dtype=np.float32)
        mask[self.max_candidates] = 1.0
        return {"state": state, "cand_feats": cand_feats, "mask": mask}

    def _state_vector(self):
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
        return np.concatenate(
            [hand_vector, table_vector, deck_count, ilp_hand_norm, meld_flags]
        ).astype(np.float32)

    def _generate_candidates(self, min_val, ignore_tbl):
        if self.generating_candidates:
            gen = self.env.generate_candidate_moves
        elif self.exhaustive_candidates:
            gen = self.env.enumerate_candidate_moves
        else:
            gen = self.env.solve_candidate_moves
        return gen(
            max_candidates=self.max_candidates,
            min_play_value=min_val,
            ignore_table=ignore_tbl,
        )[: self.max_candidates]

    def _compute_obs(self):
        """Compute the full Dict observation. Triggers ILP candidate generation."""
        self._sync_env_hand(self.ppo_player)
        min_val, ignore_tbl = self._meld_params(self.ppo_player)
        self.last_candidates = self._generate_candidates(min_val, ignore_tbl)
        self._last_candidate_count = len(self.last_candidates)

        state = self._state_vector()

        cand_feats = np.zeros((self.max_candidates, self.cand_feat_dim), dtype=np.float32)
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
        vector = np.zeros(self.tile_dim, dtype=np.float32)
        for tile in tiles:
            if tile.is_joker:
                vector[52] += 1.0  # joker slot (only present when tile_dim==53)
            else:
                index = COLORS.index(tile.color) * 13 + (tile.number - 1)
                vector[index] += 1.0
        return vector / 2.0
