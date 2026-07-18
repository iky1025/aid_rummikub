"""Interactive play backend: human vs the distilled student, or spectate the
student vs the greedy-ILP baseline.

This reuses the exact observation encoding and candidate generation that the
model was trained/evaluated with (ppo_env / rummikub_env), but drives the two
seats manually so a human can take one of them. Default rules match the
training regime: initial-meld 30, max_candidates 20, non-exhaustive candidates
(solve_many).

Used by play_server.py (web GUI) and playable headless from a REPL.
"""

from collections import Counter

import numpy as np
import torch

from rummikub_env import RummikubEnv
from rummikub_solver import COLORS, flatten
from ppo_env import STATE_DIM, CAND_FEAT_DIM
from ppo_model import DistillStudent

MAX_CANDIDATES = 20


def tiles_to_vec(tiles):
    """Same /2.0-scaled 52-dim encoding used in ppo_env."""
    v = np.zeros(52, dtype=np.float32)
    for t in tiles:
        v[COLORS.index(t.color) * 13 + (t.number - 1)] += 1.0
    return v / 2.0


def tile_dict(t):
    return {"color": t.color, "number": t.number, "label": f"{t.color}{t.number}"}


def _multiset_diff(before, after):
    """Tiles in `before` that are not in `after` (played tiles)."""
    c = Counter(after)
    played = []
    for t in before:
        if c[t] > 0:
            c[t] -= 1
        else:
            played.append(t)
    return played


class Student:
    """Thin wrapper around DistillStudent for inference + aux hand prediction."""

    def __init__(self, model_path, obs_dim=STATE_DIM):
        self.model = DistillStudent(
            obs_dim=obs_dim, cand_feat_dim=CAND_FEAT_DIM,
            max_candidates=MAX_CANDIDATES,
        )
        sd = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(sd)
        self.model.eval()

    def score(self, state, cand_feats, mask):
        """Return (logits[21], aux_pred_counts[52]) for a single observation."""
        st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        cf = torch.tensor(cand_feats, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model.forward_actor(st, cf).squeeze(0).numpy()
            aux = self.model.forward_aux(st).squeeze(0).numpy()
        masked = logits.copy()
        masked[np.asarray(mask) == 0] = -1e9
        return logits, masked, aux


class GameSession:
    """One 2-player game. Seat indices 0/1. One seat is the model, the other is
    the human (human-vs-model) or the greedy baseline (spectate)."""

    def __init__(self, student, seed=0, meld=30, model_seat=0,
                 human_seat=1, spectate=False, student_margin=0.0):
        self.student = student
        self.meld = meld
        self.model_seat = model_seat
        self.human_seat = human_seat        # in spectate, this seat = greedy
        self.spectate = spectate
        self.student_margin = student_margin

        self.renv = RummikubEnv(seed=seed, hand_size=14)
        self.renv.reset(table_sets=[], shuffle=True)
        hand0 = list(self.renv.hand)
        hand1 = []
        for _ in range(14):
            hand1.append(self.renv.draw_tile())
        self.hands = [hand0, hand1]
        self.first_meld_done = [False, False]
        self.current = 0                     # seat 0 moves first
        self.turn = 0
        self.max_turns = 100
        self.outcome = None                  # None | 'model' | 'human' | 'draw'
        self.log = []                        # human-readable move history
        self.last_model = None               # decision detail of model's last turn

    # ---- rules helpers -------------------------------------------------
    def _meld_params(self, pid):
        if self.first_meld_done[pid] or self.meld <= 0:
            return 0, False
        return self.meld, True

    def _gen_candidates(self, pid):
        mv, ig = self._meld_params(pid)
        self.renv.hand = list(self.hands[pid])
        cands = self.renv.solve_candidate_moves(
            max_candidates=MAX_CANDIDATES, min_play_value=mv, ignore_table=ig,
        )[:MAX_CANDIDATES]
        return cands, ig

    def _obs_for(self, pid):
        """Build the exact ppo_env observation from seat `pid`'s perspective."""
        opp = 1 - pid
        hand_v = tiles_to_vec(self.hands[pid])
        table_v = tiles_to_vec(flatten(self.renv.table_sets))
        deck_c = np.array([len(self.renv.deck) / 104.0], dtype=np.float32)
        opp_c = np.array([len(self.hands[opp]) / 14.0], dtype=np.float32)
        meld_f = np.array([
            1.0 if self.first_meld_done[pid] else 0.0,
            1.0 if self.first_meld_done[opp] else 0.0,
        ], dtype=np.float32)
        state = np.concatenate([hand_v, table_v, deck_c, opp_c, meld_f]).astype(np.float32)

        cands, ig = self._gen_candidates(pid)
        cand_feats = np.zeros((MAX_CANDIDATES, CAND_FEAT_DIM), dtype=np.float32)
        mask = np.zeros(MAX_CANDIDATES + 1, dtype=np.float32)
        for i, r in enumerate(cands):
            next_table = []
            for s in r.selected_sets:
                next_table.extend(s.completed_tiles)
            cand_feats[i] = np.concatenate(
                [tiles_to_vec(r.remaining_hand), tiles_to_vec(next_table)]
            ).astype(np.float32)
            mask[i] = 1.0
        mask[MAX_CANDIDATES] = 1.0
        return state, cand_feats, mask, cands, ig

    def _describe_candidate(self, pid, r):
        played = _multiset_diff(self.hands[pid], r.remaining_hand)
        sets = [[tile_dict(t) for t in s.completed_tiles] for s in r.selected_sets]
        return {
            "played": [tile_dict(t) for t in played],
            "played_count": len(played),
            "sets": sets,
        }

    def _apply(self, pid, action, cands, ig):
        """Apply seat pid's action. action == MAX_CANDIDATES means draw.
        Returns a short description string."""
        self.renv.hand = list(self.hands[pid])
        if action >= len(cands):
            tile = self.renv.draw_tile()
            self.hands[pid] = list(self.renv.hand)
            return "draw" + (f" {tile.color}{tile.number}" if tile else " (deck empty)")
        r = cands[action]
        played = _multiset_diff(self.hands[pid], r.remaining_hand)
        self.renv.apply_solution(r, append_to_table=ig)
        self.hands[pid] = list(self.renv.hand)
        self.first_meld_done[pid] = True
        lbl = " ".join(f"{t.color}{t.number}" for t in played)
        return f"play {len(played)} tiles [{lbl}]"

    def _check_end(self):
        if len(self.hands[self.model_seat]) == 0:
            self.outcome = "model"
        elif len(self.hands[self.human_seat]) == 0:
            self.outcome = "human"
        elif self.turn >= self.max_turns:
            m = len(self.hands[self.model_seat])
            h = len(self.hands[self.human_seat])
            self.outcome = "model" if m < h else "human" if h < m else "draw"

    # ---- turn drivers --------------------------------------------------
    def model_turn(self):
        """Run the model's seat. Records decision detail in self.last_model."""
        pid = self.model_seat
        state, cand_feats, mask, cands, ig = self._obs_for(pid)
        logits, masked, aux = self.student.score(state, cand_feats, mask)
        best = int(np.argmax(masked))
        # optional winner's-curse guard vs greedy candidate 0
        if (self.student_margin > 0 and mask[0] > 0 and best != MAX_CANDIDATES
                and best != 0
                and float(masked[best] - masked[0]) <= self.student_margin):
            best = 0

        ranking = []
        for i in range(len(cands)):
            d = self._describe_candidate(pid, cands[i])
            d["logit"] = float(logits[i])
            d["index"] = i
            ranking.append(d)
        ranking.sort(key=lambda d: -d["logit"])
        # Collapse candidates that play the SAME hand tiles but only rearrange
        # the table differently — indistinguishable to a player (the opponent
        # rearranges the table anyway). Keep the highest-logit representative,
        # unless the chosen move is in the group (so its highlight survives).
        deduped, seen = [], {}
        for d in ranking:  # sorted desc, so first per group = highest logit
            sig = tuple(sorted(t["label"] for t in d["played"]))
            if sig in seen:
                if d["index"] == best:
                    deduped[seen[sig]] = d
                continue
            seen[sig] = len(deduped)
            deduped.append(d)
        ranking = deduped
        draw_logit = float(logits[MAX_CANDIDATES])

        self.turn += 1
        desc = self._apply(pid, best, cands, ig)
        self.log.append(f"[model] {desc}")
        self._check_end()

        self.last_model = {
            "chose_draw": bool(best >= len(cands)),
            "chosen_index": int(best),
            "greedy_index": 0 if mask[0] > 0 else MAX_CANDIDATES,
            "matches_greedy": bool((best == 0) or (best >= len(cands) and mask[0] == 0)),
            "ranking": ranking,
            "draw_logit": draw_logit,
            "n_candidates": len(cands),
            "aux_pred": (np.clip(aux, 0, None) * 2.0).tolist(),  # 52 counts
            "desc": desc,
        }
        self.current = 1 - self.current
        return self.last_model

    def greedy_turn(self):
        """Run the greedy-ILP seat (spectate mode). Max-play = candidate 0."""
        pid = self.human_seat
        cands, ig = self._gen_candidates(pid)
        action = 0 if cands else MAX_CANDIDATES
        self.turn += 1
        desc = self._apply(pid, action, cands, ig)
        self.log.append(f"[greedy] {desc}")
        self._check_end()
        self.current = 1 - self.current
        return desc

    def human_candidates(self):
        """Candidates offered to the human seat, plus the model's 'what I'd do
        in your shoes' hint (running the net on the human's position)."""
        pid = self.human_seat
        state, cand_feats, mask, cands, ig = self._obs_for(pid)
        logits, masked, _ = self.student.score(state, cand_feats, mask)
        best = int(np.argmax(masked))
        # Collapse table-only rearrangements (same tiles leave the hand) into a
        # single choice, keeping the highest-logit representative.
        options, groups = [], {}
        for i, r in enumerate(cands):
            d = self._describe_candidate(pid, r)
            d["index"] = i
            d["model_logit"] = float(logits[i])
            sig = tuple(sorted(f"{t.color}{t.number}" for t in r.remaining_hand))
            if sig in groups:
                j = groups[sig]
                if d["model_logit"] > options[j]["model_logit"]:
                    options[j] = d
            else:
                groups[sig] = len(options)
                options.append(d)
        return {
            "options": options,
            "draw_index": MAX_CANDIDATES,
            "draw_logit": float(logits[MAX_CANDIDATES]),
            "model_hint_index": best,        # what the model would pick here
        }

    def human_move(self, action):
        pid = self.human_seat
        cands, ig = self._gen_candidates(pid)
        self.turn += 1
        desc = self._apply(pid, action, cands, ig)
        self.log.append(f"[human] {desc}")
        self._check_end()
        self.current = 1 - self.current
        return desc

    # ---- serialization -------------------------------------------------
    def public_state(self, reveal_hands=False):
        return {
            "turn": self.turn,
            "current": self.current,
            "model_seat": self.model_seat,
            "human_seat": self.human_seat,
            "spectate": self.spectate,
            "table": [[tile_dict(t) for t in s] for s in self.renv.table_sets],
            "deck_count": len(self.renv.deck),
            "human_hand": [tile_dict(t) for t in sorted(
                self.hands[self.human_seat], key=lambda t: (t.color, t.number))],
            "model_hand_count": len(self.hands[self.model_seat]),
            "human_hand_count": len(self.hands[self.human_seat]),
            "first_meld_done": list(self.first_meld_done),
            "meld": self.meld,
            "outcome": self.outcome,
            "log": self.log[-40:],
            "model_hand": ([tile_dict(t) for t in sorted(
                self.hands[self.model_seat], key=lambda t: (t.color, t.number))]
                if (reveal_hands or self.spectate or self.outcome) else None),
        }
