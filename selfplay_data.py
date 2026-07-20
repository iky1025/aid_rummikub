"""R10: self-play decision dataset for distillation (Stage 0/1).

Plays mirror pairs (same seed, both seats) of teacher-vs-greedy games and
records every decision where the teacher had at least one playable candidate.
Records per decision:

  state        (STATE_DIM,) f32   observation state vector
  cand_feats   (C, 104)     u8    candidate next-state features, raw counts
                                  (obs values * 2 — undo the /2.0 scaling)
  mask         (C + 1,)     u8    valid-action mask
  action       ()           i16   teacher's choice (candidate index or C=draw)
  n_candidates ()           i16   number of real candidates
  opp_hand     (52,)        u8    TRUE opponent hand counts (aux label only —
                                  never shown to the teacher unless --teacher
                                  is an oracle mode)
  events       (K, 4)       i16   last K opponent turn events, oldest first:
                                  (drew, hand_before, hand_after, pre_meld),
                                  -1-padded. Compact history for future
                                  sequence encoders; tables are not stored.
  outcome      ()           i8    game result from the agent's seat: +1/-1/0
  net          ()           i16   win_margin - loss_margin of the game
  seat         ()           i8    agent seat (0/1), seed: pair seed

Deviation labels need no extra field: candidate 0 is always the greedy
max-play (solve_many orders max-k first), so `action != 0` marks a decision
where the teacher deviated from greedy.

Usage:
  # Stage 0: greedy self-play (label = greedy's own move)
  python selfplay_data.py --teacher greedy --pairs 4000 --seed 100000 \
      --initial-meld-value 30 --out data/stage0 --workers 8

  # Stage 1: fair combo teacher
  python selfplay_data.py --teacher rollout --consistent --greedy-margin 1.0 \
      --endgame-search --determinizations 8 --rollout-turns 24 \
      --candidate-cap 4 --pairs 1000 --seed 200000 \
      --initial-meld-value 30 --out data/stage1 --workers 8
"""

import argparse
import os
import time

import numpy as np

from ppo_env import RummikubPPOEnv

EVENT_HISTORY_LEN = 6


def make_teacher(args):
    """Return fn(env, obs) -> action. Mirrors eval_mirror.make_policy."""
    max_candidates = args.max_candidates

    if args.teacher == "greedy":
        def teacher(env, obs):
            if obs["mask"][:max_candidates].sum() > 0:
                return 0
            return max_candidates
        return teacher

    if args.teacher == "rollout":
        from rollout_agent import RolloutPolicy
        rollout = RolloutPolicy(
            n_determinizations=args.determinizations,
            max_rollout_turns=args.rollout_turns,
            candidate_cap=args.candidate_cap,
            greedy_margin=args.greedy_margin,
            oracle=args.oracle,
            consistent=args.consistent,
            endgame_search=args.endgame_search,
            search_nodes=args.search_nodes,
            seed=args.seed,
        )
        def teacher(env, obs):
            return rollout.select_action(env)
        teacher.rollout = rollout
        return teacher

    raise ValueError(f"unknown teacher: {args.teacher}")


def make_student_actor(args):
    """R10 DAgger: the STUDENT plays the game (visiting its own state
    distribution, including the states its mistakes create) while the teacher
    only labels. Mirrors eval_mirror's student policy incl. the margin gate
    and optional history features."""
    import torch
    from distill import EVENT_FEAT_DIM, event_feats
    from ppo_env import CAND_FEAT_DIM, STATE_DIM
    from ppo_model import DistillStudent

    obs_dim = STATE_DIM + (EVENT_FEAT_DIM if args.actor_history else 0)
    model = DistillStudent(
        obs_dim=obs_dim,
        cand_feat_dim=CAND_FEAT_DIM,
        max_candidates=args.max_candidates,
    )
    model.load_state_dict(
        torch.load(args.actor_model, map_location="cpu", weights_only=True))
    model.eval()
    margin = args.actor_margin

    def actor(env, obs):
        state = obs["state"]
        if args.actor_history:
            state = np.concatenate(
                [state, event_feats(_event_history(env)[None])[0]]
            ).astype(np.float32)
        state_t = torch.tensor(state, dtype=torch.float32)
        cand_t = torch.tensor(obs["cand_feats"], dtype=torch.float32)
        mask_t = torch.tensor(obs["mask"], dtype=torch.float32)
        with torch.no_grad():
            logits = model.forward_actor(
                state_t.unsqueeze(0), cand_t.unsqueeze(0)).squeeze(0)
        logits[mask_t == 0] = -1e9
        best = int(torch.argmax(logits).item())
        if margin > 0 and mask_t[0] > 0 and best != 0 \
                and float(logits[best] - logits[0]) <= margin:
            return 0
        return best
    return actor


def _event_history(env):
    out = np.full((EVENT_HISTORY_LEN, 4), -1, dtype=np.int16)
    events = env.opponent_events[-EVENT_HISTORY_LEN:]
    for i, e in enumerate(events):
        out[i] = (int(e["drew"]), e["hand_before"], e["hand_after"],
                  int(e["pre_meld"]))
    return out


def play_game(teacher, seed, seat, args, actor=None):
    env = RummikubPPOEnv(
        max_candidates=args.max_candidates,
        max_turns=args.max_turns,
        seed=seed,
        ppo_player=seat,
        opponent="ilp",
        initial_meld_value=args.initial_meld_value,
        with_jokers=getattr(args, "with_jokers", False),
    )
    obs, _ = env.reset(seed=seed)
    records = []
    done = False
    info = {}
    turn = 0
    n_actions = args.max_candidates + 1
    while not done:
        n_cands = len(env.last_candidates)
        if n_cands > 0:
            # The teacher always provides the label (and its evaluations);
            # with a separate actor (DAgger) the game then follows the
            # actor's move instead. Labelling first is safe: the teacher
            # only reads env and its own RNG.
            action = teacher(env, obs)
            # R10: teacher's per-candidate evaluations (rollout avg scores /
            # endgame win-forcing vote fractions), NaN where not evaluated.
            cand_scores = np.full(n_actions, np.nan, dtype=np.float32)
            cand_votes = np.full(n_actions, np.nan, dtype=np.float32)
            ev = getattr(getattr(teacher, "rollout", None), "last_eval", None)
            if ev is not None:
                for i, s in (ev["scores"] or {}).items():
                    cand_scores[i] = s
                for i, v in (ev["votes"] or {}).items():
                    cand_votes[i] = v
            opp_hand = env.tiles_to_vector(env.hands[env.ilp_player]) * 2.0
            records.append({
                "cand_scores": cand_scores,
                "cand_votes": cand_votes,
                "state": obs["state"].astype(np.float32),
                "cand_feats": np.rint(obs["cand_feats"] * 2.0).astype(np.uint8),
                "mask": obs["mask"].astype(np.uint8),
                "action": np.int16(action),
                "n_candidates": np.int16(n_cands),
                "opp_hand": np.rint(opp_hand).astype(np.uint8),
                "events": _event_history(env),
                "turn": np.int16(turn),
            })
            if actor is not None:
                played = actor(env, obs)
                records[-1]["actor_action"] = np.int16(played)
            else:
                played = action
                records[-1]["actor_action"] = np.int16(played)
        else:
            played = env.max_candidates
        obs, _, terminated, truncated, info = env.step(played)
        done = terminated or truncated
        turn += 1

    outcome = {"win": 1, "loss": -1, "timeout": 0}[info.get("outcome", "timeout")]
    net = int(info.get("win_margin", 0)) - int(info.get("loss_margin", 0))
    for r in records:
        r["outcome"] = np.int8(outcome)
        r["net"] = np.int16(net)
        r["seat"] = np.int8(seat)
        r["seed"] = np.int32(seed)
    return records


_WORKER_TEACHER = None
_WORKER_ACTOR = None


def _pair_worker(payload):
    global _WORKER_TEACHER, _WORKER_ACTOR
    args, seed = payload
    if _WORKER_TEACHER is None:
        _WORKER_TEACHER = make_teacher(args)
        if args.actor == "student":
            _WORKER_ACTOR = make_student_actor(args)

    records = []
    for seat in (0, 1):
        records.extend(
            play_game(_WORKER_TEACHER, seed, seat, args, actor=_WORKER_ACTOR))
    if not records:
        return 0

    path = os.path.join(args.out, f"pair_{seed}.npz")
    stacked = {
        key: np.stack([r[key] for r in records])
        for key in records[0]
    }
    np.savez_compressed(path, **stacked)
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", choices=["greedy", "rollout"], required=True)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100000,
                        help="first pair seed (keep disjoint from eval seeds!)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--initial-meld-value", type=int, default=30)
    # rollout teacher knobs (same defaults as eval_mirror fair combo)
    parser.add_argument("--determinizations", type=int, default=8)
    parser.add_argument("--rollout-turns", type=int, default=24)
    parser.add_argument("--candidate-cap", type=int, default=4)
    parser.add_argument("--greedy-margin", type=float, default=1.0)
    parser.add_argument("--oracle", nargs="?", const="full",
                        choices=["full", "hand"], default=None)
    parser.add_argument("--consistent", action="store_true")
    parser.add_argument("--endgame-search", action="store_true")
    parser.add_argument("--search-nodes", type=int, default=200)
    parser.add_argument("--actor", choices=["teacher", "student"],
                        default="teacher",
                        help="who PLAYS the game; the teacher always labels. "
                             "'student' = DAgger collection on the student's "
                             "own state distribution")
    parser.add_argument("--actor-model", default="distill_s1s_dagger1.pt")
    parser.add_argument("--actor-history", action="store_true")
    parser.add_argument("--actor-margin", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--recycle-after", type=int, default=25,
                        help="restart each worker process after this many "
                             "pairs (bounds slow memory growth)")
    parser.add_argument("--with-jokers", action="store_true",
                        help="R11: 106-tile deck with 2 jokers (wildcards)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pair_seeds = [args.seed + p for p in range(args.pairs)]
    # Skip pairs already on disk — makes reruns resumable.
    todo = [s for s in pair_seeds
            if not os.path.exists(os.path.join(args.out, f"pair_{s}.npz"))]
    print(f"pairs: {args.pairs} requested, {len(todo)} to generate "
          f"({args.pairs - len(todo)} already on disk)", flush=True)

    t0 = time.time()
    total = 0
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        # Long-lived Python workers slowly grow their RSS over hours, which
        # got a run jetsam-killed on a 16GB machine. Bound that by replacing
        # the whole pool every `workers * recycle_after` pairs. (Per-worker
        # max_tasks_per_child deadlocked when all workers hit the recycle
        # boundary at once — run stalled at exactly workers*recycle_after
        # pairs, 2026-07-09.)
        chunk = args.workers * args.recycle_after
        done_count = 0
        for c0 in range(0, len(todo), chunk):
            block = todo[c0:c0 + chunk]
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for n in ex.map(_pair_worker, [(args, s) for s in block]):
                    total += n
                    done_count += 1
                    if done_count % 50 == 0:
                        rate = done_count / (time.time() - t0)
                        eta = (len(todo) - done_count) / rate
                        print(f"  {done_count}/{len(todo)} pairs, "
                              f"{total} decisions, ETA {eta / 60:.0f}min",
                              flush=True)
    else:
        for i, s in enumerate(todo):
            total += _pair_worker((args, s))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(todo)} pairs, {total} decisions", flush=True)

    print(f"done: {total} decisions in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
