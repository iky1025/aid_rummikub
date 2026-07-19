"""Pre-measurement for Stage 3: the "endgame budget".

Question: of the games our best student LOSES, how many were still winnable at
endgame entry (either hand <= threshold)? That fraction upper-bounds what a
dedicated endgame policy (Stage 3) could recover, and gates Stage 3 vs going
straight to technique (1) (midgame depth).

The greedy opponent is deterministic, so "winnable from here?" is a single-agent
DFS over OUR moves — reusing autopsy_oracle.GameSim. The snapshot carries the
TRUE opp hand + deck, so this is the ORACLE upper bound (A_oracle): it counts a
loss as recoverable if SOME line wins with full information. A follow-up
belief-mode pass (only if A_oracle is promising) would split this into what a
forward-only student can actually recover vs what needs the hidden info.

Classification of each loss:
  A (endgame-recoverable) : DFS from endgame entry -> WIN   -> Stage 3 target
  B (pre-endgame-decided) : DFS from endgame entry -> NO_WIN, or never reached
                            the endgame at all               -> (1)/luck, not Stage 3
  BUDGET                  : search budget exhausted (undecided)

Usage: python endgame_budget.py --model distill_s1s_dagger1.pt --pairs 160
"""
import argparse
import time
from collections import Counter

import torch

from ppo_env import RummikubPPOEnv, STATE_DIM, CAND_FEAT_DIM
from ppo_model import DistillStudent
from autopsy_oracle import GameSim
from rummikub_solver import RummikubILPSolver, flatten
from rollout_agent import RolloutPolicy, FULL_DECK


def load_student(model_path, max_candidates=20):
    model = DistillStudent(obs_dim=STATE_DIM, cand_feat_dim=CAND_FEAT_DIM,
                           max_candidates=max_candidates)
    model.load_state_dict(torch.load(model_path, map_location="cpu",
                                     weights_only=True))
    model.eval()
    return model


def student_action(model, obs):
    """Forward-only argmax over valid actions (matches eval_mirror student)."""
    state_t = torch.tensor(obs["state"], dtype=torch.float32).unsqueeze(0)
    cand_t = torch.tensor(obs["cand_feats"], dtype=torch.float32).unsqueeze(0)
    mask_t = torch.tensor(obs["mask"], dtype=torch.float32)
    with torch.no_grad():
        logits = model.forward_actor(state_t, cand_t).squeeze(0)
    logits[mask_t == 0] = -1e9
    return int(torch.argmax(logits).item())


def play_capture_endgame(model, seed, seat, imv, threshold):
    """Play one student-vs-greedy game; snapshot the true state at the start of
    the student's turn the first time min(my_hand, opp_hand) <= threshold."""
    env = RummikubPPOEnv(seed=seed, ppo_player=seat, opponent="ilp",
                         initial_meld_value=imv)
    obs, _ = env.reset(seed=seed)
    snap = None
    done, info = False, {}
    while not done:
        if snap is None:
            me = len(env.hands[env.ppo_player])
            op = len(env.hands[env.ilp_player])
            if min(me, op) <= threshold:
                snap = {
                    "hand": list(env.hands[env.ppo_player]),
                    "opp": list(env.hands[env.ilp_player]),
                    "table": [list(s) for s in env.env.table_sets],
                    "deck": list(env.env.deck),
                    "my_meld": env.first_meld_done[env.ppo_player],
                    "opp_meld": env.first_meld_done[env.ilp_player],
                    # belief mode needs the opponent's turn history (for
                    # information-consistent hand sampling) — deep-copied.
                    "events": [dict(e) for e in env.opponent_events],
                }
        a = student_action(model, obs)
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
    return info.get("outcome", "timeout"), snap


def belief_best_vote(policy, snap, imv, det, budget):
    """Belief-mode recoverability: over `det` information-consistent
    determinizations, what fraction can the SAME first move force a win in?

    Returns (best_vote_fraction, n_det). best_vote = max over my candidate first
    moves (plays + draw) of the fraction of determinizations in which that single
    move leads to a forced win vs greedy. Committing to one first move respects
    strategy fusion — this is what a forward-only endgame policy can actually do,
    unlike the oracle DFS which may use a different winning line per world."""
    my, table = snap["hand"], snap["table"]
    opp_count, my_meld, opp_meld = len(snap["opp"]), snap["my_meld"], snap["opp_meld"]

    unseen = Counter(FULL_DECK)
    unseen.subtract(Counter(my))
    unseen.subtract(Counter(flatten(table)))
    unseen_list = []
    for tile, c in unseen.items():
        unseen_list.extend([tile] * max(0, c))

    dets = [policy._sample_consistent(unseen_list, opp_count, snap["events"], imv)
            for _ in range(det)]

    # candidate first moves (plays) + the draw option
    plays = policy._enum_my_moves(my, table, my_meld, imv, cap=8)
    best = 0
    for rem, new_table in plays:
        wins = sum(
            policy._win_after_my_move(
                list(rem), list(opp_hand), [list(s) for s in new_table],
                list(deck), True, opp_meld, imv, [budget], {})
            for opp_hand, deck in dets)
        best = max(best, wins)
    draw_wins = 0
    for opp_hand, deck in dets:
        d, h = list(deck), list(my)
        if d:
            h.append(d.pop())
        if policy._win_after_my_move(h, list(opp_hand),
                                     [list(s) for s in table], d,
                                     my_meld, opp_meld, imv, [budget], {}):
            draw_wins += 1
    best = max(best, draw_wins)
    return best / det, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distill_s1s_dagger1.pt")
    ap.add_argument("--pairs", type=int, default=160)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--meld", type=int, default=30)
    ap.add_argument("--threshold", type=int, default=5,
                    help="endgame entry: min(hand sizes) <= this")
    ap.add_argument("--budget", type=int, default=200000)
    ap.add_argument("--belief", action="store_true",
                    help="belief mode: recoverability from OBSERVABLE info only "
                         "(consistent determinizations), not the oracle upper bound")
    ap.add_argument("--det", type=int, default=16,
                    help="belief mode: determinizations per loss")
    args = ap.parse_args()

    model = load_student(args.model)
    solver_sim = GameSim(RummikubILPSolver(), args.meld)

    losses = []
    t0 = time.time()
    wins = 0
    games = 0
    for p in range(args.pairs):
        for seat in (0, 1):
            games += 1
            outcome, snap = play_capture_endgame(
                model, args.seed + p, seat, args.meld, args.threshold)
            if outcome == "win":
                wins += 1
            else:
                losses.append((args.seed + p, seat, outcome, snap))
        if (p + 1) % 20 == 0:
            print(f"replayed {p + 1}/{args.pairs} pairs, {len(losses)} losses, "
                  f"{time.time() - t0:.0f}s", flush=True)

    print(f"\nstudent win rate {wins}/{games} = {wins / games:.3f}")
    print(f"{len(losses)} losses to classify\n", flush=True)

    if args.belief:
        policy = RolloutPolicy(consistent=True, seed=args.seed)
        votes = []
        no_eg = 0
        for seed, seat, outcome, snap in losses:
            if snap is None:
                no_eg += 1
                votes.append(0.0)
                continue
            t1 = time.time()
            frac, nd = belief_best_vote(policy, snap, args.meld, args.det,
                                        args.budget)
            votes.append(frac)
            print(f"seed={seed} seat={seat} {outcome} -> best_vote={frac:.2f} "
                  f"(det={nd}, {time.time() - t1:.0f}s)", flush=True)
        n = len(losses)
        print("\n=== belief-mode endgame budget ===")
        print(f"losses: {n} (no-endgame: {no_eg})")
        import numpy as np
        v = np.array(votes)
        print(f"best_vote distribution: min={v.min():.2f} "
              f"median={np.median(v):.2f} mean={v.mean():.2f} max={v.max():.2f}")
        for tau in (0.5, 0.8, 1.0):
            a = int((v >= tau).sum())
            print(f"A_belief(tau={tau}): {a}/{n} = {a / n:.1%}  "
                  f"(recoverable if a first move wins >= {tau:.0%} of worlds)")
        a50 = (v >= 0.5).mean()
        print("\n=== decision (belief) ===")
        if a50 >= 0.35:
            print(f"A_belief(0.5)={a50:.1%} >= 35% -> START Stage 3")
        elif a50 >= 0.15:
            print(f"A_belief(0.5)={a50:.1%} in [15%,35%) -> Stage 3 small pilot")
        else:
            print(f"A_belief(0.5)={a50:.1%} < 15% -> SKIP Stage 3, go to (1)")
        return

    cls = Counter()
    no_endgame = 0
    for seed, seat, outcome, snap in losses:
        if snap is None:
            cls["B_no_endgame"] += 1
            no_endgame += 1
            continue
        t1 = time.time()
        verdict, nodes = solver_sim.search(
            snap["hand"], snap["opp"], snap["table"], snap["deck"],
            snap["my_meld"], snap["opp_meld"], budget=args.budget)
        key = {"WIN": "A_recoverable", "NO_WIN": "B_pre_decided",
               "BUDGET": "BUDGET"}[verdict]
        cls[key] += 1
        print(f"seed={seed} seat={seat} {outcome} -> {key} "
              f"(nodes={nodes}, {time.time() - t1:.0f}s)", flush=True)

    n = len(losses)
    A = cls["A_recoverable"]
    B = cls["B_pre_decided"] + cls["B_no_endgame"]
    bud = cls["BUDGET"]
    print("\n=== endgame budget ===")
    print(f"losses classified: {dict(cls)}")
    if n:
        print(f"A (endgame-recoverable, oracle upper bound): {A}/{n} = "
              f"{A / n:.1%}")
        print(f"B (pre-endgame-decided): {B}/{n} = {B / n:.1%}  "
              f"(of which no-endgame: {no_endgame})")
        print(f"BUDGET (undecided): {bud}/{n} = {bud / n:.1%}")
        a_frac = A / n
        print("\n=== pre-registered decision ===")
        if a_frac >= 0.35:
            print(f"A={a_frac:.1%} >= 35% -> START Stage 3 (endgame budget ample)")
        elif a_frac >= 0.15:
            print(f"A={a_frac:.1%} in [15%,35%) -> Stage 3 small pilot first")
        else:
            print(f"A={a_frac:.1%} < 15% -> SKIP Stage 3, go straight to (1)")


if __name__ == "__main__":
    main()
