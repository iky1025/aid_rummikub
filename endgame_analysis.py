"""Analyze the student's late-game conservatism.

The distilled student is forward-only (no DFS at inference) — any holding is a
LEARNED behavior distilled from the search teacher's endgame holds. We ask:
  1. Does it draw / underplay (leave tiles unplayed) more as the deck depletes?
  2. Do losses come from drawing into deck-exhaustion and losing on tile count?
  3. Counterfactual: if we forbid VOLUNTARY draws in the late game (substitute the
     student's own best play), does win rate improve?
"""
import argparse
from collections import defaultdict
import numpy as np
import torch
from ppo_env import RummikubPPOEnv, STATE_DIM
from ppo_model import DistillStudent

NCAP = 20


def load_student(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m = DistillStudent(obs_dim=STATE_DIM, cand_feat_dim=52 + 52, max_candidates=20)
    m.load_state_dict(sd); m.eval()
    return m


def scores(model, obs):
    st = torch.tensor(obs["state"], dtype=torch.float32).unsqueeze(0)
    cd = torch.tensor(obs["cand_feats"], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        lg = model.forward_actor(st, cd).squeeze(0).numpy()
    lg[obs["mask"] == 0] = -1e9
    return lg


def tiles_of(obs, i):
    if i >= NCAP:
        return 0
    return max(0, round((obs["state"][:52].sum() - obs["cand_feats"][i, :52].sum()) * 2.0))


def deck_bucket(d):
    if d > 40:  return "deck>40"
    if d > 25:  return "deck26-40"
    if d > 12:  return "deck13-25"
    return "deck<=12"


def play(model, seed, seat, args, no_late_draw=False):
    env = RummikubPPOEnv(max_candidates=20, max_turns=100, seed=seed,
                         ppo_player=seat, opponent="ilp",
                         initial_meld_value=args.initial_meld_value)
    obs, _ = env.reset(seed=seed)
    done = False; info = {}; rows = []
    while not done:
        mask = obs["mask"]
        deck = len(env.env.deck)
        has_play = mask[:NCAP].sum() > 0
        if has_play:
            lg = scores(model, obs)
            a = int(lg.argmax())
            valid = np.flatnonzero(mask[:NCAP] > 0)
            maxtiles = max(tiles_of(obs, i) for i in valid)
            # counterfactual: forbid a voluntary draw late — use the best real play
            if no_late_draw and a == NCAP and deck <= args.late_deck:
                a = int(lg[:NCAP].argmax())
            chosen = tiles_of(obs, a)
            rows.append((deck, int(maxtiles), int(chosen), a == NCAP))
        else:
            a = NCAP
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
    oc = info.get("outcome", "timeout")
    return oc, len(env.hands[seat]), len(env.hands[env.ilp_player]), len(env.env.deck), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distill_s1s_dagger1.pt")
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--initial-meld-value", type=int, default=30)
    ap.add_argument("--late-deck", type=int, default=30,
                    help="deck<=this counts as 'late' for the no-draw intervention")
    args = ap.parse_args()
    model = load_student(args.model)

    base_w = force_w = ngames = 0
    losses = deck_exhausted = loss_by_count = 0
    drew_big = defaultdict(lambda: [0, 0])   # bucket -> [voluntary-draw-while-play>=3, total]
    left = defaultdict(list)                  # bucket -> maxtiles-chosen
    drew_big_by_oc = defaultdict(lambda: [0, 0])  # outcome -> ...

    for p in range(args.pairs):
        for seat in (0, 1):
            seed = args.seed + p
            oc, ml, ol, dend, rows = play(model, seed, seat, args)
            fo, *_ = play(model, seed, seat, args, no_late_draw=True)
            ngames += 1
            base_w += oc == "win"; force_w += fo == "win"
            if oc == "loss":
                losses += 1
                deck_exhausted += dend == 0
                loss_by_count += ml > ol
            for deck, mx, ch, isdraw in rows:
                b = deck_bucket(deck)
                drew_big[b][1] += 1
                left[b].append(mx - ch)
                if isdraw and mx >= 3:
                    drew_big[b][0] += 1
                o = "won" if oc == "win" else "lost"
                drew_big_by_oc[o][1] += 1
                drew_big_by_oc[o][0] += isdraw and mx >= 3

    print(f"model {args.model}  ({args.pairs} pairs, {ngames} games, meld={args.initial_meld_value})\n")
    print("=== win rate ===")
    print(f"  student (baseline)                         : {base_w/ngames:.1%}")
    print(f"  + forbid voluntary draw when deck<={args.late_deck}       : {force_w/ngames:.1%}")
    print(f"  delta                                      : {(force_w-base_w)/ngames:+.1%}\n")
    print(f"=== loss modes (of {losses} losses) ===")
    if losses:
        print(f"  ended with deck exhausted (0 left)         : {deck_exhausted}/{losses} ({deck_exhausted/losses:.0%})")
        print(f"  lost holding MORE tiles than opponent      : {loss_by_count}/{losses} ({loss_by_count/losses:.0%})\n")
    print("=== 'drew big' = voluntarily drew while a >=3-tile play was available, by deck ===")
    for b in ("deck>40", "deck26-40", "deck13-25", "deck<=12"):
        c, n = drew_big[b]
        if n: print(f"  {b:10s}: {c/n:5.1%}  ({c}/{n})   avg tiles left unplayed: {np.mean(left[b]):.2f}")
    print("\n=== 'drew big' rate, won vs lost games ===")
    for o in ("won", "lost"):
        c, n = drew_big_by_oc[o]
        if n: print(f"  {o:4s}: {c/n:.1%}  ({c}/{n})")


if __name__ == "__main__":
    main()
