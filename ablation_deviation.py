"""Ablation instrumentation: measure how ACTIVE the network is.

Replays the mirror-eval games with the distilled student and records, per
decision that has >=1 candidate, whether the network picks candidate 0 (the
greedy max-play, index 0) or deviates. Splits deviation rate by game outcome
so we can see whether deviating correlates with winning.

This isolates the network's *selection* from the solver's *candidate set*:
- deviation rate ~0  -> network is a greedy copier (solver does everything)
- deviation rate high -> network actively overrides greedy; combined with the
  70.3% win rate and the random-selection control, the gain is the network's.
"""
import argparse
import numpy as np
import torch
from ppo_env import RummikubPPOEnv
from ppo_model import DistillStudent


def load_student(model_path):
    from ppo_env import STATE_DIM
    sd = torch.load(model_path, map_location="cpu", weights_only=True)
    model = DistillStudent(obs_dim=STATE_DIM, cand_feat_dim=52 + 52, max_candidates=20)
    model.load_state_dict(sd)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distill_s1s_dagger1.pt")
    ap.add_argument("--pairs", type=int, default=160)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--initial-meld-value", type=int, default=30)
    ap.add_argument("--max-candidates", type=int, default=20)
    ap.add_argument("--max-turns", type=int, default=100)
    args = ap.parse_args()

    model = load_student(args.model)

    dec = 0            # decisions with >=1 candidate
    deviate = 0        # network chose action != 0
    draw_chosen = 0    # network chose to draw despite having plays
    per_game = []      # (deviations, decisions, outcome)

    for p in range(args.pairs):
        seed = args.seed + p
        for seat in (0, 1):
            env = RummikubPPOEnv(
                max_candidates=args.max_candidates, max_turns=args.max_turns,
                seed=seed, ppo_player=seat, opponent="ilp",
                initial_meld_value=args.initial_meld_value)
            obs, _ = env.reset(seed=seed)
            done = False
            info = {}
            g_dev, g_dec = 0, 0
            while not done:
                mask = obs["mask"]
                if mask[:args.max_candidates].sum() > 0:  # at least one real play available
                    st = torch.tensor(obs["state"], dtype=torch.float32).unsqueeze(0)
                    cd = torch.tensor(obs["cand_feats"], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        logits = model.forward_actor(st, cd).squeeze(0)
                    logits[torch.tensor(mask) == 0] = -1e9
                    action = int(torch.argmax(logits).item())
                    dec += 1
                    g_dec += 1
                    if action != 0:
                        deviate += 1
                        g_dev += 1
                    if action == args.max_candidates:  # draw index
                        draw_chosen += 1
                else:
                    action = args.max_candidates  # forced draw
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
            outcome = info.get("outcome", "timeout")
            per_game.append((g_dev, g_dec, outcome))

    print(f"model: {args.model}  pairs: {args.pairs} ({args.pairs*2} games)")
    print(f"decisions with >=1 play available : {dec}")
    print(f"network deviated from greedy (act!=0): {deviate}  ({deviate/dec:.1%})")
    print(f"  of which chose to DRAW over playing : {draw_chosen} ({draw_chosen/dec:.1%})")

    wins = [g for g in per_game if g[2] == "win"]
    losses = [g for g in per_game if g[2] == "loss"]
    def dev_rate(games):
        d = sum(g[0] for g in games); n = sum(g[1] for g in games)
        return d / n if n else float("nan"), n
    wr, wn = dev_rate(wins)
    lr, ln = dev_rate(losses)
    print(f"deviation rate in WON games  : {wr:.1%} (n={wn} decisions, {len(wins)} games)")
    print(f"deviation rate in LOST games : {lr:.1%} (n={ln} decisions, {len(losses)} games)")


if __name__ == "__main__":
    main()
