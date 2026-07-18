"""Student vs student self-play: does the deck fully exhaust / do games hit the
turn cap when BOTH players are the (conservative) distilled network?

Also a symmetry sanity check: same model on both sides over mirror pairs should
win ~50% — confirms the opponent="student" perspective-swap is correct.
"""
import argparse
import numpy as np
import torch
from ppo_env import RummikubPPOEnv, STATE_DIM
from ppo_model import DistillStudent

NCAP = 20


def make_policy(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m = DistillStudent(obs_dim=STATE_DIM, cand_feat_dim=104, max_candidates=20)
    m.load_state_dict(sd); m.eval()

    def policy(obs):
        st = torch.tensor(obs["state"], dtype=torch.float32).unsqueeze(0)
        cd = torch.tensor(obs["cand_feats"], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            lg = m.forward_actor(st, cd).squeeze(0).numpy()
        lg[obs["mask"] == 0] = -1e9
        return int(lg.argmax())
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distill_s1s_dagger1.pt")
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--initial-meld-value", type=int, default=30)
    args = ap.parse_args()
    policy = make_policy(args.model)

    outcomes = {"win": 0, "loss": 0, "timeout": 0}
    hit0 = 0; steps_all = []; mins = []
    for p in range(args.pairs):
        for seat in (0, 1):
            seed = args.seed + p
            env = RummikubPPOEnv(max_candidates=20, max_turns=100, seed=seed,
                                 ppo_player=seat, opponent="student",
                                 opponent_policy=policy,
                                 initial_meld_value=args.initial_meld_value)
            obs, _ = env.reset(seed=seed)
            done = False; info = {}; mn = 104; steps = 0
            while not done:
                mn = min(mn, len(env.env.deck))
                a = policy(obs) if obs["mask"][:NCAP].sum() > 0 else NCAP
                obs, _, term, trunc, info = env.step(a)
                done = term or trunc; steps += 1
            mn = min(mn, len(env.env.deck))
            outcomes[info.get("outcome", "timeout")] += 1
            hit0 += mn == 0; mins.append(mn); steps_all.append(steps)

    ng = sum(outcomes.values()); mins = np.array(mins)
    print(f"student vs student self-play  ({args.pairs} pairs, {ng} games, meld={args.initial_meld_value})\n")
    print(f"outcomes (main seat): win {outcomes['win']}  loss {outcomes['loss']}  timeout {outcomes['timeout']}")
    print(f"win rate (should be ~50% by symmetry): {outcomes['win']/ng:.1%}")
    print(f"\n=== deck / length ===")
    print(f"deck fully exhausted (reached 0): {hit0}/{ng} ({hit0/ng:.0%})")
    print(f"games hit 100-turn cap (timeout): {outcomes['timeout']}/{ng} ({outcomes['timeout']/ng:.0%})")
    print(f"min-deck reached  median={int(np.median(mins))}  min={mins.min()}  p10={int(np.percentile(mins,10))}  <=12: {(mins<=12).sum()}")
    print(f"avg game length (agent turns): {np.mean(steps_all):.1f}")


if __name__ == "__main__":
    main()
