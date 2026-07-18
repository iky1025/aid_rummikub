"""Diagnostic: how well does a student's opp_hand_head predict the opponent's
hidden hand? Reports val MSE, a mean-predictor baseline, and the fraction of
variance explained (1 - mse/baseline). ~0 means the head learned nothing
(e.g. aux-coef was 0); >0 means it recovers real signal about the hidden hand.
"""
import argparse, glob
import numpy as np
import torch
from ppo_model import DistillStudent
from ppo_env import STATE_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.data}/pair_*.npz"))
    S, O = [], []
    for f in files:
        with np.load(f) as z:
            S.append(z["state"].astype(np.float32))
            O.append(z["opp_hand"].astype(np.float32))
    state = np.concatenate(S); opp = np.concatenate(O)
    n = len(state); nv = int(n * args.val_frac)
    rng = np.random.default_rng(0); idx = rng.permutation(n)
    val = idx[:nv]
    xs = torch.tensor(state[val]); yt = opp[val]

    sd = torch.load(args.model, map_location="cpu", weights_only=True)
    model = DistillStudent(obs_dim=STATE_DIM, cand_feat_dim=52 + 52, max_candidates=20)
    model.load_state_dict(sd); model.eval()
    with torch.no_grad():
        pred = model.forward_aux(xs).numpy()

    mse = float(np.mean((pred - yt) ** 2))
    base_mse = float(np.mean((yt.mean(0, keepdims=True) - yt) ** 2))
    var_exp = 1.0 - mse / base_mse if base_mse > 0 else float("nan")
    # tiles the opponent actually holds (nonzero) vs predicted mass on them
    held = yt > 0
    print(f"model {args.model}  (val {nv} decisions)")
    print(f"  aux MSE            : {mse:.4f}")
    print(f"  mean-baseline MSE  : {base_mse:.4f}")
    print(f"HAND_VAR_EXPLAINED: {var_exp:.3f}  (0=learned nothing, higher=better)")


if __name__ == "__main__":
    main()
