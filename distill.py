"""R10: supervised distillation from self-play decision datasets.

Trains DistillStudent (ppo_model.py) on records produced by selfplay_data.py:
cross-entropy on the teacher's action over masked logits, plus an optional
auxiliary MSE loss predicting the opponent's hidden hand, plus optional
up-weighting of decisions where the teacher deviated from greedy (action != 0).

Usage:
  python distill.py --data data/stage0 --tag stage0 --epochs 10
  python distill.py --data data/stage1 --tag stage1 --dev-weight 4 --aux-coef 0.5
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ppo_env import STATE_DIM, CAND_FEAT_DIM
from ppo_model import DistillStudent


def load_dataset(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "pair_*.npz")))
    if not files:
        raise SystemExit(f"no pair_*.npz files in {data_dir}")
    parts = {}
    for f in files:
        with np.load(f) as z:
            for k in z.files:
                parts.setdefault(k, []).append(z[k])
    data = {k: np.concatenate(v) for k, v in parts.items()}
    n = len(data["action"])
    print(f"loaded {n} decisions from {len(files)} pairs "
          f"({data_dir})", flush=True)
    return data


def make_split(data, val_frac=0.1):
    """Split by pair seed so both seats/all decisions of a game stay together."""
    seeds = data["seed"]
    uniq = np.unique(seeds)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac))
    val_seeds = set(uniq[:n_val].tolist())
    val_mask = np.isin(seeds, list(val_seeds))
    return ~val_mask, val_mask


def to_tensors(data, idx, device):
    state = torch.tensor(data["state"][idx], dtype=torch.float32, device=device)
    cand = torch.tensor(data["cand_feats"][idx], dtype=torch.float32,
                        device=device) / 2.0
    mask = torch.tensor(data["mask"][idx], dtype=torch.float32, device=device)
    action = torch.tensor(data["action"][idx], dtype=torch.long, device=device)
    opp = torch.tensor(data["opp_hand"][idx], dtype=torch.float32,
                       device=device) / 2.0
    return state, cand, mask, action, opp


def evaluate(model, data, mask_idx, device, batch_size=2048):
    idxs = np.flatnonzero(mask_idx)
    ce_sum, correct, dev_total, dev_correct, n = 0.0, 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for s in range(0, len(idxs), batch_size):
            batch = idxs[s:s + batch_size]
            state, cand, mask, action, _ = to_tensors(data, batch, device)
            logits = model.forward_actor(state, cand)
            logits = logits.masked_fill(mask == 0, -1e9)
            ce_sum += F.cross_entropy(logits, action, reduction="sum").item()
            pred = logits.argmax(dim=1)
            correct += (pred == action).sum().item()
            dev = action != 0
            dev_total += dev.sum().item()
            dev_correct += (dev & (pred == action)).sum().item()
            n += len(batch)
    dev_acc = dev_correct / dev_total if dev_total else float("nan")
    return ce_sum / n, correct / n, dev_acc, dev_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dev-weight", type=float, default=0.0,
                        help="extra CE weight for decisions where the teacher "
                             "deviated from greedy (w = 1 + dev_weight)")
    parser.add_argument("--aux-coef", type=float, default=0.0,
                        help="weight of the opponent-hand MSE auxiliary loss")
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cpu")

    data = load_dataset(args.data)
    train_mask, val_mask = make_split(data, args.val_frac)
    train_idx = np.flatnonzero(train_mask)
    dev_frac = float((data["action"][train_idx] != 0).mean())
    print(f"train {len(train_idx)} / val {int(val_mask.sum())} decisions, "
          f"deviation fraction {dev_frac:.3f}", flush=True)

    model = DistillStudent(
        obs_dim=STATE_DIM,
        cand_feat_dim=CAND_FEAT_DIM,
        max_candidates=args.max_candidates,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    rng = np.random.default_rng(0)
    t0 = time.time()
    best_val = float("inf")
    out_path = f"distill_{args.tag}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_idx.copy()
        rng.shuffle(order)
        ce_sum, n_seen = 0.0, 0
        for s in range(0, len(order), args.batch_size):
            batch = order[s:s + args.batch_size]
            state, cand, mask, action, opp = to_tensors(data, batch, device)
            logits = model.forward_actor(state, cand)
            logits = logits.masked_fill(mask == 0, -1e9)
            ce = F.cross_entropy(logits, action, reduction="none")
            if args.dev_weight > 0:
                w = 1.0 + args.dev_weight * (action != 0).float()
                loss = (ce * w).sum() / w.sum()
            else:
                loss = ce.mean()
            if args.aux_coef > 0:
                aux = F.mse_loss(model.forward_aux(state), opp)
                loss = loss + args.aux_coef * aux
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ce_sum += ce.sum().item()
            n_seen += len(batch)

        val_ce, val_acc, val_dev_acc, n_dev = evaluate(
            model, data, val_mask, device)
        marker = ""
        if val_ce < best_val:
            best_val = val_ce
            torch.save(model.state_dict(), out_path)
            marker = " *saved"
        print(f"epoch {epoch:2d}  train_ce={ce_sum / n_seen:.4f}  "
              f"val_ce={val_ce:.4f}  val_acc={val_acc:.3f}  "
              f"val_dev_acc={val_dev_acc:.3f} (n={n_dev})  "
              f"{time.time() - t0:.0f}s{marker}", flush=True)

    print(f"best val_ce={best_val:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
