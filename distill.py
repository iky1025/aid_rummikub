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
    # Older datasets predate the teacher-evaluation fields — pad with NaN so
    # old and new shards can be mixed in one run.
    optional = ("cand_scores", "cand_votes")
    parts = {}
    for f in files:
        with np.load(f) as z:
            n = len(z["action"])
            for k in z.files:
                parts.setdefault(k, []).append(z[k])
            for k in optional:
                if k not in z.files:
                    parts.setdefault(k, []).append(
                        np.full((n, 21), np.nan, dtype=np.float32))
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


SCORE_SCALE = 50.0  # rollout WIN_SCORE; brings scores to roughly [-1.5, 1.5]


def imitation_loss(logits, mask, action, scores, args):
    """Per-row loss combining three signals (R10, post-literature-review):

    - soft CE on the teacher's per-candidate rollout scores (softmax at
      --soft-temp) where the teacher evaluated >=2 options — carries the
      *margin* of a deviation instead of just its argmax;
    - one-hot CE (with --dev-weight) on rows without evaluations;
    - masked value regression (--value-coef) tying all evaluated options,
      including draw vs play, to one calibrated scale.
    """
    masked = logits.masked_fill(mask == 0, -1e9)
    labeled = ~torch.isnan(scores) & (mask > 0)
    n_labeled = labeled.sum(dim=1)

    ce = F.cross_entropy(masked, action, reduction="none")
    if args.dev_weight > 0:
        w = 1.0 + args.dev_weight * (action != 0).float()
        onehot_loss = ce * w / w.mean()
    else:
        onehot_loss = ce

    if args.soft_temp > 0:
        target_logits = (scores.nan_to_num(-1e9) / SCORE_SCALE
                         ) / args.soft_temp
        target_logits = target_logits.masked_fill(~labeled, -1e9)
        target = torch.softmax(target_logits, dim=1)
        soft_loss = -(target * F.log_softmax(masked, dim=1)).sum(dim=1)
        use_soft = (n_labeled >= 2).float()
        row_loss = use_soft * soft_loss + (1 - use_soft) * onehot_loss
    else:
        row_loss = onehot_loss

    loss = row_loss.mean()
    if args.value_coef > 0 and labeled.any():
        err = (logits - scores.nan_to_num(0.0) / SCORE_SCALE) ** 2
        loss = loss + args.value_coef * err[labeled].mean()
    return loss, ce


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
    parser.add_argument("--soft-temp", type=float, default=0.0,
                        help=">0: soft CE against softmax(teacher rollout "
                             "scores / temp) on rows with >=2 evaluations")
    parser.add_argument("--value-coef", type=float, default=0.0,
                        help=">0: masked MSE regression of logits to teacher "
                             "scores / 50 on evaluated entries")
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
    # Unify the two teacher-evaluation channels into one score matrix:
    # rollout scores as-is; endgame win-forcing vote fractions scaled to the
    # same units (fraction x WIN_SCORE ~ forced-win expected value).
    merged = np.where(np.isnan(data["cand_scores"]),
                      data["cand_votes"] * SCORE_SCALE,
                      data["cand_scores"]).astype(np.float32)
    data["scores_merged"] = merged
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
            scores = torch.tensor(data["scores_merged"][batch],
                                  dtype=torch.float32, device=device)
            logits = model.forward_actor(state, cand)
            loss, ce = imitation_loss(logits, mask, action, scores, args)
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
