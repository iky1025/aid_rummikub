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
            n_actions = z["mask"].shape[1]  # max_candidates + 1, from the data
            for k in z.files:
                parts.setdefault(k, []).append(z[k])
            for k in optional:
                if k not in z.files:
                    parts.setdefault(k, []).append(
                        np.full((n, n_actions), np.nan, dtype=np.float32))
    data = {k: np.concatenate(v) for k, v in parts.items()}
    n = len(data["action"])
    print(f"loaded {n} decisions from {len(files)} pairs "
          f"({data_dir})", flush=True)
    return data


def dedupe_candidates(data):
    """Audit Finding 1: ~45% of candidate slots are byte-identical duplicates
    (arrangement-only variants — the count encoding drops the partition). Keep
    one candidate per identical cand_feats group, remap the teacher's action to
    the kept index, and compact mask/scores/votes. The teacher's action is the
    lowest index of its group (label convention), which is the one kept, so the
    remap is well defined. The draw action (index == max_candidates) is left in
    place. Soft-CE optimum is unchanged; this removes the per-pool tie floor
    from val_ce and the crowding of real moves in downstream code."""
    N, W, F = data["cand_feats"].shape
    draw = W                                   # draw action index (mask width-1)
    cf, mask, act, nc = (data["cand_feats"], data["mask"], data["action"],
                         data["n_candidates"])
    has_ev = "cand_scores" in data
    new_cf = np.zeros_like(cf)
    new_mask = np.zeros_like(mask)
    new_act = act.copy()
    new_nc = nc.copy()
    if has_ev:
        new_sc = np.full_like(data["cand_scores"], np.nan)
        new_vt = np.full_like(data["cand_votes"], np.nan)
    removed = 0
    for r in range(N):
        n = int(nc[r])
        seen, keep = {}, 0
        for i in range(n):
            key = cf[r, i].tobytes()
            j = seen.get(key)
            if j is None:
                seen[key] = keep
                new_cf[r, keep] = cf[r, i]
                new_mask[r, keep] = 1.0
                if has_ev:
                    new_sc[r, keep] = data["cand_scores"][r, i]
                    new_vt[r, keep] = data["cand_votes"][r, i]
                if int(act[r]) == i:
                    new_act[r] = keep
                keep += 1
            else:
                removed += 1
                if int(act[r]) == i:      # shouldn't happen (convention), but be safe
                    new_act[r] = j
        new_mask[r, draw] = 1.0
        if has_ev:
            new_sc[r, draw] = data["cand_scores"][r, draw]
            new_vt[r, draw] = data["cand_votes"][r, draw]
        if int(act[r]) >= n:              # draw
            new_act[r] = draw
        new_nc[r] = keep
    data["cand_feats"], data["mask"], data["action"], data["n_candidates"] = (
        new_cf, new_mask, new_act, new_nc)
    if has_ev:
        data["cand_scores"], data["cand_votes"] = new_sc, new_vt
    tot = int(nc[:N].sum())
    print(f"deduped candidates: removed {removed}/{tot} slots "
          f"({100 * removed / max(tot, 1):.1f}%)", flush=True)
    return data


def check_label_convention(data):
    """Guard the load-bearing (and *accidental*) label-index convention.

    Arrangement-only variants of the same move are byte-identical in
    cand_feats (tiles_to_vector drops the partition), so ~45% of candidate
    slots are exact duplicates. Everything downstream relies on:

      the teacher's action is always the LOWEST index of its duplicate group

    which holds because rollout_agent dedupes by remaining_hand keeping the
    first occurrence (`chosen`), and candidate 0 is its own group's first.
    torch.argmax happens to break ties the same way (lowest index), which is
    why val_acc / `action != 0` deviation stats are meaningful at all.

    This alignment is a coincidence of two independent implementations. If
    candidate ordering ever changes (e.g. enumerate_moves sorts by
    -used_hand_tile_count), it breaks *silently*. Fail loudly instead.
    """
    cf, act, nc = data["cand_feats"], data["action"], data["n_candidates"]
    viol = 0
    for r in range(len(act)):
        a, n = int(act[r]), int(nc[r])
        if n == 0 or a >= n:
            continue  # forced turn, or draw (action == max_candidates)
        if a and (cf[r, :a] == cf[r, a]).all(axis=1).any():
            viol += 1
    if viol:
        raise SystemExit(
            f"label convention broken in {viol}/{len(act)} rows: the teacher's "
            f"action is not the first index of its duplicate group. Candidate "
            f"ordering changed — val_acc and `action != 0` deviation stats are "
            f"no longer meaningful. See check_label_convention().")
    return viol


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
    if "state_ext" in data:
        state = torch.tensor(data["state_ext"][idx], dtype=torch.float32,
                             device=device)
    cand = torch.tensor(data["cand_feats"][idx], dtype=torch.float32,
                        device=device) / 2.0
    mask = torch.tensor(data["mask"][idx], dtype=torch.float32, device=device)
    action = torch.tensor(data["action"][idx], dtype=torch.long, device=device)
    opp = torch.tensor(data["opp_hand"][idx], dtype=torch.float32,
                       device=device) / 2.0
    return state, cand, mask, action, opp


SCORE_SCALE = 50.0  # rollout WIN_SCORE; brings scores to roughly [-1.5, 1.5]

EVENT_FEAT_DIM = 6 * 5  # EVENT_HISTORY_LEN x (drew, before, after, pre_meld, valid)


def event_feats(events):
    """(N, 6, 4) int16 opponent-event history -> (N, 30) float features.

    Columns per row: drew (0/1), hand_before/14, hand_after/14, pre_meld (0/1),
    valid (0/1). Fixes audit Finding 5: rows are RIGHT-ALIGNED (the most recent
    turn is always the last slot, so a given slot means the same thing every
    turn) and carry an explicit validity bit (a padded row is all-zero AND
    valid=0, so drew=0 padding is not confused with a real no-draw turn). The
    teacher's consistent determinization reads exactly this history — without it
    part of the teacher's deviations is unlearnable from the obs (imitation gap).
    Input is left-aligned (valid rows first, -1 padding at the end)."""
    f = events.astype(np.float32)
    valid = (f[..., 0] >= 0).astype(np.float32)          # (N, 6)
    f[..., 1] /= 14.0
    f[..., 2] /= 14.0
    f[valid == 0] = 0.0
    f = np.concatenate([f, valid[..., None]], axis=-1)   # (N, 6, 5)
    out = np.zeros_like(f)
    for r in range(len(f)):                              # right-align valid rows
        k = int(valid[r].sum())
        if k:
            out[r, -k:] = f[r, :k]
    return out.reshape(len(f), -1)


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
    parser.add_argument("--history", action="store_true",
                        help="append opponent-event history features (30d, "
                             "right-aligned + validity bit) to the state input")
    parser.add_argument("--dedupe", action="store_true",
                        help="audit Finding 1: drop byte-identical duplicate "
                             "candidates at load (remaps action). Off by default "
                             "to keep baselines comparable; A/B it explicitly.")
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for torch weight init (data split / batch "
                             "order stay fixed) — lets multi-seed runs average "
                             "out training-instance noise")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cpu")

    data = load_dataset(args.data)
    check_label_convention(data)
    if args.dedupe:
        dedupe_candidates(data)
    # Unify the two teacher-evaluation channels into one score matrix:
    # rollout scores as-is; endgame win-forcing vote fractions scaled to the
    # same units (fraction x WIN_SCORE ~ forced-win expected value).
    merged = np.where(np.isnan(data["cand_scores"]),
                      data["cand_votes"] * SCORE_SCALE,
                      data["cand_scores"]).astype(np.float32)
    data["scores_merged"] = merged
    # Infer dims from the data so jokered shards (state 110 / cand 106) and
    # jokerless (108 / 104) both work without a flag.
    obs_dim = data["state"].shape[1]
    cand_feat_dim = data["cand_feats"].shape[2]
    if args.history:
        data["state_ext"] = np.concatenate(
            [data["state"], event_feats(data["events"])], axis=1)
        obs_dim = data["state_ext"].shape[1]
    train_mask, val_mask = make_split(data, args.val_frac)
    train_idx = np.flatnonzero(train_mask)
    dev_frac = float((data["action"][train_idx] != 0).mean())
    print(f"train {len(train_idx)} / val {int(val_mask.sum())} decisions, "
          f"deviation fraction {dev_frac:.3f}", flush=True)

    model = DistillStudent(
        obs_dim=obs_dim,
        cand_feat_dim=cand_feat_dim,
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
