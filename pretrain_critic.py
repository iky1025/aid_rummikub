"""R12 Phase 1: wake up the critic before any RL fine-tune.

Distillation never calls ActorCritic.critic -- `--value-coef` regresses the
ACTOR logits onto teacher scores, so the critic head leaves training at its
random init (verified: weight norm 0.564 vs 0.559 fresh). Starting PPO from
that means the first advantages are pure noise, which is the fastest way to
destroy a 67.8% policy.

This fits ONLY the critic head (129 params) on a frozen state encoder, so the
policy is bit-for-bit unchanged: the actor path never sees a gradient.

Target = gamma^(steps remaining in the game) * outcome, i.e. exactly the
discounted return of the sparse reward (+1 win / -1 loss / 0 tie) that
ppo_env's reward_mode="sparse" pays.

Labels come from dagger* shards ONLY. `outcome` is the result of whoever was
PLAYING, so on teacher shards it is the teacher's value function, not the
student's -- fitting on those would warm-start the critic to the wrong policy.

Usage:
  python pretrain_critic.py --data data/dagger1 --model distill_s1s_dagger1.pt \
      --out distill_s1s_dagger1_critic.pt
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn

from ppo_model import ActorCritic


def load_shards(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "pair_*.npz")))
    if not files:
        raise SystemExit(f"no pair_*.npz in {data_dir}")
    return files


def discounted_targets(outcome, seat, gamma):
    """gamma^(remaining steps) * outcome, per game within the shard.

    A shard holds both games of a mirror pair; `seat` separates them and the
    rows of one game are already in play order, so "remaining" is just the
    count of later rows with the same seat.
    """
    tgt = np.zeros(len(outcome), dtype=np.float32)
    for s in np.unique(seat):
        idx = np.flatnonzero(seat == s)
        n = len(idx)
        # idx[-1] is the last decision of that game -> 0 steps remaining
        k = np.arange(n - 1, -1, -1, dtype=np.float32)
        tgt[idx] = (gamma ** k) * outcome[idx].astype(np.float32)
    return tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dagger1",
                    help="dagger shard dir (outcome = the ACTING policy's result)")
    ap.add_argument("--model", default="distill_s1s_dagger1.pt")
    ap.add_argument("--out", default="distill_s1s_dagger1_critic.pt")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    files = load_shards(args.data)
    # split by SHARD, not by row: rows inside one game are highly correlated,
    # so a row-level split leaks the game's outcome into validation.
    perm = rng.permutation(len(files))
    n_val = max(1, int(len(files) * args.val_frac))
    val_files = {files[i] for i in perm[:n_val]}

    def gather(sel):
        S, T = [], []
        for f in files:
            if (f in val_files) != sel:
                continue
            with np.load(f) as z:
                S.append(z["state"])
                T.append(discounted_targets(z["outcome"], z["seat"], args.gamma))
        return (torch.tensor(np.concatenate(S), dtype=torch.float32),
                torch.tensor(np.concatenate(T), dtype=torch.float32))

    Xtr, Ytr = gather(False)
    Xva, Yva = gather(True)
    print(f"shards {len(files)}  (val {len(val_files)})")
    print(f"train {len(Xtr)} decisions   val {len(Xva)}")
    print(f"target: mean {Ytr.mean():+.4f}  std {Ytr.std():.4f}")

    sd = torch.load(args.model, map_location="cpu", weights_only=True)
    obs_dim = sd["state_encoder.0.weight"].shape[1]
    cand_dim = sd["cand_encoder.0.weight"].shape[1]
    model = ActorCritic(obs_dim, cand_dim, 20)
    res = model.load_state_dict(sd, strict=False)
    if res.missing_keys:
        raise SystemExit(f"checkpoint missing {res.missing_keys}")

    # freeze everything but the critic -> the actor is provably untouched
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.critic.parameters():
        p.requires_grad_(True)
    print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    with torch.no_grad():
        Htr = model.state_encoder(Xtr)      # frozen encoder -> precompute once
        Hva = model.state_encoder(Xva)

    opt = torch.optim.Adam(model.critic.parameters(), lr=args.lr)
    lossf = nn.MSELoss()
    baseline = lossf(torch.full_like(Yva, Ytr.mean()), Yva).item()
    print(f"baseline val MSE (predict train mean): {baseline:.4f}\n")

    best, best_state = float("inf"), None
    n = len(Htr)
    for ep in range(1, args.epochs + 1):
        model.critic.train()
        idx = torch.randperm(n)
        for i in range(0, n, 4096):
            b = idx[i:i + 4096]
            opt.zero_grad()
            loss = lossf(model.critic(Htr[b]).squeeze(-1), Ytr[b])
            loss.backward()
            opt.step()
        model.critic.eval()
        with torch.no_grad():
            v = model.critic(Hva).squeeze(-1)
            vl = lossf(v, Yva).item()
            corr = np.corrcoef(v.numpy(), Yva.numpy())[0, 1]
        if vl < best:
            best, best_state = vl, {k: t.clone() for k, t in model.critic.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  val MSE {vl:.4f}  corr {corr:+.3f}")

    model.critic.load_state_dict(best_state)
    with torch.no_grad():
        v = model.critic(Hva).squeeze(-1)
        corr = np.corrcoef(v.numpy(), Yva.numpy())[0, 1]
    print(f"\nbest val MSE {best:.4f}  vs baseline {baseline:.4f} "
          f"({100 * (1 - best / baseline):+.1f}% explained)")
    print(f"corr(V, return) = {corr:+.3f}")
    print(f"critic weight norm {model.critic.weight.norm():.4f} (was 0.564 = random init)")

    out = model.state_dict()
    torch.save(out, args.out)
    print(f"\nsaved -> {args.out}")

    # prove the actor is untouched
    ref = ActorCritic(obs_dim, cand_dim, 20)
    ref.load_state_dict(sd, strict=False)
    same = all(torch.equal(out[k], ref.state_dict()[k])
               for k in out if not k.startswith("critic."))
    print(f"actor tensors identical to the distilled model: {same}")


if __name__ == "__main__":
    main()
