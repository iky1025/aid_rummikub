"""R8: Mirror-pair (duplicate) evaluation.

Each pair plays the SAME deal (same seed => same deck order) twice, with the
agent in seat 0 then seat 1, always vs the greedy-ILP opponent. Summing the
two games of a pair cancels most deal luck, so far fewer games are needed for
a significant comparison.

Sanity check: --policy greedy must come out exactly 50% (each pair is the
same deterministic game with roles swapped).

Usage:
  python eval_mirror.py --policy greedy  --pairs 100
  python eval_mirror.py --policy rollout --pairs 30 --determinizations 8
  python eval_mirror.py --policy model   --model rummikub_ppo_best_r7.pt
"""

import argparse
import math
import time

import numpy as np

from ppo_env import RummikubPPOEnv


def make_policy(args):
    """Return fn(env, obs) -> action."""
    max_candidates = args.max_candidates

    if args.policy == "greedy":
        def policy(env, obs):
            if obs["mask"][:max_candidates].sum() > 0:
                return 0  # candidate 0 = max-tiles best play (same as opponent)
            return max_candidates
        return policy

    if args.policy == "random":
        rng = np.random.default_rng(args.seed)
        def policy(env, obs):
            valid = np.flatnonzero(obs["mask"] > 0)
            return int(rng.choice(valid))
        return policy

    if args.policy == "rollout":
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
        def policy(env, obs):
            return rollout.select_action(env)
        return policy

    if args.policy == "model":
        import torch
        from ppo_model import ActorCritic

        device = torch.device("cpu")
        state_dict = torch.load(args.model, map_location=device, weights_only=True)
        model = ActorCritic(
            obs_dim=obs_dim_from_env(),
            cand_feat_dim=52 + 52,
            max_candidates=max_candidates,
        )
        model.load_state_dict(state_dict)
        model.eval()

        def policy(env, obs):
            state_t = torch.tensor(obs["state"], dtype=torch.float32)
            cand_t = torch.tensor(obs["cand_feats"], dtype=torch.float32)
            mask_t = torch.tensor(obs["mask"], dtype=torch.float32)
            with torch.no_grad():
                logits = model.forward_actor(
                    state_t.unsqueeze(0), cand_t.unsqueeze(0)
                ).squeeze(0)
            logits[mask_t == 0] = -1e9
            return int(torch.argmax(logits).item())
        return policy

    raise ValueError(f"unknown policy: {args.policy}")


def obs_dim_from_env():
    from ppo_env import STATE_DIM
    return STATE_DIM


def play_game(policy, seed, ppo_player, args):
    env = RummikubPPOEnv(
        max_candidates=args.max_candidates,
        max_turns=args.max_turns,
        seed=seed,
        ppo_player=ppo_player,
        opponent="ilp",
        initial_meld_value=args.initial_meld_value,
        exhaustive_candidates=args.exhaustive,
    )
    obs, _ = env.reset(seed=seed)
    done = False
    info = {}
    steps = 0
    while not done:
        action = policy(env, obs)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    outcome = info.get("outcome", "timeout")
    net = int(info.get("win_margin", 0)) - int(info.get("loss_margin", 0))
    return outcome, net, steps


def eval_pair(policy, seed, args):
    """Play both seats of one mirrored pair. Returns [(outcome, net, steps)] * 2."""
    return [play_game(policy, seed, seat, args) for seat in (0, 1)]


# Per-worker policy cache for --workers > 1 (built once per process).
_WORKER_POLICY = None


def _pair_worker(payload):
    global _WORKER_POLICY
    args, seed = payload
    if _WORKER_POLICY is None:
        _WORKER_POLICY = make_policy(args)
    return eval_pair(_WORKER_POLICY, seed, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["greedy", "rollout", "model", "random"],
                        required=True)
    parser.add_argument("--model", default="rummikub_ppo_best_r7.pt")
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--initial-meld-value", type=int, default=0)
    parser.add_argument("--determinizations", type=int, default=8)
    parser.add_argument("--rollout-turns", type=int, default=12)
    parser.add_argument("--candidate-cap", type=int, default=6)
    parser.add_argument("--oracle", nargs="?", const="full",
                        choices=["full", "hand"], default=None,
                        help="rollout ceiling measurement (not a fair agent): "
                             "'full'=true hand+deck, 'hand'=true hand only, "
                             "deck order sampled")
    parser.add_argument("--endgame-search", action="store_true",
                        help="rollout: bounded win-forcing DFS per "
                             "determinization near the endgame (R9-3)")
    parser.add_argument("--search-nodes", type=int, default=200)
    parser.add_argument("--exhaustive", action="store_true",
                        help="use exhaustive move enumeration for the agent's "
                             "candidates (R9)")
    parser.add_argument("--consistent", action="store_true",
                        help="rollout: information-consistent determinization "
                             "(reject samples contradicting opponent draw history)")
    parser.add_argument("--greedy-margin", type=float, default=0.0,
                        help="rollout: stick with greedy unless another "
                             "candidate wins by more than this per-rollout margin")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes (pairs are independent)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    pair_seeds = [args.seed + p for p in range(args.pairs)]
    t0 = time.time()

    executor = None
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(max_workers=args.workers)
        games_iter = executor.map(_pair_worker, [(args, s) for s in pair_seeds])
    else:
        policy = make_policy(args)
        games_iter = (eval_pair(policy, seed, args) for seed in pair_seeds)

    outcomes = {"win": 0, "loss": 0, "timeout": 0}
    pair_nets = []
    pair_patterns = {}
    total_steps = 0

    for p, games in enumerate(games_iter):
        pair_net = 0
        pattern = ""
        for outcome, net, steps in games:
            outcomes[outcome] += 1
            pair_net += net
            total_steps += steps
            pattern += {"win": "W", "loss": "L", "timeout": "T"}[outcome]
        pair_nets.append(pair_net)
        pair_patterns[pattern] = pair_patterns.get(pattern, 0) + 1
        if args.verbose:
            print(f"pair {p + 1:3d}/{args.pairs} seed={pair_seeds[p]} "
                  f"pattern={pattern} pair_net={pair_net:+d} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    if executor is not None:
        executor.shutdown()

    n_games = 2 * args.pairs
    win_rate = outcomes["win"] / n_games
    mean_net = float(np.mean(pair_nets))
    stderr = float(np.std(pair_nets, ddof=1) / math.sqrt(len(pair_nets))) \
        if len(pair_nets) > 1 else 0.0
    elapsed = time.time() - t0

    print(f"\n=== mirror eval: {args.policy} vs greedy-ILP ===")
    print(f"pairs           : {args.pairs} ({n_games} games)")
    print(f"win/loss/timeout: {outcomes['win']}/{outcomes['loss']}/{outcomes['timeout']}")
    print(f"win_rate        : {win_rate:.1%}")
    print(f"pair patterns   : {dict(sorted(pair_patterns.items()))}")
    print(f"mean pair net   : {mean_net:+.2f} ± {stderr:.2f} (SE)  "
          f"[>0 means better than greedy]")
    print(f"avg steps/game  : {total_steps / n_games:.1f}")
    print(f"elapsed         : {elapsed:.1f}s ({elapsed / n_games:.2f}s/game)")


if __name__ == "__main__":
    main()
