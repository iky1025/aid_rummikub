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
  python eval_mirror.py --policy student --model distill_s1s_dagger1.pt
"""

import argparse
import math
import time

import numpy as np

from ppo_env import RummikubPPOEnv


def _wrap_endgame(base_policy, args):
    """R10: hybrid wrapper — delegate near-endgame decisions to the rollout
    machinery's win-forcing DFS. greedy_margin=1e9 pins the delegate's 1-ply
    fallback to greedy, so the delegate adds exactly the DFS and nothing else.
    Used for `--policy student --endgame-search` (level-2 hybrid agent) and
    `--policy greedy --endgame-search` (ablation A baseline: greedy + DFS)."""
    from rollout_agent import RolloutPolicy
    delegate = RolloutPolicy(
        n_determinizations=args.determinizations,
        max_rollout_turns=args.rollout_turns,
        candidate_cap=args.candidate_cap,
        greedy_margin=1e9,
        consistent=args.consistent,
        endgame_search=True,
        search_nodes=args.search_nodes,
        seed=args.seed,
    )
    trigger = delegate.search_hand_trigger
    tau = getattr(args, "endgame_tau", 0.0)

    def policy(env, obs):
        my = len(env.hands[env.ppo_player])
        opp = len(env.hands[env.ilp_player])
        if min(my, opp) > trigger:
            return base_policy(env, obs)
        act = delegate.select_action(env)
        if tau <= 0.0:
            return act  # legacy: fully delegate (greedy fallback + any-vote DFS)
        # conservative: keep the BASE policy (student) unless the DFS found a
        # move that forces a win in >= tau of the sampled worlds.
        ev = delegate.last_eval
        if ev and ev.get("votes"):
            best_frac = max(ev["votes"].values())
            if best_frac >= tau:
                return act
        return base_policy(env, obs)
    return policy


def make_policy(args):
    """Return fn(env, obs) -> action."""
    max_candidates = args.max_candidates

    if args.policy == "greedy":
        def policy(env, obs):
            if obs["mask"][:max_candidates].sum() > 0:
                return 0  # candidate 0 = max-tiles best play (same as opponent)
            return max_candidates
        if args.endgame_search:
            policy = _wrap_endgame(policy, args)
        return policy

    if args.policy == "student":
        import numpy as np_
        import torch
        from ppo_model import DistillStudent

        wj = getattr(args, "with_jokers", False)
        obs_dim = obs_dim_from_env(wj)
        if args.student_history:
            from distill import EVENT_FEAT_DIM, event_feats
            from selfplay_data import _event_history
            obs_dim += EVENT_FEAT_DIM

        model = DistillStudent(
            obs_dim=obs_dim,
            cand_feat_dim=cand_dim_from_env(wj),
            max_candidates=max_candidates,
        )
        state_dict = torch.load(args.model, map_location="cpu",
                                weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        margin = args.student_margin

        def policy(env, obs):
            state = obs["state"]
            if args.student_history:
                ev = event_feats(_event_history(env)[None])
                state = np_.concatenate([state, ev[0]]).astype(np_.float32)
            state_t = torch.tensor(state, dtype=torch.float32)
            cand_t = torch.tensor(obs["cand_feats"], dtype=torch.float32)
            mask_t = torch.tensor(obs["mask"], dtype=torch.float32)
            with torch.no_grad():
                logits = model.forward_actor(
                    state_t.unsqueeze(0), cand_t.unsqueeze(0)
                ).squeeze(0)
            logits[mask_t == 0] = -1e9
            best = int(torch.argmax(logits).item())
            # R10: same winner's-curse guard that saved the rollout teacher
            # (v1 -> v2): stick with the greedy max-play (candidate 0) unless
            # the student's own predicted margin clears the threshold. Only
            # meaningful for value-trained students (logits in score units).
            if margin > 0 and mask_t[0] > 0 and best != 0 \
                    and float(logits[best] - logits[0]) <= margin:
                return 0
            return best
        if args.endgame_search:
            policy = _wrap_endgame(policy, args)
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


def obs_dim_from_env(with_jokers=False):
    from ppo_env import state_dim_for
    return state_dim_for(with_jokers)


def cand_dim_from_env(with_jokers=False):
    from ppo_env import cand_feat_dim_for
    return cand_feat_dim_for(with_jokers)


def play_game(policy, seed, ppo_player, args):
    env = RummikubPPOEnv(
        max_candidates=args.max_candidates,
        max_turns=args.max_turns,
        seed=seed,
        ppo_player=ppo_player,
        opponent=args.opponent,
        initial_meld_value=args.initial_meld_value,
        exhaustive_candidates=args.exhaustive,
        generating_candidates=getattr(args, "generating_candidates", False),
        with_jokers=getattr(args, "with_jokers", False),
        value_scoring=getattr(args, "value_scoring", False),
        end_on_stuck=getattr(args, "end_on_stuck", False),
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
    parser.add_argument("--policy",
                        choices=["greedy", "rollout", "model", "random", "student"],
                        required=True)
    parser.add_argument("--model", default="distill_s1s_dagger1.pt")
    parser.add_argument("--opponent", choices=["ilp", "random"], default="ilp",
                        help="baseline opponent: greedy-ILP (default) or random")
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--initial-meld-value", type=int, default=30)
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
    parser.add_argument("--generating-candidates", action="store_true",
                        help="R11: generating-DP candidate list (complete + "
                             "fast, no arrangement dups). Overrides --exhaustive.")
    parser.add_argument("--exhaustive", action="store_true",
                        help="use exhaustive move enumeration for the agent's "
                             "candidates (R9)")
    parser.add_argument("--consistent", action="store_true",
                        help="rollout: information-consistent determinization "
                             "(reject samples contradicting opponent draw history)")
    parser.add_argument("--endgame-tau", type=float, default=0.0,
                        help="conservative endgame override: only replace the "
                             "base policy's move with the DFS move when it forces "
                             "a win in >= this fraction of determinizations "
                             "(1.0 = all worlds). Below it, keep the base policy. "
                             "0.0 = legacy behaviour (delegate fully to greedy+DFS).")
    parser.add_argument("--student-history", action="store_true",
                        help="student was trained with --history (opponent-"
                             "event features appended to state)")
    parser.add_argument("--student-margin", type=float, default=0.0,
                        help="student: deviate from the greedy max-play only "
                             "when its predicted score margin exceeds this "
                             "(value-trained logits, /50 score units)")
    parser.add_argument("--greedy-margin", type=float, default=0.0,
                        help="rollout: stick with greedy unless another "
                             "candidate wins by more than this per-rollout margin")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes (pairs are independent)")
    parser.add_argument("--with-jokers", action="store_true",
                        help="R11: 106-tile deck with 2 jokers (wildcards)")
    parser.add_argument("--value-scoring", action="store_true",
                        help="official value-sum scoring (joker=30) not tile count")
    parser.add_argument("--end-on-stuck", action="store_true",
                        help="official pool-empty-and-stuck end")
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

    opp_name = {"ilp": "greedy-ILP", "random": "random"}[args.opponent]
    print(f"\n=== mirror eval: {args.policy} vs {opp_name} ===")
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
