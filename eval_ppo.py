import argparse
import time

import torch

from ppo_env import RummikubPPOEnv
from ppo_model import ActorCritic


def evaluate(model_path="rummikub_ppo_model.pt", episodes=50, seed=123, stochastic=False, verbose=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    print(f"model: {model_path}")
    print(f"episodes: {episodes}, seed: {seed}, mode: {'stochastic' if stochastic else 'deterministic'}")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    max_candidates = 10
    max_turns = 100
    obs_dim = 52 + 52 + 1
    cand_feat_dim = 52 + 52

    env = RummikubPPOEnv(
        max_candidates=max_candidates,
        max_turns=max_turns,
        seed=seed,
    )

    model = ActorCritic(
        obs_dim=obs_dim,
        cand_feat_dim=cand_feat_dim,
        max_candidates=max_candidates,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    wins = 0
    losses = 0
    timeouts = 0
    total_reward = 0.0
    total_steps = 0
    draw_actions = 0
    total_actions = 0

    t0 = time.time()

    for ep in range(episodes):
        env.reset()
        obs, cand_feats, mask = env.get_policy_inputs()
        done = False
        episode_reward = 0.0
        steps = 0
        final_info = None
        ep_draw = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            cand_tensor = torch.tensor(cand_feats, dtype=torch.float32, device=device)
            mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)

            with torch.no_grad():
                logits = model.forward_actor(obs_tensor.unsqueeze(0), cand_tensor.unsqueeze(0)).squeeze(0)
                masked_logits = logits.clone()
                masked_logits[mask_tensor == 0] = -1e9

                if stochastic:
                    probs = torch.softmax(masked_logits, dim=-1)
                    action = torch.multinomial(probs, num_samples=1).item()
                else:
                    action = torch.argmax(masked_logits).item()

            if action == max_candidates:
                ep_draw += 1

            _, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            final_info = info

            if not done:
                obs, cand_feats, mask = env.get_policy_inputs()

        total_actions += steps
        draw_actions += ep_draw

        if final_info is not None and final_info["ppo_hand_count"] == 0:
            wins += 1
            outcome = "W"
        elif final_info is not None and final_info["ilp_hand_count"] == 0:
            losses += 1
            outcome = "L"
        else:
            timeouts += 1
            outcome = "T"

        total_reward += episode_reward
        total_steps += steps

        if verbose:
            print(
                f"ep={ep + 1:3d}/{episodes} "
                f"outcome={outcome} steps={steps:3d} "
                f"reward={episode_reward:+6.2f} draw_ratio={ep_draw / steps:.2f} "
                f"ppo_hand={final_info['ppo_hand_count']:2d} "
                f"ilp_hand={final_info['ilp_hand_count']:2d}"
            )

    elapsed = time.time() - t0

    print("\n=== result ===")
    print(f"episodes      : {episodes}")
    print(f"wins          : {wins} ({wins / episodes:.1%})")
    print(f"losses        : {losses} ({losses / episodes:.1%})")
    print(f"timeouts      : {timeouts} ({timeouts / episodes:.1%})")
    print(f"avg_reward    : {total_reward / episodes:+.3f}")
    print(f"avg_steps     : {total_steps / episodes:.2f}")
    print(f"draw_ratio    : {draw_actions / total_actions:.3f}")
    print(f"elapsed       : {elapsed:.1f}s ({elapsed / episodes:.2f}s/ep)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rummikub_ppo_model.pt", help="path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=50, help="number of episodes")
    parser.add_argument("--seed", type=int, default=123, help="env seed")
    parser.add_argument("--stochastic", action="store_true", help="sample from policy instead of argmax")
    parser.add_argument("--verbose", action="store_true", help="print per-episode result")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        seed=args.seed,
        stochastic=args.stochastic,
        verbose=args.verbose,
    )
