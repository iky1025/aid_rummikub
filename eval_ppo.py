import torch
import time

from ppo_env import RummikubPPOEnv
from ppo_model import ActorCritic


def evaluate(model_path="rummikub_ppo_model.pt", episodes=20, seed=123):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(model_path, map_location=device)

    max_candidates = 20 
    max_turns = 100
    obs_dim = 52 + 52 + 2
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
    total_reward = 0.0
    total_steps = 0
    start_time = time.time()

    for _ in range(episodes):
        episode_idx = _ + 1
        env.reset()
        obs, cand_feats, mask = env.get_policy_inputs()
        done = False
        episode_reward = 0.0
        steps = 0
        final_info = None

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            cand_tensor = torch.tensor(cand_feats, dtype=torch.float32, device=device)
            mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)

            with torch.no_grad():
                logits = model.forward_actor(obs_tensor.unsqueeze(0), cand_tensor.unsqueeze(0)).squeeze(0)
                masked_logits = logits.clone()
                masked_logits[mask_tensor == 0] = -1e9
                action = torch.argmax(masked_logits).item()

            _, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            final_info = info

            if not done:
                obs, cand_feats, mask = env.get_policy_inputs()

        if final_info is not None and final_info["ppo_hand_count"] == 0:
            wins += 1

        total_reward += episode_reward
        total_steps += steps
        elapsed = time.time() - start_time
        avg_time_per_ep = elapsed / episode_idx
        eta = avg_time_per_ep * (episodes - episode_idx)
        print(
            f"[eval] episode={episode_idx}/{episodes}, "
            f"episode_reward={episode_reward:.2f}, "
            f"steps={steps}, "
            f"elapsed={elapsed:.1f}s, "
            f"eta={eta:.1f}s"
        )

    print(f"episodes={episodes}")
    print(f"win_rate={wins / episodes:.3f}")
    print(f"avg_reward={total_reward / episodes:.3f}")
    print(f"avg_steps={total_steps / episodes:.2f}")


if __name__ == "__main__":
    evaluate()
