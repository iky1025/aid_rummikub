import torch

from ppo_env import RummikubPPOEnv
from ppo_model import ActorCritic


def evaluate(model_path="rummikub_ppo_model.pt", episodes=10, seed=123):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_candidates = 10
    max_turns = 100
    obs_dim = 53 + 53 + 1
    action_dim = max_candidates + 1

    env = RummikubPPOEnv(
        max_candidates=max_candidates,
        max_turns=max_turns,
        seed=seed,
    )

    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    wins = 0
    total_reward = 0.0
    total_steps = 0

    for _ in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        final_info = None

        while not done:
            mask = env.get_action_mask()

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)

            with torch.no_grad():
                logits, _ = model.forward(obs_tensor)
                masked_logits = logits.clone()
                masked_logits[mask_tensor == 0] = -1e9
                action = torch.argmax(masked_logits).item()

            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            final_info = info

        if final_info is not None and final_info["ppo_hand_count"] == 0:
            wins += 1

        total_reward += episode_reward
        total_steps += steps

    print(f"episodes={episodes}")
    print(f"win_rate={wins / episodes:.3f}")
    print(f"avg_reward={total_reward / episodes:.3f}")
    print(f"avg_steps={total_steps / episodes:.2f}")


if __name__ == "__main__":
    evaluate()
