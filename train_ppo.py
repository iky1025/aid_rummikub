import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ppo_env import RummikubPPOEnv
from ppo_model import ActorCritic


def compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95):
    advantages = []
    gae = 0.0

    values = values + [last_value]

    for step in reversed(range(len(rewards))):
        if dones[step]:
            next_non_terminal = 0.0
        else:
            next_non_terminal = 1.0

        delta = rewards[step] + gamma * values[step + 1] * next_non_terminal - values[step]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.insert(0, gae)

    returns = []
    for adv, value in zip(advantages, values[:-1]):
        returns.append(adv + value)

    return advantages, returns


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_candidates = 10
    max_turns = 100

    env = RummikubPPOEnv(
        max_candidates=max_candidates,
        max_turns=max_turns,
        seed=42,
    )

    obs_dim = 53 + 53 + 1
    action_dim = max_candidates + 1

    model = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=3e-4)

    n_steps = 100
    total_updates = 10
    batch_size = 5
    ppo_epochs = 2

    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    value_coef = 0.5
    entropy_coef = 0.01

    obs = env.reset()

    for update in range(total_updates):
        obs_list = []
        action_list = []
        reward_list = []
        done_list = []
        log_prob_list = []
        value_list = []
        mask_list = []

        episode_rewards = []
        current_episode_reward = 0.0

        for _ in range(n_steps):
            action_mask = env.get_action_mask()

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=device)

            with torch.no_grad():
                action, log_prob, _, value = model.act(
                    obs_tensor,
                    mask_tensor,
                )

            action_int = action.item()
            next_obs, reward, done, info = env.step(action_int)

            obs_list.append(obs)
            action_list.append(action_int)
            reward_list.append(reward)
            done_list.append(done)
            log_prob_list.append(log_prob.item())
            value_list.append(value.item())
            mask_list.append(action_mask)

            current_episode_reward += reward

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                obs = env.reset()
            else:
                obs = next_obs

        last_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            _, last_value_tensor = model.forward(last_obs_tensor)
            last_value = last_value_tensor.squeeze(-1).item()

        advantages, returns = compute_gae(
            rewards=reward_list,
            values=value_list,
            dones=done_list,
            last_value=last_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(action_list, dtype=torch.long, device=device)
        old_log_probs_tensor = torch.tensor(log_prob_list, dtype=torch.float32, device=device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        masks_tensor = torch.tensor(np.array(mask_list), dtype=torch.float32, device=device)

        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )

        data_size = n_steps
        indices = np.arange(data_size)

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, data_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_masks = masks_tensor[batch_indices]

                new_log_probs, entropy, values = model.evaluate_actions(
                    batch_obs,
                    batch_actions,
                    batch_masks,
                )

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                unclipped_loss = ratio * batch_advantages
                clipped_loss = torch.clamp(
                    ratio,
                    1.0 - clip_range,
                    1.0 + clip_range,
                ) * batch_advantages

                actor_loss = -torch.min(unclipped_loss, clipped_loss).mean()
                critic_loss = F.mse_loss(values, batch_returns)
                entropy_loss = entropy.mean()

                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if len(episode_rewards) > 0:
            avg_reward = sum(episode_rewards) / len(episode_rewards)
        else:
            avg_reward = current_episode_reward

        print(
            f"update={update + 1}, "
            f"avg_episode_reward={avg_reward:.2f}, "
            f"ppo_hand_count={info['ppo_hand_count']}, "
            f"ilp_hand_count={info['ilp_hand_count']}, "
            f"last_deck_count={info['deck_count']}, "
            f"candidate_count={info['candidate_count']}"
        )

        if (update + 1) % 50 == 0:
            torch.save(model.state_dict(), "rummikub_ppo_model.pt")
            print("model saved: rummikub_ppo_model.pt")


if __name__ == "__main__":
    train()
