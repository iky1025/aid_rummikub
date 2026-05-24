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
        next_non_terminal = 0.0 if dones[step] else 1.0
        delta = rewards[step] + gamma * values[step + 1] * next_non_terminal - values[step]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.insert(0, gae)

    returns = [adv + value for adv, value in zip(advantages, values[:-1])]
    return advantages, returns


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_candidates = 20
    max_turns = 100

    env = RummikubPPOEnv(
        max_candidates=max_candidates,
        max_turns=max_turns,
        seed=42,
    )

    obs_dim = 52 + 52 + 1
    cand_feat_dim = 52 + 52
    action_dim = max_candidates + 1

    model = ActorCritic(
        obs_dim=obs_dim,
        cand_feat_dim=cand_feat_dim,
        max_candidates=max_candidates,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=3e-4)

    n_steps = 100
    total_updates = 30
    batch_size = 5
    ppo_epochs = 3

    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    value_coef = 0.5
    entropy_coef = 0.01

    env.reset()
    obs, cand_feats, action_mask = env.get_policy_inputs()

    for update in range(total_updates):
        obs_list = []
        cand_feat_list = []
        action_list = []
        reward_list = []
        done_list = []
        log_prob_list = []
        value_list = []
        mask_list = []

        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0.0
        current_episode_steps = 0
        info = None
        draw_count = 0

        for _ in range(n_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            cand_tensor = torch.tensor(cand_feats, dtype=torch.float32, device=device)
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=device)

            with torch.no_grad():
                action, log_prob, _, value = model.act(obs_tensor, cand_tensor, mask_tensor)

            action_int = action.item()
            if action_int == max_candidates:
                draw_count += 1
            next_obs, reward, done, info = env.step(action_int)

            obs_list.append(obs)
            cand_feat_list.append(cand_feats)
            action_list.append(action_int)
            reward_list.append(reward)
            done_list.append(done)
            log_prob_list.append(log_prob.item())
            value_list.append(value.item())
            mask_list.append(action_mask)

            current_episode_reward += reward
            current_episode_steps += 1

            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_steps)
                current_episode_reward = 0.0
                current_episode_steps = 0
                env.reset()

            obs, cand_feats, action_mask = env.get_policy_inputs()

        last_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            last_value = model.forward_value(last_obs_tensor).item()

        advantages, returns = compute_gae(
            rewards=reward_list,
            values=value_list,
            dones=done_list,
            last_value=last_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
        cand_tensor = torch.tensor(np.array(cand_feat_list), dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(action_list, dtype=torch.long, device=device)
        old_log_probs_tensor = torch.tensor(log_prob_list, dtype=torch.float32, device=device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        masks_tensor = torch.tensor(np.array(mask_list), dtype=torch.float32, device=device)

        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        data_size = n_steps
        indices = np.arange(data_size)
        actor_losses = []
        critic_losses = []
        entropy_values = []
        total_losses = []

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, data_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_obs = obs_tensor[batch_indices]
                batch_cands = cand_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_masks = masks_tensor[batch_indices]

                new_log_probs, entropy, values = model.evaluate_actions(
                    batch_obs,
                    batch_cands,
                    batch_actions,
                    batch_masks,
                )

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                unclipped_loss = ratio * batch_advantages
                clipped_loss = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * batch_advantages

                actor_loss = -torch.min(unclipped_loss, clipped_loss).mean()
                critic_loss = F.mse_loss(values, batch_returns)
                entropy_loss = entropy.mean()

                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_loss
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_values.append(entropy_loss.item())
                total_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else current_episode_reward
        avg_episode_turns = (
            sum(episode_lengths) / len(episode_lengths)
            if episode_lengths
            else float(current_episode_steps)
        )
        draw_rate = draw_count / float(n_steps)
        avg_actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else 0.0
        avg_critic_loss = sum(critic_losses) / len(critic_losses) if critic_losses else 0.0
        avg_entropy = sum(entropy_values) / len(entropy_values) if entropy_values else 0.0
        avg_total_loss = sum(total_losses) / len(total_losses) if total_losses else 0.0

        if info is None:
            info = {
                "ppo_hand_count": len(env.hands[env.ppo_player]),
                "ilp_hand_count": len(env.hands[env.ilp_player]),
                "deck_count": len(env.env.deck),
                "candidate_count": 0,
            }

        print(
            f"update={update + 1}, "
            f"avg_episode_reward={avg_reward:.2f}, "
            f"avg_episode_turns={avg_episode_turns:.2f}, "
            f"draw_count={draw_count}, "
            f"draw_rate={draw_rate:.2f}, "
            f"actor_loss={avg_actor_loss:.4f}, "
            f"critic_loss={avg_critic_loss:.4f}, "
            f"entropy={avg_entropy:.4f}, "
            f"total_loss={avg_total_loss:.4f}"
        )

        if (update + 1) % 10 == 0:
            torch.save(model.state_dict(), "rummikub_ppo_model.pt")
            print("model saved: rummikub_ppo_model.pt")


if __name__ == "__main__":
    train()
