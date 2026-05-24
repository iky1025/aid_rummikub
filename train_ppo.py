import csv
import os
import time

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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_candidates = 10
    max_turns = 100

    env = RummikubPPOEnv(
        max_candidates=max_candidates,
        max_turns=max_turns,
        seed=seed,
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

    n_steps = 512
    total_updates = 50
    batch_size = 64
    ppo_epochs = 4
    save_every = 10
    model_path = "rummikub_ppo_model.pt"
    log_path = "train_log.csv"

    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    value_coef = 0.1
    entropy_coef = 0.01

    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "update", "elapsed_sec", "steps_total",
        "episodes", "win_rate", "loss_rate", "timeout_rate",
        "avg_episode_reward", "avg_episode_length",
        "draw_action_ratio", "avg_candidate_count",
        "actor_loss", "critic_loss", "entropy",
        "best_avg_reward",
    ])

    best_avg_reward = -float("inf")
    steps_total = 0
    train_start = time.time()

    env.reset()
    obs, cand_feats, action_mask = env.get_policy_inputs()
    current_episode_length = 0

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
        win_count = 0
        loss_count = 0
        timeout_count = 0
        draw_action_count = 0
        candidate_count_sum = 0
        current_episode_reward = 0.0
        info = None

        for _ in range(n_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            cand_tensor = torch.tensor(cand_feats, dtype=torch.float32, device=device)
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=device)

            with torch.no_grad():
                action, log_prob, _, value = model.act(obs_tensor, cand_tensor, mask_tensor)

            action_int = action.item()
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
            current_episode_length += 1
            steps_total += 1
            if action_int == max_candidates:
                draw_action_count += 1
            candidate_count_sum += info.get("candidate_count", 0)

            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                if info["ppo_hand_count"] == 0:
                    win_count += 1
                elif info["ilp_hand_count"] == 0:
                    loss_count += 1
                else:
                    timeout_count += 1
                current_episode_reward = 0.0
                current_episode_length = 0
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

        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        entropy_sum = 0.0
        update_steps = 0

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

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                actor_loss_sum += actor_loss.item()
                critic_loss_sum += critic_loss.item()
                entropy_sum += entropy_loss.item()
                update_steps += 1

        episodes = len(episode_rewards)
        avg_reward = sum(episode_rewards) / episodes if episodes else current_episode_reward
        avg_length = sum(episode_lengths) / episodes if episodes else 0.0
        win_rate = win_count / episodes if episodes else 0.0
        loss_rate = loss_count / episodes if episodes else 0.0
        timeout_rate = timeout_count / episodes if episodes else 0.0
        draw_ratio = draw_action_count / n_steps
        avg_candidate_count = candidate_count_sum / n_steps
        avg_actor_loss = actor_loss_sum / max(1, update_steps)
        avg_critic_loss = critic_loss_sum / max(1, update_steps)
        avg_entropy = entropy_sum / max(1, update_steps)
        elapsed = time.time() - train_start

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(model.state_dict(), f"rummikub_ppo_best.pt")

        print(
            f"upd={update + 1:3d}/{total_updates} "
            f"t={elapsed:6.1f}s steps={steps_total:5d} "
            f"eps={episodes:2d} W/L/T={win_count}/{loss_count}/{timeout_count} "
            f"rew={avg_reward:7.2f} len={avg_length:5.1f} "
            f"draw={draw_ratio:.2f} cand={avg_candidate_count:4.1f} "
            f"a={avg_actor_loss:+.3f} c={avg_critic_loss:.3f} ent={avg_entropy:.3f} "
            f"best={best_avg_reward:7.2f}"
        )

        log_writer.writerow([
            update + 1, f"{elapsed:.2f}", steps_total,
            episodes, f"{win_rate:.3f}", f"{loss_rate:.3f}", f"{timeout_rate:.3f}",
            f"{avg_reward:.3f}", f"{avg_length:.2f}",
            f"{draw_ratio:.3f}", f"{avg_candidate_count:.2f}",
            f"{avg_actor_loss:.4f}", f"{avg_critic_loss:.4f}", f"{avg_entropy:.4f}",
            f"{best_avg_reward:.3f}",
        ])
        log_file.flush()

        if (update + 1) % save_every == 0 or (update + 1) == total_updates:
            torch.save(model.state_dict(), model_path)
            print(f"  saved: {model_path}")

    log_file.close()
    print(f"\ndone. log: {log_path}, model: {model_path}, best: rummikub_ppo_best.pt")


if __name__ == "__main__":
    train()
