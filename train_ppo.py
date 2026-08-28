import argparse
import json
from datetime import datetime
from pathlib import Path

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


def train(
    n_steps=1000,
    total_updates=20,
    target_episodes=None,
    model_path="rummikub_ppo_model.pt",
    resume_model_path=None,
    initial_completed_episodes=0,
    status_path=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42)
    torch.manual_seed(42)

    max_candidates = 10
    max_turns = 100
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    value_coef = 0.5
    entropy_coef = 0.01

    env = RummikubPPOEnv(
        max_candidates=max_candidates,
        max_turns=max_turns,
        seed=42,
        reward_gamma=gamma,
    )

    obs_dim = env.OBS_DIM
    cand_feat_dim = env.CAND_FEAT_DIM

    model = ActorCritic(
        obs_dim=obs_dim,
        cand_feat_dim=cand_feat_dim,
        max_candidates=max_candidates,
    ).to(device)

    if resume_model_path is not None:
        state_dict = torch.load(resume_model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"model resumed: {resume_model_path}")

    optimizer = Adam(model.parameters(), lr=3e-4)

    batch_size = 64
    ppo_epochs = 4

    env.reset()
    obs, cand_feats, action_mask = env.get_policy_inputs()
    completed_episodes = initial_completed_episodes
    current_episode_reward = 0.0

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
        episode_wins = 0
        episode_losses = 0
        episode_timeouts = 0
        draw_action_count = 0
        info = None
        stop_training = False
        candidate_counts = []
        raw_candidate_counts = []
        pool_candidate_counts = []
        duplicate_candidate_counts = []
        strategy_candidate_counts = []

        for step in range(n_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            cand_tensor = torch.tensor(cand_feats, dtype=torch.float32, device=device)
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=device)

            with torch.no_grad():
                action, log_prob, _, value = model.act(obs_tensor, cand_tensor, mask_tensor)

            action_int = action.item()
            if action_int == max_candidates:
                draw_action_count += 1
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
            candidate_counts.append(info.get("candidate_count", 0))
            raw_candidate_counts.append(info.get("raw_candidate_count", 0))
            pool_candidate_counts.append(info.get("pool_candidate_count", 0))
            duplicate_candidate_counts.append(
                info.get("duplicate_candidate_count", 0)
            )
            strategy_candidate_counts.append(
                info.get("strategy_candidate_count", 0)
            )

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                completed_episodes += 1
                if info.get("winner") == "ppo":
                    episode_wins += 1
                elif info.get("winner") == "ilp":
                    episode_losses += 1
                else:
                    episode_timeouts += 1
                env.reset()
                if (
                    target_episodes is not None
                    and completed_episodes >= target_episodes
                ):
                    stop_training = True

            obs, cand_feats, action_mask = env.get_policy_inputs()

            if (step + 1) % 100 == 0:
                print(
                    f"update={update + 1}/{total_updates}, "
                    f"step={step + 1}/{n_steps}",
                    flush=True,
                )

            if stop_training:
                break

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

        data_size = len(reward_list)
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

                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite PPO loss detected")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_values.append(entropy_loss.item())
                total_losses.append(loss.item())

        avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else current_episode_reward

        if info is None:
            info = {
                "ppo_hand_count": len(env.hands[env.ppo_player]),
                "ilp_hand_count": len(env.hands[env.ilp_player]),
                "deck_count": len(env.env.deck),
                "candidate_count": 0,
            }

        print(
            f"update={update + 1}, "
            f"completed_episodes={completed_episodes}, "
            f"avg_episode_reward={avg_reward:.2f}, "
            f"wins={episode_wins}, "
            f"losses={episode_losses}, "
            f"timeouts={episode_timeouts}, "
            f"draw_rate={draw_action_count / data_size:.3f}, "
            f"ppo_hand_count={info['ppo_hand_count']}, "
            f"ilp_hand_count={info['ilp_hand_count']}, "
            f"last_deck_count={info['deck_count']}, "
            f"avg_candidates={np.mean(candidate_counts):.2f}, "
            f"avg_raw_candidates={np.mean(raw_candidate_counts):.2f}, "
            f"avg_candidate_pool={np.mean(pool_candidate_counts):.2f}, "
            f"avg_duplicate_candidates={np.mean(duplicate_candidate_counts):.2f}, "
            f"avg_strategy_solutions={np.mean(strategy_candidate_counts):.2f}, "
            f"actor_loss={np.mean(actor_losses):.4f}, "
            f"critic_loss={np.mean(critic_losses):.4f}, "
            f"entropy={np.mean(entropy_values):.4f}, "
            f"total_loss={np.mean(total_losses):.4f}"
        )

        if status_path is not None:
            Path(status_path).write_text(
                json.dumps(
                    {
                        "stage": "training",
                        "detail": (
                            f"{completed_episodes}/{target_episodes} episodes"
                        ),
                        "updated_at": datetime.now().astimezone().isoformat(),
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                encoding="utf-8",
            )

        torch.save(model.state_dict(), model_path)
        print(f"model saved: {model_path}")

        if stop_training:
            break

    return {
        "completed_episodes": completed_episodes,
        "model_path": model_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--total-updates", type=int, default=20)
    parser.add_argument("--target-episodes", type=int)
    parser.add_argument("--model-path", default="rummikub_ppo_model.pt")
    parser.add_argument("--resume-model-path")
    parser.add_argument("--initial-completed-episodes", type=int, default=0)
    parser.add_argument("--status-path")
    args = parser.parse_args()
    train(
        n_steps=args.n_steps,
        total_updates=args.total_updates,
        target_episodes=args.target_episodes,
        model_path=args.model_path,
        resume_model_path=args.resume_model_path,
        initial_completed_episodes=args.initial_completed_episodes,
        status_path=args.status_path,
    )
