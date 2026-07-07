import argparse
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

from ppo_env import RummikubPPOEnv
from ppo_model import ActorCritic


def make_env_fn(seed, alternate_first_player=False, initial_meld_value=0):
    def _init():
        return RummikubPPOEnv(
            seed=seed,
            alternate_first_player=alternate_first_player,
            initial_meld_value=initial_meld_value,
        )
    return _init


def compute_gae_vec(rewards, values, dones, last_values, gamma=0.99, gae_lambda=0.95):
    """
    Per-env GAE.
    Inputs are np arrays of shape (T, N).
    last_values is shape (N,).
    Returns advantages, returns of shape (T, N).
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_values = last_values
        else:
            next_values = values[t + 1]
        next_non_terminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_envs = args.n_envs
    env_fns = [
        make_env_fn(
            seed + i,
            alternate_first_player=args.alternate_first_player,
            initial_meld_value=args.initial_meld_value,
        )
        for i in range(n_envs)
    ]
    if args.use_subproc:
        vec_env = SubprocVecEnv(env_fns, start_method=args.start_method)
    else:
        vec_env = DummyVecEnv(env_fns)
    print(f"vec_env: {'SubprocVecEnv' if args.use_subproc else 'DummyVecEnv'} with {n_envs} envs")

    max_candidates = 20
    obs_dim = 52 + 52 + 1 + 1 + 1 + 1  # hand + table + deck + opp_hand + meld_ppo + meld_opp
    cand_feat_dim = 52 + 52

    model = ActorCritic(
        obs_dim=obs_dim,
        cand_feat_dim=cand_feat_dim,
        max_candidates=max_candidates,
    ).to(device)

    if args.resume:
        print(f"resuming from: {args.resume}")
        state_dict = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    optimizer = Adam(model.parameters(), lr=args.lr_init)

    n_steps = args.n_steps
    total_updates = args.total_updates
    batch_size = args.batch_size
    ppo_epochs = args.ppo_epochs
    save_every = args.save_every

    suffix = f"_{args.tag}" if args.tag else ""
    model_path = f"rummikub_ppo_model{suffix}.pt"
    best_path = f"rummikub_ppo_best{suffix}.pt"
    log_path = f"train_log{suffix}.csv"

    gamma = 0.99
    gae_lambda = 0.95
    clip_range = args.clip_range
    value_coef = 0.1
    entropy_coef = 0.01

    lr_scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args.lr_final / args.lr_init,
        total_iters=total_updates,
    )

    print(
        f"config: n_envs={n_envs} n_steps={n_steps} total_updates={total_updates} "
        f"batch_size={batch_size} ppo_epochs={ppo_epochs} "
        f"lr={args.lr_init}->{args.lr_final} clip={clip_range}"
    )
    print(f"experience per update: {n_envs * n_steps} steps")
    print(f"output: {model_path}, {best_path}, {log_path}")

    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "update", "elapsed_sec", "steps_total",
        "episodes", "win_rate", "loss_rate", "timeout_rate",
        "avg_episode_reward", "avg_episode_length",
        "draw_action_ratio", "forced_draw_ratio", "chosen_draw_ratio",
        "pre_meld_ratio",
        "pre_meld_forced_ratio", "pre_meld_chosen_ratio",
        "post_meld_forced_ratio", "post_meld_chosen_ratio",
        "avg_candidate_count",
        "avg_win_margin", "avg_loss_margin", "expected_score",
        "actor_loss", "critic_loss", "entropy",
        "lr", "best_expected_score",
    ])

    # R8: best checkpoint is picked by expected_score (win/loss margin), not
    # avg_reward — avg_reward is dominated by uncontrollable forced-draw
    # penalties, so it mostly measures deal luck.
    best_expected_score = -float("inf")
    steps_total = 0
    train_start = time.time()

    obs = vec_env.reset()
    current_episode_rewards = np.zeros(n_envs, dtype=np.float32)
    current_episode_lengths = np.zeros(n_envs, dtype=np.int32)

    for update in range(total_updates):
        # buffers shaped (T, N, ...)
        obs_state_buf = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        obs_cand_buf = np.zeros((n_steps, n_envs, max_candidates, cand_feat_dim), dtype=np.float32)
        obs_mask_buf = np.zeros((n_steps, n_envs, max_candidates + 1), dtype=np.float32)
        action_buf = np.zeros((n_steps, n_envs), dtype=np.int64)
        reward_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        done_buf = np.zeros((n_steps, n_envs), dtype=bool)
        log_prob_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        value_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

        episode_rewards = []
        episode_lengths = []
        win_count = 0
        loss_count = 0
        timeout_count = 0
        draw_action_count = 0
        forced_draw_count = 0
        # R7: split draws by meld phase
        pre_meld_forced_count = 0
        pre_meld_chosen_count = 0
        post_meld_forced_count = 0
        post_meld_chosen_count = 0
        pre_meld_turn_count = 0
        candidate_count_sum = 0
        win_margin_sum = 0      # wins only (for avg_win_margin display)
        loss_margin_sum = 0     # losses only (for avg_loss_margin display)
        net_score_sum = 0       # all episodes (for expected_score)

        for step in range(n_steps):
            state = obs["state"]                # (N, obs_dim)
            cand_feats = obs["cand_feats"]      # (N, K, cand_dim)
            mask = obs["mask"]                  # (N, K+1)

            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            cand_t = torch.tensor(cand_feats, dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=device)

            with torch.no_grad():
                logits = model.forward_actor(state_t, cand_t)
                values = model.forward_value(state_t)
                masked_logits = logits.clone()
                masked_logits[mask_t == 0] = -1e9
                dist = torch.distributions.Categorical(logits=masked_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            action_np = action.cpu().numpy()

            obs_state_buf[step] = state
            obs_cand_buf[step] = cand_feats
            obs_mask_buf[step] = mask
            action_buf[step] = action_np
            log_prob_buf[step] = log_prob.cpu().numpy()
            value_buf[step] = values.cpu().numpy()

            next_obs, rewards, dones, infos = vec_env.step(action_np)
            reward_buf[step] = rewards
            done_buf[step] = dones

            draw_mask = action_np == max_candidates
            draw_action_count += int(draw_mask.sum())
            # forced draw = chose draw AND no valid candidate
            cand_mask_sum = mask[:, :max_candidates].sum(axis=1)  # (N,)
            forced_mask = (cand_mask_sum == 0) & draw_mask
            forced_draw_count += int(forced_mask.sum())
            # R7: meld-phase split
            pre_meld_arr = np.array(
                [bool(info.get("pre_meld", False)) for info in infos],
                dtype=bool,
            )
            pre_meld_turn_count += int(pre_meld_arr.sum())
            pre_meld_forced_count += int((pre_meld_arr & draw_mask & (cand_mask_sum == 0)).sum())
            pre_meld_chosen_count += int((pre_meld_arr & draw_mask & (cand_mask_sum > 0)).sum())
            post_meld_forced_count += int((~pre_meld_arr & draw_mask & (cand_mask_sum == 0)).sum())
            post_meld_chosen_count += int((~pre_meld_arr & draw_mask & (cand_mask_sum > 0)).sum())
            for info in infos:
                candidate_count_sum += int(info.get("candidate_count", 0))

            current_episode_rewards += rewards
            current_episode_lengths += 1
            steps_total += n_envs

            for i, done in enumerate(dones):
                if done:
                    episode_rewards.append(float(current_episode_rewards[i]))
                    episode_lengths.append(int(current_episode_lengths[i]))
                    outcome = infos[i].get("outcome")
                    win_m = int(infos[i].get("win_margin", 0))
                    loss_m = int(infos[i].get("loss_margin", 0))
                    if outcome == "win":
                        win_count += 1
                        win_margin_sum += win_m
                    elif outcome == "loss":
                        loss_count += 1
                        loss_margin_sum += loss_m
                    else:
                        timeout_count += 1
                    # expected_score includes timeouts (signed net)
                    net_score_sum += win_m - loss_m
                    current_episode_rewards[i] = 0.0
                    current_episode_lengths[i] = 0

            obs = next_obs

        # bootstrap last values
        state_t = torch.tensor(obs["state"], dtype=torch.float32, device=device)
        with torch.no_grad():
            last_values = model.forward_value(state_t).cpu().numpy()

        advantages, returns = compute_gae_vec(
            rewards=reward_buf,
            values=value_buf,
            dones=done_buf,
            last_values=last_values,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # flatten (T, N, ...) -> (T*N, ...)
        flat = n_steps * n_envs
        obs_state_flat = obs_state_buf.reshape(flat, obs_dim)
        obs_cand_flat = obs_cand_buf.reshape(flat, max_candidates, cand_feat_dim)
        obs_mask_flat = obs_mask_buf.reshape(flat, max_candidates + 1)
        actions_flat = action_buf.reshape(flat)
        old_log_probs_flat = log_prob_buf.reshape(flat)
        advantages_flat = advantages.reshape(flat)
        returns_flat = returns.reshape(flat)

        # to tensors
        obs_state_t = torch.tensor(obs_state_flat, dtype=torch.float32, device=device)
        obs_cand_t = torch.tensor(obs_cand_flat, dtype=torch.float32, device=device)
        obs_mask_t = torch.tensor(obs_mask_flat, dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions_flat, dtype=torch.long, device=device)
        old_log_probs_t = torch.tensor(old_log_probs_flat, dtype=torch.float32, device=device)
        advantages_t = torch.tensor(advantages_flat, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns_flat, dtype=torch.float32, device=device)

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        indices = np.arange(flat)
        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        entropy_sum = 0.0
        update_steps = 0

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, flat, batch_size):
                end = start + batch_size
                b = indices[start:end]

                new_log_probs, entropy, values = model.evaluate_actions(
                    obs_state_t[b],
                    obs_cand_t[b],
                    actions_t[b],
                    obs_mask_t[b],
                )

                ratio = torch.exp(new_log_probs - old_log_probs_t[b])
                unclipped = ratio * advantages_t[b]
                clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages_t[b]

                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = F.mse_loss(values, returns_t[b])
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
        avg_reward = sum(episode_rewards) / episodes if episodes else 0.0
        avg_length = sum(episode_lengths) / episodes if episodes else 0.0
        win_rate = win_count / episodes if episodes else 0.0
        loss_rate = loss_count / episodes if episodes else 0.0
        timeout_rate = timeout_count / episodes if episodes else 0.0
        total_actions = n_steps * n_envs
        draw_ratio = draw_action_count / total_actions
        forced_draw_ratio = forced_draw_count / total_actions
        chosen_draw_ratio = draw_ratio - forced_draw_ratio
        # R7 phase-split ratios
        pre_meld_ratio = pre_meld_turn_count / total_actions
        pre_meld_forced_ratio = pre_meld_forced_count / total_actions
        pre_meld_chosen_ratio = pre_meld_chosen_count / total_actions
        post_meld_forced_ratio = post_meld_forced_count / total_actions
        post_meld_chosen_ratio = post_meld_chosen_count / total_actions
        avg_candidate_count = candidate_count_sum / total_actions
        avg_win_margin = win_margin_sum / win_count if win_count else 0.0
        avg_loss_margin = loss_margin_sum / loss_count if loss_count else 0.0
        expected_score = net_score_sum / episodes if episodes else 0.0
        avg_actor_loss = actor_loss_sum / max(1, update_steps)
        avg_critic_loss = critic_loss_sum / max(1, update_steps)
        avg_entropy = entropy_sum / max(1, update_steps)
        elapsed = time.time() - train_start

        current_lr = lr_scheduler.get_last_lr()[0]
        lr_scheduler.step()

        if expected_score > best_expected_score and episodes > 0:
            best_expected_score = expected_score
            torch.save(model.state_dict(), best_path)

        print(
            f"upd={update + 1:3d}/{total_updates} "
            f"t={elapsed:6.1f}s steps={steps_total:6d} "
            f"eps={episodes:3d} W/L/T={win_count}/{loss_count}/{timeout_count} "
            f"rew={avg_reward:7.2f} len={avg_length:5.1f} "
            f"draw={draw_ratio:.2f}(pre.f={pre_meld_forced_ratio:.2f} "
            f"pre.c={pre_meld_chosen_ratio:.2f} "
            f"post.f={post_meld_forced_ratio:.2f} "
            f"post.c={post_meld_chosen_ratio:.2f}) "
            f"premeld={pre_meld_ratio:.2f} "
            f"wm={avg_win_margin:4.1f} lm={avg_loss_margin:4.1f} es={expected_score:+.2f} "
            f"a={avg_actor_loss:+.3f} cl={avg_critic_loss:.2f} ent={avg_entropy:.3f} "
            f"lr={current_lr:.1e} best_es={best_expected_score:+.2f}"
        )

        log_writer.writerow([
            update + 1, f"{elapsed:.2f}", steps_total,
            episodes, f"{win_rate:.3f}", f"{loss_rate:.3f}", f"{timeout_rate:.3f}",
            f"{avg_reward:.3f}", f"{avg_length:.2f}",
            f"{draw_ratio:.3f}", f"{forced_draw_ratio:.3f}", f"{chosen_draw_ratio:.3f}",
            f"{pre_meld_ratio:.3f}",
            f"{pre_meld_forced_ratio:.3f}", f"{pre_meld_chosen_ratio:.3f}",
            f"{post_meld_forced_ratio:.3f}", f"{post_meld_chosen_ratio:.3f}",
            f"{avg_candidate_count:.2f}",
            f"{avg_win_margin:.2f}", f"{avg_loss_margin:.2f}", f"{expected_score:.2f}",
            f"{avg_actor_loss:.4f}", f"{avg_critic_loss:.4f}", f"{avg_entropy:.4f}",
            f"{current_lr:.2e}", f"{best_expected_score:.3f}",
        ])
        log_file.flush()

        if (update + 1) % save_every == 0 or (update + 1) == total_updates:
            torch.save(model.state_dict(), model_path)
            print(f"  saved: {model_path}")

    log_file.close()
    vec_env.close()
    print(f"\ndone. log: {log_path}, model: {model_path}, best: {best_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=8,
                        help="number of parallel envs")
    parser.add_argument("--use-subproc", action="store_true", default=True,
                        help="use SubprocVecEnv (default)")
    parser.add_argument("--no-subproc", dest="use_subproc", action="store_false",
                        help="use DummyVecEnv (single process) for debugging")
    parser.add_argument("--start-method", type=str, default="forkserver",
                        choices=["spawn", "fork", "forkserver"],
                        help="multiprocessing start method")
    parser.add_argument("--n-steps", type=int, default=128,
                        help="rollout steps PER ENV per update")
    parser.add_argument("--total-updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--lr-init", type=float, default=3e-4)
    parser.add_argument("--lr-final", type=float, default=3e-5)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--alternate-first-player", action="store_true",
                        help="R7: alternate ppo_player between 0 and 1 per reset")
    parser.add_argument("--initial-meld-value", type=int, default=0,
                        help="R7: initial meld threshold (default 0=disabled; "
                             "30 enables standard Rummikub initial meld rule)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
