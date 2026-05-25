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
    obs_dim = 52 + 52 + 1 + 1
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
    forced_draw_actions = 0
    total_actions = 0
    win_margins = []        # from wins only
    loss_margins = []       # from losses only
    net_scores = []         # all episodes (win_m - loss_m)

    t0 = time.time()

    for ep in range(episodes):
        obs_dict, _ = env.reset(seed=seed + ep)
        done = False
        episode_reward = 0.0
        steps = 0
        final_info = None
        ep_draw = 0
        ep_forced_draw = 0

        while not done:
            state = obs_dict["state"]
            cand_feats = obs_dict["cand_feats"]
            mask = obs_dict["mask"]

            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            cand_t = torch.tensor(cand_feats, dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=device)

            with torch.no_grad():
                logits = model.forward_actor(state_t.unsqueeze(0), cand_t.unsqueeze(0)).squeeze(0)
                masked_logits = logits.clone()
                masked_logits[mask_t == 0] = -1e9

                if stochastic:
                    probs = torch.softmax(masked_logits, dim=-1)
                    action = torch.multinomial(probs, num_samples=1).item()
                else:
                    action = torch.argmax(masked_logits).item()

            n_valid_candidates = int(mask[:max_candidates].sum())
            if action == max_candidates:
                ep_draw += 1
                if n_valid_candidates == 0:
                    ep_forced_draw += 1

            obs_dict, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            final_info = info

        total_actions += steps
        draw_actions += ep_draw
        forced_draw_actions += ep_forced_draw

        outcome_flag = final_info.get("outcome") if final_info is not None else None
        win_m = int(final_info.get("win_margin", 0)) if final_info else 0
        loss_m = int(final_info.get("loss_margin", 0)) if final_info else 0

        if outcome_flag == "win":
            wins += 1
            outcome = "W"
            win_margins.append(win_m)
        elif outcome_flag == "loss":
            losses += 1
            outcome = "L"
            loss_margins.append(loss_m)
        else:
            timeouts += 1
            outcome = "T"
        net_scores.append(win_m - loss_m)

        total_reward += episode_reward
        total_steps += steps

        if verbose:
            print(
                f"ep={ep + 1:3d}/{episodes} "
                f"outcome={outcome} steps={steps:3d} "
                f"reward={episode_reward:+6.2f} "
                f"draw={ep_draw / steps:.2f}(f={ep_forced_draw / steps:.2f}) "
                f"ppo_hand={final_info['ppo_hand_count']:2d} "
                f"ilp_hand={final_info['ilp_hand_count']:2d}"
            )

    elapsed = time.time() - t0

    avg_win_margin = sum(win_margins) / len(win_margins) if win_margins else 0.0
    avg_loss_margin = sum(loss_margins) / len(loss_margins) if loss_margins else 0.0
    expected_score = sum(net_scores) / episodes if episodes else 0.0
    forced_ratio = forced_draw_actions / total_actions
    chosen_ratio = draw_actions / total_actions - forced_ratio

    print("\n=== result ===")
    print(f"episodes        : {episodes}")
    print(f"wins            : {wins} ({wins / episodes:.1%})")
    print(f"  avg margin    : {avg_win_margin:.2f} (opponent tiles left)")
    print(f"losses          : {losses} ({losses / episodes:.1%})")
    print(f"  avg margin    : {avg_loss_margin:.2f} (own tiles left)")
    print(f"timeouts        : {timeouts} ({timeouts / episodes:.1%})")
    print(f"expected_score  : {expected_score:+.3f}  (win_m_sum - loss_m_sum) / eps")
    print(f"avg_reward      : {total_reward / episodes:+.3f}")
    print(f"avg_steps       : {total_steps / episodes:.2f}")
    print(f"draw_ratio      : {draw_actions / total_actions:.3f}")
    print(f"  forced        : {forced_ratio:.3f}")
    print(f"  chosen        : {chosen_ratio:.3f}")
    print(f"elapsed         : {elapsed:.1f}s ({elapsed / episodes:.2f}s/ep)")


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
