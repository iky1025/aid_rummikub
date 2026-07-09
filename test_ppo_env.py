import numpy as np

from ppo_env import RummikubPPOEnv


def main():
    env = RummikubPPOEnv(
        max_candidates=20,
        max_turns=30,
        seed=42,
    )

    obs = env.reset()
    obs, cand_feats, mask = env.get_policy_inputs()

    print("initial observation shape:", obs.shape)
    print("initial candidate feature shape:", cand_feats.shape)
    print("initial state:")
    env.env.render()

    done = False
    total_reward = 0.0
    step_count = 0

    while not done:
        valid_actions = np.where(mask == 1)[0]
        action = np.random.choice(valid_actions)

        obs, reward, done, info = env.step(action)

        total_reward += reward
        step_count += 1

        print("\n==============================")
        print("step:", step_count)
        print("selected action:", action)
        print("reward:", reward)
        print("done:", done)
        print("info:", info)
        print("observation shape:", obs.shape)

        env.env.render()

        if not done:
            obs, cand_feats, mask = env.get_policy_inputs()

    print("\n=== episode finished ===")
    print("total steps:", step_count)
    print("total reward:", total_reward)


if __name__ == "__main__":
    main()
