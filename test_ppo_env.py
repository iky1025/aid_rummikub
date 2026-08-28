import numpy as np

from ppo_env import RummikubPPOEnv


def main():
    env = RummikubPPOEnv(
        max_candidates=10,
        max_turns=30,
        seed=42,
    )

    obs = env.reset()
    obs, cand_feats, mask = env.get_policy_inputs()

    assert obs.shape == (env.OBS_DIM,)
    assert cand_feats.shape == (env.max_candidates, env.CAND_FEAT_DIM)
    assert mask[-1] == 1
    _assert_unique_candidate_features(cand_feats, mask)

    paired_env = RummikubPPOEnv(
        max_candidates=10,
        max_turns=30,
        seed=42,
        ppo_player=1,
    )
    paired_env.reset()
    assert env.dealt_hands == paired_env.dealt_hands

    assert env._potential_shaping(0.0, 2.0) > 0.0
    assert env._potential_shaping(0.0, -2.0) < 0.0

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
            _assert_unique_candidate_features(cand_feats, mask)

    print("\n=== episode finished ===")
    print("total steps:", step_count)
    print("total reward:", total_reward)


def _assert_unique_candidate_features(cand_feats, mask):
    candidate_count = int(mask[:-1].sum())
    feature_keys = {
        cand_feats[index].tobytes()
        for index in range(candidate_count)
    }
    assert len(feature_keys) == candidate_count


if __name__ == "__main__":
    main()
