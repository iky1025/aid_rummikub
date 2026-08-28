import math

from ppo_env import RummikubPPOEnv
from rummikub_solver import parse_tiles


def main():
    env = RummikubPPOEnv(max_candidates=10, max_turns=100, seed=7)
    assert env.REWARD_VERSION == 2
    assert env.win_reward == 20.0
    assert env.shaping_scale == 0.1

    env.hands = [parse_tiles("R5 R6"), parse_tiles("Y1 K1")]
    env.initial_meld_done = [False, False]
    connected_potential = env._potential()

    env.hands[env.ppo_player] = parse_tiles("R5 B6")
    isolated_potential = env._potential()
    assert connected_potential > isolated_potential

    env.initial_meld_done[env.ppo_player] = True
    assert math.isclose(
        env._potential() - isolated_potential,
        0.5,
        abs_tol=1e-9,
    )

    assert env._terminal_reward(True, False, False) == 20.0
    assert env._terminal_reward(False, True, False) == -20.0

    env.hands = [parse_tiles("R1 R2"), parse_tiles("B1 B2 B3 B4")]
    assert env._terminal_reward(False, False, True) == 4.0
    env.hands = [parse_tiles("R1 R2 R3 R4 R5 R6 R7"), parse_tiles("B1")]
    assert env._terminal_reward(False, False, True) == -10.0

    assert math.isclose(
        env._potential_shaping(3.0, 0.0),
        -0.3,
        abs_tol=1e-9,
    )
    print("connected_potential:", connected_potential)
    print("isolated_potential:", isolated_potential)
    print("reward_v2: ok")


if __name__ == "__main__":
    main()
