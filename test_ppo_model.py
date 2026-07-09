import numpy as np
import torch

from ppo_env import RummikubPPOEnv
from ppo_model import ActorCritic


def main():
    env = RummikubPPOEnv(
        max_candidates=20,
        max_turns=30,
        seed=42,
    )

    env.reset()
    obs, cand_feats, mask = env.get_policy_inputs()

    model = ActorCritic(
        obs_dim=106,
        cand_feat_dim=104,
        max_candidates=20,
    )

    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    cand_tensor = torch.tensor(cand_feats, dtype=torch.float32)
    mask_tensor = torch.tensor(mask, dtype=torch.float32)

    action, log_prob, entropy, value = model.act(
        obs_tensor,
        cand_tensor,
        mask_tensor,
    )

    print("obs shape:", obs_tensor.shape)
    print("cand_feats shape:", cand_tensor.shape)
    print("mask:", mask)
    print("action:", action.item())
    print("log_prob:", log_prob.item())
    print("entropy:", entropy.item())
    print("value:", value.item())

    valid_actions = np.where(mask == 1)[0]
    print("valid_actions:", valid_actions)

    if action.item() in valid_actions:
        print("ok: selected action is valid.")
    else:
        print("error: selected action is invalid.")


if __name__ == "__main__":
    main()
