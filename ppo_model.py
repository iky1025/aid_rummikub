import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs):
        x = self.shared(obs)

        logits = self.actor(x)
        value = self.critic(x)

        return logits, value

    def act(self, obs, action_mask):
        logits, value = self.forward(obs)

        logits = logits.clone()
        logits[action_mask == 0] = -1e9

        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    def evaluate_actions(self, obs, actions, action_masks):
        logits, values = self.forward(obs)

        logits = logits.clone()
        logits[action_masks == 0] = -1e9

        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values.squeeze(-1)
