import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, cand_feat_dim, max_candidates):
        super().__init__()

        self.max_candidates = max_candidates

        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.cand_encoder = nn.Sequential(
            nn.Linear(cand_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.score_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.draw_head = nn.Linear(128, 1)
        self.critic = nn.Linear(128, 1)

    def forward_actor(self, obs, cand_feats):
        state_embed = self.state_encoder(obs)
        cand_embed = self.cand_encoder(cand_feats)

        state_expand = state_embed.unsqueeze(1).expand(-1, cand_embed.size(1), -1)
        joint = torch.cat([state_expand, cand_embed], dim=-1)

        cand_logits = self.score_head(joint).squeeze(-1)
        draw_logit = self.draw_head(state_embed)
        return torch.cat([cand_logits, draw_logit], dim=1)

    def forward_value(self, obs):
        state_embed = self.state_encoder(obs)
        return self.critic(state_embed).squeeze(-1)

    def act(self, obs, cand_feats, action_mask):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            cand_feats = cand_feats.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)

        logits = self.forward_actor(obs, cand_feats)
        values = self.forward_value(obs)

        masked_logits = logits.clone()
        masked_logits[action_mask == 0] = -1e9

        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.squeeze(0), log_prob.squeeze(0), entropy.squeeze(0), values.squeeze(0)

    def evaluate_actions(self, obs, cand_feats, actions, action_masks):
        logits = self.forward_actor(obs, cand_feats)
        values = self.forward_value(obs)

        masked_logits = logits.clone()
        masked_logits[action_masks == 0] = -1e9

        dist = Categorical(logits=masked_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values
