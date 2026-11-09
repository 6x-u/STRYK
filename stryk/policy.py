from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist

from .obs import OBS_DIM
from .act import ACT_DIM


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.logstd = nn.Parameter(torch.full((act_dim,), -0.4))
        self.value = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0.0)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.constant_(self.value.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        mu = self.mu(h).tanh()
        logstd = self.logstd.expand_as(mu).clamp(-4.0, 2.0)
        value = self.value(h).squeeze(-1)
        return mu, logstd, value

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        mu, logstd, value = self.forward(obs)
        std = logstd.exp()
        d = dist.Normal(mu, std)
        if deterministic:
            act = mu
            logp = d.log_prob(act).sum(-1)
            ent = torch.zeros_like(logp)
        else:
            act = d.sample()
            act = act.clamp(-1.0, 1.0)
            logp = d.log_prob(act).sum(-1)
            ent = d.entropy().sum(-1)
        return act, logp, value, ent

    def evaluate(self, obs: torch.Tensor, act: torch.Tensor):
        mu, logstd, value = self.forward(obs)
        std = logstd.exp()
        d = dist.Normal(mu, std)
        logp = d.log_prob(act).sum(-1)
        ent = d.entropy().sum(-1)
        return logp, value, ent

    def mean_std(self) -> float:
        return float(self.logstd.exp().mean().detach().cpu())


def np_sample(policy: ActorCritic, obs_np: np.ndarray, device: str, deterministic: bool = False):
    with torch.no_grad():
        o = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        if o.ndim == 1:
            o = o.unsqueeze(0)
        act, logp, val, ent = policy.sample(o, deterministic=deterministic)
        return (act.cpu().numpy().astype(np.float32),
                logp.cpu().numpy().astype(np.float32),
                val.cpu().numpy().astype(np.float32),
                ent.cpu().numpy().astype(np.float32))
