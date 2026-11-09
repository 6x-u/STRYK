from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributions as dist

from .obs_dynamic import ENT_DIM, GLOBAL_DIM, MAX_ENTITIES
from .act import ACT_DIM

# hybrid discrete head — masks keep the policy honest about what is possible :)
ACTION_FAMILIES = ("move", "control", "pass_short", "pass_long_through", "shoot", "defend", "skill", "support")
DISCRETE_DIM = len(ACTION_FAMILIES)


def action_mask(world, idx: int) -> torch.Tensor:
    me = world.players[idx]
    b = world.ball.pos
    d = float(((me.pos[0] - b[0]) ** 2 + (me.pos[1] - b[1]) ** 2) ** 0.5)
    rd = world.dist_to_goal(me.pos, me.side)
    mates = sum(1 for q, p in enumerate(world.players) if p.side == me.side and q != idx)
    has_bandwidd = mates > 0
    m = torch.zeros(DISCRETE_DIM, dtype=torch.float32)
    m[0] = 1.0  # move always fine
    m[1] = 1.0 if d < 2.5 else 0.2  # control needs the ball close
    m[2] = 1.0 if (d < 1.8 and has_bandwidd) else 0.1
    m[3] = 1.0 if (d < 2.0 and has_bandwidd) else 0.1
    m[4] = 1.0 if (d < 2.2 and rd < 28.0) else 0.05  # no 40m rockets pls
    m[5] = 1.0  # defend / press is mostly positioning
    m[6] = 1.0 if d < 2.8 else 0.3
    m[7] = 1.0  # support run is always legal
    return m


class EntityEncoder(nn.Module):
    def __init__(self, ent_dim: int = ENT_DIM, hidden: int = 128, heads: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, ent: torch.Tensor, mask: torch.Tensor):
        # ent [B,E,D], mask [B,E] (1 = alive)
        h = self.proj(ent)
        pad = (mask < 0.5)
        # self-attention over the squad glide :) — pooling happens after
        a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        h = self.norm(h + a)
        return h  # [B,E,H]


class MultiAgentPolicy(nn.Module):
    """Shared actor (all roles share weights) + centralized critic."""

    def __init__(self, hidden: int = 128, ent_dim: int = ENT_DIM, glob_dim: int = GLOBAL_DIM):
        super().__init__()
        self.encoder = EntityEncoder(ent_dim, hidden)
        # actor: take the self entity slot (index of agent in entities is ambiguous,
        # so we mean-pool attended set + self slot feature from input — kept simple)
        self.actor_trunk = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.cont_mu = nn.Linear(hidden, ACT_DIM)
        self.cont_logstd = nn.Parameter(torch.full((ACT_DIM,), -0.4))
        self.disc_logits = nn.Linear(hidden, DISCRETE_DIM)
        # centralized critic: global + pooled entities
        self.critic_trunk = nn.Sequential(
            nn.Linear(glob_dim + hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.value = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.cont_mu.weight, gain=0.01)
        nn.init.constant_(self.cont_mu.bias, 0.0)

    def _pooled(self, ent: torch.Tensor, mask: torch.Tensor):
        h = self.encoder(ent, mask)  # [B,E,H]
        m = mask.unsqueeze(-1)
        pooled = (h * m).sum(dim=1) / (m.sum(dim=1) + 1e-6)
        # self feature: entity slot 0 is the ball, so use mean of alive players as proxy
        # — in practice the agent identity rides in ent[... , is_self]; pool weighted by is_self
        w_self = ent[..., 9] * mask  # is_self channel
        w_self = w_self.unsqueeze(-1)
        selff = (h * w_self).sum(dim=1) / (w_self.sum(dim=1) + 1e-6)
        # fallback: if no is_self flagged (e.g. global view), pooled alone is fine
        selff = torch.where(w_self.sum(dim=1) > 0, selff, pooled)
        return torch.cat([pooled, selff], dim=-1), pooled

    def forward(self, ent: torch.Tensor, mask: torch.Tensor, glob: torch.Tensor):
        fused, pooled = self._pooled(ent, mask)
        h = self.actor_trunk(fused)
        mu = self.cont_mu(h).tanh()
        logits = self.disc_logits(h)
        ch = self.critic_trunk(torch.cat([glob, pooled], dim=-1))
        value = self.value(ch).squeeze(-1)
        return mu, logits, value

    def sample(self, ent, mask, glob, amask: torch.Tensor, deterministic: bool = False):
        mu, logits, value = self.forward(ent, mask, glob)
        std = self.cont_logstd.exp().expand_as(mu)
        dc = dist.Normal(mu, std)
        dd = dist.Categorical(logits=logits + (1.0 - amask) * -1e4)
        if deterministic:
            cont = mu
            disc = torch.argmax(dd.logits, dim=-1)
        else:
            cont = dc.sample().clamp(-1.0, 1.0)
            disc = dd.sample()
        logp = dc.log_prob(cont).sum(-1) + dd.log_prob(disc)
        ent_c = dc.entropy().sum(-1) + dd.entropy()
        return cont, disc, logp, value, ent_c

    def evaluate(self, ent, mask, glob, amask, cont, disc):
        mu, logits, value = self.forward(ent, mask, glob)
        std = self.cont_logstd.exp().expand_as(mu)
        dc = dist.Normal(mu, std)
        dd = dist.Categorical(logits=logits + (1.0 - amask) * -1e4)
        logp = dc.log_prob(cont).sum(-1) + dd.log_prob(disc)
        ent_c = dc.entropy().sum(-1) + dd.entropy()
        return logp, value, ent_c

    def mean_std(self) -> float:
        return float(self.cont_logstd.exp().mean().detach().cpu())


def np_sample_multi(policy: MultiAgentPolicy, batch: dict, device: str, deterministic: bool = False):
    with torch.no_grad():
        ent = torch.as_tensor(batch["entities"], dtype=torch.float32, device=device)
        msk = torch.as_tensor(batch["mask"], dtype=torch.float32, device=device)
        glo = torch.as_tensor(batch["global"], dtype=torch.float32, device=device)
        ams = torch.as_tensor(batch["action_mask"], dtype=torch.float32, device=device)
        cont, disc, logp, val, ent_c = policy.sample(ent, msk, glo, ams, deterministic=deterministic)
        return (cont.cpu().numpy().astype("float32"), disc.cpu().numpy().astype("int64"),
                logp.cpu().numpy().astype("float32"), val.cpu().numpy().astype("float32"),
                ent_c.cpu().numpy().astype("float32"))
