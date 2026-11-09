import torch

from stryk.policy import ActorCritic
from stryk.obs import OBS_DIM
from stryk.act import ACT_DIM


def test_policy_forward_and_update_shape():
    p = ActorCritic()
    t = torch.randn(8, OBS_DIM)
    mu, logstd, val = p(t)
    assert mu.shape == (8, ACT_DIM)
    assert val.shape == (8,)
    act = torch.randn(8, ACT_DIM).clamp(-1, 1)
    logp, val2, ent = p.evaluate(t, act)
    assert logp.shape == (8,)
    loss = -logp.mean() + val2.mean()
    loss.backward()
    assert p.mu.weight.grad is not None
