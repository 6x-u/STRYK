import numpy as np
import torch

from stryk.obs_dynamic import build_entities, MAX_ENTITIES, ENT_DIM, GLOBAL_DIM
from stryk.policy_multi import MultiAgentPolicy
from stryk.multi_env import MultiEnv


def test_entities_shapes_all_modes():
    for mode in ("1v1", "2v2", "3v3", "5v5", "11v11"):
        env = MultiEnv(mode, "wembley_fc27", 3)
        obs = env.reset()
        n = len(env.home_ids())
        assert obs["entities"].shape == (n, MAX_ENTITIES, ENT_DIM)
        assert obs["global"].shape == (n, GLOBAL_DIM)
        assert obs["mask"].shape == (n, MAX_ENTITIES)


def test_multi_policy_sample_masked():
    env = MultiEnv("2v2", "etihad_fc27", 5)
    obs = env.reset()
    pol = MultiAgentPolicy(hidden=32)
    ent = torch.as_tensor(obs["entities"])
    msk = torch.as_tensor(obs["mask"])
    glo = torch.as_tensor(obs["global"])
    ams = torch.as_tensor(obs["action_mask"])
    cont, disc, logp, val, entc = pol.sample(ent, msk, glo, ams)
    assert cont.shape == (2, 10)
    assert val.shape == (2,)
    obs2, r, done, info = env.step(cont.detach().numpy(), disc.numpy())
    assert r.shape == (2,)
