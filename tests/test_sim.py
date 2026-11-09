import numpy as np

from stryk.config import MODES
from stryk.env import StrykEnv
from stryk.obs import OBS_DIM
from stryk.sim import World
from stryk.act import zero_action, decode


def test_modes():
    assert set(MODES) == {"11v11", "7v7", "5v5", "3v3", "2v2", "1v1"}
    assert MODES["11v11"].per_side == 11
    assert MODES["5v5"].per_side == 5
    assert MODES["3v3"].per_side == 3
    assert MODES["7v7"].per_side == 7
    assert MODES["1v1"].per_side == 1


def test_world_kickoff_and_step():
    w = World("5v5", "wembley_fc27", 42)
    assert len(w.players) == 10
    acts = [decode(zero_action()) for _ in w.players]
    w.step(acts)
    assert np.isfinite(w.ball.pos).all()
    assert w.tick > 0


def test_env_obs_and_step():
    env = StrykEnv("1v1", "dome_fc27", 7)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,)
    obs2, r, done, info = env.step(np.zeros(env.action_space_dim, dtype=np.float32))
    assert obs2.shape == (OBS_DIM,)
    assert isinstance(r, float)
    assert isinstance(done, bool)
