from stryk.curriculum import STAGES, Curriculum
from stryk.reward import RewardTracker
from stryk.config import MODES


def test_six_phases_map_to_modes():
    assert len(STAGES) == 6
    names = [s["name"] for s in STAGES]
    assert names[0] == "ball-control"
    assert names[-1] == "elite-league"
    for s in STAGES:
        assert s["mode"] in MODES


def test_curriculum_promotes():
    c = Curriculum(start=0)
    for _ in range(30):
        c.record(True)
    assert c.stage > 0


def test_reward_breakdown_shape():
    tr = RewardTracker()
    class W:
        pass
    # smoke: update with a dummy world-ish object is covered in train smoke;
    # here just check summary keys exist
    s = tr.summary()
    for k in ("pass_completion", "shot_accuracy", "possession_pct", "turnovers"):
        assert k in s
