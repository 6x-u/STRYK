from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from .stadiums import STADIUMS, StadiumSpec
from .version import __version__

PITCH_LEN = 105.0
PITCH_WID = 68.0
GOAL_WID = 7.32
GOAL_HGT = 2.44
BALL_R = 0.11
BALL_MASS = 0.43
GRAV = 9.81


@dataclass
class ModeSpec:
    name: str          # 11v11 | 7v7 | 5v5 | 3v3 | 2v2 | 1v1
    per_side: int
    formation: str
    episode_sec: float
    phys_hz: float = 240.0
    env_hz: float = 60.0


# formations live in tactics.py, keep them in sync when you add a mode
MODES: dict[str, ModeSpec] = {
    "11v11": ModeSpec("11v11", 11, "4-3-3", 540.0),
    "7v7": ModeSpec("7v7", 7, "3-2-1", 300.0),
    "5v5": ModeSpec("5v5", 5, "2-1-1", 300.0),
    "3v3": ModeSpec("3v3", 3, "1-1-1", 180.0),
    "2v2": ModeSpec("2v2", 2, "1-1", 150.0),
    "1v1": ModeSpec("1v1", 1, "1", 120.0),
}


@dataclass
class TrainCfg:
    model_name: str = "STRYK"  # files become {model_name}_latest.pt etc — kaggle run sets merobot
    mode: str = "5v5"
    stadium: str = "wembley_fc27"
    seed: int = 2026
    envs: int = 1
    rollout_steps: int = 2048
    update_epochs: int = 4
    minibatch: int = 256
    lr: float = 3e-4
    clip: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad: float = 0.5
    total_updates: int = 200
    save_every: int = 20
    eval_every: int = 20
    stage: int = 0
    curriculum: bool = True
    selfplay: bool = True
    opponent_style: str = "balanced"
    device: str = "cpu"
    checkpoint: str = "checkpoints/last.pt"
    telemetry_csv: str = "checkpoints/telemetry.csv"
    # -- mappo / wall-clock knobs, defaults keep old single-agent loads happy
    use_mappo: bool = False
    train_hours: float = 0.0  # 0 = run until total_updates, else wall-clock cap
    models_root: str = "Models"
    hidden: int = 256
    ent_dim: int = 128
    obs_version: int = 2
    act_version: int = 2

    def stadium_spec(self) -> StadiumSpec:
        return STADIUMS[self.stadium]

    def mode_spec(self) -> ModeSpec:
        return MODES[self.mode]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainCfg":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def load_config(path: str | Path) -> TrainCfg:
    p = Path(path)
    if p.suffix == ".json":
        return TrainCfg.from_dict(json.loads(p.read_text(encoding="utf-8-sig")))
    raise ValueError(f"unsupported config format: {p.suffix}, use .json")


def stadium_names() -> list[str]:
    return sorted(STADIUMS.keys())


def model_layout(root: str | Path = "Models") -> dict[str, Path]:
    # canonical layout the spec asks for — one place to find everything :)
    r = Path(root)
    return {
        "checkpoints": r / "Checkpoints",
        "best": r / "Best",
        "exports": r / "Exports",
        "metadata": r / "Metadata",
    }
