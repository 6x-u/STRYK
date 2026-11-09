from __future__ import annotations

import copy
import random

from .opponents import STYLES


class League:
    def __init__(self, seed: int = 2026, snapshot_every: int = 50):
        self.rng = random.Random(int(seed))
        self.pool: list[dict] = [{"style": "balanced", "difficulty": 0.0, "state": None, "wins": 0, "games": 0}]
        self.snapshot_every = int(snapshot_every)
        self.steps = 0

    def step(self):
        self.steps += 1

    def maybe_snapshot(self, policy, step: int, difficulty: float):
        if self.steps % self.snapshot_every != 0:
            return
        state = copy.deepcopy({k: v.detach().cpu() for k, v in policy.state_dict().items()})
        self.pool.append({"style": self.rng.choice(STYLES),
                          "difficulty": float(min(difficulty + 0.1, 1.0)),
                          "state": state, "wins": 0, "games": 0, "at": int(step)})
        if len(self.pool) > 12:
            self.pool.pop(1)  # keep seed + recent

    def sample(self) -> tuple[str, float]:
        e = self.rng.choice(self.pool)
        return e["style"], float(e["difficulty"])

    def record(self, home_won: bool):
        # credits the last sampled foe implicitly via difficulty bump on loss
        if not home_won:
            self.pool[-1]["wins"] += 1
        self.pool[-1]["games"] += 1

    def summary(self) -> list[dict]:
        return [{"style": e["style"], "difficulty": e["difficulty"],
                 "wins": e["wins"], "games": e["games"]} for e in self.pool]
