from __future__ import annotations

from collections import deque

# 6 phases — start tiny or the team melts lol
# mode per phase pins the curriculum to a real ruleset :)
STAGES: list[dict] = [
    {"name": "ball-control", "mode": "1v1", "episode_sec": 60.0, "difficulty": 0.0, "gate": 0.45,
     "focus": "carry, touch, turn"},
    {"name": "duel-1v1", "mode": "1v1", "episode_sec": 90.0, "difficulty": 0.2, "gate": 0.50,
     "focus": "dribble vs press, intercept"},
    {"name": "duo-2v2", "mode": "2v2", "episode_sec": 150.0, "difficulty": 0.35, "gate": 0.50,
     "focus": "pass, support, shape"},
    {"name": "tactics-5v5", "mode": "5v5", "episode_sec": 300.0, "difficulty": 0.55, "gate": 0.55,
     "focus": "chance creation, cover, press"},
    {"name": "full-11v11", "mode": "11v11", "episode_sec": 540.0, "difficulty": 0.75, "gate": 0.58,
     "focus": "formation, stamina, organised attack"},
    {"name": "elite-league", "mode": "11v11", "episode_sec": 540.0, "difficulty": 0.95, "gate": 0.60,
     "focus": "elite self-play, unseen styles"},
]


class Curriculum:
    def __init__(self, start: int = 0, enabled: bool = True, window: int = 40):
        self.stage = int(start)
        self.enabled = enabled
        self.window = deque(maxlen=window)
        self.updates = 0
        self.promotions = 0
        self.demotions = 0

    def current(self) -> dict:
        return STAGES[max(0, min(self.stage, len(STAGES) - 1))]

    def mode_for(self) -> str:
        return self.current()["mode"]

    def record(self, success: bool):
        self.window.append(1.0 if success else 0.0)
        self.updates += 1
        if not self.enabled or len(self.window) < 10:
            return
        rate = sum(self.window) / len(self.window)
        gate = float(self.current()["gate"])
        if rate >= gate and self.stage < len(STAGES) - 1:
            self.stage += 1
            self.promotions += 1
            self.window.clear()
        elif rate < gate - 0.25 and self.stage > 0:
            self.stage -= 1
            self.demotions += 1
            self.window.clear()

    def summary(self) -> dict:
        return {"stage": self.stage, "name": self.current()["name"],
                "updates": self.updates, "promotions": self.promotions,
                "demotions": self.demotions,
                "rate": (sum(self.window) / len(self.window)) if self.window else 0.0,
                "mode": self.mode_for()}
