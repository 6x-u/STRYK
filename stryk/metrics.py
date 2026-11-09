from __future__ import annotations

import numpy as np


class EpisodeStats:
    """Per-episode ledger. Win rate alone never tells the story :)"""

    def __init__(self):
        self.goals_for = 0
        self.goals_against = 0
        self.passes_tried = 0
        self.passes_done = 0
        self.shots = 0
        self.on_target = 0
        self.poss_home = 0.0
        self.poss_away = 0.0
        self.turnovers = 0
        self.def_wins = 0
        self.def_chances = 0
        self.collisions = 0  # teammates stacking
        self.episodes = 0

    def add_episode(self, world, tracker) -> dict:
        self.goals_for += int(world.score_home)
        self.goals_against += int(world.score_away)
        s = tracker.summary() if tracker is not None else {}
        self.passes_tried += int(s.get("passes", 0))
        self.passes_done += int(s.get("passes", 0) * s.get("pass_completion", 0.0))
        self.shots += int(s.get("shots", 0))
        self.on_target += int(s.get("shots", 0) * s.get("shot_accuracy", 0.0))
        self.poss_home += float(s.get("possession_pct", 0.0))
        self.turnovers += int(s.get("turnovers", 0))
        self.def_wins += int(s.get("def_wins", 0))
        self.episodes += 1
        return self.summary(world)

    def summary(self, world=None) -> dict:
        ep = max(self.episodes, 1)
        pt = max(self.passes_tried, 1)
        st = max(self.shots, 1)
        return {
            "win_rate": 1.0 if (world is not None and world.score_home > world.score_away and self.episodes) else (self.goals_for / max(self.goals_for + self.goals_against, 1)),
            "goal_diff": (self.goals_for - self.goals_against) / ep,
            "goals_for": self.goals_for / ep,
            "goals_against": self.goals_against / ep,
            "pass_completion": self.passes_done / pt,
            "shot_accuracy": self.on_target / st,
            "possession_pct": (self.poss_home / ep) if self.episodes else 0.0,
            "defensive_success": self.def_wins / max(self.def_chances, self.def_wins, 1),
            "turnovers_per_ep": self.turnovers / ep,
            "coordination": 1.0 - min(self.collisions / max(ep, 1) / 10.0, 1.0),
            "episodes": self.episodes,
        }
