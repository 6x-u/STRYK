from __future__ import annotations

import numpy as np

from .act import decode, zero_action
from .obs_dynamic import build_entities, MAX_ENTITIES
from .opponents import scripted_action
from .policy_multi import action_mask
from .reward import RewardTracker
from .sim import World


class MultiEnv:
    """All home agents act at once (shared policy); away is league/scripted."""

    def __init__(self, mode: str = "5v5", stadium: str = "wembley_fc27", seed: int = 2026,
                 stage: int = 0, style: str = "balanced", difficulty: float = 0.4,
                 episode_sec: float | None = None):
        self.world = World(mode, stadium, seed)
        self.mode = mode
        self.stage = int(stage)
        self.style = style
        self.difficulty = float(difficulty)
        self.episode_sec = float(episode_sec if episode_sec is not None else self.world.mode.episode_sec)
        self.steps = 0
        self.rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        self.tracker = RewardTracker(home_side=0)
        self.dt = 1.0 / 60.0

    def reset(self, seed: int | None = None):
        self.world.reset(seed, True)
        self.steps = 0
        self.tracker.reset()
        return self.observe()

    def home_ids(self) -> list[int]:
        per = self.world.mode.per_side
        return list(range(per))

    def observe(self) -> dict:
        ids = self.home_ids()
        ents, masks, globs, ams = [], [], [], []
        for i in ids:
            e, m, g = build_entities(self.world, i, stage=self.stage)
            ents.append(e)
            masks.append(m)
            globs.append(g)
            ams.append(action_mask(self.world, i).numpy())
        return {
            "entities": np.stack(ents).astype(np.float32),   # [N, MAX, D]
            "mask": np.stack(masks).astype(np.float32),
            "global": np.stack(globs).astype(np.float32),
            "action_mask": np.stack(ams).astype(np.float32),
        }

    def step(self, cont: np.ndarray, disc: np.ndarray):
        # cont [N,10], disc [N] — decode hybrid into sim dicts
        cont = np.asarray(cont, dtype=np.float32)
        disc = np.asarray(disc).reshape(-1)
        n_total = len(self.world.players)
        actions = [decode(zero_action()) for _ in range(n_total)]
        for k, pid in enumerate(self.home_ids()):
            d = int(disc[k]) if k < len(disc) else 0
            c = np.array(cont[k], dtype=np.float32, copy=True)
            if d == 2:  # pass_short
                c[3] = 1.0
            elif d == 3:  # long/through
                c[3] = 1.0
                c[2] = float(np.clip(c[2], 0.7, 1.0))
            elif d == 4:  # shoot
                c[4] = 1.0
            elif d == 5:  # defend/press
                c[8] = 1.0
                c[9] = 1.0
            elif d == 6:  # skill
                c[7] = 1.0
            elif d == 7:  # support run: push forward off-ball
                c[0] = float(np.clip(c[0] + 0.5, -1, 1))
            actions[pid] = decode(c)
        per = self.world.mode.per_side
        for i in range(per, n_total):
            a = decode(scripted_action(self.world, i, self.style, self.difficulty, self.rng))
            actions[i] = a
        self.world.step(actions)
        self.steps += 1
        broke = self.tracker.update(self.world, self.dt)
        # per-agent reward: shared team reward + tiny carrier bonus so somebody learns to touch it
        team_r = float(broke["total"]) / max(len(self.home_ids()), 1)
        rewards = np.full(len(self.home_ids()), team_r, dtype=np.float32)
        if self.world.ball.last_touch in self.home_ids():
            k = self.home_ids().index(self.world.ball.last_touch)
            rewards[k] += 0.005
        done = self.steps >= int(self.episode_sec * 60.0)
        return self.observe(), rewards, float(done), {
            "score": (self.world.score_home, self.world.score_away),
            "breakdown": broke,
        }
