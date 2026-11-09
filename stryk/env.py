from __future__ import annotations

import numpy as np

from .config import MODES
from .act import decode, ACT_DIM, scripted_chase
from .obs import build_obs, OBS_DIM
from .opponents import scripted_action
from .sim import World


class StrykEnv:
    def __init__(self, mode: str = "5v5", stadium: str = "wembley_fc27", seed: int = 2026,
                 stage: int = 0, opponent_style: str = "balanced", difficulty: float = 0.4,
                 episode_sec: float | None = None, opponent_policy=None):
        self.world = World(mode, stadium, seed)
        self.mode = mode
        self.stage = int(stage)
        self.style = opponent_style
        self.difficulty = float(difficulty)
        self.opponent_policy = opponent_policy  # torch policy or None -> scripted
        self.episode_sec = float(episode_sec if episode_sec is not None else MODES[mode].episode_sec)
        self.steps = 0
        self.rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        self.last_reward = 0.0
        self.home_idx = 0
        self._sync_home()

    def _sync_home(self):
        per = self.world.mode.per_side
        self.home_idx = 0  # controls idx 0; others filled by scripted chase for sparse-stage
        self.home_members = list(range(per))

    def reset(self, seed: int | None = None, stage: int | None = None):
        if stage is not None:
            self.stage = int(stage)
        self.world.reset(seed, True)
        self.steps = 0
        self.last_reward = 0.0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        return build_obs(self.world, self.home_idx, stage=self.stage)

    def _fill_teammates(self, actions: list[dict]):
        for i in self.home_members[1:]:
            a = decode(scripted_chase(self.world, i))
            actions[i] = a
        per = self.world.mode.per_side
        for i in range(per, len(self.world.players)):
            if self.opponent_policy is not None:
                actions[i] = self._opp_policy_action(i)
            else:
                a = decode(scripted_action(self.world, i, self.style, self.difficulty, self.rng))
                actions[i] = a

    def _opp_policy_action(self, i: int) -> dict:
        return decode(scripted_action(self.world, i, self.style, self.difficulty, self.rng))

    def step(self, act: np.ndarray):
        actions = [decode(np.zeros(ACT_DIM, dtype=np.float32)) for _ in self.world.players]
        actions[self.home_idx] = decode(np.asarray(act, dtype=np.float32))
        self._fill_teammates(actions)
        prev_x = float(self.world.ball.pos[0])
        prev_goal_d = self.world.dist_to_goal(self.world.players[self.home_idx].pos, 0)
        prev_score = self.world.score_home - self.world.score_away
        self.world.step(actions)
        self.steps += 1
        new_score = self.world.score_home - self.world.score_away
        reward = self._reward(prev_x, prev_goal_d, prev_score, prev_score != new_score)
        done = self.steps >= int(self.episode_sec * 60.0)
        info = {"score": (self.world.score_home, self.world.score_away),
                "ball_speed": self.world.ball.speed, "kicks": self.world.kicks}
        return self._obs(), float(reward), bool(done), info

    def _reward(self, prev_x: float, prev_d: float, prev_s: int, scored: bool) -> float:
        b = self.world.ball
        gx = self.world.field.half_len
        # dense progress toward attacking goal (home attacks +X)
        prog = float(np.clip((float(b.pos[0]) - (-20.0)) / 72.0, -1.0, 1.0)) * 0.02
        box = 0.01 if (abs(float(b.pos[0]) - gx) < 20.0 and abs(float(b.pos[1])) < 20.0) else 0.0
        height = 0.004 if 0.5 < float(b.pos[2]) < 2.44 else 0.0
        # stage gates: early stages must not reward long shots
        stage_w = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}.get(self.stage, 1.0)
        r = (prog + box + height) * stage_w
        if scored:
            r += 1.0
        # ball out-of-bounds tempo penalty (keeps play alive)
        if abs(float(b.pos[0])) > self.world.field.half_len + 6 and abs(float(b.pos[1])) < 34:
            r -= 0.005
        return float(r)

    @property
    def observation_space_dim(self) -> int:
        return OBS_DIM

    @property
    def action_space_dim(self) -> int:
        return ACT_DIM
