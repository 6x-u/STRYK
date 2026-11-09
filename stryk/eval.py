from __future__ import annotations

import time

import numpy as np
import torch

from .config import TrainCfg
from .env import StrykEnv
from .policy import ActorCritic, np_sample
from .checkpoint import load_checkpoint


def evaluate(checkpoint: str, mode: str, stadium: str, episodes: int = 5, seed: int = 2026) -> dict:
    dev = "cpu"
    policy = ActorCritic().to(dev)
    load_checkpoint(checkpoint, policy, None, device=dev)
    policy.eval()
    wins, goals, rewards = 0, 0, []
    for e in range(episodes):
        env = StrykEnv(mode, stadium, seed + e)
        obs, _ = env.reset()
        done, rew = False, 0.0
        while not done:
            act, _, _, _ = np_sample(policy, obs, dev, deterministic=True)
            obs, r, done, _ = env.step(act.reshape(-1))
            rew += float(r)
        rewards.append(rew)
        goals += int(env.world.score_home + env.world.score_away)
        wins += int(env.world.score_home > env.world.score_away)
    return {"episodes": episodes, "wins": wins, "goals": goals,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0}


def bench(mode: str, stadium: str, seconds: int = 3, seed: int = 2026) -> dict:
    env = StrykEnv(mode, stadium, seed)
    env.reset()
    steps = seconds * 60
    t0 = time.time()
    for _ in range(steps):
        env.step(np.zeros(env.action_space_dim, dtype=np.float32))
    real = time.time() - t0
    return {"mode": mode, "stadium": stadium, "sim_sec": float(steps) / 60.0,
            "real_sec": real, "speedup": (float(steps) / 60.0) / max(real, 1e-6)}
