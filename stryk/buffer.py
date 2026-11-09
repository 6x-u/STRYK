from __future__ import annotations

import numpy as np


class RolloutBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.logp = np.zeros((size,), dtype=np.float32)
        self.rew = np.zeros((size,), dtype=np.float32)
        self.done = np.zeros((size,), dtype=np.float32)
        self.val = np.zeros((size,), dtype=np.float32)
        self.adv = np.zeros((size,), dtype=np.float32)
        self.ret = np.zeros((size,), dtype=np.float32)
        self.size = int(size)
        self.ptr = 0

    def add(self, obs, act, logp, rew, done, val):
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.logp[i] = float(logp)
        self.rew[i] = float(rew)
        self.done[i] = float(done)
        self.val[i] = float(val)
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.size

    def compute_gae(self, last_val: float, gamma: float = 0.99, lam: float = 0.95):
        adv = 0.0
        for t in reversed(range(self.ptr)):
            nonterm = 1.0 - self.done[t]
            next_nonterm = next_nonterm_placeholder(t, self.ptr, self.done)
            nextv = last_val if t == self.ptr - 1 else self.val[t + 1]
            delta = self.rew[t] + gamma * nextv * nonterm - self.val[t]
            adv = delta + gamma * lam * nonterm * adv
            self.adv[t] = adv
            self.ret[t] = adv + self.val[t]
        self.adv = (self.adv - self.adv.mean()) / (self.adv.std() + 1e-8)

    def flat(self) -> dict:
        n = self.ptr
        return {"obs": self.obs[:n], "act": self.act[:n], "logp": self.logp[:n],
                "ret": self.ret[:n], "adv": self.adv[:n]}

    def reset(self):
        self.ptr = 0


def next_nonterm_placeholder(t: int, ptr: int, done: np.ndarray) -> float:
    return 1.0 - done[t] if t + 1 < ptr else 1.0
