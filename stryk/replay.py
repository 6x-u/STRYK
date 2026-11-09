from __future__ import annotations

from collections import deque

import numpy as np


class ReplayStore:
    """Ring of raw transitions for eyeballing. trim it or ram fills up fast :)"""

    def __init__(self, capacity: int = 20000):
        self.buf: deque = deque(maxlen=int(capacity))
        self.frames = 0

    def push(self, step: dict):
        self.buf.append(step)
        self.frames += 1

    def snapshot(self, max_items: int = 512) -> list[dict]:
        return list(self.buf)[-int(max_items):]

    def save_npz(self, path: str):
        items = self.snapshot(1024)
        if not items:
            return path
        ent = np.stack([x["entities"] for x in items])
        act = np.stack([x["actions"] for x in items]) if "actions" in items[0] else np.zeros((len(items), 10), dtype=np.float32)
        rew = np.array([x["reward"] for x in items], dtype=np.float32)
        np.savez_compressed(path, entities=ent, actions=act, rewards=rew)
        return path
