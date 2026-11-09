from __future__ import annotations

import numpy as np

# 10 continuous dims, all in [-1,1] except gates mapped through sigmoid-like thresholds.
ACT_DIM = 10
ACT_NAMES = ("mx", "my", "sprint", "pass", "shoot", "shoot_power", "shoot_lift", "skill", "tackle", "press")


def decode(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if a.shape[0] < ACT_DIM:
        a = np.pad(a, (0, ACT_DIM - a.shape[0]))
    move = np.array([np.clip(a[0], -1, 1), np.clip(a[1], -1, 1)], dtype=np.float32)
    return {
        "move": move,
        "sprint": bool(a[2] > 0.4),
        "pass": bool(a[3] > 0.55),
        "shoot": bool(a[4] > 0.55),
        "shoot_power": float(np.clip(a[5], -1, 1)) * 0.5 + 0.5,
        "shoot_lift": float(np.clip(a[6], -1, 1)),
        "skill": bool(a[7] > 0.6),
        "tackle": bool(a[8] > 0.6),
        "press": bool(a[9] > 0.2),
    }


def zero_action() -> np.ndarray:
    return np.zeros(ACT_DIM, dtype=np.float32)


def scripted_chase(world, idx: int) -> np.ndarray:
    # chaser for sanity checks before the policy knows anything :)
    me = world.players[idx]
    b = world.ball.pos
    d = b[:2] - me.pos
    n = float(np.linalg.norm(d))
    u = d / n if n > 1e-4 else np.zeros(2, dtype=np.float32)
    a = zero_action()
    a[0], a[1] = u[0], u[1]
    a[2] = 1.0 if n > 7.0 else -1.0
    if n < 2.2:
        tg = world.shoot_target(me.side)
        to_g = tg - b[:2]
        from_me = b[:2] - me.pos
        cos = float(np.dot(from_me, to_g) / (np.linalg.norm(from_me) * np.linalg.norm(to_g) + 1e-6))
        if cos > 0.7:
            a[4] = 1.0
            a[5] = float(np.clip(np.linalg.norm(to_g) / 22.0, -1, 1))
    return a
