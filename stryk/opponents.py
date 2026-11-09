from __future__ import annotations

import numpy as np

from .act import zero_action, ACT_DIM
from .tactics import best_pass_target, press_target

STYLES = ("balanced", "press", "counter", "possession", "longball")


def style_params(style: str, difficulty: float) -> dict:
    d = float(np.clip(difficulty, 0.0, 1.0))
    base = {
        "balanced": {"press_burst": 0.5, "pass_fwd": 0.4, "skill": 0.05, "speed": 0.75},
        "press": {"press_burst": 1.0, "pass_fwd": 0.35, "skill": 0.1, "speed": 0.9},
        "counter": {"press_burst": 0.25, "pass_fwd": 0.9, "skill": 0.08, "speed": 0.85},
        "possession": {"press_burst": 0.45, "pass_fwd": 0.25, "skill": 0.2, "speed": 0.7},
        "longball": {"press_burst": 0.3, "pass_fwd": 0.7, "skill": 0.03, "speed": 0.8},
    }[style]
    base["speed"] = base["speed"] * (0.7 + 0.4 * d)
    return base


def scripted_action(world, idx: int, style: str, difficulty: float, rng: np.random.Generator) -> np.ndarray:
    """Style-driven scripted foe (pre-policy baseline and league filler)."""
    prm = style_params(style, difficulty)
    me = world.players[idx]
    b = world.ball.pos
    a = zero_action()
    tgt_player = press_target(world, idx)
    if tgt_player is not None and rng.random() < prm["press_burst"]:
        foe_pos = world.players[tgt_player].pos
        d = foe_pos - me.pos
    else:
        d = b[:2] - me.pos
    n = float(np.linalg.norm(d))
    u = d / n if n > 1e-4 else np.zeros(2, dtype=np.float32)
    a[0], a[1] = u[0] * prm["speed"] + rng.normal(0, 0.05), u[1] * prm["speed"] + rng.normal(0, 0.05)
    a[2] = 1.0 if n > 6.0 else -1.0
    behind = b[:2] - (world.shoot_target(me.side) - b[:2]) * 0.28
    if n > 5.5:
        w = behind - me.pos
        wn = float(np.linalg.norm(w))
        if wn > 1e-4:
            a[0], a[1] = (w / wn)[0], (w / wn)[1]
    if n < 2.2:
        tg = world.shoot_target(me.side)
        to_g = tg - b[:2]
        from_me = b[:2] - me.pos
        cos = float(np.dot(from_me, to_g) / (np.linalg.norm(from_me) * np.linalg.norm(to_g) + 1e-6))
        if cos > 0.7 and rng.random() < 0.6 + 0.4 * difficulty:
            a[4] = 1.0
            a[5] = float(np.clip(np.linalg.norm(to_g) / 22.0, -1, 1))
        elif rng.random() < prm["skill"]:
            a[7] = 1.0
    # passing weight pushes along forward axis
    a[0] += float(rng.normal(0, 0.03))
    return np.clip(a, -1, 1).astype(np.float32)


_OPPONENT_REGISTRY: dict[str, object] = {}


def register_opponent_style(name: str, fn):
    _OPPONENT_REGISTRY[name] = fn
    return fn
