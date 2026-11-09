from __future__ import annotations

import numpy as np

FORMATIONS = {
    "4-3-3": [(-38, -15), (-22, 0), (0, -18), (0, 18), (12, -10), (12, 0), (12, 10),
              (28, -12), (28, 0), (28, 12), (40, 0)],
    "3-2-1": [(-30, 0), (-12, -10), (-12, 10), (0, 0), (8, -8), (8, 8), (22, 0)],
    "2-1-1": [(-25, 0), (-5, -12), (-5, 12), (10, 0), (22, 0)],
    "1-1-1": [(-20, 0), (0, 0), (18, 0)],
    "1-1": [(-15, 0), (15, 0)],
    "1": [(0, 0)],
}


def formation(name: str, side: int, stadium_len: float = 105.0) -> list[np.ndarray]:
    pts = FORMATIONS.get(name, FORMATIONS["4-3-3"])
    s = 1.0 if side == 0 else -1.0
    return [np.array([p[0] * s, p[1]], dtype=np.float32) for p in pts]


def pass_lane_score(ball: np.ndarray, start: np.ndarray, end: np.ndarray,
                   foes: list[np.ndarray], width: float = 2.2) -> float:
    # higher = cleaner lane — cheap perp scan, good enough for shaping :)
    seg = end - start
    L = float(np.linalg.norm(seg))
    if L < 1e-4:
        return 0.0
    u = seg / L
    score = 1.0
    for f in foes:
        rel = f - start
        along = float(np.dot(rel, u))
        perp = float(np.linalg.norm(rel - u * along))
        if 0.0 < along < L and perp < width:
            score *= 1.0 - max(0.0, 1.0 - perp / width) * 0.7
    return float(np.clip(score, 0.0, 1.0))


def best_pass_target(world, idx: int) -> np.ndarray:
    me = world.players[idx]
    best, bs = me.pos.copy(), -1e9
    foes = [p.pos for p in world.players if p.side != me.side]
    for j, p in enumerate(world.players):
        if j == idx or p.side != me.side:
            continue
        lane = pass_lane_score(world.ball.pos[:2], me.pos, p.pos, foes)
        fwd = float(p.pos[0] - me.pos[0]) * (1.0 if me.side == 0 else -1.0)
        s = lane * 10.0 + fwd * 0.05
        if s > bs:
            bs, best = s, p.pos.copy()
    return best


def press_target(world, idx: int) -> int | None:
    me = world.players[idx]
    bx, by = float(world.ball.pos[0]), float(world.ball.pos[1])
    myd = float(np.hypot(me.pos[0] - bx, me.pos[1] - by))
    for j, p in enumerate(world.players):
        if p.side == me.side:
            d = float(np.hypot(p.pos[0] - bx, p.pos[1] - by))
            if d < myd - 0.05:
                return None
    best, bd = None, 1e9
    for j, p in enumerate(world.players):
        if p.side != me.side:
            d = float(np.hypot(p.pos[0] - bx, p.pos[1] - by))
            if d < bd:
                bd, best = d, j
    return best
