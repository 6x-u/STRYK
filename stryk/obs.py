from __future__ import annotations

import numpy as np

OBS_DIM = 53
MAX_SORT = 4


def _norm(x: float, s: float) -> float:
    return float(np.clip(x / s, -1.0, 1.0))


def build_obs(world, idx: int, stage: int = 0) -> np.ndarray:
    b = world.ball
    me = world.players[idx]
    o = np.zeros(OBS_DIM, dtype=np.float32)
    o[0:3] = np.array([_norm(b.pos[0], 52.5), _norm(b.pos[1], 34.0), _norm(b.pos[2], 3.0)])
    o[3:6] = np.array([_norm(b.vel[0], 30.0), _norm(b.vel[1], 30.0), _norm(b.vel[2], 30.0)])
    o[6] = float(np.clip(np.linalg.norm(b.ang) / 60.0, 0.0, 1.0))
    o[7:9] = np.array([_norm(me.pos[0], 52.5), _norm(me.pos[1], 34.0)])
    o[9:11] = np.array([_norm(me.vel[0], 9.0), _norm(me.vel[1], 9.0)])
    o[11] = float(me.stamina)
    o[12] = float(np.cos(me.heading))
    o[13] = float(np.sin(me.heading))
    gx = world.field.half_len if me.side == 0 else -world.field.half_len
    tg = world.shoot_target(me.side)
    o[14] = _norm(gx - b.pos[0], 52.5)
    o[15] = _norm(tg[1] - b.pos[1], 34.0)
    mates, foes = [], []
    for j, p in enumerate(world.players):
        if j == idx:
            continue
        d = float(np.hypot(p.pos[0] - b.pos[0], p.pos[1] - b.pos[1]))
        (mates if p.side == me.side else foes).append((d, j, p))
    mates.sort(key=lambda t: t[0])
    foes.sort(key=lambda t: t[0])
    for k, (_, _, p) in enumerate(mates[:MAX_SORT]):
        o[23 + k * 2] = _norm(p.pos[0] - me.pos[0], 52.5)
        o[24 + k * 2] = _norm(p.pos[1] - me.pos[1], 34.0)
    for k, (_, _, p) in enumerate(foes[:MAX_SORT]):
        o[31 + k * 2] = _norm(p.pos[0] - me.pos[0], 52.5)
        o[32 + k * 2] = _norm(p.pos[1] - me.pos[1], 34.0)
    if foes:
        _, _, nf = foes[0]
        o[16] = _norm(nf.pos[0] - b.pos[0], 52.5)
        o[17] = _norm(nf.pos[1] - b.pos[1], 34.0)
        o[18] = _norm(float(np.hypot(nf.pos[0] - me.pos[0], nf.pos[1] - me.pos[1])), 50.0)
        o[44:48] = np.array([_norm(nf.vel[0], 9.0), _norm(nf.vel[1], 9.0),
                             _norm(nf.heading, np.pi), float(nf.stamina)])
    o[19] = float(np.clip(world.tick / 5400.0, 0.0, 1.0))
    o[20] = float(np.clip(np.linalg.norm(b.vel) / 30.0, 0.0, 1.0))
    o[21] = float(np.clip(world.score_home - world.score_away, -11, 11) / 11.0)
    o[22] = float(me.side)
    o[39] = _norm(float(np.hypot(me.pos[0] - b.pos[0], me.pos[1] - b.pos[1])), 50.0)
    o[40] = _norm(world.dist_to_goal(me.pos, me.side), 60.0)
    o[41:44] = np.array([_norm(b.ang[0], 60.0), _norm(b.ang[1], 60.0), _norm(b.ang[2], 60.0)])
    carrier = 1.0 if float(np.hypot(me.pos[0] - b.pos[0], me.pos[1] - b.pos[1])) < 1.2 else 0.0
    o[48] = carrier
    o[49] = float(stage) / 4.0
    o[50:53] = np.array([float(world.field.turf_grip), float(world.field.restitution), carrier])
    return o.astype(np.float32)
