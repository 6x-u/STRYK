from __future__ import annotations

import numpy as np

from .physics import wind_at, aero_acc

# entity vector — keep it flat, the encoder does the heavy lifting :)
# [x,y,z, vx,vy,vz, stamina, cos_h, sin_h, is_self, type_id, side,
#  dist_ball, dist_goal, reach, role]
ENT_DIM = 16
# global vector: ball future + team shape + phase. critic-only, actor never sees it.
GLOBAL_DIM = 24
MAX_ENTITIES = 24  # 1 ball + 22 players + 1 pad slot; 1v1 pads the rest
ENTITY_TYPES = {"ball": 0, "self": 1, "mate": 2, "foe": 3}
OBS_VERSION = 2


def _n(x: float, s: float) -> float:
    return float(np.clip(x / s, -1.0, 1.0))


def predict_ball(world, times=(0.5, 1.0, 2.0)) -> np.ndarray:
    # dumb rollout with real aero — drag matters more than the integrators :)
    pos = np.array(world.ball.pos, dtype=np.float32)
    vel = np.array(world.ball.vel, dtype=np.float32)
    ang = np.array(world.ball.ang, dtype=np.float32)
    dt = 0.05
    out = []
    t = 0.0
    idx = 0
    targets = sorted(times)
    acc_t = 0.0
    for k in range(int(max(times) / dt) + 1):
        if idx < len(targets) and acc_t >= targets[idx] - 1e-6:
            out.append(np.array([pos[0], pos[1], max(float(pos[2]), 0.0)], dtype=np.float32))
            idx += 1
        w = wind_at(float(world.tick + acc_t), float(pos[0]), float(pos[1]),
                    amp=float(world.field.spec.wind_amp), swirl=float(world.field.spec.bowl_swirl))
        vel = (vel + aero_acc(vel, ang, w, float(world.field.spec.drag_mult)) * dt).astype(np.float32)
        vel[2] -= 9.81 * dt
        pos = (pos + vel * dt).astype(np.float32)
        if pos[2] < 0.11:
            pos[2] = 0.11
            vel[2] = -vel[2] * 0.5
        acc_t += dt
        t += dt
    while len(out) < len(targets):
        out.append(pos.copy())
    return np.stack(out).astype(np.float32)  # [3,3]


def build_entities(world, idx: int, stage: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (entities [MAX, ENT_DIM], mask [MAX], global [GLOBAL_DIM])."""
    me = world.players[idx]
    b = world.ball
    ent = np.zeros((MAX_ENTITIES, ENT_DIM), dtype=np.float32)
    mask = np.zeros((MAX_ENTITIES,), dtype=np.float32)
    gx = world.field.half_len if me.side == 0 else -world.field.half_len
    # ball entity (A: ball intelligence)
    ent[0] = np.array([
        _n(b.pos[0], 52.5), _n(b.pos[1], 34.0), _n(b.pos[2], 3.0),
        _n(b.vel[0], 30.0), _n(b.vel[1], 30.0), _n(b.vel[2], 30.0),
        float(np.clip(np.linalg.norm(b.ang) / 60.0, 0, 1)), 1.0, 0.0,
        0.0, ENTITY_TYPES["ball"] / 3.0, float(b.pos[2] > 0.3),
        0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float32)
    mask[0] = 1.0
    # players (B: player intelligence)
    cursor = 1
    for j, p in enumerate(world.players):
        if cursor >= MAX_ENTITIES:
            break
        is_self = 1.0 if j == idx else 0.0
        typ = ENTITY_TYPES["self"] if j == idx else (ENTITY_TYPES["mate"] if p.side == me.side else ENTITY_TYPES["foe"])
        role = float(j % 11) / 10.0
        d_ball = float(np.hypot(p.pos[0] - b.pos[0], p.pos[1] - b.pos[1]))
        d_goal = float(abs(float(p.pos[0]) - gx))
        speed = float(np.linalg.norm(p.vel))
        reach = float(np.clip(1.0 - d_ball / (6.0 + speed * 1.2), 0.0, 1.0))
        ent[cursor] = np.array([
            _n(p.pos[0], 52.5), _n(p.pos[1], 34.0), 0.0,
            _n(p.vel[0], 9.0), _n(p.vel[1], 9.0), 0.0,
            float(p.stamina), float(np.cos(p.heading)), float(np.sin(p.heading)),
            is_self, float(typ) / 3.0, float(p.side),
            _n(d_ball, 50.0), _n(d_goal, 60.0), reach, role,
        ], dtype=np.float32)
        mask[cursor] = 1.0
        cursor += 1
    # global (C + D: team + tactical) — critic only
    future = predict_ball(world)  # [3,3] xyz at 0.5/1/2 s
    mates = [p for q, p in enumerate(world.players) if p.side == me.side and q != idx]
    foes = [p for p in world.players if p.side != me.side]
    # open space: centroid of mates behind ball line
    if mates:
        cx = float(np.mean([p.pos[0] for p in mates]))
        cy = float(np.mean([p.pos[1] for p in mates]))
        spread = float(np.mean([float(np.hypot(p.pos[0] - cx, p.pos[1] - cy)) for p in mates]))
    else:
        cx, cy, spread = 0.0, 0.0, 0.0
    # shooting angle from ball to goal mouth
    mouth_l = np.array([gx, -3.66], dtype=np.float32)
    mouth_r = np.array([gx, 3.66], dtype=np.float32)
    bx = np.array([b.pos[0], b.pos[1]], dtype=np.float32)
    v1 = mouth_l - bx
    v2 = mouth_r - bx
    cos_ang = float(np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6))
    shoot_ang = float(np.arccos(np.clip(cos_ang, -1, 1)))
    # best receiver = max lane + forward (cheap reuse of tactics)
    try:
        from .tactics import best_pass_target
        br = best_pass_target(world, idx)
        best_rx = np.array([_n(br[0], 52.5), _n(br[1], 34.0)], dtype=np.float32)
    except Exception:
        best_rx = np.zeros(2, dtype=np.float32)
    last_side = -1.0
    if b.last_touch >= 0 and b.last_touch < len(world.players):
        last_side = float(world.players[b.last_touch].side == me.side)
    g = np.zeros(GLOBAL_DIM, dtype=np.float32)
    g[0:3] = np.array([_n(future[0, 0], 52.5), _n(future[0, 1], 34.0), _n(future[0, 2], 3.0)])
    g[3:6] = np.array([_n(future[1, 0], 52.5), _n(future[1, 1], 34.0), _n(future[1, 2], 3.0)])
    g[6:9] = np.array([_n(future[2, 0], 52.5), _n(future[2, 1], 34.0), _n(future[2, 2], 3.0)])
    g[9] = float(np.clip(np.linalg.norm(b.vel) / 30.0, 0, 1))
    g[10] = float(b.last_touch >= 0 and world.players[b.last_touch].side == me.side)
    g[11] = last_side
    g[12] = _n(cx, 52.5)
    g[13] = _n(cy, 34.0)
    g[14] = float(np.clip(spread / 20.0, 0, 1))
    g[15] = float(len(mates)) / 11.0
    g[16] = float(len(foes)) / 11.0
    g[17] = float(shoot_ang / np.pi)
    g[18:20] = best_rx
    g[20] = _n(gx - b.pos[0], 52.5)
    g[21] = float(np.clip(world.tick / 5400.0, 0, 1))
    g[22] = float(np.clip((world.score_home - world.score_away if me.side == 0 else world.score_away - world.score_home), -11, 11) / 11.0)
    g[23] = float(stage) / 5.0
    return ent, mask, g.astype(np.float32)


def dynamic_obs_dict(world, idx: int, stage: int = 0) -> dict:
    e, m, g = build_entities(world, idx, stage=stage)
    return {"entities": e, "mask": m, "global": g, "obs_version": OBS_VERSION}
