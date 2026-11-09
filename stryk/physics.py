from __future__ import annotations

import numpy as np

BALL_R = 0.11
BALL_MASS = 0.43
GRAV = 9.81
RHO = 1.225
CD = 0.47
CL = 0.35
_AREA = np.pi * BALL_R * BALL_R


def wind_at(t: float, x: float, y: float, amp: float = 1.0, swirl: float = 1.0) -> np.ndarray:
    s = np.sin(x * 0.045 + t * 0.33) * np.cos(y * 0.06 - t * 0.21)
    edge = min(abs(x) / 52.5, 1.0) * 0.6 + min(abs(y) / 34.0, 1.0) * 0.4
    a = (0.9 + edge * 1.4) * amp * 0.65
    return np.array([s * a * 0.55, (1.0 - s) * a * 0.45, 0.0], dtype=np.float32) * swirl


def aero_acc(vel: np.ndarray, ang: np.ndarray, wind: np.ndarray, drag_mult: float = 1.0) -> np.ndarray:
    rel = vel - wind
    v = float(np.linalg.norm(rel))
    if v < 1e-4:
        return np.zeros((3,), dtype=np.float32)
    k_d = 0.5 * RHO * CD * _AREA / BALL_MASS * drag_mult
    a_drag = rel * (-k_d * v)
    re = v * BALL_R * 2.0 / 1.47e-5
    if 2.0e5 < re < 3.5e5:
        dip = 1.0 - 0.18 * np.sin((re - 2.0e5) / 1.5e5 * np.pi)
        a_drag = a_drag * float(dip)
    k_m = 0.5 * RHO * CL * _AREA / BALL_MASS * drag_mult
    a_mag = np.cross(ang, rel) * k_m
    return (a_drag + a_mag).astype(np.float32)


def spin_decay(ang: np.ndarray, dt: float) -> np.ndarray:
    d = np.exp(-dt * (0.55 + 0.015 * float(np.linalg.norm(ang))))
    return (ang * d).astype(np.float32)


def ground_step(pos: np.ndarray, vel: np.ndarray, ang: np.ndarray, dt: float,
                turf_grip: float, restitution: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.array(pos, dtype=np.float32, copy=True)
    vel = np.array(vel, dtype=np.float32, copy=True)
    ang = np.array(ang, dtype=np.float32, copy=True)
    if pos[2] - BALL_R <= 0:
        pos[2] = BALL_R
        if vel[2] < 0:
            rest = restitution * (0.55 + 0.45 * turf_grip)
            vel[2] = -vel[2] * rest
            if abs(vel[2]) < 0.8:
                vel[2] = 0.0
            keep = 0.62 + 0.30 * turf_grip
            ang[0] *= keep
            ang[1] *= keep
        vx, vy = float(vel[0]), float(vel[1])
        sp = (vx * vx + vy * vy) ** 0.5
        if sp > 1e-4:
            mu = 0.18 + 0.42 * turf_grip
            dec = mu * GRAV * dt
            ns = max(sp - dec, 0.0)
            if ns < 1e-4:
                vel[0] = vel[1] = 0.0
            else:
                f = ns / sp
                vel[0] *= f
                vel[1] *= f
        wt_x, wt_y = vy / BALL_R, -vx / BALL_R
        k = float(np.clip(dt * 6.0, 0.0, 1.0))
        ang[0] += (wt_x - ang[0]) * k
        ang[1] += (wt_y - ang[1]) * k
        if sp < 0.05:
            ang[0] = ang[1] = 0.0
    else:
        if pos[2] > 0.4:
            ang[2] *= 0.99 ** (dt * 60.0)
    return pos, vel, ang
