from __future__ import annotations

import numpy as np

from .config import MODES, PITCH_LEN, PITCH_WID, GOAL_WID, GOAL_HGT, BALL_R
from .physics import wind_at, aero_acc, spin_decay, ground_step
from .stadiums import get_stadium
from .tactics import formation
from .utils import seeded_rng


class Ball:
    def __init__(self):
        self.pos = np.array([0.0, 0.0, BALL_R], dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.ang = np.zeros(3, dtype=np.float32)
        self.last_touch = -1
        self.last_touch_t = -1.0

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))

    def kick(self, direction: np.ndarray, power: float, spin: np.ndarray, who: int, t: float):
        d = np.asarray(direction, dtype=np.float32)
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return
        d = d / n
        self.vel = d * float(np.clip(power, 0.0, 42.0))
        self.ang = self.ang * 0.35 + np.asarray(spin, dtype=np.float32)
        self.last_touch = who
        self.last_touch_t = t


class Player:
    __slots__ = ("pos", "vel", "heading", "stamina", "cooldown", "side", "id",
                 "radius", "max_walk", "max_sprint", "accel", "z", "vz")

    def __init__(self, pos, pid: int, side: int):
        self.pos = np.array(pos, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.heading = 3.14159265 if side else 0.0
        self.stamina = 1.0
        self.cooldown = 0.0
        self.side = side
        self.id = pid
        self.radius = 0.28
        self.max_walk = 4.2
        self.max_sprint = 8.4
        self.accel = 11.0
        self.z = 0.0
        self.vz = 0.0


class Field:
    def __init__(self, stadium: str):
        s = get_stadium(stadium)
        self.spec = s
        self.half_len = s.length / 2.0
        self.half_wid = s.width / 2.0
        self.turf_grip = s.turf_grip
        self.restitution = s.restitution

    def goal_scored(self, p: np.ndarray) -> int:
        in_mouth = abs(float(p[1])) <= self.half_wid and (p[2] - BALL_R) < GOAL_HGT and p[2] > 0
        if not in_mouth:
            return 0
        if p[0] + BALL_R < -self.half_len:
            return 1
        if p[0] - BALL_R > self.half_len:
            return 2
        return 0


class World:
    def __init__(self, mode: str = "5v5", stadium: str = "wembley_fc27", seed: int = 2026):
        self.mode = MODES[mode]
        self.field = Field(stadium)
        self.rng = seeded_rng(seed)
        self.ball = Ball()
        self.players: list[Player] = []
        self.tick = 0.0
        self.score_home = 0
        self.score_away = 0
        self.kicks = 0
        self.reset(seed, kickoff=True)

    # ---- setup ----
    def reset(self, seed: int | None = None, kickoff: bool = True):
        if seed is not None:
            self.rng = seeded_rng(seed)
        self.tick = 0.0
        per = self.mode.per_side
        self.players = []
        home_form = formation(self.mode.formation, 0, self.field.half_len * 2)
        away_form = formation(self.mode.formation, 1, self.field.half_len * 2)
        for i in range(per):
            self.players.append(Player(home_form[i], i, 0))
        for i in range(per):
            self.players.append(Player(away_form[i], per + i, 1))
        self.ball = Ball()
        if kickoff:
            idx = int(self.rng.integers(0, per))
            self.players[idx].pos[:] = np.array([-0.9, 0.0], dtype=np.float32)
            self.ball.pos[:] = np.array([-0.15, 0.0, BALL_R], dtype=np.float32)

    # ---- queries ----
    def closest_player(self, p: np.ndarray) -> int:
        best, bd = -1, 1e18
        for i, pl in enumerate(self.players):
            d = float(np.sum((pl.pos - p[:2]) ** 2))
            if d < bd:
                bd, best = d, i
        return best

    def dist_to_goal(self, p: np.ndarray, side: int) -> float:
        gx = self.field.half_len if side == 0 else -self.field.half_len
        return float(abs(float(p[0]) - gx))

    def shoot_target(self, side: int) -> np.ndarray:
        gx = self.field.half_len if side == 0 else -self.field.half_len
        top = self.field.half_wid * 0.42 * (-1.0 if self.ball.pos[1] > 0 else 1.0)
        return np.array([gx, float(top)], dtype=np.float32)

    def try_strike(self, idx: int, direction: np.ndarray, power: float, spin: np.ndarray) -> bool:
        pl = self.players[idx]
        if pl.cooldown > 0:
            return False
        d = self.ball.pos[:2] - pl.pos
        if float(np.linalg.norm(d)) > 1.55 or abs(self.ball.pos[2] - BALL_R) > 1.9:
            return False
        w = 0.72 + 0.28 * pl.stamina
        self.ball.kick(direction, power * w, spin, idx, self.tick)
        pl.cooldown = 0.22
        self.kicks += 1
        return True

    # ---- stepping ----
    def _drive(self, pl: Player, wish_dir: np.ndarray, wish_speed: float, sprint: bool, dt: float):
        want = float(np.clip(wish_speed, 0.0, pl.max_sprint if sprint else pl.max_walk))
        if sprint and pl.stamina < 0.12:
            want = pl.max_walk
        wd = np.asarray(wish_dir, dtype=np.float32)
        wl = float(np.linalg.norm(wd))
        if wl > 1.0:
            wd = wd / wl
        elif wl > 1e-4:
            wd = wd / wl
        else:
            wd = np.zeros(2, dtype=np.float32)
        spd = float(np.linalg.norm(pl.vel))
        turn_max = 4.5 * (1.0 - 0.55 * min(spd / pl.max_sprint, 1.0))
        if wl > 0.05:
            from .utils import ang_diff
            want_ang = float(np.arctan2(float(wd[1]), float(wd[0])))
            d = ang_diff(want_ang, pl.heading)
            max_turn = turn_max * dt * (0.45 + 0.55 * wl)
            pl.heading += float(np.clip(d, -max_turn, max_turn))
        want_vel = np.array([np.cos(pl.heading) * want * wl, np.sin(pl.heading) * want * wl], dtype=np.float32)
        dv = want_vel - pl.vel
        dv_len = float(np.linalg.norm(dv))
        max_dv = pl.accel * dt * (0.72 if sprint else 1.0) * (0.75 + 0.25 * pl.stamina)
        if dv_len > 1e-4:
            f = min(1.0, max_dv / dv_len)
            pl.vel = pl.vel + dv * f

    def _step_phys(self, dt: float):
        self.tick += dt
        w = wind_at(self.tick, float(self.ball.pos[0]), float(self.ball.pos[1]),
                    amp=self.field.spec.wind_amp, swirl=self.field.spec.bowl_swirl)
        acc = aero_acc(self.ball.vel, self.ball.ang, w, drag_mult=self.field.spec.drag_mult)
        self.ball.vel = (self.ball.vel + acc * dt).astype(np.float32)
        self.ball.vel[2] -= 9.81 * dt
        self.ball.pos = (self.ball.pos + self.ball.vel * dt).astype(np.float32)
        self.ball.ang = spin_decay(self.ball.ang, dt)
        self.ball.pos, self.ball.vel, self.ball.ang = ground_step(
            self.ball.pos, self.ball.vel, self.ball.ang, dt,
            self.field.turf_grip, self.field.restitution)
        # posts / crossbar push
        self._posts()
        # players: integrate + separate
        for i, pl in enumerate(self.players):
            pl.pos = (pl.pos + pl.vel * dt * (0.65 + 0.35 * self.field.turf_grip)).astype(np.float32)
            if float(np.linalg.norm(pl.vel)) > 8.0:
                pl.stamina = float(np.clip(pl.stamina - dt * 0.045, 0, 1))
            elif float(np.linalg.norm(pl.vel)) > 2.0:
                pl.stamina = float(np.clip(pl.stamina - dt * 0.012, 0, 1))
            else:
                pl.stamina = float(np.clip(pl.stamina + dt * 0.028, 0, 1))
            pl.cooldown = max(0.0, pl.cooldown - dt)
        for i in range(len(self.players)):
            for j in range(i + 1, len(self.players)):
                a, b = self.players[i], self.players[j]
                d = b.pos - a.pos
                dd = float(np.linalg.norm(d))
                rr = a.radius + b.radius
                if 0 < dd < rr:
                    n = d / dd
                    push = (rr - dd) * 0.5
                    a.pos = (a.pos - n * push).astype(np.float32)
                    b.pos = (b.pos + n * push).astype(np.float32)
        self.ball.pos[0] = float(np.clip(self.ball.pos[0], -self.field.half_len - 14, self.field.half_len + 14))
        self.ball.pos[1] = float(np.clip(self.ball.pos[1], -self.field.half_wid - 14, self.field.half_wid + 14))
        self.ball.pos[2] = max(float(self.ball.pos[2]), BALL_R)
        g = self.field.goal_scored(self.ball.pos)
        if g == 1:
            self.score_away += 1
            self.reset(None, True)
        elif g == 2:
            self.score_home += 1
            self.reset(None, True)

    def _posts(self):
        gh = GOAL_WID / 2.0
        rr = BALL_R + 0.06
        for side in (0, 1):
            gx = -self.field.half_len if side == 0 else self.field.half_len
            for gy in (-gh, gh):
                dx = float(self.ball.pos[0]) - gx
                dy = float(self.ball.pos[1]) - gy
                if abs(dx) < rr and abs(dy) < rr and float(self.ball.pos[2]) < GOAL_HGT + 0.1:
                    d = (dx * dx + dy * dy) ** 0.5
                    if d < rr and d > 1e-5:
                        nx, ny = dx / d, dy / d
                        self.ball.pos[0] = float(gx + nx * rr)
                        self.ball.pos[1] = float(gy + ny * rr)
                        vn = float(self.ball.vel[0] * nx + self.ball.vel[1] * ny)
                        if vn < 0:
                            self.ball.vel[0] -= 1.55 * vn * nx
                            self.ball.vel[1] -= 1.55 * vn * ny
                            self.ball.vel[:2] *= 0.82
                            self.ball.vel[2] *= 0.9

    def step(self, actions: list[dict]):
        """One env step (1/60 s = four 240 Hz substeps)."""
        sub = 1.0 / self.mode.phys_hz
        for _ in range(4):
            for pl, a in zip(self.players, actions):
                ws = pl.max_sprint if a.get("sprint", False) else pl.max_walk
                self._drive(pl, a.get("move", np.zeros(2)), ws, bool(a.get("sprint", False)), sub)
            for i, a in enumerate(actions):
                pl = self.players[i]
                if a.get("shoot", False) and pl.cooldown <= 0:
                    tg = self.shoot_target(pl.side)
                    d2 = np.array([tg[0] - self.ball.pos[0], tg[1] - self.ball.pos[1],
                                   float(a.get("shoot_lift", 0.0)) * 0.35 + 0.12], dtype=np.float32)
                    pw = 10.0 + float(a.get("shoot_power", 0.5)) * 22.0
                    spin = np.array([-d2[1] * 18.0 * float(a.get("shoot_lift", 0.0)),
                                     d2[0] * 18.0 * float(a.get("shoot_lift", 0.0)),
                                     d2[0] * 4.0], dtype=np.float32)
                    self.try_strike(i, d2, pw, spin)
                elif a.get("skill", False) and pl.cooldown <= 0:
                    dirv = np.array([np.cos(pl.heading), np.sin(pl.heading), 0.05], dtype=np.float32)
                    spin = np.array([-dirv[1] * 22.0, dirv[0] * 22.0, 0.0], dtype=np.float32)
                    self.try_strike(i, dirv, 6.5, spin)
            self._step_phys(sub)
