from __future__ import annotations

import numpy as np


class RewardTracker:
    """5-pillar shaping. Every pillar pays on outcomes, not on wagging the stick :)

    scoring / possession / passing / positioning / defense
    + penalties for gifting, brain-dead losses, shape collapse, crowding.
    """

    def __init__(self, home_side: int = 0, w_scoring=1.0, w_poss=0.15, w_pass=0.10,
                 w_pos=0.05, w_def=0.10):
        self.home = home_side
        self.w = {"scoring": w_scoring, "possession": w_poss, "passing": w_pass,
                  "positioning": w_pos, "defense": w_def}
        self.reset()

    def reset(self):
        self.poss_time = 0.0
        self.poss_total = 0.0
        self.passes_tried = 0
        self.passes_done = 0
        self.shots = 0
        self.on_target = 0
        self.turnovers = 0
        self.def_wins = 0
        self.def_chances = 0
        self._last_side = None
        self._last_touch_idx = -1
        self._last_kick_t = -99.0
        self._last_kick_side = None
        self._last_kick_idx = -1
        self._goal_check = (0, 0)

    def update(self, world, dt: float) -> dict:
        # --- detect kicks / touches from Ball.last_touch
        b = world.ball
        broke = {"scoring": 0.0, "possession": 0.0, "passing": 0.0,
                 "positioning": 0.0, "defense": 0.0, "penalty": 0.0, "total": 0.0}
        prev_h, prev_a = self._goal_check
        dh, da = world.score_home - prev_h, world.score_away - prev_a
        self._goal_check = (world.score_home, world.score_away)
        # goal scored — only praise the attacking side, no free lunch for the loser
        if dh > 0:
            broke["scoring"] += self.w["scoring"] * 1.0 if self.home == 0 else -1.0
        if da > 0:
            broke["scoring"] += self.w["scoring"] * 1.0 if self.home == 1 else -1.0
        # new touch = someone did something
        if b.last_touch != self._last_touch_idx and b.last_touch >= 0:
            side = world.players[b.last_touch].side
            if self._last_kick_side is not None:
                # previous kicker -> this toucher: classify the pass
                if side == self._last_kick_side:
                    if self._last_kick_side == self.home:
                        self.passes_done += 1
                        broke["passing"] += self.w["passing"] * 0.05
                    else:
                        broke["defense"] += self.w["defense"] * 0.02  # let them pass, whatever
                else:
                    # interception / turnover
                    if self._last_kick_side == self.home:
                        self.turnovers += 1
                        broke["penalty"] -= 0.08  # gifted it away, ouch
                    else:
                        self.def_wins += 1
                        broke["defense"] += self.w["defense"] * 0.08
            # shot heuristic: last kick was powerful + forward + inside box zone
            if self._last_kick_side is not None and side == self._last_kick_side:
                gap = float(world.tick - self._last_kick_t)
                if gap > 0.25 and float(np.linalg.norm(b.vel)) > 8.0:
                    self.shots += 1
                    gx = world.field.half_len if side == 0 else -world.field.half_len
                    near = abs(float(b.pos[0]) - gx) < 20 and abs(float(b.pos[1])) < 20
                    if near:
                        self.on_target += 1
            self._last_touch_idx = int(b.last_touch)
        # possession timer — hold the ball, get paid a little, slowly
        cur_side = world.players[b.last_touch].side if b.last_touch >= 0 else self._last_side
        if cur_side is not None:
            if cur_side == self.home:
                self.poss_time += dt
                broke["possession"] += self.w["possession"] * dt * 0.05
            if self._last_side is not None and cur_side != self._last_side:
                if cur_side != self.home:
                    broke["penalty"] -= 0.02  # ball flipped sides
            self._last_side = cur_side
        self.poss_total += dt
        # track kick origin for next transition
        kicks = getattr(world, "kicks", 0)
        if kicks != getattr(self, "_kicks_seen", 0):
            last = b.last_touch
            if last >= 0:
                self._last_kick_idx = int(last)
                self._last_kick_side = int(world.players[last].side)
                self._last_kick_t = float(world.tick)
                if self._last_kick_side == self.home:
                    self.passes_tried += 1
            self._kicks_seen = kicks
        # positioning: keep shape sane + don't crowd the carrier
        if world.players:
            mates = [p for p in world.players if p.side == self.home]
            if len(mates) >= 2:
                cx = float(np.mean([p.pos[0] for p in mates]))
                cy = float(np.mean([p.pos[1] for p in mates]))
                spread = float(np.mean([float(np.hypot(p.pos[0] - cx, p.pos[1] - cy)) for p in mates]))
                if 4.0 < spread < 22.0:
                    broke["positioning"] += self.w["positioning"] * 0.01
                else:
                    broke["penalty"] -= 0.02  # too tight or lost shape
            # crowding penalty: too many bodies stacked on the carrier
            if b.last_touch >= 0:
                cp = world.players[b.last_touch].pos
                close = sum(1 for p in mates if float(np.hypot(p.pos[0] - cp[0], p.pos[1] - cp[1])) < 2.0)
                if close > 3:
                    broke["penalty"] -= 0.015
        broke["total"] = sum(v for k, v in broke.items() if k != "total")
        return broke

    def summary(self) -> dict:
        pt = max(self.passes_tried, 1)
        st = max(self.shots, 1)
        return {
            "pass_completion": self.passes_done / pt,
            "shot_accuracy": self.on_target / st,
            "possession_pct": self.poss_time / max(self.poss_total, 1e-6),
            "turnovers": self.turnovers,
            "def_wins": self.def_wins,
            "passes": self.passes_tried,
            "shots": self.shots,
        }
