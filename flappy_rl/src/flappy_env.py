"""
Flappy Bird Environment modeled as a Markov Decision Process (MDP).
Headless (no display) for fast RL training.
"""

import numpy as np
import random

# ── Game constants ──────────────────────────────────────────────────────────
SCREEN_W = 288
SCREEN_H = 512

GRAVITY = 1.0
FLAP_V = -9.0
MAX_V = 10.0

PIPE_VX = -4.0
PIPE_GAP = 150          # vertical gap size between top and bottom pipe
PIPE_SPACING = 220      # horizontal distance between pipe pairs
MAX_GAP_JUMP = 120      # max vertical change from one pipe gap to the next

PIPE_W = 52
BIRD_X = 60
BIRD_W = 34
BIRD_H = 24
PIPE_SPAWN_X = SCREEN_W


class FlappyBirdEnv:
    """
    Headless Flappy Bird environment.

    State:
        dx - horizontal distance to next pipe
        dy - vertical distance from bird to gap center
        vy - bird vertical velocity

    Actions:
        0 - do nothing
        1 - flap

    Reward:
        +1   for surviving each frame
        +5   for passing a pipe
        -100 for collision
    """

    DX_BINS = np.linspace(0, SCREEN_W, 20)
    DY_BINS = np.linspace(-SCREEN_H // 2, SCREEN_H // 2, 20)
    VY_BINS = np.linspace(-15, 15, 10)

    def __init__(self, pipe_gap: int = PIPE_GAP, pipe_speed: float = PIPE_VX):
        self.pipe_gap = pipe_gap
        self.pipe_speed = pipe_speed
        self.reset()

    def reset(self):
        self.bird_y = SCREEN_H // 2
        self.bird_vy = 0.0
        self.score = 0
        self.frame = 0
        self.done = False

        min_y = 70
        max_y = SCREEN_H - 70 - self.pipe_gap
        self.last_gap_y = random.randint(min_y, max_y)

        self.pipes = [
            self._new_pipe(SCREEN_W + 50),
            self._new_pipe(SCREEN_W + 50 + PIPE_SPACING),
        ]

        return self._get_state()

    def step(self, action: int):
        assert not self.done, "Call reset() before step()."

        if action == 1:
            self.bird_vy = FLAP_V

        self.bird_vy = min(self.bird_vy + GRAVITY, MAX_V)
        self.bird_y += self.bird_vy
        self.frame += 1

        reward = 1.0

        for p in self.pipes:
            p["x"] += self.pipe_speed

        for i, p in enumerate(self.pipes):
            if p["x"] + PIPE_W < 0:
                other = self.pipes[1 - i]
                self.pipes[i] = self._new_pipe(other["x"] + PIPE_SPACING)

        for p in self.pipes:
            if not p["passed"] and p["x"] + PIPE_W < BIRD_X:
                p["passed"] = True
                self.score += 1
                reward += 5.0

        if self._collision():
            self.done = True
            reward = -100.0

        return self._get_state(), reward, self.done, {"score": self.score}

    def get_state_continuous(self):
        dx, dy, vy = self._raw_state()
        return dx, dy, vy

    def _new_pipe(self, x: float):
        min_y = 70
        max_y = SCREEN_H - 70 - self.pipe_gap

        low = max(min_y, self.last_gap_y - MAX_GAP_JUMP)
        high = min(max_y, self.last_gap_y + MAX_GAP_JUMP)

        gap_y = random.randint(low, high)
        self.last_gap_y = gap_y

        return {
            "x": x,
            "gap_y": gap_y,
            "passed": False,
        }

    def _nearest_pipe(self):
        candidates = [p for p in self.pipes if p["x"] + PIPE_W >= BIRD_X]

        if not candidates:
            candidates = self.pipes

        return min(candidates, key=lambda p: p["x"])

    def _raw_state(self):
        p = self._nearest_pipe()

        dx = p["x"] - BIRD_X
        gap_center = p["gap_y"] + self.pipe_gap // 2
        dy = self.bird_y - gap_center
        vy = self.bird_vy

        return dx, dy, vy

    def _get_state(self):
        dx, dy, vy = self._raw_state()

        dx_d = int(np.digitize(dx, self.DX_BINS))
        dy_d = int(np.digitize(dy, self.DY_BINS))
        vy_d = int(np.digitize(vy, self.VY_BINS))

        return dx_d, dy_d, vy_d

    def _collision(self):
        if self.bird_y - BIRD_H // 2 <= 0:
            return True

        if self.bird_y + BIRD_H // 2 >= SCREEN_H:
            return True

        for p in self.pipes:
            bird_left = BIRD_X - BIRD_W // 2
            bird_right = BIRD_X + BIRD_W // 2
            bird_top = self.bird_y - BIRD_H // 2
            bird_bottom = self.bird_y + BIRD_H // 2

            pipe_left = p["x"]
            pipe_right = p["x"] + PIPE_W

            pipe_gap_top = p["gap_y"]
            pipe_gap_bottom = p["gap_y"] + self.pipe_gap

            if bird_right > pipe_left and bird_left < pipe_right:
                if bird_top < pipe_gap_top or bird_bottom > pipe_gap_bottom:
                    return True

        return False