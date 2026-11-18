"""Reset-only joint progress watchdog; no ROS, commands or persistent logging."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ResetStall:
    index: int
    joint: str
    seconds: float
    command_gap_deg: float
    goal_gap_deg: float


class ResetStallGuard:
    def __init__(self, start, goal, indices, names, started_at, *,
                 grace_seconds=1.0, window_seconds=2.0, min_progress_deg=0.5,
                 command_gap_deg=3.0, goal_gap_deg=3.0):
        self.start = self._vector(start)
        self.goal = self._vector(goal)
        self.indices = tuple(indices)
        self.names = tuple(names)
        if len(self.indices) != len(self.names) or len(set(self.indices)) != len(self.indices):
            raise ValueError('stall joints must have unique indices and matching names')
        if any(not isinstance(i, int) or not 0 <= i < 21 for i in self.indices):
            raise ValueError('stall joint index is invalid')
        values = [started_at, grace_seconds, window_seconds, min_progress_deg,
                  command_gap_deg, goal_gap_deg]
        if not all(math.isfinite(float(x)) for x in values):
            raise ValueError('reset stall settings must be finite')
        if grace_seconds < 0 or any(x <= 0 for x in values[2:]):
            raise ValueError('reset stall thresholds must be positive')
        self.started_at = float(started_at)
        self.grace = float(grace_seconds)
        self.window = float(window_seconds)
        self.minimum_progress = float(min_progress_deg)
        self.command_gap = float(command_gap_deg)
        self.goal_gap = float(goal_gap_deg)
        self.direction = np.sign(self.goal - self.start)
        self._watch = {}
        self._last_time = self.started_at

    @staticmethod
    def _vector(value):
        vector = np.asarray(value, dtype=float)
        if vector.shape != (21,) or not np.all(np.isfinite(vector)):
            raise ValueError('reset stall feedback must contain 21 finite joint angles')
        return vector.copy()

    def check(self, now, measured, command):
        now = float(now)
        if not math.isfinite(now) or now < self._last_time:
            raise ValueError('reset stall clock is invalid')
        self._last_time = now
        measured, command = self._vector(measured), self._vector(command)
        if now - self.started_at < self.grace:
            self._watch.clear()
            return None
        for index, name in zip(self.indices, self.names):
            direction = self.direction[index]
            goal_gap = direction * (self.goal[index] - measured[index])
            command_gap = direction * (command[index] - measured[index])
            if direction == 0 or goal_gap < self.goal_gap or command_gap < self.command_gap:
                self._watch.pop(index, None)
                continue
            position = float(direction * measured[index])
            previous = self._watch.get(index)
            if previous is None:
                self._watch[index] = (now, position)
                continue
            progress_at, progress_position = previous
            # Only new net progress toward this joint's goal rearms the timer.
            # Another joint moving or repeated encoder oscillation cannot hide a stall.
            if position - progress_position >= self.minimum_progress:
                self._watch[index] = (now, position)
            elif now - progress_at >= self.window:
                return ResetStall(index, name, now - progress_at,
                                  float(command_gap), float(goal_gap))
        return None
