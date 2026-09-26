"""Bound command-reference motion independently of measured-position error."""

from __future__ import annotations

import math


class JointTargetRateLimiter:
    """Limit targets, not actual joint velocity. Commit only after successful sends.

    Elapsed time bounds speed; the per-send bound prevents a pause or slow
    inference from accumulating a large catch-up step. The first send holds the
    measured pose. A separate tracking-error envelope prevents reference windup.
    """

    def __init__(self, velocity_deg_s: float, step_deg: float):
        for name, value in (("velocity_deg_s", velocity_deg_s), ("step_deg", step_deg)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.velocity_deg_s = velocity_deg_s
        self.step_deg = step_deg
        self.reset()

    def reset(self) -> None:
        self._targets: dict[str, float] = {}
        self._time: float | None = None

    def limit(
        self,
        goals: dict[str, float],
        present: dict[str, float],
        joint_limits: dict[str, tuple[float, float]],
        max_error: float | dict[str, float],
        now: float,
    ) -> dict[str, float]:
        if not math.isfinite(now) or (self._time is not None and now < self._time):
            raise ValueError("Target limiter needs a finite, monotonic timestamp")
        elapsed = 0.0 if self._time is None else now - self._time
        budget = min(self.velocity_deg_s * elapsed, self.step_deg)
        targets = {}
        for name, goal in goals.items():
            measured = present[name]
            error = max_error[name] if isinstance(max_error, dict) else max_error
            if not all(math.isfinite(x) for x in (goal, measured, error)) or error <= 0:
                raise ValueError(f"Invalid target, feedback or tracking-error limit for {name}")
            previous = self._targets.get(name, measured)
            step = budget if name in self._targets else 0.0
            minimum, maximum = joint_limits.get(name, (-math.inf, math.inf))
            lower = max(previous - step, measured - error, minimum)
            upper = min(previous + step, measured + error, maximum)
            if lower > upper:
                raise ValueError(
                    f"Cannot satisfy target rate, tracking error and joint limits for {name}; "
                    "stopping commands instead of jumping to a new reference"
                )
            targets[name] = max(lower, min(upper, goal))
        return targets

    def commit(self, sent: dict[str, float], now: float) -> None:
        self._targets = dict(sent)
        self._time = now
