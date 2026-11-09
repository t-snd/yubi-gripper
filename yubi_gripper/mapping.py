"""Map YUBI encoder angle [rad] to Dynamixel goal position [tick]."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TICKS_PER_REV = 4096  # XM460-W350-R: 0.088 deg/tick


@dataclass
class GripperMapping:
    """Linear map between encoder angle and motor ticks.

    ``encoder_closed_rad`` -> ``motor_closed_tick`` and
    ``encoder_open_rad``   -> ``motor_open_tick``. Values outside the encoder
    range are clipped (when ``clip`` is True) so the motor never goes beyond
    the calibrated closed/open positions.

    The YUBI encoder reports 0 rad when closed and about 0.94 rad (URDF limit)
    when fully open. Calibrate the motor ticks on the robot-side gripper with
    ``yubi-gripper calibrate``.
    """

    motor_closed_tick: int
    motor_open_tick: int
    encoder_closed_rad: float = 0.0
    encoder_open_rad: float = 0.94
    clip: bool = True

    def __post_init__(self) -> None:
        if math.isclose(self.encoder_open_rad, self.encoder_closed_rad):
            raise ValueError("encoder_open_rad and encoder_closed_rad must differ")
        if self.motor_open_tick == self.motor_closed_tick:
            raise ValueError("motor_open_tick and motor_closed_tick must differ")

    def ratio(self, rad):
        r = (np.asarray(rad, dtype=float) - self.encoder_closed_rad) / (
            self.encoder_open_rad - self.encoder_closed_rad
        )
        return np.clip(r, 0.0, 1.0) if self.clip else r

    def to_tick(self, rad):
        tick = self.motor_closed_tick + self.ratio(rad) * (
            self.motor_open_tick - self.motor_closed_tick
        )
        out = np.rint(tick).astype(np.int64)
        return int(out) if out.ndim == 0 else out

    def tick_to_rad(self, tick):
        """Inverse map (encoder-equivalent angle), for comparing tracking."""
        r = (np.asarray(tick, dtype=float) - self.motor_closed_tick) / (
            self.motor_open_tick - self.motor_closed_tick
        )
        out = self.encoder_closed_rad + r * (self.encoder_open_rad - self.encoder_closed_rad)
        return float(out) if np.ndim(out) == 0 else out

    @property
    def motor_stroke_deg(self) -> float:
        return abs(self.motor_open_tick - self.motor_closed_tick) * 360.0 / TICKS_PER_REV
