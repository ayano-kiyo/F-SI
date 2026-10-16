"""Deterministic kinematic bicycle dynamics in the source CSV plane."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class VehicleState:
    x: float
    y: float
    heading_rad: float
    speed_mps: float = 0.0
    acceleration_mps2: float = 0.0
    yaw_rate_rad_s: float = 0.0


@dataclass(frozen=True, slots=True)
class ControlInput:
    steering: float
    throttle: float
    brake: float


class KinematicBicycle:
    WHEELBASE_M = 3.55
    MAX_STEER_RAD = 0.42
    MAX_SPEED_MPS = 92.0

    def step(self, state: VehicleState, control: ControlInput, dt: float) -> VehicleState:
        if dt <= 0:
            raise ValueError("The dynamics timestep must be positive")
        steering = max(-1.0, min(1.0, control.steering))
        throttle = max(0.0, min(1.0, control.throttle))
        brake = max(0.0, min(1.0, control.brake))
        acceleration = throttle * 13.5 - brake * 22.0 - 0.018 * state.speed_mps - 0.0016 * state.speed_mps**2
        speed = max(0.0, min(self.MAX_SPEED_MPS, state.speed_mps + acceleration * dt))
        yaw_rate = speed / self.WHEELBASE_M * math.tan(steering * self.MAX_STEER_RAD)
        mid_heading = state.heading_rad + yaw_rate * dt * 0.5
        return VehicleState(
            x=state.x + speed * math.cos(mid_heading) * dt,
            y=state.y + speed * math.sin(mid_heading) * dt,
            heading_rad=(state.heading_rad + yaw_rate * dt + math.pi) % (2 * math.pi) - math.pi,
            speed_mps=speed,
            acceleration_mps2=acceleration,
            yaw_rate_rad_s=yaw_rate,
        )
