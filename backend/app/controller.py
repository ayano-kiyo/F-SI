"""Controller contracts: baseline pure-pursuit and the trained RL policy."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from .dynamics import ControlInput, VehicleState
from .planner import RoutePlan

@dataclass(frozen=True, slots=True)
class ControllerContext:
    vehicle: VehicleState
    plan: RoutePlan

class Controller(Protocol):
    name: str
    available: bool

    def compute(self, context: ControllerContext) -> ControlInput: ...

class BaselineController:
    name = "baseline"
    available = True

    def compute(self, context: ControllerContext) -> ControlInput:
        vehicle, plan = context.vehicle, context.plan
        dx, dy = plan.target_x - vehicle.x, plan.target_y - vehicle.y
        distance = max(math.hypot(dx, dy), 1.0)
        target_angle = math.atan2(dy, dx)
        alpha = (target_angle - vehicle.heading_rad + math.pi) % (2 * math.pi) - math.pi
        lookahead = max(distance, 4.0 + vehicle.speed_mps * 0.18)
        steering_angle = math.atan2(2 * self_wheelbase() * math.sin(alpha), lookahead)
        steering = max(-1.0, min(1.0, steering_angle / 0.42))

        speed_error = plan.target_speed_mps - vehicle.speed_mps
        if speed_error < -1.5:
            brake = max(0.0, min(1.0, 0.18 + (-speed_error) * 0.12))
            throttle = 0.0
        else:
            brake = 0.0
            throttle = max(0.12, min(1.0, 0.35 + speed_error * 0.18))
        return ControlInput(steering=steering, throttle=throttle, brake=brake)

def self_wheelbase() -> float:
    return 3.55

class NeuralNetworkController:
    """Live solo driver for a legacy trained PPO racing policy.

    The policy consumes the legacy 25-dim observation used in solo training
    (speed, reference speed, lateral offset, heading, last steer, half width,
    11 lidar rays, 6 curvature samples) and emits [steer, pedal] in [-1, 1].
    Positive pedal is throttle, negative pedal is brake.
    Torch stays lazily imported so baseline installs remain lightweight.
    """

    name = "neural_network"

    def __init__(self, policy=None, deterministic: bool = True):
        self.policy = policy
        self.deterministic = deterministic
        self.available = policy is not None
        self._prev_steer: float = 0.0
        self._fast_track = None
        self._fast_track_name: str | None = None

    def attach(self, policy) -> None:
        self.policy = policy
        self.available = policy is not None
        self._prev_steer = 0.0

    def reset_episode(self) -> None:
        self._prev_steer = 0.0

    def compute(self, context: ControllerContext) -> ControlInput:
        if self.policy is None:
            raise RuntimeError("No neural-network checkpoint is configured")
        if getattr(self.policy, "obs_dim", 25) != 25:
            raise RuntimeError("Shared-policy self-play checkpoints must be run in race mode")
        import torch

        from .rl.fastrack import get_fast_track
        from .rl.observations import observe_live
        from .track_geometry import load_track

        vehicle = context.vehicle
        if self._fast_track is None:
            raise RuntimeError("Neural controller needs bind_track() before driving")
        obs = observe_live(self._fast_track, vehicle.x, vehicle.y,
                           vehicle.heading_rad, vehicle.speed_mps, self._prev_steer)
        tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            if self.deterministic:
                mean, _, _ = self.policy.forward(tensor)
                action = torch.tanh(mean)
            else:
                action, _, _, _ = self.policy.act(tensor)
            values = action.reshape(-1).tolist()
        steer = max(-1.0, min(1.0, float(values[0])))
        pedal = max(-1.0, min(1.0, float(values[1])))
        self._prev_steer = steer
        if pedal >= 0:
            return ControlInput(steering=steer, throttle=float(pedal), brake=0.0)
        return ControlInput(steering=steer, throttle=0.0, brake=float(-pedal))

    def bind_track(self, geometry) -> None:
        from .rl.fastrack import get_fast_track
        self._fast_track = get_fast_track(geometry)
        self._fast_track_name = geometry.name
        self._prev_steer = 0.0
