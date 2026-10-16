"""Simulation state, baseline driving, and loaded PPO self-play races."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .controller import BaselineController, ControllerContext, NeuralNetworkController
from .dynamics import ControlInput, KinematicBicycle, VehicleState
from .planner import JEVPlanner
from .track_geometry import TrackGeometry


class SimulationEngine:
    FIXED_DT_S = 1.0 / 60.0
    RACE_CONTROL_STEPS = 3

    def __init__(self, track: TrackGeometry):
        self.track = track
        self.planner = JEVPlanner(track)
        self.dynamics = KinematicBicycle()
        self.baseline = BaselineController()
        self.neural = NeuralNetworkController()
        self.neural.bind_track(track)
        self.track_name = track.name
        self.checkpoint_path: str | None = None
        self.race_env = None
        self._race_phase = 0
        self._race_actions = []
        self._race_finished = False
        self._race_finished_elapsed_s = 0.0
        self.controller_name = "baseline"
        self.reset()

    def _make_race_env(self) -> None:
        import numpy as np
        from .rl.env import RacingVecEnv, ResetConfig

        cars_per_race = int(getattr(self.neural.policy, "cars_per_race", 4))
        self.race_env = RacingVecEnv(
            [self.track],
            num_envs=1,
            cars_per_race=cars_per_race,
            seed=42,
            reset_config=ResetConfig(
                randomize_along=True,
                speed_fraction=(0.05, 0.12),
                heading_noise_rad=0.025,
                lateral_offset_m=0.45,
            ),
            auto_reset=False,
        )
        self._race_actions = np.zeros((cars_per_race, 2), dtype=np.float32)
        self._race_phase = 0
        self._race_finished = False
        self._race_finished_elapsed_s = 0.0

    def set_track(self, track: TrackGeometry) -> None:
        """Swap circuits and reset the current solo car or multi-car grid."""
        self.track = track
        self.track_name = track.name
        self.planner = JEVPlanner(track)
        self.neural.bind_track(track)
        if self.neural.policy is not None and getattr(self.neural.policy, "obs_dim", 25) > 25:
            self._make_race_env()
        self.reset(keep_track=True)

    def load_checkpoint(self, checkpoint: str | Path) -> dict:
        """Attach a legacy solo policy or a versioned shared self-play policy."""
        from .rl.policy_io import load_policy

        path = Path(checkpoint)
        policy = load_policy(path)
        self.neural.attach(policy)
        self.neural.bind_track(self.track)
        self.checkpoint_path = str(path)
        if policy.obs_dim > 25:
            self._make_race_env()
        else:
            self.race_env = None
        self.controller_name = "neural_network"
        self.reset(keep_track=True)
        return {
            "checkpoint": str(path),
            "controller": self.neural.name,
            "available": self.neural.available,
            "race_mode": getattr(policy, "race_mode", "solo"),
            "cars_per_race": getattr(policy, "cars_per_race", 1),
            "observation_version": getattr(policy, "observation_version", 1),
        }

    def reset(self, keep_track: bool = False) -> None:
        if self.race_env is not None and self.controller_name == "neural_network":
            self.race_env.reset()
            self._race_phase = 0
            self._race_actions.fill(0.0)
            self._race_finished = False
            self._race_finished_elapsed_s = 0.0
            self._sync_primary_from_race()
        else:
            first = self.planner.selected.points[0]
            second = self.planner.selected.points[1]
            heading = math.atan2(second[1] - first[1], second[0] - first[0])
            self.vehicle = VehicleState(first[0], first[1], heading)
        self.sim_time_s = 0.0
        self.lap = 0
        self.lap_time_s = 0.0
        self.best_lap_time_s: float | None = None
        self._previous_distance_m = 0.0
        self._last_control = ControlInput(steering=0.0, throttle=0.0, brake=0.0)
        self._last_plan = self.planner.plan(self.vehicle.x, self.vehicle.y, self.vehicle.speed_mps)
        self._last_projection = self.track.project(self.vehicle.x, self.vehicle.y)
        self.neural.reset_episode()
        self.paused = False
        self.simulation_speed = 1.0
        if not keep_track and self.controller_name != "neural_network":
            self.controller_name = "baseline"

    def _sync_primary_from_race(self) -> None:
        if self.race_env is None:
            return
        env = self.race_env
        self.vehicle = VehicleState(
            x=float(env.x[0, 0]),
            y=float(env.y[0, 0]),
            heading_rad=float(env.heading[0, 0]),
            speed_mps=float(env.speed[0, 0]),
            acceleration_mps2=float(env.acceleration[0, 0]),
            yaw_rate_rad_s=float(env.yaw_rate[0, 0]),
        )
        self._last_projection = self.track.project(self.vehicle.x, self.vehicle.y)
        self._last_plan = self.planner.plan(self.vehicle.x, self.vehicle.y, self.vehicle.speed_mps)

    @property
    def controller(self):
        return self.baseline if self.controller_name == "baseline" else self.neural

    def set_speed(self, speed: float) -> None:
        if speed not in (0.25, 0.5, 1.0, 2.0, 4.0):
            raise ValueError("Supported simulation speeds are 0.25, 0.5, 1, 2, and 4")
        self.simulation_speed = speed

    def set_controller(self, name: str) -> None:
        if name == "neural_network" and not self.neural.available:
            raise ValueError("Neural-network control is unavailable: no checkpoint is configured")
        if name not in ("baseline", "neural_network"):
            raise ValueError("Controller must be 'baseline' or 'neural_network'")
        if name == "baseline":
            self.controller_name = "baseline"
            self.race_env = None
        else:
            self.controller_name = "neural_network"
            if getattr(self.neural.policy, "obs_dim", 25) > 25 and self.race_env is None:
                self._make_race_env()
        self.reset(keep_track=True)

    def _step_race(self, dt: float) -> None:
        self.sim_time_s += dt
        self.lap_time_s += dt
        if self._race_finished:
            # Hold the final order long enough to read it, then keep the preview
            # alive with another seeded grid instead of leaving a dead race on screen.
            self._race_finished_elapsed_s += dt
            if self._race_finished_elapsed_s >= 2.5:
                self.race_env.reset()
                self._race_phase = 0
                self._race_actions.fill(0.0)
                self._race_finished = False
                self._race_finished_elapsed_s = 0.0
                self.lap_time_s = 0.0
                self._sync_primary_from_race()
        else:
            if self._race_phase == 0:
                import torch

                observations = self.race_env.observe()
                device = next(self.neural.policy.parameters()).device
                tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
                with torch.inference_mode():
                    mean, _, _ = self.neural.policy.forward(tensor)
                    self._race_actions = torch.tanh(mean).cpu().numpy()
            self._race_phase += 1
            if self._race_phase >= self.RACE_CONTROL_STEPS:
                _, _, _, _ = self.race_env.step(self._race_actions)
                self._race_phase = 0
                self._race_finished = self.race_env.last_results[0] is not None
                if self._race_finished:
                    self._race_finished_elapsed_s = 0.0
                self._sync_primary_from_race()
        self._last_control = ControlInput(
            steering=float(self._race_actions[0, 0]),
            throttle=max(0.0, float(self._race_actions[0, 1])),
            brake=max(0.0, -float(self._race_actions[0, 1])),
        )

    def step(self, dt: float = FIXED_DT_S) -> None:
        if self.paused:
            return
        if self.race_env is not None and self.controller_name == "neural_network":
            self._step_race(dt)
            return
        self.sim_time_s += dt
        self.lap_time_s += dt
        plan = self.planner.plan(self.vehicle.x, self.vehicle.y, self.vehicle.speed_mps)
        control = self.controller.compute(ControllerContext(self.vehicle, plan))
        self.vehicle = self.dynamics.step(self.vehicle, control, dt)
        self._last_control = control
        self._last_plan = self.planner.plan(self.vehicle.x, self.vehicle.y, self.vehicle.speed_mps)
        self._last_projection = self.track.project(self.vehicle.x, self.vehicle.y)

        race_index = self.track.nearest_race_line_index(self.vehicle.x, self.vehicle.y)
        distance = self.track.race_line_distance_at(race_index)
        if self._previous_distance_m > self.track.race_line_length_m * 0.82 and distance < self.track.race_line_length_m * 0.18:
            self.lap += 1
            if self.lap_time_s > 1.0:
                if self.best_lap_time_s is None or self.lap_time_s < self.best_lap_time_s:
                    self.best_lap_time_s = self.lap_time_s
            self.lap_time_s = 0.0
        self._previous_distance_m = distance

    def _race_telemetry(self) -> tuple[list[dict], dict, dict]:
        env = self.race_env
        ft = env.tracks[env.track_idx[0]]
        vehicles = []
        for car in range(env.cars_per_race):
            action = self._race_actions[car]
            fraction = min(1.0, max(0.0, float(env.lap_progress[0, car]) / ft.race_length_m))
            vehicles.append({
                "id": f"car-{car + 1}",
                "x": float(env.x[0, car]),
                "y": float(env.y[0, car]),
                "heading_rad": float(env.heading[0, car]),
                "speed_mps": float(env.speed[0, car]),
                "acceleration_mps2": float(env.acceleration[0, car]),
                "yaw_rate_rad_s": float(env.yaw_rate[0, car]),
                "steering": float(action[0]),
                "throttle": max(0.0, float(action[1])),
                "brake": max(0.0, -float(action[1])),
                "progress_m": float(env.lap_progress[0, car]),
                "fraction": fraction,
                "status": str(env.reason[0, car]),
                "position": int(env.rank[0, car]) or None,
                "target_waypoint": {"x": self._last_plan.target_x, "y": self._last_plan.target_y},
                "route_id": self._last_plan.route_id,
            })
        primary_index = next(
            (index for index, vehicle in enumerate(vehicles)
             if env.active[0, index] and env.speed[0, index] > 0.5),
            next((index for index in range(env.cars_per_race) if env.active[0, index]), 0),
        )
        primary = vehicles[primary_index]
        progress = {
            "distance_m": primary["progress_m"],
            "fraction": primary["fraction"],
            "lap": 1 if env.finished[0, primary_index] else 0,
            "lap_time_s": (float(env.finish_step[0, primary_index]) / 20.0 if env.finished[0, primary_index] else self.lap_time_s),
            "best_lap_time_s": self.best_lap_time_s,
            "off_track": str(env.reason[0, primary_index]) == "off_track",
            "lateral_error_m": float(env.lateral[0, primary_index]),
        }
        race = env.last_results[0] or {
            "track": ft.name,
            "finished": False,
            "cars": [
                {"car": index, "position": int(env.rank[0, index]) or None,
                 "status": str(env.reason[0, index]), "progress_m": float(env.lap_progress[0, index])}
                for index in range(env.cars_per_race)
            ],
            "collisions": int(env.collision_count[0]),
        }
        race["finished"] = env.last_results[0] is not None
        return vehicles, progress, race

    def snapshot(self) -> dict[str, Any]:
        projection = self.track.project(self.vehicle.x, self.vehicle.y)
        progress_index = self.track.nearest_race_line_index(self.vehicle.x, self.vehicle.y)
        progress_m = self.track.race_line_distance_at(progress_index)
        off_track = projection.signed_lateral_m > projection.width_left_m or projection.signed_lateral_m < -projection.width_right_m
        plan = self._last_plan

        if self.race_env is not None and self.controller_name == "neural_network":
            vehicles, race_progress, race = self._race_telemetry()
            primary = next(
                (vehicle for vehicle in vehicles if vehicle["status"] == "racing" and vehicle["speed_mps"] > 0.5),
                vehicles[0],
            )
            primary_vehicle = {
                "id": primary["id"], "x": primary["x"], "y": primary["y"], "heading_rad": primary["heading_rad"],
                "speed_mps": primary["speed_mps"], "acceleration_mps2": primary["acceleration_mps2"],
                "yaw_rate_rad_s": primary["yaw_rate_rad_s"], "steering": primary["steering"],
                "throttle": primary["throttle"], "brake": primary["brake"],
                "target_waypoint": primary["target_waypoint"], "route_id": primary["route_id"],
            }
            progress = race_progress
        else:
            vehicles = [{
                "id": "car-1", "x": self.vehicle.x, "y": self.vehicle.y,
                "heading_rad": self.vehicle.heading_rad, "speed_mps": self.vehicle.speed_mps,
                "acceleration_mps2": self.vehicle.acceleration_mps2,
                "yaw_rate_rad_s": self.vehicle.yaw_rate_rad_s,
                "steering": self._last_control.steering, "throttle": self._last_control.throttle,
                "brake": self._last_control.brake, "progress_m": progress_m,
                "fraction": min(1.0, max(0.0, progress_m / self.track.race_line_length_m)),
                "status": "racing", "position": 1,
                "target_waypoint": {"x": plan.target_x, "y": plan.target_y}, "route_id": plan.route_id,
            }]
            primary_vehicle = {
                "x": self.vehicle.x,
                "y": self.vehicle.y,
                "heading_rad": self.vehicle.heading_rad,
                "speed_mps": self.vehicle.speed_mps,
                "acceleration_mps2": self.vehicle.acceleration_mps2,
                "yaw_rate_rad_s": self.vehicle.yaw_rate_rad_s,
                "steering": self._last_control.steering,
                "throttle": self._last_control.throttle,
                "brake": self._last_control.brake,
                "target_waypoint": {"x": plan.target_x, "y": plan.target_y},
                "route_id": plan.route_id,
            }
            progress = {
                "distance_m": progress_m,
                "fraction": min(1.0, max(0.0, progress_m / self.track.race_line_length_m)),
                "lap": self.lap,
                "lap_time_s": self.lap_time_s,
                "best_lap_time_s": self.best_lap_time_s,
                "off_track": off_track,
                "lateral_error_m": projection.signed_lateral_m,
            }
            race = None

        result = {
            "type": "telemetry",
            "sim_time_s": self.sim_time_s,
            "paused": self.paused,
            "simulation_speed": self.simulation_speed,
            "track": {"name": self.track.name, "length_m": self.track.length_m},
            "vehicle": primary_vehicle,
            "vehicles": vehicles,
            "planner": {"name": "JEV", "route_cost": plan.route_cost, "route_id": plan.route_id},
            "controller": {
                "name": self.controller_name, "available": self.controller.available,
                "checkpoint": self.checkpoint_path,
            },
            "progress": progress,
            "planned_trajectory": plan.trajectory,
        }
        if race is not None:
            result["race"] = race
        return result
