"""Vectorized shared-policy self-play races (NumPy physics, batched PPO)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .fastrack import FastTrack, get_fast_track
from ..track_geometry import TrackGeometry

BASE_OBS_DIM = 25
MAX_CARS_PER_RACE = 4
OPPONENT_SLOTS = MAX_CARS_PER_RACE - 1
OPPONENT_FEATURES = 6
OBS_VERSION = 2
OBS_DIM = BASE_OBS_DIM + OPPONENT_SLOTS * OPPONENT_FEATURES
ACT_DIM = 2

_LIDAR_ANGLES = np.deg2rad(np.linspace(-100.0, 100.0, 11)).astype(np.float64)
_CURVATURE_GAPS = (10.0, 20.0, 35.0, 55.0, 80.0, 110.0)
_DT = 1.0 / 60.0
_CONTROL_SUBSTEPS = 3
_WHEELBASE = 3.55
_MAX_STEER = 0.42
_MAX_SPEED = 92.0
_CAR_HALF_LENGTH_M = 2.25
_CAR_HALF_WIDTH_M = 0.95
_GRID_SPACING_M = 8.0


@dataclass
class ResetConfig:
    randomize_along: bool = True
    speed_fraction: tuple[float, float] = (0.0, 0.12)
    heading_noise_rad: float = 0.025
    lateral_offset_m: float = 0.45


class RacingVecEnv:
    """N independent race worlds, each with 2-4 cars sharing one PPO policy.

    Observations, actions, rewards, and termination flags are flattened to
    num_envs * cars_per_race rows for the actor-critic. Each world advances
    all of its cars simultaneously and resets as a group after the race ends.
    """

    def __init__(
        self,
        tracks: list[TrackGeometry],
        num_envs: int = 8,
        cars_per_race: int = 4,
        seed: int = 0,
        reset_config: ResetConfig | None = None,
        max_steps: int = 7200,
        auto_reset: bool = True,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if not 1 <= cars_per_race <= MAX_CARS_PER_RACE:
            raise ValueError(f"cars_per_race must be between 1 and {MAX_CARS_PER_RACE}")
        if not tracks:
            raise ValueError("need at least one track")
        self.geometries = list(tracks)
        self.tracks = [get_fast_track(track) for track in tracks]
        self.num_envs = num_envs
        self.num_races = num_envs
        self.cars_per_race = cars_per_race
        self.num_agents = num_envs * cars_per_race
        self.obs_dim = BASE_OBS_DIM if cars_per_race == 1 else OBS_DIM
        self.max_steps = max_steps
        self.auto_reset = auto_reset
        self.reset_config = reset_config or ResetConfig()
        self.rng = np.random.default_rng(seed)

        shape = (num_envs, cars_per_race)
        self.x = np.zeros(shape, dtype=np.float64)
        self.y = np.zeros(shape, dtype=np.float64)
        self.heading = np.zeros(shape, dtype=np.float64)
        self.speed = np.zeros(shape, dtype=np.float64)
        self.acceleration = np.zeros(shape, dtype=np.float64)
        self.yaw_rate = np.zeros(shape, dtype=np.float64)
        # Keep each vectorized world on one circuit. The initial assignment is
        # shuffled from the seeded RNG, and eight default worlds cover all eight
        # training tracks. Changing tracks only between worlds prevents the live
        # browser scene from being remounted after every short episode.
        track_assignments = np.tile(
            np.arange(len(self.tracks), dtype=np.int64),
            int(np.ceil(num_envs / len(self.tracks))),
        )[:num_envs]
        self.rng.shuffle(track_assignments)
        self.track_idx = track_assignments
        self.along = np.zeros(shape, dtype=np.float64)
        self.lateral = np.zeros(shape, dtype=np.float64)
        self.half_width = np.full(shape, 6.0, dtype=np.float64)
        self.steps = np.zeros(shape, dtype=np.int64)
        self.race_steps = np.zeros(num_envs, dtype=np.int64)
        self.prev_along = np.zeros(shape, dtype=np.float64)
        self.prev_steer = np.zeros(shape, dtype=np.float64)
        self.off_track_time = np.zeros(shape, dtype=np.float64)
        self.lap_progress = np.zeros(shape, dtype=np.float64)
        self.active = np.ones(shape, dtype=bool)
        self.finished = np.zeros(shape, dtype=bool)
        self.finish_step = np.full(shape, -1, dtype=np.int64)
        self.rank = np.zeros(shape, dtype=np.int64)
        self.reason = np.full(shape, "racing", dtype=object)
        self.collision_count = np.zeros(num_envs, dtype=np.int64)
        self.last_results: list[dict | None] = [None] * num_envs

    @property
    def active_mask(self) -> np.ndarray:
        return self.active.reshape(-1).copy()

    @staticmethod
    def _race_line_pose(ft: FastTrack, along_m: float) -> tuple[float, float, float]:
        distance = along_m % ft.race_length_m
        index = int(np.searchsorted(ft.race_cumulative, distance, side="right") - 1)
        index = max(0, min(index, len(ft.race_line) - 1))
        nxt = (index + 1) % len(ft.race_line)
        segment_length = max(float(np.linalg.norm(ft.race_line[nxt] - ft.race_line[index])), 1e-9)
        fraction = np.clip((distance - ft.race_cumulative[index]) / segment_length, 0.0, 1.0)
        point = ft.race_line[index] + fraction * (ft.race_line[nxt] - ft.race_line[index])
        direction = ft.race_line[nxt] - ft.race_line[index]
        heading = math.atan2(float(direction[1]), float(direction[0]))
        return float(point[0]), float(point[1]), heading

    def _randomize(self, race_indices: np.ndarray) -> None:
        cfg = self.reset_config
        for race in np.asarray(race_indices, dtype=np.int64):
            # Keep each parallel world on its seeded circuit; randomize the grid
            # start below so every episode still explores a different section.
            track_index = int(self.track_idx[race])
            ft = self.tracks[track_index]
            self.track_idx[race] = track_index
            start = self.rng.uniform(0.0, ft.race_length_m) if cfg.randomize_along else 0.0
            self.race_steps[race] = 0
            self.collision_count[race] = 0
            self.last_results[race] = None

            for car in range(self.cars_per_race):
                along = start - car * _GRID_SPACING_M
                px, py, heading = self._race_line_pose(ft, along)
                side = -1.0 if car % 2 == 0 else 1.0
                lateral_offset = side * cfg.lateral_offset_m
                px += -math.sin(heading) * lateral_offset
                py += math.cos(heading) * lateral_offset
                heading += self.rng.uniform(-cfg.heading_noise_rad, cfg.heading_noise_rad)
                center_along, lat, half = ft.project(px, py)
                vmax = ft.speed_at(along)
                frac = self.rng.uniform(*cfg.speed_fraction)

                self.x[race, car] = px
                self.y[race, car] = py
                self.heading[race, car] = heading
                self.speed[race, car] = vmax * frac
                self.acceleration[race, car] = 0.0
                self.yaw_rate[race, car] = 0.0
                self.along[race, car] = center_along
                self.lateral[race, car] = lat
                self.half_width[race, car] = half
                self.steps[race, car] = 0
                self.prev_along[race, car] = center_along
                self.prev_steer[race, car] = 0.0
                self.off_track_time[race, car] = 0.0
                self.lap_progress[race, car] = 0.0
                self.active[race, car] = True
                self.finished[race, car] = False
                self.finish_step[race, car] = -1
                self.rank[race, car] = 0
                self.reason[race, car] = "racing"

    def reset(self, race_indices: np.ndarray | None = None) -> np.ndarray:
        if race_indices is None:
            race_indices = np.arange(self.num_envs)
        self._randomize(np.asarray(race_indices))
        return self.observe()

    def observe(self) -> np.ndarray:
        from .observations import observe_race

        obs = np.zeros((self.num_envs, self.cars_per_race, self.obs_dim), dtype=np.float32)
        for race in range(self.num_envs):
            ft = self.tracks[self.track_idx[race]]
            for car in range(self.cars_per_race):
                opponents = []
                for other in range(self.cars_per_race):
                    if other == car:
                        continue
                    opponents.append({
                        "x": self.x[race, other],
                        "y": self.y[race, other],
                        "heading": self.heading[race, other],
                        "speed": self.speed[race, other],
                        "active": self.active[race, other],
                    })
                if self.cars_per_race == 1:
                    from .observations import observe_live
                    obs[race, car] = observe_live(
                        ft, self.x[race, car], self.y[race, car], self.heading[race, car],
                        self.speed[race, car], self.prev_steer[race, car],
                    )
                else:
                    obs[race, car] = observe_race(
                        ft, self.x[race, car], self.y[race, car], self.heading[race, car],
                        self.speed[race, car], self.prev_steer[race, car], opponents,
                    )
        return obs.reshape(self.num_agents, self.obs_dim)

    @staticmethod
    def _rectangles_overlap(
        x1: float, y1: float, heading1: float, x2: float, y2: float, heading2: float,
    ) -> bool:
        """Separating-axis test for two simple oriented car rectangles."""
        delta_x, delta_y = x2 - x1, y2 - y1
        axes = (
            (math.cos(heading1), math.sin(heading1)),
            (-math.sin(heading1), math.cos(heading1)),
            (math.cos(heading2), math.sin(heading2)),
            (-math.sin(heading2), math.cos(heading2)),
        )
        for axis_x, axis_y in axes:
            distance = abs(delta_x * axis_x + delta_y * axis_y)
            radius1 = (
                _CAR_HALF_LENGTH_M * abs(math.cos(heading1) * axis_x + math.sin(heading1) * axis_y)
                + _CAR_HALF_WIDTH_M * abs(-math.sin(heading1) * axis_x + math.cos(heading1) * axis_y)
            )
            radius2 = (
                _CAR_HALF_LENGTH_M * abs(math.cos(heading2) * axis_x + math.sin(heading2) * axis_y)
                + _CAR_HALF_WIDTH_M * abs(-math.sin(heading2) * axis_x + math.cos(heading2) * axis_y)
            )
            if distance > radius1 + radius2:
                return False
        return True

    def _race_result(self, race: int) -> dict:
        cars = []
        order = sorted(
            range(self.cars_per_race),
            key=lambda car: (
                0 if self.finished[race, car] else 1,
                self.finish_step[race, car] if self.finished[race, car] else -self.lap_progress[race, car],
                car,
            ),
        )
        for position, car in enumerate(order, start=1):
            if self.rank[race, car] == 0:
                self.rank[race, car] = position
            finished = bool(self.finished[race, car])
            cars.append({
                "car": car,
                "position": int(self.rank[race, car]),
                "status": str(self.reason[race, car]),
                "progress_m": float(self.lap_progress[race, car]),
                "lap_time_s": float(self.finish_step[race, car] / 20.0) if finished else None,
            })
        return {
            "track": self.tracks[self.track_idx[race]].name,
            "steps": int(self.race_steps[race]),
            "finishes": int(self.finished[race].sum()),
            "collisions": int(self.collision_count[race]),
            "cars": cars,
        }

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        actions = np.asarray(actions, dtype=np.float64).reshape(self.num_envs, self.cars_per_race, ACT_DIM)
        actions = np.clip(actions, -1.0, 1.0)
        steer = actions[:, :, 0]
        pedal = actions[:, :, 1]
        throttle = np.clip(pedal, 0.0, 1.0)
        brake = np.clip(-pedal, 0.0, 1.0)
        active_before = self.active.copy()
        rewards = np.zeros((self.num_envs, self.cars_per_race), dtype=np.float32)
        dones = ~active_before
        collision_hits = np.zeros_like(self.active)

        for _ in range(_CONTROL_SUBSTEPS):
            moving = self.active & ~collision_hits
            accel = throttle * 13.5 - brake * 22.0 - 0.018 * self.speed - 0.0016 * self.speed ** 2
            accel = np.where(moving, accel, 0.0)
            new_speed = np.clip(self.speed + accel * _DT, 0.0, _MAX_SPEED)
            yaw = new_speed / _WHEELBASE * np.tan(steer * _MAX_STEER)
            yaw = np.where(moving, yaw, 0.0)
            mid = self.heading + yaw * _DT * 0.5
            self.x = np.where(moving, self.x + new_speed * np.cos(mid) * _DT, self.x)
            self.y = np.where(moving, self.y + new_speed * np.sin(mid) * _DT, self.y)
            self.heading = np.where(
                moving,
                (self.heading + yaw * _DT + math.pi) % (2 * math.pi) - math.pi,
                self.heading,
            )
            self.speed = np.where(moving, new_speed, self.speed)
            self.acceleration = np.where(moving, accel, self.acceleration)
            self.yaw_rate = np.where(moving, yaw, self.yaw_rate)

            for race in range(self.num_envs):
                active_cars = np.flatnonzero(self.active[race] & ~collision_hits[race])
                for left_index, left in enumerate(active_cars):
                    for right in active_cars[left_index + 1:]:
                        if self._rectangles_overlap(
                            self.x[race, left], self.y[race, left], self.heading[race, left],
                            self.x[race, right], self.y[race, right], self.heading[race, right],
                        ):
                            collision_hits[race, left] = True
                            collision_hits[race, right] = True
                            self.collision_count[race] += 1
                            self.speed[race, left] = 0.0
                            self.speed[race, right] = 0.0

        infos: list[dict] = []
        for race in range(self.num_envs):
            ft = self.tracks[self.track_idx[race]]
            self.race_steps[race] += 1
            for car in range(self.cars_per_race):
                if not active_before[race, car]:
                    infos.append({
                        "reason": str(self.reason[race, car]), "active": False,
                        "track": ft.name, "progress_m": float(self.lap_progress[race, car]),
                    })
                    continue

                along, lateral, half = ft.project(float(self.x[race, car]), float(self.y[race, car]))
                delta = along - self.prev_along[race, car]
                if delta < -ft.length_m * 0.5:
                    delta += ft.length_m
                elif delta > ft.length_m * 0.5:
                    delta -= ft.length_m
                progress_delta = max(delta, 0.0)
                self.lap_progress[race, car] += progress_delta
                off_track = abs(lateral) > half
                self.off_track_time[race, car] = (
                    self.off_track_time[race, car] + 1.0 / 20.0 if off_track else 0.0
                )

                progress_reward = progress_delta / 12.0
                speed_bonus = 0.02 * (self.speed[race, car] / 60.0)
                edge_penalty = 1.0 if off_track else 0.0
                wall_penalty = 2.0 if abs(lateral) > half + 3.0 else 0.0
                steering_penalty = 0.02 * abs(steer[race, car] - self.prev_steer[race, car])
                rewards[race, car] = progress_reward + speed_bonus - edge_penalty - wall_penalty - steering_penalty
                self.steps[race, car] += 1

                reason = ""
                if collision_hits[race, car]:
                    reason = "collision"
                    rewards[race, car] -= 18.0
                elif self.off_track_time[race, car] > 2.5 or abs(lateral) > half + 8.0:
                    reason = "off_track"
                    rewards[race, car] -= 10.0
                elif self.lap_progress[race, car] >= ft.race_length_m * 0.999:
                    reason = "finished"
                    self.finished[race, car] = True
                    self.finish_step[race, car] = self.race_steps[race]
                    rank_at_finish = int(self.finished[race].sum())
                    rewards[race, car] += (18.0, 9.0, 3.0, 0.0)[rank_at_finish - 1]
                elif self.speed[race, car] < 1.0 and self.steps[race, car] > 200:
                    reason = "stalled"
                    rewards[race, car] -= 3.0
                elif self.race_steps[race] >= self.max_steps:
                    reason = "time"
                    rewards[race, car] -= 6.0

                if reason:
                    self.active[race, car] = False
                    self.reason[race, car] = reason
                    self.speed[race, car] = 0.0
                    self.acceleration[race, car] = 0.0
                    self.yaw_rate[race, car] = 0.0
                    dones[race, car] = True
                self.prev_along[race, car] = along
                self.along[race, car] = along
                self.lateral[race, car] = lateral
                self.half_width[race, car] = half
                self.prev_steer[race, car] = steer[race, car]
                infos.append({
                    "reason": reason or "racing",
                    "active": bool(self.active[race, car]),
                    "track": ft.name,
                    "along_m": float(along),
                    "lateral_m": float(lateral),
                    "progress_m": float(self.lap_progress[race, car]),
                    "speed_mps": float(self.speed[race, car]),
                })

            race_done = not bool(self.active[race].any())
            if race_done:
                result = self._race_result(race)
                self.last_results[race] = result
                start = race * self.cars_per_race
                for car in range(self.cars_per_race):
                    dones[race, car] = True
                    infos[start + car]["position"] = int(self.rank[race, car])
                infos[start]["race_result"] = result
                if self.auto_reset:
                    self._randomize(np.array([race]))

        obs = self.observe()
        return obs, rewards.reshape(-1), dones.reshape(-1), infos

    def eval_episode(self, policy, track_name: str, max_steps: int = 7200, seed: int = 42) -> dict:
        """Run a seeded one-lap self-play race without mutating training state."""
        from .policy_io import policy_device
        import torch

        geometry = next(track for track in self.geometries if track.name == track_name)
        race = RacingVecEnv(
            [geometry],
            num_envs=1,
            cars_per_race=self.cars_per_race,
            seed=seed,
            reset_config=ResetConfig(
                randomize_along=False,
                speed_fraction=(0.0, 0.0),
                heading_noise_rad=0.0,
                lateral_offset_m=0.45,
            ),
            max_steps=max_steps,
            auto_reset=False,
        )
        race.reset()
        device = policy_device(policy)
        for _ in range(max_steps):
            obs = race.observe()
            with torch.inference_mode():
                mean, _, _ = policy.forward(torch.as_tensor(obs, dtype=torch.float32, device=device))
                actions = torch.tanh(mean).cpu().numpy()
            _, _, _, infos = race.step(actions)
            result = next((info.get("race_result") for info in infos if info.get("race_result")), None)
            if result:
                return result
        return race._race_result(0)
