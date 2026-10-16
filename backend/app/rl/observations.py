"""Build policy observations for solo and self-play cars."""

from __future__ import annotations

import math

import numpy as np

from .env import (
    BASE_OBS_DIM,
    OBS_DIM,
    OPPONENT_FEATURES,
    OPPONENT_SLOTS,
    _CURVATURE_GAPS,
    _LIDAR_ANGLES,
)
from .fastrack import FastTrack


def _base_observation(
    track: FastTrack, x: float, y: float, heading: float, speed: float, prev_steer: float,
) -> np.ndarray:
    along, lateral, half = track.project(x, y)
    curvature = track.curvature_ahead(along, _CURVATURE_GAPS)
    rays = track.lidar(x, y, heading, _LIDAR_ANGLES)
    reference_speed = track.speed_at(along)
    obs = np.empty(BASE_OBS_DIM, dtype=np.float32)
    obs[0] = speed / 92.0
    obs[1] = (speed - reference_speed) / 40.0
    obs[2] = np.clip(lateral / max(half, 1e-3), -2.0, 2.0) * 0.5
    obs[3] = math.sin(heading) * 0.5 + 0.5
    obs[4] = math.cos(heading) * 0.5 + 0.5
    obs[5] = np.clip(prev_steer, -1.0, 1.0) * 0.5
    obs[6] = np.clip(reference_speed / 92.0, 0.0, 1.0)
    obs[7] = np.clip(half / 12.0, 0.0, 1.0)
    obs[8:19] = np.clip(rays / 80.0, 0.0, 1.0)
    obs[19:25] = np.clip(curvature * 40.0, -1.0, 1.0)
    return obs


def observe_live(
    track: FastTrack, x: float, y: float, heading: float, speed: float, prev_steer: float = 0.0,
) -> np.ndarray:
    """Return the legacy 25-value observation used by solo checkpoints."""
    return _base_observation(track, x, y, heading, speed, prev_steer)


def observe_race(
    track: FastTrack,
    x: float,
    y: float,
    heading: float,
    speed: float,
    prev_steer: float,
    opponents: list[dict],
) -> np.ndarray:
    """Return 25 track/vehicle values plus three normalized opponent slots.

    Opponent features are longitudinal distance, lateral distance, relative
    speed, relative heading sine/cosine, and an active flag. Slots are ordered
    by distance so the shared policy has a stable nearest-opponent convention.
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[:BASE_OBS_DIM] = _base_observation(track, x, y, heading, speed, prev_steer)
    ranked = sorted(
        opponents,
        key=lambda other: (float(other["x"]) - x) ** 2 + (float(other["y"]) - y) ** 2,
    )[:OPPONENT_SLOTS]
    for index, other in enumerate(ranked):
        dx = float(other["x"]) - x
        dy = float(other["y"]) - y
        longitudinal = dx * math.cos(heading) + dy * math.sin(heading)
        lateral = -dx * math.sin(heading) + dy * math.cos(heading)
        heading_delta = float(other["heading"]) - heading
        offset = BASE_OBS_DIM + index * OPPONENT_FEATURES
        obs[offset] = np.clip(longitudinal / 120.0, -1.0, 1.0)
        obs[offset + 1] = np.clip(lateral / 30.0, -1.0, 1.0)
        obs[offset + 2] = np.clip((float(other["speed"]) - speed) / 40.0, -1.0, 1.0)
        obs[offset + 3] = math.sin(heading_delta)
        obs[offset + 4] = math.cos(heading_delta)
        obs[offset + 5] = 1.0 if bool(other.get("active", True)) else 0.0
    return obs
