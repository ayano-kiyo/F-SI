"""Numpy-accelerated track queries for RL (same semantics as track_geometry)."""

from __future__ import annotations

import math

import numpy as np

from ..track_geometry import TrackGeometry

_MAX_LIDAR_M = 80.0
_MAX_LAT_ACCEL = 17.0
_MAX_SPEED = 86.0
_MIN_SPEED = 24.0

_cache: dict[str, "FastTrack"] = {}

def get_fast_track(geometry: TrackGeometry) -> "FastTrack":
    key = geometry.name
    cached = _cache.get(key)
    if cached is None:
        cached = FastTrack(geometry)
        _cache[key] = cached
    return cached

class FastTrack:
    def __init__(self, geometry: TrackGeometry):
        self.name = geometry.name
        points = np.array([(p.x, p.y) for p in geometry.points], dtype=np.float64)
        self.center = points
        self.width_right = np.array([p.width_right for p in geometry.points], dtype=np.float64)
        self.width_left = np.array([p.width_left for p in geometry.points], dtype=np.float64)
        count = len(points)
        after = np.roll(points, -1, axis=0)
        before = np.roll(points, 1, axis=0)
        tangent = after - before
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12
        self.tangent = tangent
        self.normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        seg = after - points
        seg_len = np.linalg.norm(seg, axis=1)
        self.seg_len = seg_len
        self.cumulative = np.zeros(count)
        self.cumulative[1:] = np.cumsum(seg_len[:-1])
        self.length_m = float(seg_len.sum())
        self.curvature = self._signed_curvature(points)
        self.speed_profile = self._speed_envelope(self.curvature, seg_len)
        self.left_edge = points + self.normal * self.width_left[:, None]
        self.right_edge = points - self.normal * self.width_right[:, None]
        race = np.array(geometry.race_line, dtype=np.float64)
        self.race_line = race
        race_seg = np.linalg.norm(np.roll(race, -1, axis=0) - race, axis=1)
        self.race_cumulative = np.zeros(len(race))
        self.race_cumulative[1:] = np.cumsum(race_seg[:-1])
        self.race_length_m = float(race_seg.sum())

    @staticmethod
    def _signed_curvature(points: np.ndarray) -> np.ndarray:
        prev = np.roll(points, 1, axis=0)
        nxt = np.roll(points, -1, axis=0)
        ab = points - prev
        bc = nxt - points
        ca = nxt - prev
        cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
        denom = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ca, axis=1)
        return 2.0 * cross / np.maximum(denom, 1e-8)

    @staticmethod
    def _speed_envelope(curvature: np.ndarray, seg_len: np.ndarray) -> np.ndarray:
        raw = np.sqrt(_MAX_LAT_ACCEL / np.maximum(np.abs(curvature), 0.002))
        speed = np.clip(raw, _MIN_SPEED, _MAX_SPEED)
        for _ in range(3):
            for index in range(len(speed) - 1, -1, -1):
                nxt = (index + 1) % len(speed)
                reachable = math.sqrt(speed[nxt] ** 2 + 2 * 8.5 * seg_len[index])
                if reachable < speed[index]:
                    speed[index] = reachable
        return speed

    def project(self, x: float, y: float) -> tuple[float, float, float]:
        """Return (distance_along_m, signed_lateral_m, half_width_m)."""
        px, py = x, y
        ax = self.center[:, 0]
        ay = self.center[:, 1]
        bx = np.roll(ax, -1)
        by = np.roll(ay, -1)
        dx = bx - ax
        dy = by - ay
        len_sq = dx * dx + dy * dy + 1e-12
        frac = np.clip(((px - ax) * dx + (py - ay) * dy) / len_sq, 0.0, 1.0)
        cx = ax + frac * dx
        cy = ay + frac * dy
        ox = px - cx
        oy = py - cy
        dist_sq = ox * ox + oy * oy
        best = int(np.argmin(dist_sq))
        length = math.sqrt(len_sq[best])
        lateral = (-dy[best] * ox[best] + dx[best] * oy[best]) / length
        if lateral >= 0:
            half = self.width_left[best] + frac[best] * (self.width_left[(best + 1) % len(ax)] - self.width_left[best])
        else:
            half = self.width_right[best] + frac[best] * (self.width_right[(best + 1) % len(ax)] - self.width_right[best])
        along = float(self.cumulative[best] + frac[best] * self.seg_len[best])
        return along, float(lateral), float(half)

    def nearest_index(self, x: float, y: float) -> int:
        diff = self.center - np.array([x, y])
        return int(np.argmin(diff[:, 0] ** 2 + diff[:, 1] ** 2))

    def speed_at(self, along_m: float) -> float:
        idx = int(np.searchsorted(self.cumulative, along_m % self.length_m, side="right") - 1) % len(self.speed_profile)
        return float(self.speed_profile[idx])

    def curvature_ahead(self, along_m: float, gaps_m: tuple[float, ...]) -> np.ndarray:
        base = along_m % self.length_m
        out = np.empty(len(gaps_m))
        for i, gap in enumerate(gaps_m):
            s = (base + gap) % self.length_m
            idx = int(np.searchsorted(self.cumulative, s, side="right") - 1) % len(self.curvature)
            out[i] = self.curvature[idx]
        return out

    def lidar(self, x: float, y: float, heading: float, angles: np.ndarray) -> np.ndarray:
        dirs = np.stack([np.cos(heading + angles), np.sin(heading + angles)], axis=1)
        origin = np.array([x, y])
        best = np.full(len(angles), _MAX_LIDAR_M)
        for edge in (self.left_edge, self.right_edge):
            a = edge
            b = np.roll(edge, -1, axis=0)
            e = b - a
            ao = a - origin
            denom = dirs[:, 0, None] * e[None, :, 1] - dirs[:, 1, None] * e[None, :, 0]
            valid_denom = np.abs(denom) > 1e-9
            t = np.full_like(denom, np.inf)
            u = np.full_like(denom, np.inf)
            cross_e = ao[:, 0] * e[:, 1] - ao[:, 1] * e[:, 0]
            cross_d = ao[None, :, 0] * dirs[:, None, 1] - ao[None, :, 1] * dirs[:, None, 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(valid_denom, cross_e[None, :] / denom, np.inf)
                u = np.where(valid_denom, cross_d / denom, np.inf)
            hit = valid_denom & (t > 0.0) & (u >= 0.0) & (u <= 1.0)
            nearest = np.where(hit, t, np.inf).min(axis=1)
            best = np.minimum(best, nearest)
        return np.clip(best, 0.0, _MAX_LIDAR_M)
