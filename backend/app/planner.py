"""JEV (project-specific jerk/envelope/value) route scoring for local route choices."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .track_geometry import TrackGeometry


@dataclass(frozen=True, slots=True)
class RoutePlan:
    route_id: str
    route_cost: float
    target_index: int
    target_x: float
    target_y: float
    target_speed_mps: float
    trajectory: list[dict[str, float]]


@dataclass(frozen=True, slots=True)
class RouteCandidate:
    route_id: str
    lateral_offset_m: float
    points: list[tuple[float, float]]
    cost: float
    speed_profile_mps: list[float]
    cumulative_m: list[float]
    length_m: float


class JEVPlanner:
    """Pick the lowest-cost feasible offset of the supplied, optimized race line.

    The JEV score combines curvature energy, curvature change (a spatial jerk proxy),
    deviation from the source race line, and hard track-width feasibility. It is an
    explicitly simple route decision layer, not a learned or venue-specific planner.
    """

    OFFSETS_M = (-0.9, 0.0, 0.9)
    MAX_LATERAL_ACCEL_MPS2 = 17.0
    MAX_SPEED_MPS = 86.0
    COMFORT_BRAKE_MPS2 = 8.5

    def __init__(self, track: TrackGeometry):
        self.track = track
        self.candidates = [self._build_candidate(offset) for offset in self.OFFSETS_M]
        self.selected = min(self.candidates, key=lambda candidate: candidate.cost)

    def _build_candidate(self, offset_m: float) -> RouteCandidate:
        reference = self.track.race_line
        count = len(reference)
        points: list[tuple[float, float]] = []
        feasible = True
        lateral_energy = 0.0
        for index, (x, y) in enumerate(reference):
            before, after = reference[(index - 1) % count], reference[(index + 1) % count]
            dx, dy = after[0] - before[0], after[1] - before[1]
            tangent_length = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / tangent_length, dx / tangent_length
            px, py = x + nx * offset_m, y + ny * offset_m
            projection = self.track.project(px, py)
            if projection.signed_lateral_m < -projection.width_right_m + 0.25 or projection.signed_lateral_m > projection.width_left_m - 0.25:
                feasible = False
            lateral_energy += projection.signed_lateral_m**2
            points.append((px, py))

        curvature = self._curvatures(points)
        ds = self.track.race_line_length_m / count
        curvature_energy = sum(value * value for value in curvature) / count
        curvature_change = sum((curvature[(i + 1) % count] - curvature[i]) ** 2 for i in range(count)) / count
        lateral_cost = lateral_energy / count
        cost = 1000.0 * curvature_energy + 12.0 * curvature_change / max(ds, 1.0) + 0.025 * lateral_cost
        if not feasible:
            cost += 1_000_000.0

        speed = [
            min(self.MAX_SPEED_MPS, max(24.0, math.sqrt(self.MAX_LATERAL_ACCEL_MPS2 / max(abs(kappa), 0.002))))
            for kappa in curvature
        ]
        # Circular backward passes form a conservative braking envelope over the lap.
        for _ in range(3):
            for index in range(count - 1, -1, -1):
                next_index = (index + 1) % count
                segment = math.dist(points[index], points[next_index])
                reachable = math.sqrt(speed[next_index] ** 2 + 2 * self.COMFORT_BRAKE_MPS2 * segment)
                speed[index] = min(speed[index], reachable)

        cumulative = [0.0]
        for first, second in zip(points, points[1:]):
            cumulative.append(cumulative[-1] + math.dist(first, second))
        length = cumulative[-1] + math.dist(points[-1], points[0])
        return RouteCandidate(
            route_id=f"jeV-offset-{offset_m:+.1f}",
            lateral_offset_m=offset_m,
            points=points,
            cost=cost,
            speed_profile_mps=speed,
            cumulative_m=cumulative,
            length_m=length,
        )

    @staticmethod
    def _curvatures(points: list[tuple[float, float]]) -> list[float]:
        result: list[float] = []
        count = len(points)
        for index in range(count):
            ax, ay = points[(index - 1) % count]
            bx, by = points[index]
            cx, cy = points[(index + 1) % count]
            ab = math.dist((ax, ay), (bx, by))
            bc = math.dist((bx, by), (cx, cy))
            ca = math.dist((cx, cy), (ax, ay))
            area2 = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
            result.append(2 * area2 / max(ab * bc * ca, 1e-8))
        return result

    @staticmethod
    def _advance_index(candidate: RouteCandidate, start: int, distance_m: float) -> int:
        index = start % len(candidate.points)
        traveled = 0.0
        while traveled < distance_m:
            next_index = (index + 1) % len(candidate.points)
            traveled += math.dist(candidate.points[index], candidate.points[next_index])
            index = next_index
        return index

    def plan(self, x: float, y: float, speed_mps: float) -> RoutePlan:
        candidate = self.selected
        start = min(
            range(len(candidate.points)),
            key=lambda i: (candidate.points[i][0] - x) ** 2 + (candidate.points[i][1] - y) ** 2,
        )
        lookahead_m = max(9.0, min(34.0, 7.0 + speed_mps * 0.42))
        target_index = self._advance_index(candidate, start, lookahead_m)
        trajectory = []
        for offset in range(10, 181, 10):
            index = self._advance_index(candidate, start, float(offset))
            px, py = candidate.points[index]
            trajectory.append({"x": px, "y": py})
        tx, ty = candidate.points[target_index]
        return RoutePlan(
            route_id=candidate.route_id,
            route_cost=candidate.cost,
            target_index=target_index,
            target_x=tx,
            target_y=ty,
            target_speed_mps=candidate.speed_profile_mps[target_index],
            trajectory=trajectory,
        )

    def payload(self) -> dict:
        return {
            "name": "JEV",
            "description": "Curvature, spatial-jerk, lateral deviation, and track-width route score",
            "selected_route": self.selected.route_id,
            "route_cost": self.selected.cost,
            "candidate_costs": {candidate.route_id: candidate.cost for candidate in self.candidates},
        }
