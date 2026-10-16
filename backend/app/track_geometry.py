"""Load and query the local metric TUMFTM track data."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True, slots=True)
class TrackPoint:
    x: float
    y: float
    width_right: float
    width_left: float

@dataclass(frozen=True, slots=True)
class Projection:
    x: float
    y: float
    segment_index: int
    fraction: float
    signed_lateral_m: float
    width_right_m: float
    width_left_m: float
    distance_m: float

class TrackGeometry:
    """Closed 2D track geometry in the source's meter-valued local plane."""

    def __init__(self, points: list[TrackPoint], race_line: list[tuple[float, float]], name: str = "Silverstone"):
        if len(points) < 3 or len(race_line) < 3:
            raise ValueError("A closed circuit needs at least three track and race-line points")
        self.name = name
        self.points = points
        self.race_line = race_line
        self.length_m = self._closed_length(points)
        self.race_line_length_m = self._closed_length_xy(race_line)
        self._cumulative = self._cumulative_distances(race_line)

    @staticmethod
    def _closed_length(points: list[TrackPoint]) -> float:
        return TrackGeometry._closed_length_xy([(p.x, p.y) for p in points])

    @staticmethod
    def _closed_length_xy(points: list[tuple[float, float]]) -> float:
        return sum(math.dist(a, b) for a, b in zip(points, points[1:] + points[:1]))

    @staticmethod
    def _cumulative_distances(points: list[tuple[float, float]]) -> list[float]:
        result = [0.0]
        for first, second in zip(points, points[1:]):
            result.append(result[-1] + math.dist(first, second))
        return result

    @classmethod
    def from_csv(cls, track_csv: Path, race_line_csv: Path, name: str = "Silverstone") -> "TrackGeometry":
        points: list[TrackPoint] = []
        with track_csv.open(newline="", encoding="utf-8") as source:
            for row in csv.reader(source):
                if not row or row[0].lstrip().startswith("#"):
                    continue
                points.append(TrackPoint(float(row[0]), float(row[1]), float(row[2]), float(row[3])))

        race_line: list[tuple[float, float]] = []
        with race_line_csv.open(newline="", encoding="utf-8") as source:
            for row in csv.reader(source):
                if not row or row[0].lstrip().startswith("#"):
                    continue
                race_line.append((float(row[0]), float(row[1])))
        return cls(points, race_line, name)

    def point_at(self, index: int) -> TrackPoint:
        return self.points[index % len(self.points)]

    def tangent_at(self, index: int) -> tuple[float, float]:
        before = self.point_at(index - 1)
        after = self.point_at(index + 1)
        dx, dy = after.x - before.x, after.y - before.y
        length = math.hypot(dx, dy) or 1.0
        return dx / length, dy / length

    def project(self, x: float, y: float) -> Projection:
        """Project to the closest centerline segment, retaining signed edge distances."""
        best: Projection | None = None
        best_d2 = math.inf
        count = len(self.points)
        for index, a in enumerate(self.points):
            b = self.points[(index + 1) % count]
            dx, dy = b.x - a.x, b.y - a.y
            length_sq = dx * dx + dy * dy
            if length_sq <= 1e-12:
                continue
            fraction = max(0.0, min(1.0, ((x - a.x) * dx + (y - a.y) * dy) / length_sq))
            px, py = a.x + fraction * dx, a.y + fraction * dy
            offset_x, offset_y = x - px, y - py
            d2 = offset_x * offset_x + offset_y * offset_y
            if d2 >= best_d2:
                continue
            length = math.sqrt(length_sq)
            signed = (-dy * offset_x + dx * offset_y) / length
            right = a.width_right + fraction * (b.width_right - a.width_right)
            left = a.width_left + fraction * (b.width_left - a.width_left)
            best_d2 = d2
            best = Projection(px, py, index, fraction, signed, right, left, math.sqrt(d2))
        if best is None:
            raise ValueError("Track geometry has no nonzero segments")
        return best

    def nearest_race_line_index(self, x: float, y: float) -> int:
        return min(range(len(self.race_line)), key=lambda i: (self.race_line[i][0] - x) ** 2 + (self.race_line[i][1] - y) ** 2)

    def race_line_distance_at(self, index: int) -> float:
        return self._cumulative[index % len(self.race_line)]

    def to_payload(self) -> dict:
        return {
            "name": self.name,
            "label": track_label(self.name),
            "length_m": self.length_m,
            "race_line_length_m": self.race_line_length_m,
            "coordinate_system": "local_metric_xy",
            "elevation_available": False,
            "centerline": [
                {"x": p.x, "y": p.y, "width_right_m": p.width_right, "width_left_m": p.width_left}
                for p in self.points
            ],
            "race_line": [{"x": x, "y": y} for x, y in self.race_line],
        }

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRACKS_DIR = DATA_DIR / "tracks"
RACELINES_DIR = DATA_DIR / "racelines"
DEFAULT_TRACK_NAME = "Silverstone"

TRACK_LABELS = {
    "Austin": "Austin (COTA)",
    "BrandsHatch": "Brands Hatch",
    "Budapest": "Budapest (Hungaroring)",
    "Catalunya": "Catalunya",
    "Hockenheim": "Hockenheim",
    "IMS": "Indianapolis",
    "Melbourne": "Melbourne",
    "MexicoCity": "Mexico City",
    "Montreal": "Montreal",
    "Monza": "Monza",
    "MoscowRaceway": "Moscow Raceway",
    "Norisring": "Norisring",
    "Nuerburgring": "Nurburgring",
    "Oschersleben": "Oschersleben",
    "Sakhir": "Sakhir (Bahrain)",
    "SaoPaulo": "Sao Paulo (Interlagos)",
    "Sepang": "Sepang",
    "Shanghai": "Shanghai",
    "Silverstone": "Silverstone",
    "Sochi": "Sochi",
    "Spa": "Spa-Francorchamps",
    "Spielberg": "Spielberg (Red Bull Ring)",
    "Suzuka": "Suzuka",
    "YasMarina": "Yas Marina",
    "Zandvoort": "Zandvoort",
}

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_track_cache: dict[str, TrackGeometry] = {}

def track_label(name: str) -> str:
    return TRACK_LABELS.get(name, name)

def available_track_names() -> list[str]:
    """Names with both a track CSV and a raceline CSV present."""
    if not TRACKS_DIR.is_dir() or not RACELINES_DIR.is_dir():
        return [DEFAULT_TRACK_NAME]
    names: list[str] = []
    for path in TRACKS_DIR.glob("*.csv"):
        stem = path.stem
        if not _NAME_RE.match(stem):
            continue
        if (RACELINES_DIR / f"{stem}.csv").is_file():
            names.append(stem)
    names.sort()
    if DEFAULT_TRACK_NAME in names:
        names.remove(DEFAULT_TRACK_NAME)
        names.insert(0, DEFAULT_TRACK_NAME)
    return names or [DEFAULT_TRACK_NAME]

def load_track(name: str) -> TrackGeometry:
    """Load (and cache) a track by name; raises ValueError for unknown names."""
    cleaned = (name or "").strip()
    if not _NAME_RE.match(cleaned):
        raise ValueError(f"Unknown track: {name!r}")
    if cleaned in _track_cache:
        return _track_cache[cleaned]
    track_csv = TRACKS_DIR / f"{cleaned}.csv"
    race_csv = RACELINES_DIR / f"{cleaned}.csv"
    if not track_csv.is_file() or not race_csv.is_file():
        available = ", ".join(available_track_names())
        raise ValueError(f"Unknown track {cleaned!r}. Available tracks: {available}")
    geometry = TrackGeometry.from_csv(track_csv, race_csv, name=cleaned)
    _track_cache[cleaned] = geometry
    return geometry

def list_track_metadata() -> list[dict]:
    return [
        {
            "name": geometry.name,
            "label": track_label(geometry.name),
            "length_m": geometry.length_m,
            "race_line_length_m": geometry.race_line_length_m,
            "centerline_points": len(geometry.points),
            "race_line_points": len(geometry.race_line),
        }
        for geometry in (load_track(name) for name in available_track_names())
    ]

DEFAULT_TRACK = load_track(DEFAULT_TRACK_NAME)
