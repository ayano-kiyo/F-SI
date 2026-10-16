"""Record a neural rollout as ghost polyline points for the visualizer."""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from .observations import observe_live
from .fastrack import get_fast_track
from .policy_io import load_policy
from ..dynamics import KinematicBicycle, ControlInput, VehicleState
from ..track_geometry import load_track

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a neural ghost lap")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--track", type=str, default="Silverstone")
    parser.add_argument("--max-steps", type=int, default=14400)
    parser.add_argument("--every", type=int, default=6)
    parser.add_argument("--out", type=str, default="")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    policy = load_policy(args.checkpoint)
    if policy.obs_dim != 25:
        raise SystemExit("self-play checkpoints contain multiple cars; use the simulator race view instead")
    geometry = load_track(args.track)
    ft = get_fast_track(geometry)
    dynamics = KinematicBicycle()
    px, py = float(ft.race_line[0][0]), float(ft.race_line[0][1])
    qx, qy = float(ft.race_line[1][0]), float(ft.race_line[1][1])
    state = VehicleState(px, py, math.atan2(qy - py, qx - px))
    prev_steer = 0.0
    points: list[list[float]] = []
    for step in range(args.max_steps):
        obs = observe_live(ft, state.x, state.y, state.heading_rad, state.speed_mps, prev_steer)
        with torch.inference_mode():
            mean, _, _ = policy.forward(torch.as_tensor(obs).unsqueeze(0))
            action = torch.tanh(mean).reshape(-1).tolist()
        steer = max(-1.0, min(1.0, float(action[0])))
        pedal = max(-1.0, min(1.0, float(action[1])))
        prev_steer = steer
        control = ControlInput(steer, max(0.0, pedal), max(0.0, -pedal))
        for _ in range(3):
            state = dynamics.step(state, control, 1.0 / 60.0)
        if step % args.every == 0:
            points.append([round(state.x, 2), round(state.y, 2)])
        along, lat, half = ft.project(state.x, state.y)
        if abs(lat) > half + 8.0:
            print(f"off track at step {step}; recorded {len(points)} points", flush=True)
            break
    print(f"ghost points: {len(points)}", flush=True)
    if args.out:
        import json
        from pathlib import Path
        Path(args.out).write_text(json.dumps({"track": args.track, "points": points}))
        print(f"wrote {args.out}", flush=True)
    else:
        print(points[:5], "...", points[-3:] if points else [])

if __name__ == "__main__":
    main()
