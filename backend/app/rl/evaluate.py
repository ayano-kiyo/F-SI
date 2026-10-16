"""Evaluate a checkpoint: full-lap attempts + baseline comparison."""

from __future__ import annotations

import argparse
import time

import torch

from .env import RacingVecEnv
from .policy_io import load_policy
from ..track_geometry import available_track_names, load_track

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a racing checkpoint")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--tracks", nargs="*", default=["Silverstone", "Monza", "Spa"])
    parser.add_argument("--max-steps", type=int, default=14400)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    available = set(available_track_names())
    tracks = [t for t in args.tracks if t in available] or ["Silverstone"]
    policy = load_policy(args.checkpoint, device=args.device)
    cars_per_race = int(getattr(policy, "cars_per_race", 1))
    env = RacingVecEnv([load_track(t) for t in tracks], num_envs=1,
                       cars_per_race=cars_per_race, seed=42, auto_reset=False)
    print(f"checkpoint: {args.checkpoint}", flush=True)
    for name in tracks:
        t0 = time.perf_counter()
        result = env.eval_episode(policy, name, max_steps=args.max_steps)
        dt = time.perf_counter() - t0
        if cars_per_race == 1:
            status = "LAP" if result["finishes"] else result["cars"][0]["status"]
            car = result["cars"][0]
            print(f"{name:15s} {status:10s} progress={car['progress_m']:7.0f}m "
                  f"({dt:.1f}s)", flush=True)
        else:
            order = ", ".join(
                f"P{car['position']} car-{car['car'] + 1} {car['status']} "
                f"progress={car['progress_m']:.0f}m "
                f"lap_time={car['lap_time_s']:.2f}s" if car["lap_time_s"] is not None else
                f"P{car['position']} car-{car['car'] + 1} {car['status']} "
                f"progress={car['progress_m']:.0f}m lap_time=n/a"
                for car in result["cars"]
            )
            finish_rate = result["finishes"] / cars_per_race
            print(f"{name:15s} finish_rate={finish_rate:.0%} "
                  f"finishes={result['finishes']}/{cars_per_race} "
                  f"collisions={result['collisions']} order=[{order}] ({dt:.1f}s)", flush=True)

if __name__ == "__main__":
    main()
