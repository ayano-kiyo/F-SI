"""Train one shared PPO policy through parallel multi-car self-play races."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .env import OBS_DIM, OBS_VERSION, RacingVecEnv
from .ppo import ActorCritic, PPOConfig, RolloutBuffer, ppo_update
from ..track_geometry import available_track_names, load_track

DEFAULT_TRACKS = ["Silverstone", "Monza", "Spa", "Suzuka", "Catalunya", "Austin", "Sepang", "Spielberg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train F-SI shared-policy racing PPO")
    parser.add_argument("--tracks", nargs="*", default=DEFAULT_TRACKS)
    parser.add_argument("--num-races", "--num-envs", dest="num_races", type=int, default=8,
                        help="independent race worlds (default: 8; --num-envs remains an alias)")
    parser.add_argument("--cars-per-race", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--total-steps", type=int, default=2_000_000,
                        help="active car transitions collected across all race worlds")
    parser.add_argument("--rollout-steps", type=int, default=512,
                        help="20 Hz steps per PPO update")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--name", type=str, default="ppo-racing")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=8,
                        help="PPO updates between deterministic race evaluations; 0 disables evaluation")
    parser.add_argument("--eval-track", type=str, default="Silverstone")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--preview-file", type=str, default="checkpoints/live-race.json",
                        help="atomic telemetry file consumed by the browser simulator")
    parser.add_argument("--preview-every", type=int, default=1,
                        help="training environment steps between live preview updates (default: every step)")
    return parser.parse_args()


def _checkpoint_payload(policy, optimizer, args, tracks, iteration, global_step) -> dict:
    return {
        "checkpoint_version": 2,
        "observation_version": OBS_VERSION,
        "observation_layout": {
            "base_features": 25,
            "opponent_slots": 3,
            "opponent_features_per_slot": [
                "relative_longitudinal_position",
                "relative_lateral_position",
                "relative_speed",
                "relative_heading_sin",
                "relative_heading_cos",
                "active",
            ],
            "total_features": OBS_DIM,
        },
        "race_mode": "shared_policy_self_play",
        "cars_per_race": args.cars_per_race,
        "num_races": args.num_races,
        "race_settings": {
            "laps": 1,
            "max_steps_per_race": 7200,
            "control_rate_hz": 20,
            "randomized_track_and_grid_start": True,
        },
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "global_step": global_step,
        "tracks": tracks,
        "obs_dim": OBS_DIM,
    }


def _live_preview_payload(env, actions: np.ndarray, env_step: int, global_step: int, running: bool) -> dict:
    """Build browser telemetry for the first training race world."""
    race = 0
    track_index = int(env.track_idx[race])
    geometry = env.geometries[track_index]
    fast_track = env.tracks[track_index]
    cars = []
    for car in range(env.cars_per_race):
        action = actions[race * env.cars_per_race + car]
        fraction = float(np.clip(env.lap_progress[race, car] / fast_track.race_length_m, 0.0, 1.0))
        cars.append({
            "id": f"car-{car + 1}",
            "x": float(env.x[race, car]),
            "y": float(env.y[race, car]),
            "heading_rad": float(env.heading[race, car]),
            "speed_mps": float(env.speed[race, car]),
            "acceleration_mps2": float(env.acceleration[race, car]),
            "yaw_rate_rad_s": float(env.yaw_rate[race, car]),
            "steering": float(action[0]),
            "throttle": max(0.0, float(action[1])),
            "brake": max(0.0, -float(action[1])),
            "progress_m": float(env.lap_progress[race, car]),
            "fraction": fraction,
            "status": str(env.reason[race, car]),
            "position": int(env.rank[race, car]) or None,
            "target_waypoint": {"x": float(env.x[race, car]), "y": float(env.y[race, car])},
            "route_id": "shared-policy-training",
        })
    # Keep the existing singular telemetry useful even after the grid car
    # retires: show a live driver while any car in the pack is still racing.
    primary_index = next(
        (index for index in range(env.cars_per_race)
         if env.active[race, index] and env.speed[race, index] > 0.5),
        next((index for index in range(env.cars_per_race) if env.active[race, index]), 0),
    )
    primary = cars[primary_index]
    return {
        "running": running,
        "updated_at": time.time(),
        "telemetry": {
            "type": "telemetry",
            "sim_time_s": float(env_step / 20.0),
            "paused": False,
            "simulation_speed": 1.0,
            "track": {"name": geometry.name, "length_m": geometry.length_m},
            "vehicle": {"id": primary["id"], **{key: primary[key] for key in (
                "x", "y", "heading_rad", "speed_mps", "acceleration_mps2", "yaw_rate_rad_s",
                "steering", "throttle", "brake", "target_waypoint", "route_id",
            )}},
            "vehicles": cars,
            "race": {
                "track": geometry.name,
                "finished": False,
                "collisions": int(env.collision_count[race]),
                "training": True,
                "training_step": int(global_step),
                "race_index": race,
                "num_races": int(env.num_races),
                "cars": [
                    {"car": index, "position": car["position"], "status": car["status"],
                     "progress_m": car["progress_m"]}
                    for index, car in enumerate(cars)
                ],
            },
            "planner": {"name": "PPO", "route_cost": 0.0, "route_id": "shared-policy-training"},
            "controller": {"name": "training", "available": True, "checkpoint": None},
            "progress": {
                "distance_m": primary["progress_m"],
                "fraction": primary["fraction"],
                "lap": 0,
                "lap_time_s": float(env.race_steps[race] / 20.0),
                "best_lap_time_s": None,
                "off_track": primary["status"] == "off_track",
                "lateral_error_m": float(env.lateral[race, primary_index]),
            },
            "planned_trajectory": [],
        },
    }


def _write_live_preview(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")))
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.num_races < 1 or args.total_steps < 1 or args.rollout_steps < 1 or args.preview_every < 1:
        raise SystemExit("--num-races, --total-steps, --rollout-steps, and --preview-every must be positive")
    if args.save_every < 1 or args.eval_every < 0:
        raise SystemExit("--save-every must be positive and --eval-every cannot be negative")

    available = set(available_track_names())
    unknown = [name for name in args.tracks if name not in available]
    if unknown:
        raise SystemExit(f"unknown track(s): {', '.join(unknown)}")
    track_names = args.tracks or ["Silverstone"]
    geometries = [load_track(name) for name in track_names]
    if args.eval_track not in available:
        raise SystemExit(f"unknown evaluation track: {args.eval_track}")

    print(
        f"tracks: {track_names} | races: {args.num_races} | cars/race: {args.cars_per_race} "
        f"| active-agent steps: {args.total_steps} | device: {args.device}",
        flush=True,
    )
    env = RacingVecEnv(
        geometries,
        num_envs=args.num_races,
        cars_per_race=args.cars_per_race,
        seed=args.seed,
    )
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = PPOConfig(lr=args.lr, rollout_steps=args.rollout_steps, total_steps=args.total_steps)
    policy = ActorCritic(obs_dim=OBS_DIM).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    obs = env.reset()
    global_step = 0
    env_step = 0
    iteration = 0
    completed_races = 0
    finish_count = 0
    collision_count = 0
    recent_results: list[dict] = []
    start = time.perf_counter()
    preview_path = Path(args.preview_file)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.unlink(missing_ok=True)
    preview_actions = np.zeros((env.num_agents, 2), dtype=np.float32)
    _write_live_preview(
        preview_path, _live_preview_payload(env, preview_actions, env_step, global_step, True),
    )

    while global_step < args.total_steps:
        buffer = RolloutBuffer(config.rollout_steps, env.num_agents, OBS_DIM)
        for _ in range(config.rollout_steps):
            active_before = env.active_mask
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.inference_mode():
                action, log_prob, value, _ = policy.act(obs_t)
            action_np = action.cpu().numpy()
            next_obs, reward, done, infos = env.step(action_np)
            env_step += 1
            preview_actions = action_np
            if env_step % args.preview_every == 0:
                _write_live_preview(
                    preview_path,
                    _live_preview_payload(env, preview_actions, env_step, global_step + int(active_before.sum()), True),
                )
            buffer.add(
                obs, action_np, log_prob.cpu().numpy(), reward, done.astype(np.float32),
                value.cpu().numpy(), valid=active_before,
            )
            global_step += int(active_before.sum())
            for info in infos:
                result = info.get("race_result")
                if not result:
                    continue
                completed_races += 1
                finish_count += int(result["finishes"])
                collision_count += int(result["collisions"])
                recent_results.append(result)
                if len(recent_results) > 100:
                    recent_results.pop(0)
            obs = next_obs

        with torch.inference_mode():
            _, _, last_values, _ = policy.act(torch.as_tensor(obs, dtype=torch.float32, device=device))
        advantages, returns = buffer.advantages(
            last_values.cpu().numpy(), done.astype(np.float32), config.gamma, config.lam,
        )
        stats = ppo_update(policy, optimizer, buffer, advantages, returns, config, device)
        iteration += 1
        finished_cars = sum(int(result["finishes"]) for result in recent_results)
        cars_observed = sum(len(result["cars"]) for result in recent_results)
        recent_collisions = sum(int(result["collisions"]) for result in recent_results)
        record = {
            "iteration": iteration,
            "global_step": global_step,
            "completed_races": completed_races,
            "recent_finish_rate": finished_cars / max(1, cars_observed),
            "recent_collisions_per_race": recent_collisions / max(1, len(recent_results)),
            "mean_reward_step": float(buffer.rewards[buffer.valid].mean()) if buffer.valid.any() else 0.0,
            "mean_advantage": float(advantages[buffer.valid].mean()) if buffer.valid.any() else 0.0,
            **{key: float(value) for key, value in stats.items()},
            "elapsed_s": time.perf_counter() - start,
        }
        history.append(record)
        print(
            f"iter={iteration} car_steps={global_step} races={completed_races} "
            f"finish_rate={record['recent_finish_rate']:.2f} "
            f"collisions/race={record['recent_collisions_per_race']:.2f} "
            f"rew/step={record['mean_reward_step']:.3f} vloss={stats['value_loss']:.3f} "
            f"elapsed={record['elapsed_s']:.0f}s",
            flush=True,
        )

        if args.eval_every and iteration % args.eval_every == 0:
            evaluation = env.eval_episode(policy, args.eval_track)
            record["evaluation"] = evaluation
            eval_order = ", ".join(
                f"P{car['position']} car-{car['car'] + 1} {car['status']} "
                f"lap_time={car['lap_time_s']:.2f}s" if car["lap_time_s"] is not None else
                f"P{car['position']} car-{car['car'] + 1} {car['status']} lap_time=n/a"
                for car in evaluation["cars"]
            )
            eval_finish_rate = evaluation["finishes"] / args.cars_per_race
            print(
                f"  eval {args.eval_track}: finish_rate={eval_finish_rate:.0%} "
                f"finishes={evaluation['finishes']}/{args.cars_per_race} "
                f"collisions={evaluation['collisions']} order=[{eval_order}]",
                flush=True,
            )

        if iteration % args.save_every == 0 or global_step >= args.total_steps:
            payload = _checkpoint_payload(policy, optimizer, args, track_names, iteration, global_step)
            path = checkpoint_dir / f"{args.name}-iter{iteration:04d}.pt"
            torch.save(payload, path)
            latest = checkpoint_dir / f"{args.name}-latest.pt"
            torch.save(payload, latest)
            (checkpoint_dir / f"{args.name}-history.json").write_text(json.dumps(history, indent=1))
            print(f"  saved {path}", flush=True)

    _write_live_preview(
        preview_path, _live_preview_payload(env, preview_actions, env_step, global_step, False),
    )
    print(
        f"done in {time.perf_counter() - start:.0f}s; latest: "
        f"{checkpoint_dir / (args.name + '-latest.pt')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
