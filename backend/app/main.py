"""FastAPI endpoints and fixed wall-clock WebSocket simulation loop."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .simulation import SimulationEngine
from .track_geometry import (
    DEFAULT_TRACK_NAME,
    available_track_names,
    list_track_metadata,
    load_track,
    track_label,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BACKEND_DIR / "checkpoints"
LIVE_PREVIEW_PATH = CHECKPOINT_DIR / "live-race.json"
engine = SimulationEngine(load_track(DEFAULT_TRACK_NAME))

if os.getenv("SIMULATOR_AUTO_LOAD_CHECKPOINT", "").lower() == "true":
    default_checkpoint = CHECKPOINT_DIR / "ppo-racing-latest.pt"
    if default_checkpoint.is_file():
        engine.load_checkpoint(default_checkpoint)


def _current_snapshot() -> dict:
    """Prefer an active PPO training world for the browser live view."""
    try:
        payload = json.loads(LIVE_PREVIEW_PATH.read_text())
        age_s = time.time() - float(payload.get("updated_at", 0.0))
        if payload.get("running") and age_s < 30.0:
            telemetry = payload.get("telemetry")
            if isinstance(telemetry, dict):
                return telemetry
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return engine.snapshot()

async def _simulation_loop() -> None:
    """One server-side fixed-step clock shared by all telemetry clients."""
    fixed_dt = engine.FIXED_DT_S
    previous = time.perf_counter()
    accumulator = 0.0
    while True:
        now = time.perf_counter()
        elapsed = min(0.25, max(0.0, now - previous))
        previous = now
        if engine.paused:
            accumulator = 0.0
            await asyncio.sleep(fixed_dt)
            continue
        else:
            accumulator += elapsed * engine.simulation_speed
            steps = 0
            while accumulator >= fixed_dt and steps < 16:
                engine.step(fixed_dt)
                accumulator -= fixed_dt
                steps += 1
            # Under overload, discard excessive wall-clock debt instead of letting
            # the simulator run ahead of real time in an unbounded catch-up loop.
            if steps == 16 and accumulator >= fixed_dt:
                accumulator = min(accumulator, fixed_dt)

        # Wake when the next fixed step is due instead of polling the clock 1,000
        # times a second. The accumulator still catches up if the event loop wakes late.
        until_step = max(0.0, fixed_dt - accumulator) / engine.simulation_speed
        await asyncio.sleep(min(until_step, fixed_dt))

@asynccontextmanager
async def lifespan(_app: FastAPI):
    simulation_task = asyncio.create_task(_simulation_loop())
    try:
        yield
    finally:
        simulation_task.cancel()
        try:
            await simulation_task
        except asyncio.CancelledError:
            pass

app = FastAPI(title="F-SI Autonomous Simulator", version="0.1.0", lifespan=lifespan)
_raw_origins = [value.strip() for value in os.getenv("SIMULATOR_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if value.strip()]
# A bare "*" must be passed through alone for Starlette to treat it as allow-all
# (needed for Vercel preview URLs, which change per deployment). The API carries
# no auth, so allow-all is acceptable for this public simulation service.
origins = ["*"] if "*" in _raw_origins else _raw_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def _set_track(name: str) -> dict:
    geometry = load_track(name)
    engine.set_track(geometry)
    state = engine.snapshot()
    return {
        "track": {
            "name": geometry.name,
            "label": track_label(geometry.name),
            "length_m": geometry.length_m,
            "race_line_length_m": geometry.race_line_length_m,
        },
        "planner": engine.planner.payload(),
        "telemetry": state,
    }

@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "simulator": "ready", "track": engine.track.name}

@app.get("/api/track")
async def track(name: str | None = None) -> dict:
    if name:
        return load_track(name).to_payload()
    return engine.track.to_payload()

@app.get("/api/tracks")
async def tracks() -> dict:
    metadata = list_track_metadata()
    return {"tracks": metadata, "active": engine.track.name, "default": DEFAULT_TRACK_NAME}

@app.post("/api/track")
async def select_track(payload: dict) -> dict:
    name = (payload or {}).get("name", "")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Provide a track name, e.g. {\"name\": \"Monza\"}")
    try:
        return _set_track(name.strip())
    except ValueError as error:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=str(error))

@app.get("/api/rl/checkpoints")
async def rl_checkpoints() -> dict:
    directory = CHECKPOINT_DIR
    items = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.pt")):
            items.append({"name": path.name, "size_bytes": path.stat().st_size,
                          "modified_s": path.stat().st_mtime})
    latest = None
    for suffix in ("ppo-racing-latest.pt", "smoke-latest.pt"):
        candidate = directory / suffix
        if candidate.is_file():
            latest = candidate.name
            break
    return {"checkpoints": items, "active": engine.checkpoint_path, "latest": latest}

@app.post("/api/rl/checkpoint")
async def rl_load_checkpoint(payload: dict) -> dict:
    name = (payload or {}).get("name", "")
    if not isinstance(name, str) or not name.strip():
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Provide a checkpoint name, e.g. {\"name\": \"ppo-racing-latest.pt\"}")
    cleaned = name.strip().replace("/", "").replace("\\", "")
    path = CHECKPOINT_DIR / cleaned
    try:
        info = engine.load_checkpoint(path)
    except Exception as error:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Could not load {cleaned}: {error}")
    return info

@app.get("/api/rl/ghost")
async def rl_ghost(track: str | None = None, steps: int = 3600, every: int = 6) -> dict:
    if not engine.neural.available:
        from fastapi import HTTPException
        raise HTTPException(status_code=409, detail="Load a checkpoint first via POST /api/rl/checkpoint")
    if engine.race_env is not None:
        from fastapi import HTTPException
        raise HTTPException(status_code=409, detail="Self-play checkpoints run as a multi-car race, not as a solo ghost")
    import math
    import torch
    from .dynamics import ControlInput, KinematicBicycle, VehicleState
    from .rl.fastrack import get_fast_track
    from .rl.observations import observe_live
    geometry = engine.track if not track else load_track(track)
    ft = get_fast_track(geometry)
    dynamics = KinematicBicycle()
    px, py = float(ft.race_line[0][0]), float(ft.race_line[0][1])
    qx, qy = float(ft.race_line[1][0]), float(ft.race_line[1][1])
    state = VehicleState(px, py, math.atan2(qy - py, qx - px))
    prev_steer = 0.0
    points: list[dict] = []
    for step in range(max(60, min(steps, 14400))):
        obs = observe_live(ft, state.x, state.y, state.heading_rad, state.speed_mps, prev_steer)
        with torch.inference_mode():
            mean, _, _ = engine.neural.policy.forward(torch.as_tensor(obs).unsqueeze(0))
            action = torch.tanh(mean).reshape(-1).tolist()
        steer = max(-1.0, min(1.0, float(action[0])))
        pedal = max(-1.0, min(1.0, float(action[1])))
        prev_steer = steer
        control = ControlInput(steer, max(0.0, pedal), max(0.0, -pedal))
        for _ in range(3):
            state = dynamics.step(state, control, 1.0 / 60.0)
        if step % max(1, every) == 0:
            points.append({"x": round(state.x, 2), "y": round(state.y, 2)})
        along, lat, half = ft.project(state.x, state.y)
        if abs(lat) > half + 8.0:
            break
    return {"track": geometry.name, "points": points, "checkpoint": engine.checkpoint_path}

@app.get("/api/simulator")
async def simulator_info() -> dict:
    state = _current_snapshot()
    return {
        "fixed_timestep_s": engine.FIXED_DT_S,
        "telemetry_rate_hz": 20,
        "controller": state["controller"],
        "neural_network_ready": engine.neural.available,
        "race": state.get("race"),
        "vehicles": state.get("vehicles", []),
        "planner": engine.planner.payload(),
        "track": {"name": engine.track.name, "label": track_label(engine.track.name)},
        "available_tracks": available_track_names(),
    }

async def _apply_command(websocket: WebSocket, command: dict) -> None:
    command_type = command.get("type")
    try:
        if command_type == "ping":
            return
        if command_type == "pause":
            engine.paused = True
        elif command_type == "resume":
            engine.paused = False
        elif command_type == "reset":
            engine.reset()
        elif command_type == "set_speed":
            engine.set_speed(float(command.get("value")))
        elif command_type == "set_controller":
            engine.set_controller(str(command.get("value")))
        elif command_type == "set_track":
            value = command.get("value")
            if not isinstance(value, str) or not value.strip():
                raise ValueError("set_track needs a track name, e.g. {\"type\": \"set_track\", \"value\": \"Monza\"}")
            engine.set_track(load_track(value.strip()))
        else:
            raise ValueError(f"Unknown command type: {command_type}")
    except (TypeError, ValueError) as error:
        await websocket.send_json({"type": "error", "message": str(error)})

@app.websocket("/ws/simulation")
async def simulation_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    receive_task = asyncio.create_task(websocket.receive_json())
    try:
        while True:
            done, _ = await asyncio.wait({receive_task}, timeout=0.05)
            if receive_task in done:
                try:
                    message = receive_task.result()
                except WebSocketDisconnect:
                    break
                if isinstance(message, dict):
                    await _apply_command(websocket, message)
                receive_task = asyncio.create_task(websocket.receive_json())
            await websocket.send_json(_current_snapshot())
    except WebSocketDisconnect:
        pass
    finally:
        if not receive_task.done():
            receive_task.cancel()
