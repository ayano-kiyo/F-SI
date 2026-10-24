# F-SI Autonomous Racing Simulator

A browser-based, deterministic first slice of a 3D autonomous racing simulator. The local build connects a FastAPI fixed-step vehicle simulation to a React/Three.js visualization over WebSocket. It ships all 25 TUMFTM circuits (Silverstone default) with published 2D centerlines, measured track widths, and optimized race line; the scene is deliberately flat because the source does not provide venue elevation data.

## Project status

- Road and runoff meshes are generated per circuit from the source centerline and left/right widths; switch tracks from the top-right picker without restarting.
- A JEV route optimizer compares feasible lateral variants of the source race line. The baseline pure-pursuit controller follows the selected route and curvature speed envelope.
- A deterministic kinematic bicycle model advances at a fixed 1/60-second simulation timestep, independent of browser rendering. The WebSocket streams actual state at 20 Hz.
- The browser loads the CC0 current ground-effect racer GLB. A procedural 3D open-wheel car remains visible if that asset fails to load.
- A PyTorch actor-critic is trained with PPO in parallel four-car self-play races. Checkpoints can be loaded into the browser to watch the cars race on the selected circuit.
- Scenic verge, kerbs, barriers, grass, and simplified grandstand structures are generated at runtime. They are illustrative scenery, not a venue-accurate 3D replica.

## Requirements

- Python 3.10+
- Node.js 20+

## Run locally

In one terminal:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` and `/ws` to FastAPI on port 8000. The WebSocket controls pause, resume, reset, simulation speed, and controller selection. Open **Train / Race** to start an autonomous 2-, 3-, or 4-car race on the selected circuit without training a policy first; a loaded shared self-play checkpoint is used when available, otherwise the built-in baseline controller drives each car. Use **Return to solo** to leave the race. Browser rendering runs independently from the server's fixed-step physics.

Optional frontend proxy/API settings are listed in `frontend/.env.example`.

## Train a self-play policy

The training command runs from the repository root, installs the optional NumPy/PyTorch training packages into the active Python environment, and starts eight parallel race worlds with four cars per race:

```bash
cd backend && (test -x .venv/bin/python || python3 -m venv .venv) && .venv/bin/python -m pip install -r requirements-rl.txt && .venv/bin/python -m app.rl.train --num-races 8 --cars-per-race 4 --total-steps 2000000
```

The policy is a shared 256-by-256 actor-critic trained with PPO. Each car observes the track and up to three nearby opponents. Race episodes randomize their start point and track from the default eight-circuit set; completed episodes report finish order, lap completions, and collisions. Use `--cars-per-race 2` or `3` to reduce the field, `--num-races` to change the number of parallel race worlds, and `--total-steps` to adjust training duration. `--num-envs` remains an alias for `--num-races`.

When the simulator backend and browser are running, the trainer writes a live snapshot of race world 1 to `backend/checkpoints/live-race.json`, and the browser displays it while training is active. Once training stops, refresh the checkpoint list and load a saved model to watch its standalone race.

The multi-car environment design references [F1TENTH Gym](https://f1tenth-gym.readthedocs.io/en/stable/), and the independent race batching follows the vectorized-environment pattern described in the [Stable-Baselines3 guide](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html).

Training writes versioned checkpoints and a JSON history under `backend/checkpoints`. Open the simulator's Train / Race panel, refresh checkpoints after training, and load a checkpoint to view the self-play race. Legacy 25-value solo checkpoints still run one car. The panel's copy button provides the default command above. The command uses the backend virtual environment so it also works when the system Python is managed by Homebrew or another package manager.

## Verify

```bash
cd backend && python -m unittest discover -s tests -v
cd frontend && npm run build
```

## Architecture

```text
backend/app/
  track_geometry.py  CSV parsing, local metric geometry, projections
  planner.py         JEV route choice, curvature speed envelope
  controller.py      baseline controller and legacy solo neural policy
  dynamics.py        fixed-step kinematic bicycle model
  simulation.py      state, lap timing, solo and multi-car races, controls and reset
  main.py            REST metadata and WebSocket telemetry/control endpoint
  rl/                PyTorch PPO, multi-car race worlds, observations and evaluation
frontend/src/
  components/        R3F scene, track, car, telemetry, camera and controls
  hooks/             WebSocket state and UI control client
```

## Coordinate convention

The source CSV columns `x_m,y_m` are used as a local planar coordinate system in meters. Three.js maps them to `(x, 0, -y)` with positive Y as up. Vehicle heading is measured in the CSV plane from +x toward +y. This transform is applied consistently to the road, boundaries, race line, planned trajectory, and vehicle. The source README describes the points as smoothed GPS-derived centerlines but does not identify a geodetic CRS or provide elevation. No elevation is synthesized.

The TUM source has 1,178 Silverstone centerline points and 1,161 optimized race-line points. Track widths are stored as right and left widths from the local centerline. The supplied widths vary by location, and source documentation cautions that accuracy varies with GPS and image data quality.

## Telemetry and WebSocket schema

Connect to `ws://localhost:8000/ws/simulation` (the Vite proxy makes the browser use `/ws/simulation`). The server emits `type: "telemetry"` JSON objects at 20 Hz. Positions are local meters in CSV coordinates; speed is m/s; steering is normalized to `[-1, 1]`; throttle and brake are `[0, 1]`; angles are radians.

```json
{
  "type": "telemetry",
  "sim_time_s": 12.5,
  "paused": false,
  "simulation_speed": 1.0,
  "track": { "name": "Silverstone", "length_m": 5886.8 },
  "vehicle": {
    "x": 123.4, "y": -45.6, "heading_rad": 0.8,
    "speed_mps": 41.2, "acceleration_mps2": 2.1,
    "yaw_rate_rad_s": 0.18, "steering": 0.12,
    "throttle": 0.6, "brake": 0.0,
    "target_waypoint": { "x": 135.0, "y": -35.0 },
    "route_id": "jeV-offset-0.0"
  },
  "vehicles": [
    {
      "id": "car-1", "x": 123.4, "y": -45.6, "heading_rad": 0.8,
      "speed_mps": 41.2, "progress_m": 345.2, "fraction": 0.0587,
      "status": "racing", "position": null
    }
  ],
  "planner": { "name": "JEV", "route_cost": 0.034 },
  "controller": { "name": "baseline", "available": true },
  "progress": {
    "distance_m": 345.2, "fraction": 0.0587, "lap": 0,
    "lap_time_s": 12.5, "best_lap_time_s": null,
    "off_track": false, "lateral_error_m": 0.4
  },
  "planned_trajectory": [{ "x": 130.0, "y": -40.0 }],
  "race": null
}
```

List tracks with `GET /api/tracks` and fetch one with `GET /api/track?name=Monza` (or `POST /api/track` with `{"name":"Monza"}`). Client commands are JSON objects: `{"type":"pause"}`, `{"type":"resume"}`, `{"type":"reset"}`, `{"type":"set_speed","value":2}`, `{"type":"set_controller","value":"baseline"}`, `{"type":"set_track","value":"Monza"}`, `{"type":"start_race","cars_per_race":4}`, and `{"type":"stop_race"}`. Race fields must contain 2–4 cars. List checkpoints with `GET /api/rl/checkpoints` and load one with `POST /api/rl/checkpoint`.

## Assets and source licenses

- Circuit data: [TUMFTM/racetrack-database](https://github.com/TUMFTM/racetrack-database), pinned at commit `e59595d1f3573b30d1ded6a08984935b957688e0`. Its repository declares LGPL-3.0; the upstream README says its smoothed centerlines were derived from OpenStreetMap and widths were extracted from satellite imagery. The upstream data-quality caveat is retained here. A copy of its license is at `backend/data/TUMFTM-LICENSE.txt`.
- Car and grandstand models: [Current ground-effect racer](https://3dassets.dev/assets/grand-prix-circuit-and-open-wheel-ground-effect-racer-02c3b491) and [Six-tier grandstand module](https://3dassets.dev/assets/grand-prix-circuit-and-open-wheel-grandstand-straight-0ea66f7b) from the [Formula Race Cars and Grand Prix Circuit pack](https://3dassets.dev/packs/grand-prix-circuit-and-open-wheel), stored under `frontend/public/assets`. The pack declares CC0 1.0 Universal and provides meter-scale, +Y-up GLBs. No Formula 1 team or manufacturer branding is included.

## Deployment

Build the frontend with `npm run build` in `frontend`; publish `frontend/dist` to any static host. Deploy `backend` with Python 3.10+ and run `uvicorn app.main:app --host 0.0.0.0 --port $PORT` from that directory. Configure the static host's `/api` and `/ws` reverse proxies to the Python service, enabling WebSocket upgrades and setting `VITE_API_BASE_URL` only if the API is hosted on a separate origin. For cross-origin deployments, set `SIMULATOR_CORS_ORIGINS` to a comma-separated allowlist of frontend origins. A public browser preview needs both deployed services and the WebSocket proxy; the repository itself does not provision hosting.

### Vercel frontend with a Cloudflare backend

The backend can run as a Cloudflare Container behind a Worker and Durable Object. This keeps the existing FastAPI, NumPy, PyTorch, fixed-step simulation loop, and WebSocket endpoint intact. Cloudflare Python Workers include FastAPI and support pure Python or PyEmscripten wheels; this deployment uses a Container for the native PyTorch dependency and long-running service.

Cloudflare Containers require a Workers Paid plan and are billed for active container resources. The configuration uses a `standard-1` instance (0.5 vCPU, 4 GiB memory, 8 GB disk); see Cloudflare's current [Containers pricing](https://developers.cloudflare.com/containers/platform/pricing/) and [instance limits](https://developers.cloudflare.com/containers/platform/limits/) before deploying.

From `backend`, install the Wrangler dependency and deploy the Worker:

```bash
npm install
npx wrangler login
npm run deploy
```

The Worker forwards HTTP and WebSocket requests to one shared simulator container. Set the Vercel frontend project's `VITE_API_BASE_URL` environment variable to the deployed Worker origin, then redeploy the frontend. The Worker passes `SIMULATOR_CORS_ORIGINS` to FastAPI; it is `*` in the starter configuration so Vercel preview URLs work. Since the API is public, narrow this to the production Vercel origin if preview deployments do not need access.

For local container development, start Docker Desktop or another Docker-compatible engine and run `npm run dev` from `backend`. The backend's regular Python startup instructions remain available above for non-Cloudflare hosts.

For a split-origin deployment, set `VITE_API_BASE_URL=https://your-api.example` at frontend build time and set `SIMULATOR_CORS_ORIGINS=https://your-frontend.example` on the backend. The Python service needs an ASGI-capable host, HTTP and WebSocket support, and access to `backend/data`; the frontend needs a static host that serves the `dist` output with SPA fallback. Keep `/api` and `/ws` on the same backend origin. These settings are demonstrated locally by Vite's development and preview proxies.

## Local performance check

At a 1440 × 900 desktop viewport, an `agent-browser` Chrome trace over six seconds recorded 395 animation-frame callbacks. Callback-handler duration averaged 1.76 ms, p95 was 5.37 ms, and the longest was 8.74 ms in that run. This measures JavaScript callback work, not a guarantee of end-to-end FPS on every GPU. The production build splits the UI, R3F, Three.js, and renderer into separate chunks; the Three.js chunk is 704 KB uncompressed (181 KB gzip) and triggers Vite's default large-chunk advisory.

## Limitations

The dynamics are a deterministic kinematic baseline, not a tire, suspension, or full F1 powertrain model. The self-play collision check uses simplified oriented rectangles. JEV is a project-specific route scoring heuristic around the supplied race line, not an externally trained planner. Training is CPU-heavy and rewards are an initial racing objective, so lap performance should be checked on evaluation races before treating a checkpoint as a strong driver. The scene is a genuine 3D rendering of the supplied 2D geometry plus explicitly simplified scenery, not a fully accurate Silverstone replica.
