import type { Telemetry, VehicleTelemetry } from '../types'

export type TelemetryFrame = {
  receivedAtMs: number
  telemetry: Telemetry
}

export type TelemetryFrameBuffer = { current: TelemetryFrame[] }
export type VisualVehicleState = Pick<VehicleTelemetry, 'x' | 'y' | 'heading_rad' | 'speed_mps' | 'yaw_rate_rad_s'>
export type VisualRacePose = Record<string, VisualVehicleState>

const MAX_BUFFERED_FRAMES = 16
const DISCONTINUITY_GAP_MS = 1000
const VISUAL_DELAY_MS = 100

function vehicles(telemetry: Telemetry): VehicleTelemetry[] {
  if (telemetry.vehicles?.length) return telemetry.vehicles
  return [{
    id: 'car-1',
    ...telemetry.vehicle,
    progress_m: telemetry.progress.distance_m,
    fraction: telemetry.progress.fraction,
    status: 'racing',
    position: 1,
  }]
}

export function pushTelemetryFrame(frames: TelemetryFrame[], telemetry: Telemetry, receivedAtMs: number) {
  const previous = frames.at(-1)
  const jumpedBetweenEpisodes = previous && vehicles(telemetry).some((car) => {
    const before = vehicles(previous.telemetry).find((candidate) => candidate.id === car.id)
    return before && Math.hypot(car.x - before.x, car.y - before.y) > 45
  })

  // The websocket broadcasts at 20 Hz, while the training preview may not
  // advance on every broadcast. Keep only distinct simulation samples so
  // repeated snapshots do not evict the movement history used for playback.
  if (previous && !jumpedBetweenEpisodes && Math.abs(telemetry.sim_time_s - previous.telemetry.sim_time_s) < 1e-5) return

  if (previous && (
    telemetry.sim_time_s < previous.telemetry.sim_time_s ||
    receivedAtMs - previous.receivedAtMs > DISCONTINUITY_GAP_MS ||
    jumpedBetweenEpisodes
  )) {
    frames.length = 0
  }

  frames.push({ receivedAtMs, telemetry })
  if (frames.length > MAX_BUFFERED_FRAMES) frames.splice(0, frames.length - MAX_BUFFERED_FRAMES)
}

export function sampleVisualVehicles(frames: TelemetryFrame[], nowMs: number): VisualRacePose {
  if (frames.length === 0) return {}

  const first = frames[0]
  const last = frames[frames.length - 1]
  const latest = last.telemetry
  const playbackRate = latest.paused ? 0 : latest.simulation_speed
  const estimatedSimTime = latest.sim_time_s + Math.max(0, nowMs - last.receivedAtMs) / 1000 * playbackRate
  const targetSimTime = latest.paused
    ? latest.sim_time_s
    : estimatedSimTime - VISUAL_DELAY_MS / 1000 * playbackRate
  if (targetSimTime <= first.telemetry.sim_time_s) return poseMap(first.telemetry)
  if (targetSimTime >= latest.sim_time_s) return poseMap(latest)

  let afterIndex = 1
  while (afterIndex < frames.length && frames[afterIndex].telemetry.sim_time_s < targetSimTime) afterIndex += 1

  const before = frames[afterIndex - 1]
  const after = frames[afterIndex]
  const interval = after.telemetry.sim_time_s - before.telemetry.sim_time_s
  if (interval <= 0) return poseMap(after.telemetry)

  const amount = Math.min(1, Math.max(0, (targetSimTime - before.telemetry.sim_time_s) / interval))
  const beforeCars = new Map(vehicles(before.telemetry).map((car) => [car.id, car]))
  const afterCars = new Map(vehicles(after.telemetry).map((car) => [car.id, car]))
  const ids = new Set([...beforeCars.keys(), ...afterCars.keys()])
  const poses: VisualRacePose = {}

  for (const id of ids) {
    const start = beforeCars.get(id)
    const end = afterCars.get(id)
    if (!start || !end) {
      poses[id] = pose(end ?? start!)
      continue
    }
    const headingDelta = shortestAngle(start.heading_rad, end.heading_rad)
    const endHeading = start.heading_rad + headingDelta
    poses[id] = {
      x: hermite(start.x, end.x, start.speed_mps * Math.cos(start.heading_rad), end.speed_mps * Math.cos(endHeading), interval, amount),
      y: hermite(start.y, end.y, start.speed_mps * Math.sin(start.heading_rad), end.speed_mps * Math.sin(endHeading), interval, amount),
      heading_rad: hermite(start.heading_rad, endHeading, start.yaw_rate_rad_s, end.yaw_rate_rad_s, interval, amount),
      speed_mps: start.speed_mps + (end.speed_mps - start.speed_mps) * amount,
      yaw_rate_rad_s: start.yaw_rate_rad_s + (end.yaw_rate_rad_s - start.yaw_rate_rad_s) * amount,
    }
  }
  return poses
}

export function sampleVisualVehicle(frames: TelemetryFrame[], nowMs: number): VisualVehicleState | null {
  const poses = sampleVisualVehicles(frames, nowMs)
  return poses['car-1'] ?? null
}

function poseMap(telemetry: Telemetry): VisualRacePose {
  return Object.fromEntries(vehicles(telemetry).map((car) => [car.id, pose(car)]))
}

function pose(car: VehicleTelemetry): VisualVehicleState {
  return {
    x: car.x,
    y: car.y,
    heading_rad: car.heading_rad,
    speed_mps: car.speed_mps,
    yaw_rate_rad_s: car.yaw_rate_rad_s,
  }
}

function shortestAngle(start: number, end: number) {
  return ((end - start + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI
}

function hermite(start: number, end: number, startVelocity: number, endVelocity: number, duration: number, amount: number) {
  const t2 = amount * amount
  const t3 = t2 * amount
  const startWeight = 2 * t3 - 3 * t2 + 1
  const startTangentWeight = t3 - 2 * t2 + amount
  const endWeight = -2 * t3 + 3 * t2
  const endTangentWeight = t3 - t2

  return startWeight * start
    + startTangentWeight * duration * startVelocity
    + endWeight * end
    + endTangentWeight * duration * endVelocity
}
