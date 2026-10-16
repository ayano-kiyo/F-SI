export type TrackPoint = {
  x: number
  y: number
  width_right_m: number
  width_left_m: number
}

export type XY = { x: number; y: number }

export type TrackPayload = {
  name: string
  label?: string
  length_m: number
  race_line_length_m: number
  coordinate_system: string
  elevation_available: boolean
  centerline: TrackPoint[]
  race_line: XY[]
}

export type TrackSummary = {
  name: string
  label: string
  length_m: number
  race_line_length_m: number
  centerline_points: number
  race_line_points: number
}

export type VehicleTelemetry = {
  id: string
  x: number
  y: number
  heading_rad: number
  speed_mps: number
  acceleration_mps2: number
  yaw_rate_rad_s: number
  steering: number
  throttle: number
  brake: number
  progress_m: number
  fraction: number
  status: string
  position: number | null
  target_waypoint: XY
  route_id: string
}

export type RaceTelemetry = {
  track: string
  finished: boolean
  steps?: number
  finishes?: number
  collisions: number
  training?: boolean
  training_step?: number
  race_index?: number
  num_races?: number
  cars: Array<{
    car: number
    position: number | null
    status: string
    progress_m: number
    lap_time_s?: number | null
  }>
}

export type CheckpointSummary = {
  name: string
  size_bytes: number
  modified_s: number
}

export type Telemetry = {
  type: 'telemetry'
  sim_time_s: number
  paused: boolean
  simulation_speed: number
  track: { name: string; length_m: number }
  vehicle: {
    id?: string
    x: number
    y: number
    heading_rad: number
    speed_mps: number
    acceleration_mps2: number
    yaw_rate_rad_s: number
    steering: number
    throttle: number
    brake: number
    target_waypoint: XY
    route_id: string
  }
  vehicles?: VehicleTelemetry[]
  race?: RaceTelemetry
  planner: { name: string; route_cost: number; route_id: string }
  controller: { name: string; available: boolean; checkpoint?: string | null }
  progress: {
    distance_m: number
    fraction: number
    lap: number
    lap_time_s: number
    best_lap_time_s: number | null
    off_track: boolean
    lateral_error_m: number
  }
  planned_trajectory: XY[]
}

export type SocketStatus = 'connecting' | 'live' | 'offline'
export type CameraMode = 'follow' | 'onboard' | 'orbit' | 'track' | 'race'
