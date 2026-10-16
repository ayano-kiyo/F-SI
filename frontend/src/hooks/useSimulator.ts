import { useCallback, useEffect, useRef, useState } from 'react'
import type { CheckpointSummary, SocketStatus, Telemetry, TrackPayload, TrackSummary } from '../types'
import { pushTelemetryFrame, type TelemetryFrame } from '../utils/telemetryInterpolation'

function resolveApiBase(): string {
  const envBase = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')
  if (typeof window === 'undefined') return envBase
  try {
    // Runtime override so a deployed Vercel build can point at a backend
    // without a rebuild: /?backend=https://f-si-simulator-api.onrender.com
    // The value persists in localStorage; env (when baked in) wins otherwise.
    const param = new URLSearchParams(window.location.search).get('backend')
    if (param) {
      const cleaned = param.replace(/\/$/, '')
      try { window.localStorage.setItem('fsi-backend', cleaned) } catch { /* private mode */ }
      return cleaned
    }
    if (envBase) return envBase
    const stored = window.localStorage.getItem('fsi-backend')
    if (stored) return stored.replace(/\/$/, '')
  } catch { /* URL/storage unavailable */ }
  return envBase
}

const apiBase = resolveApiBase()
const demoMode = import.meta.env.VITE_DEMO_MODE === 'true'

type DemoPath = { points: TrackPayload['race_line']; cumulative: number[]; length: number }

function buildDemoPath(track: TrackPayload): DemoPath {
  const points = track.race_line
  const cumulative = [0]
  for (let index = 1; index < points.length; index += 1) {
    cumulative.push(cumulative[index - 1] + Math.hypot(
      points[index].x - points[index - 1].x,
      points[index].y - points[index - 1].y,
    ))
  }
  const last = points.at(-1)!
  const first = points[0]
  return { points, cumulative, length: cumulative.at(-1)! + Math.hypot(first.x - last.x, first.y - last.y) }
}

function pointOnDemoPath(path: DemoPath, distance: number) {
  const target = ((distance % path.length) + path.length) % path.length
  let low = 0
  let high = path.cumulative.length - 1
  while (low < high) {
    const middle = Math.ceil((low + high) / 2)
    if (path.cumulative[middle] <= target) low = middle
    else high = middle - 1
  }
  const start = path.points[low]
  const end = path.points[(low + 1) % path.points.length]
  const segmentStart = path.cumulative[low]
  const segmentEnd = low + 1 < path.cumulative.length ? path.cumulative[low + 1] : path.length
  const fraction = segmentEnd === segmentStart ? 0 : (target - segmentStart) / (segmentEnd - segmentStart)
  const x = start.x + (end.x - start.x) * fraction
  const y = start.y + (end.y - start.y) * fraction
  return { x, y, heading: Math.atan2(end.y - start.y, end.x - start.x) }
}

function clamp(value: number, low: number, high: number) {
  return Math.min(high, Math.max(low, value))
}

function socketUrl() {
  const base = apiBase ? new URL(apiBase) : new URL(window.location.href)
  base.protocol = base.protocol === 'https:' ? 'wss:' : 'ws:'
  base.pathname = '/ws/simulation'
  base.search = ''
  base.hash = ''
  return base.toString()
}

export function useSimulator() {
  const [track, setTrack] = useState<TrackPayload | null>(null)
  const [trackList, setTrackList] = useState<TrackSummary[]>([])
  const [checkpoints, setCheckpoints] = useState<CheckpointSummary[]>([])
  const [latestCheckpoint, setLatestCheckpoint] = useState<string | null>(null)
  const [activeCheckpoint, setActiveCheckpoint] = useState<string | null>(null)
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null)
  const [socketStatus, setSocketStatus] = useState<SocketStatus>('connecting')
  const [error, setError] = useState<string | null>(null)
  const [switchingTrack, setSwitchingTrack] = useState(false)
  const socketRef = useRef<WebSocket | null>(null)
  const telemetryFrames = useRef<TelemetryFrame[]>([])
  const lastTelemetryUiUpdateMs = useRef(0)
  const lastUiTelemetry = useRef<Telemetry | null>(null)
  const activeTrackName = useRef<string | null>(null)
  const demoPaused = useRef(false)
  const demoSpeed = useRef(1)
  const demoElapsed = useRef(0)
  const demoDistance = useRef(0)

  const loadTrack = useCallback(async (name: string | null, signal?: AbortSignal) => {
    const url = name ? `${apiBase}/api/track?name=${encodeURIComponent(name)}` : `${apiBase}/api/track`
    const response = await fetch(url, { signal })
    if (!response.ok) throw new Error(`Track service returned ${response.status}`)
    const payload = (await response.json()) as TrackPayload
    activeTrackName.current = payload.name
    telemetryFrames.current.length = 0
    lastUiTelemetry.current = null
    setTelemetry(null)
    setTrack(payload)
    return payload
  }, [])

  const refreshTrackList = useCallback(async (signal?: AbortSignal) => {
    const response = await fetch(`${apiBase}/api/tracks`, { signal })
    if (!response.ok) return
    const payload = (await response.json()) as { tracks: TrackSummary[]; active: string }
    setTrackList(payload.tracks ?? [])
  }, [])

  const refreshCheckpoints = useCallback(async (signal?: AbortSignal) => {
    const response = await fetch(`${apiBase}/api/rl/checkpoints`, { signal })
    if (!response.ok) return
    const payload = await response.json() as {
      checkpoints: CheckpointSummary[]
      latest: string | null
      active: string | null
    }
    setCheckpoints(payload.checkpoints ?? [])
    setLatestCheckpoint(payload.latest ?? null)
    setActiveCheckpoint(payload.active?.split(/[\\/]/).at(-1) ?? null)
  }, [])

  useEffect(() => {
    const abortController = new AbortController()
    if (demoMode) {
      fetch('/demo/silverstone.json', { signal: abortController.signal })
        .then((response) => {
          if (!response.ok) throw new Error(`Demo track returned ${response.status}`)
          return response.json() as Promise<TrackPayload>
        })
        .then((payload) => {
          activeTrackName.current = payload.name
          setTrack(payload)
          setTrackList([{
            name: payload.name,
            label: payload.label ?? payload.name,
            length_m: payload.length_m,
            race_line_length_m: payload.race_line_length_m,
            centerline_points: payload.centerline.length,
            race_line_points: payload.race_line.length,
          }])
          setError(null)
        })
        .catch((reason: unknown) => {
          if (!abortController.signal.aborted) setError(reason instanceof Error ? reason.message : 'Could not load demo track')
        })
      return () => abortController.abort()
    }
    Promise.all([
      loadTrack(null, abortController.signal),
      refreshTrackList(abortController.signal),
      refreshCheckpoints(abortController.signal),
    ]).catch(
      (reason: unknown) => {
        if (!abortController.signal.aborted) {
          setError(reason instanceof Error ? reason.message : 'Could not load track geometry')
        }
      },
    )
    return () => abortController.abort()
  }, [loadTrack, refreshTrackList, refreshCheckpoints])

  useEffect(() => {
    if (demoMode) return
    let disposed = false
    let retryTimer: number | undefined
    let heartbeatTimer: number | undefined
    let attempt = 0
    let activeSocket: WebSocket | null = null

    const connect = () => {
      if (disposed) return
      setSocketStatus('connecting')
      const socket = new WebSocket(socketUrl())
      activeSocket = socket
      socketRef.current = socket
      socket.onopen = () => {
        attempt = 0
        setError(null)
        setSocketStatus('live')
        if (heartbeatTimer) window.clearInterval(heartbeatTimer)
        heartbeatTimer = window.setInterval(() => {
          if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: 'ping' }))
        }, 30_000)
      }
      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data as string) as Telemetry | { type: 'error'; message: string }
          if (message.type === 'error') {
            setError(message.message)
            setSwitchingTrack(false)
            return
          }
          if (message.type !== 'telemetry') return
          if (message.track?.name && activeTrackName.current && message.track.name !== activeTrackName.current) {
            // Backend switched circuits (e.g. another client); reload geometry once.
            activeTrackName.current = message.track.name
            loadTrack(message.track.name).catch(() => {})
            return
          }
          if (switchingTrackRef.current) switchingTrackRef.current = false
          setSwitchingTrack(false)
          const receivedAtMs = performance.now()
          pushTelemetryFrame(telemetryFrames.current, message, receivedAtMs)

          // Keep every packet for frame interpolation, but avoid re-rendering the
          // whole React HUD and 3D scene at the simulator's full 20 Hz rate.
          const previousUiTelemetry = lastUiTelemetry.current
          const controlsChanged = previousUiTelemetry && (
            message.paused !== previousUiTelemetry.paused ||
            message.simulation_speed !== previousUiTelemetry.simulation_speed ||
            message.controller.name !== previousUiTelemetry.controller.name
          )
          const trackChanged = previousUiTelemetry && message.track?.name !== previousUiTelemetry.track?.name
          const simulationReset = previousUiTelemetry && message.sim_time_s < previousUiTelemetry.sim_time_s
          if (
            !previousUiTelemetry ||
            receivedAtMs - lastTelemetryUiUpdateMs.current >= 100 ||
            controlsChanged ||
            trackChanged ||
            simulationReset
          ) {
            lastTelemetryUiUpdateMs.current = receivedAtMs
            lastUiTelemetry.current = message
            setTelemetry(message)
          }
        } catch {
          setError('Received malformed telemetry from the simulator')
        }
      }
      socket.onerror = () => setSocketStatus('offline')
      socket.onclose = () => {
        if (disposed) return
        if (heartbeatTimer) window.clearInterval(heartbeatTimer)
        heartbeatTimer = undefined
        telemetryFrames.current.length = 0
        setSocketStatus('offline')
        const delay = Math.min(5000, 500 * 2 ** attempt)
        attempt += 1
        retryTimer = window.setTimeout(connect, delay)
      }
    }

    connect()
    return () => {
      disposed = true
      if (retryTimer) window.clearTimeout(retryTimer)
      if (heartbeatTimer) window.clearInterval(heartbeatTimer)
      activeSocket?.close()
      socketRef.current = null
    }
  }, [loadTrack])

  useEffect(() => {
    if (!demoMode || !track) return
    const path = buildDemoPath(track)
    let lastTickMs = performance.now()

    const tick = () => {
      const nowMs = performance.now()
      const delta = Math.min(0.15, Math.max(0, (nowMs - lastTickMs) / 1000))
      lastTickMs = nowMs
      let fraction = (demoDistance.current % path.length) / path.length
      const speed = 58 + Math.sin(fraction * Math.PI * 10 - 0.8) * 15 + Math.sin(fraction * Math.PI * 34) * 4
      if (!demoPaused.current) {
        demoElapsed.current += delta * demoSpeed.current
        demoDistance.current += delta * speed * demoSpeed.current
      }

      const elapsed = demoElapsed.current
      const lap = Math.floor(demoDistance.current / path.length)
      const distance = demoDistance.current % path.length
      fraction = distance / path.length
      const point = pointOnDemoPath(path, distance)
      const previousPoint = pointOnDemoPath(path, distance - 2)
      const nextPoint = pointOnDemoPath(path, distance + 2)
      const headingChange = Math.atan2(
        Math.sin(nextPoint.heading - previousPoint.heading),
        Math.cos(nextPoint.heading - previousPoint.heading),
      )
      const lapTime = elapsed % (path.length / 72)
      const vehicle = {
        id: 'car-1',
        x: point.x,
        y: point.y,
        heading_rad: point.heading,
        speed_mps: speed,
        acceleration_mps2: Math.sin(elapsed * 0.8) * 3.2,
        yaw_rate_rad_s: headingChange / 4,
        steering: clamp(headingChange * 2.4, -0.95, 0.95),
        throttle: clamp(0.78 + Math.sin(elapsed * 1.7) * 0.2, 0, 1),
        brake: clamp(Math.sin(elapsed * 1.7 + 2.5) * 0.26, 0, 1),
        progress_m: distance,
        fraction,
        status: 'racing',
        position: 1,
        target_waypoint: pointOnDemoPath(path, distance + 32),
        route_id: 'sample-race-line',
      }
      const trajectory = Array.from({ length: 19 }, (_, index) => {
        const future = pointOnDemoPath(path, distance + index * 8)
        return { x: future.x, y: future.y }
      })
      const sample: Telemetry = {
        type: 'telemetry',
        sim_time_s: elapsed,
        paused: demoPaused.current,
        simulation_speed: demoSpeed.current,
        track: { name: track.name, length_m: track.length_m },
        vehicle,
        vehicles: [vehicle],
        planner: { name: 'sample_race_line', route_cost: 0, route_id: 'sample-race-line' },
        controller: { name: 'demo_autopilot', available: false },
        progress: {
          distance_m: distance,
          fraction,
          lap,
          lap_time_s: lapTime,
          best_lap_time_s: lap > 0 ? path.length / 72 : null,
          off_track: false,
          lateral_error_m: 0,
        },
        planned_trajectory: trajectory,
      }
      pushTelemetryFrame(telemetryFrames.current, sample, nowMs)
      setTelemetry(sample)
    }

    tick()
    const timer = window.setInterval(tick, 50)
    return () => window.clearInterval(timer)
  }, [track])

  const switchingTrackRef = useRef(false)

  const send = useCallback((command: Record<string, string | number>) => {
    if (demoMode) {
      if (command.type === 'pause') demoPaused.current = true
      else if (command.type === 'resume') demoPaused.current = false
      else if (command.type === 'reset') {
        demoElapsed.current = 0
        demoDistance.current = 0
        telemetryFrames.current.length = 0
      } else if (command.type === 'set_speed' && typeof command.value === 'number') {
        demoSpeed.current = clamp(command.value, 0.25, 4)
      }
      setTelemetry((current) => current ? {
        ...current,
        paused: demoPaused.current,
        simulation_speed: demoSpeed.current,
      } : current)
      return
    }
    const socket = socketRef.current
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(command))
  }, [])

  const selectTrack = useCallback(
    async (name: string) => {
      if (!name || name === activeTrackName.current || switchingTrackRef.current) return
      switchingTrackRef.current = true
      setSwitchingTrack(true)
      setError(null)
      send({ type: 'set_track', value: name })
      // The sim loop resets instantly; fetch the new mesh and clear stale frames.
      try {
        await loadTrack(name)
      } catch (reason: unknown) {
        setError(reason instanceof Error ? reason.message : 'Could not switch track')
      } finally {
        switchingTrackRef.current = false
        setSwitchingTrack(false)
      }
    },
    [loadTrack, send],
  )

  const loadCheckpoint = useCallback(async (name: string) => {
    const response = await fetch(`${apiBase}/api/rl/checkpoint`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    })
    const payload = await response.json() as { checkpoint?: string; detail?: string }
    if (!response.ok) throw new Error(payload.detail ?? `Checkpoint service returned ${response.status}`)
    setActiveCheckpoint(name)
    await refreshCheckpoints()
  }, [refreshCheckpoints])

  return {
    track, trackList, telemetry, telemetryFrames, socketStatus, error, send, selectTrack, switchingTrack,
    checkpoints, latestCheckpoint, activeCheckpoint, refreshCheckpoints, loadCheckpoint,
  }
}
