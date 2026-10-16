import { lazy, Suspense, useEffect, useState } from 'react'
import { Maximize2, Minimize2 } from 'lucide-react'
import { CameraControls, SimulationControls } from './components/SimulationControls'
import { TelemetryPanel } from './components/TelemetryPanel'
import { RaceStandings } from './components/RaceStandings'
import { TrainingPanel } from './components/TrainingPanel'
import { TrackSelector } from './components/TrackSelector'
import { useSimulator } from './hooks/useSimulator'
import type { CameraMode } from './types'

const TrackScene = lazy(() => import('./components/TrackScene').then((module) => ({ default: module.TrackScene })))
const demoMode = import.meta.env.VITE_DEMO_MODE === 'true'

export default function App() {
  const {
    track, trackList, telemetry, telemetryFrames, error, send, selectTrack, switchingTrack,
    checkpoints, latestCheckpoint, activeCheckpoint, refreshCheckpoints, loadCheckpoint,
  } = useSimulator()
  const [cameraMode, setCameraMode] = useState<CameraMode>('follow')
  const [isFullscreen, setIsFullscreen] = useState(Boolean(document.fullscreenElement))

  useEffect(() => {
    const syncFullscreen = () => setIsFullscreen(Boolean(document.fullscreenElement))
    document.addEventListener('fullscreenchange', syncFullscreen)
    return () => document.removeEventListener('fullscreenchange', syncFullscreen)
  }, [])

  useEffect(() => {
    const hasRace = (telemetry?.vehicles?.length ?? 1) > 1
    setCameraMode((current) => hasRace ? 'race' : current === 'race' ? 'follow' : current)
  }, [telemetry?.vehicles?.length])

  const toggleFullscreen = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen()
      else await document.documentElement.requestFullscreen()
    } catch {
      // Fullscreen may be unavailable in embedded browsers; the simulator remains usable.
    }
  }

  const activeLabel = trackList.find((t) => t.name === (track?.name ?? telemetry?.track.name))?.label
    ?? track?.label ?? track?.name ?? telemetry?.track.name ?? 'CIRCUIT'

  return <main className="simulator-shell">
    <section className="scene-stage" aria-label={`${activeLabel} circuit simulator`}>
      {track
        ? <Suspense fallback={<div className="scene-loading"><div className="loading-pulse" />Preparing 3D renderer…</div>}><TrackScene key={track.name} track={track} telemetryFrames={telemetryFrames} cameraMode={cameraMode} vehicleCount={telemetry?.vehicles?.length ?? 1} /></Suspense>
        : <div className="scene-loading"><div className="loading-pulse" />{error ? 'Track service unavailable' : 'Loading circuit geometry…'}</div>}

      <header className="top-hud">
        <TrackSelector tracks={trackList} activeName={track?.name ?? telemetry?.track.name ?? null} switching={switchingTrack} onSelect={selectTrack} />
        <button className="fullscreen-button hud-surface" onClick={toggleFullscreen} aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'} title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}>
          {isFullscreen ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
        </button>
      </header>

      {demoMode
        ? <div className="demo-banner hud-surface"><strong>DEMO MODE · SAMPLE TELEMETRY</strong><span>Local preview · no live simulator connected</span></div>
        : <TrainingPanel
          checkpoints={checkpoints}
          latestCheckpoint={latestCheckpoint}
          activeCheckpoint={activeCheckpoint}
          onRefresh={refreshCheckpoints}
          onLoad={loadCheckpoint}
        />}
      <RaceStandings telemetry={telemetry} />
      {telemetry?.progress.off_track && <div className="track-alert">OFF TRACK <span>· {telemetry.progress.lateral_error_m.toFixed(1)} m</span></div>}
      {error && <div className="service-error" role="status">{error}</div>}

      <div className="bottom-hud">
        <TelemetryPanel telemetry={telemetry} />
        <CameraControls mode={cameraMode} onChange={setCameraMode} />
        <SimulationControls telemetry={telemetry} send={send} demoMode={demoMode} />
      </div>

      <div className="scene-note">{activeLabel.toUpperCase()} · FLAT 2D SOURCE GEOMETRY{demoMode ? ' · SAMPLE' : ''}</div>
    </section>
  </main>
}
