import { useState } from 'react'
import { ChevronDown, Pause, Play, RotateCcw } from 'lucide-react'
import type { CameraMode, Telemetry } from '../types'

const speedOptions = [0.25, 0.5, 1, 2, 4]

export function SimulationControls({ telemetry, send, demoMode = false }: { telemetry: Telemetry | null; send: (command: Record<string, string | number>) => void; demoMode?: boolean }) {
  const [speedOpen, setSpeedOpen] = useState(false)
  const speed = telemetry?.simulation_speed ?? 1
  const paused = telemetry?.paused ?? false
  const trainingLive = telemetry?.race?.training ?? false

  return <div className="simulation-hud hud-surface" aria-label="Simulation controls">
    <span className="driver-mode"><i /> {demoMode
      ? 'DEMO AUTO'
      : telemetry?.race?.training
      ? 'TRAINING LIVE'
      : telemetry?.race ? telemetry.controller.name === 'neural_network' ? 'SELF-PLAY' : 'MULTI-CAR' : telemetry?.controller.name === 'neural_network' ? 'NEURAL' : 'BASELINE'}</span>
    <div className="simulation-actions">
      <button className="sim-action pause-action" onClick={() => send({ type: paused ? 'resume' : 'pause' })} disabled={trainingLive} aria-label={paused ? 'Resume simulation' : 'Pause simulation'} title={trainingLive ? 'Training is controlled by the terminal process' : paused ? 'Resume' : 'Pause'}>
        {paused ? <Play size={15} fill="currentColor" /> : <Pause size={15} fill="currentColor" />}
      </button>
      <button className="sim-action" onClick={() => send({ type: 'reset' })} disabled={trainingLive} aria-label="Reset simulation" title={trainingLive ? 'Training is controlled by the terminal process' : 'Reset simulation'}><RotateCcw size={14} /></button>
      <div className="speed-picker">
        <button className="speed-select" onClick={() => setSpeedOpen((open) => !open)} disabled={trainingLive} aria-expanded={speedOpen} aria-label={`Simulation speed ${speed} times`}>
          {speed}× <ChevronDown size={13} />
        </button>
        {speedOpen && <div className="speed-menu">{speedOptions.map((value) => <button key={value} className={value === speed ? 'selected' : ''} onClick={() => { send({ type: 'set_speed', value }); setSpeedOpen(false) }}>{value}×</button>)}</div>}
      </div>
    </div>
  </div>
}

export function CameraControls({ mode, onChange }: { mode: CameraMode; onChange: (mode: CameraMode) => void }) {
  const modes: { value: CameraMode; label: string }[] = [
    { value: 'race', label: 'Race' },
    { value: 'follow', label: 'Follow' },
    { value: 'onboard', label: 'Onboard' },
    { value: 'orbit', label: 'Orbit' },
    { value: 'track', label: 'Track' },
  ]

  return <div className="camera-controls hud-surface" role="group" aria-label="Camera view">
    {modes.map((item) => <button key={item.value} className={mode === item.value ? 'active' : ''} onClick={() => onChange(item.value)} aria-pressed={mode === item.value}>{item.label}</button>)}
  </div>
}
