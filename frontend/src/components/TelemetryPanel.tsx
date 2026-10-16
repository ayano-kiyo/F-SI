import type { Telemetry } from '../types'

function formatTime(seconds: number | null | undefined) {
  if (seconds == null) return '--:--.---'
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds - minutes * 60
  return `${String(minutes).padStart(2, '0')}:${remainder.toFixed(3).padStart(6, '0')}`
}

function InputMeter({ label, value, accent }: { label: string; value: number; accent: 'throttle' | 'brake' }) {
  const percent = Math.round(Math.min(1, Math.max(0, value)) * 100)
  return <div className="input-meter">
    <div className="input-meter-label"><span>{label}</span><span>{percent}%</span></div>
    <div className="meter-track"><span className={`meter-fill ${accent}`} style={{ width: `${percent}%` }} /></div>
  </div>
}

export function TelemetryPanel({ telemetry }: { telemetry: Telemetry | null }) {
  const vehicle = telemetry?.vehicle
  const progress = telemetry?.progress
  const speed = Math.round((vehicle?.speed_mps ?? 0) * 3.6)
  const steeringDegrees = (vehicle?.steering ?? 0) * 24.1
  const fraction = Math.min(100, Math.max(0, (progress?.fraction ?? 0) * 100))

  return <section className="telemetry-hud hud-surface" aria-label="Live vehicle telemetry">
    <div className="telemetry-heading"><span>{vehicle?.id ? `VEHICLE · ${vehicle.id.toUpperCase()}` : 'VEHICLE'}</span><span className="telemetry-route">{telemetry?.planner.route_id ?? 'WAITING FOR DATA'}</span></div>
    <div className="telemetry-primary">
      <div className="speed-readout"><strong>{speed}</strong><span>KM/H</span></div>
      <div className="lap-readout">
        <div className="lap-time-row"><span>LAP {String((progress?.lap ?? 0) + 1).padStart(2, '0')}</span><strong>{formatTime(progress?.lap_time_s ?? 0)}</strong></div>
        <div className="progress-track"><span style={{ width: `${fraction}%` }} /></div>
        <div className="progress-caption"><span>TRACK PROGRESS</span><span>{fraction.toFixed(1)}%</span></div>
      </div>
    </div>
    <div className="telemetry-details">
      <div className="telemetry-detail"><span>STEER</span><strong>{steeringDegrees.toFixed(1)}°</strong></div>
      <div className="telemetry-detail"><span>ACCEL</span><strong>{((vehicle?.acceleration_mps2 ?? 0) / 9.80665).toFixed(2)} G</strong></div>
      <InputMeter label="THR" value={vehicle?.throttle ?? 0} accent="throttle" />
      <InputMeter label="BRK" value={vehicle?.brake ?? 0} accent="brake" />
    </div>
  </section>
}
