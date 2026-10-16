import type { Telemetry, VehicleTelemetry } from '../types'

export function RaceStandings({ telemetry }: { telemetry: Telemetry | null }) {
  const vehicles = telemetry?.vehicles ?? []
  if (vehicles.length < 2) return null
  const ordered = [...vehicles].sort((left, right) => {
    if (left.position != null && right.position != null) return left.position - right.position
    if (left.status === 'finished' && right.status !== 'finished') return -1
    if (right.status === 'finished' && left.status !== 'finished') return 1
    return right.progress_m - left.progress_m
  })
  const collisions = telemetry?.race?.collisions ?? 0

  return <aside className="race-standings hud-surface" aria-label="Race standings">
    <div className="race-standings-heading">
      <span>{telemetry?.race?.training
        ? `LIVE TRAINING · WORLD ${(telemetry.race.race_index ?? 0) + 1}/${telemetry.race.num_races ?? 1}`
        : 'RACE ORDER'}</span>
      <span>{collisions} CONTACT{collisions === 1 ? '' : 'S'}</span>
    </div>
    <ol>
      {ordered.map((vehicle: VehicleTelemetry, index) => <li key={vehicle.id}>
        <span className="race-position">{vehicle.position ?? index + 1}</span>
        <span className="race-car-name"><i style={{ backgroundColor: carColor(vehicle.id) }} />{vehicle.id}</span>
        <span className={'race-car-status ' + (vehicle.status === 'racing' ? 'is-racing' : '')}>{vehicle.status.toUpperCase()}</span>
      </li>)}
    </ol>
    {telemetry?.race?.finished && <div className="race-finished">RACE FINISHED</div>}
  </aside>
}

function carColor(id: string) {
  const match = Number(id.replace('car-', '')) - 1
  return ['#dce5df', '#e56a43', '#64b5d6', '#d5b45f'][Math.max(0, match) % 4]
}
