import { memo, Suspense, useEffect, useMemo, useRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { AdaptiveDpr, PerformanceMonitor, Stars } from '@react-three/drei'
import * as THREE from 'three'
import { Car } from './Car'
import { SimulationCamera } from './SimulationCamera'
import { Track } from './Track'
import type { CameraMode, TrackPayload, XY } from '../types'
import { sampleVisualVehicles, type TelemetryFrameBuffer, type VisualRacePose } from '../utils/telemetryInterpolation'

const VisualPoseSampler = memo(function VisualPoseSampler({ telemetryFrames, visualPose }: {
  telemetryFrames: TelemetryFrameBuffer
  visualPose: { current: VisualRacePose }
}) {
  useFrame(() => {
    visualPose.current = sampleVisualVehicles(telemetryFrames.current, performance.now())
  }, -1)
  return null
})

const AdaptiveRenderingQuality = memo(function AdaptiveRenderingQuality() {
  const regress = useThree((state) => state.performance.regress)
  return <PerformanceMonitor onDecline={() => regress()}>
    <AdaptiveDpr />
  </PerformanceMonitor>
})

const StaticStars = memo(function StaticStars() {
  return <Stars radius={4500} depth={50} count={400} factor={1.7} saturation={0} fade speed={0} />
})

const LiveTrajectoryLine = memo(function LiveTrajectoryLine({ telemetryFrames }: {
  telemetryFrames: TelemetryFrameBuffer
}) {
  const line = useMemo(() => new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: '#f2f3f3', transparent: true, opacity: 0.96 }),
  ), [])
  const previousPoints = useRef<XY[] | null>(null)

  useFrame(() => {
    const points = telemetryFrames.current.at(-1)?.telemetry.planned_trajectory
    if (!points) {
      previousPoints.current = null
      line.visible = false
      return
    }
    if (points === previousPoints.current) return
    previousPoints.current = points
    line.visible = points.length > 1
    if (!line.visible) return
    line.geometry.setFromPoints(points.map((point) => new THREE.Vector3(point.x, 0.4, -point.y)))
    line.geometry.computeBoundingSphere()
  })

  useEffect(() => () => {
    line.geometry.dispose()
    line.material.dispose()
  }, [line])

  return <primitive object={line} />
})

export const TrackScene = memo(function TrackScene({ track, telemetryFrames, cameraMode, vehicleIds }: {
  track: TrackPayload
  telemetryFrames: TelemetryFrameBuffer
  cameraMode: CameraMode
  vehicleIds: string[]
}) {
  const visualPose = useRef<VisualRacePose>({})
  const fallbackPosition = track.race_line[0]
  const visibleVehicleIds = vehicleIds.length ? vehicleIds : ['car-1']
  const carColors = ['#dce5df', '#e56a43', '#64b5d6', '#d5b45f']
  return <Canvas
    dpr={[0.65, 1]}
    performance={{ min: 0.5, max: 1, debounce: 750 }}
    camera={{ position: [fallbackPosition.x - 19, 10, -fallbackPosition.y + 3], fov: 48, near: 0.1, far: 8000 }}
    gl={{ antialias: true, powerPreference: 'high-performance' }}
    onCreated={({ gl }) => { gl.toneMapping = THREE.ACESFilmicToneMapping; gl.toneMappingExposure = 1.12 }}
  >
    <AdaptiveRenderingQuality />
    <color attach="background" args={['#111719']} />
    <fog attach="fog" args={['#111719', 850, 2900]} />
    <ambientLight intensity={0.82} />
    <hemisphereLight args={['#dce4da', '#27332c', 1.3]} />
    <directionalLight position={[650, 1100, 500]} intensity={3.4} />
    <directionalLight position={[-800, 500, -600]} intensity={1.1} color="#95b0aa" />
    <StaticStars />
    <Track track={track} />
    <LiveTrajectoryLine telemetryFrames={telemetryFrames} />
    <VisualPoseSampler telemetryFrames={telemetryFrames} visualPose={visualPose} />
    <Suspense fallback={null}>
      {visibleVehicleIds.map((vehicleId, index) => <Car
        key={vehicleId}
        visualPoses={visualPose}
        vehicleId={vehicleId}
        markerColor={carColors[index % carColors.length]}
        fallbackPosition={fallbackPosition}
      />)}
    </Suspense>
    <SimulationCamera mode={cameraMode} visualPose={visualPose} track={track} fallbackPosition={fallbackPosition} />
  </Canvas>
})
