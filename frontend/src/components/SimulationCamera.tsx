import { memo, useMemo, useRef } from 'react'
import { OrbitControls } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import * as THREE from 'three'
import type { CameraMode, TrackPayload, XY } from '../types'
import type { VisualRacePose } from '../utils/telemetryInterpolation'

export const SimulationCamera = memo(function SimulationCamera({ mode, visualPose, track, fallbackPosition }: {
  mode: CameraMode
  visualPose: { current: VisualRacePose }
  track: TrackPayload
  fallbackPosition: XY
}) {
  const { camera } = useThree()
  const controls = useRef<OrbitControlsImpl>(null)
  const center = useMemo(() => {
    const xs = track.centerline.map((point) => point.x)
    const ys = track.centerline.map((point) => point.y)
    return new THREE.Vector3((Math.min(...xs) + Math.max(...xs)) / 2, 0, -(Math.min(...ys) + Math.max(...ys)) / 2)
  }, [track])
  const forward = useMemo(() => new THREE.Vector3(), [])
  const car = useMemo(() => new THREE.Vector3(), [])
  const desiredPosition = useMemo(() => new THREE.Vector3(), [])
  const lookAt = useMemo(() => new THREE.Vector3(), [])
  const currentTarget = useMemo(() => new THREE.Vector3(), [])
  const offset = useMemo(() => new THREE.Vector3(), [])
  const side = useMemo(() => new THREE.Vector3(), [])
  const raceFocus = useRef<THREE.Vector3 | null>(null)
  const raceForward = useRef<THREE.Vector3 | null>(null)

  useFrame((_, delta) => {
    if (mode === 'orbit') {
      camera.up.set(0, 1, 0)
      if (controls.current) controls.current.target.lerp(center, 1 - Math.exp(-delta * 2))
      controls.current?.update()
      return
    }
    const vehicle = visualPose.current['car-1']
    const x = vehicle?.x ?? fallbackPosition.x
    const y = vehicle?.y ?? fallbackPosition.y
    const heading = vehicle?.heading_rad ?? 0
    forward.set(Math.cos(heading), 0, -Math.sin(heading))
    car.set(x, 0, -y)
    if (mode === 'track') {
      desiredPosition.set(center.x, 2250, center.z)
      lookAt.copy(center)
      camera.up.set(0, 0, -1)
    } else if (mode === 'race') {
      const poses = Object.values(visualPose.current)
      const movingPoses = poses.filter((pose) => pose.speed_mps > 0.5)
      const framedPoses = movingPoses.length ? movingPoses : poses
      const focusX = framedPoses.length ? framedPoses.reduce((sum, pose) => sum + pose.x, 0) / framedPoses.length : x
      const focusY = framedPoses.length ? framedPoses.reduce((sum, pose) => sum + pose.y, 0) / framedPoses.length : y
      const spread = framedPoses.reduce((largest, pose) => Math.max(largest, Math.hypot(pose.x - focusX, pose.y - focusY)), 0)
      const headingPoses = movingPoses.length ? movingPoses : framedPoses
      if (headingPoses.length) {
        const headingX = headingPoses.reduce((sum, pose) => sum + Math.cos(pose.heading_rad), 0)
        const headingY = headingPoses.reduce((sum, pose) => sum + Math.sin(pose.heading_rad), 0)
        const raceHeading = Math.atan2(headingY, headingX)
        const targetForward = offset.set(Math.cos(raceHeading), 0, -Math.sin(raceHeading))
        if (!raceForward.current) raceForward.current = targetForward.clone()
        else raceForward.current.lerp(targetForward, 1 - Math.exp(-delta * 4)).normalize()
        forward.copy(raceForward.current)
      }
      const height = Math.max(10, spread * 0.28 + 7)
      const trailDistance = Math.max(22, spread * 0.55 + 12)
      const sideDistance = Math.max(10, spread * 0.3 + 6)
      side.set(-forward.z, 0, forward.x)
      const targetFocus = offset.set(focusX, 0, -focusY)
      if (!raceFocus.current) raceFocus.current = targetFocus.clone()
      else raceFocus.current.lerp(targetFocus, 1 - Math.exp(-delta * 5))
      car.copy(raceFocus.current)
      desiredPosition.copy(car)
        .addScaledVector(forward, -trailDistance)
        .addScaledVector(side, sideDistance)
        .add(offset.set(0, height, 0))
      lookAt.copy(car).add(offset.set(0, 0.5, 0))
      camera.up.set(0, 1, 0)
    } else if (mode === 'onboard') {
      desiredPosition.copy(car).add(offset.set(0, 1.35, 0)).addScaledVector(forward, 0.22)
      lookAt.copy(car).add(offset.set(0, 1.2, 0)).addScaledVector(forward, 38)
    } else {
      desiredPosition.copy(car).addScaledVector(forward, -13).add(offset.set(0, 5.8, 0))
      lookAt.copy(car).addScaledVector(forward, 8).add(offset.set(0, 0.9, 0))
      camera.up.set(0, 1, 0)
    }
    const alpha = 1 - Math.exp(-delta * (mode === 'onboard' ? 12 : 7))
    camera.position.lerp(desiredPosition, alpha)
    camera.getWorldDirection(currentTarget).multiplyScalar(16).add(camera.position)
    currentTarget.lerp(lookAt, alpha)
    camera.lookAt(currentTarget)
  })

  return mode === 'orbit'
    ? <OrbitControls ref={controls} enableDamping dampingFactor={0.08} minDistance={24} maxDistance={3600} target={center} />
    : null
})
