import { memo, useMemo } from 'react'
import { Line } from '@react-three/drei'
import * as THREE from 'three'
import type { TrackPayload, XY } from '../types'
import { GrandstandAsset } from './GrandstandAsset'

type Side = 'left' | 'right'

function createRibbon(track: TrackPayload, extraWidth = 0, surfaceHeight = 0.015, edgeOnly = false, kerbs = false) {
  const count = track.centerline.length
  const positions: number[] = []
  const colors: number[] = []
  const indices: number[] = []
  const kerbRed = new THREE.Color('#e74742')
  const kerbWhite = new THREE.Color('#e4e4dc')
  const edgeColor = new THREE.Color('#798488')
  for (let i = 0; i < count; i += 1) {
    const previous = track.centerline[(i - 1 + count) % count]
    const current = track.centerline[i]
    const next = track.centerline[(i + 1) % count]
    const dx = next.x - previous.x
    const dy = next.y - previous.y
    const length = Math.hypot(dx, dy) || 1
    const nx = -dy / length
    const ny = dx / length
    const sideSpecs: [Side, number][] = edgeOnly
      ? [['left', current.width_left_m + extraWidth], ['right', -current.width_right_m - extraWidth]]
      : [['left', current.width_left_m + extraWidth], ['right', -current.width_right_m - extraWidth]]
    for (const [, signedWidth] of sideSpecs) {
      positions.push(current.x + nx * signedWidth, surfaceHeight, -(current.y + ny * signedWidth))
    }
  }
  if (kerbs) {
    // Two narrow painted bands follow the actual source edges. Vertex colors alternate by circuit segment.
    const kerbPositions: number[] = []
    const kerbColors: number[] = []
    const kerbIndices: number[] = []
    for (let i = 0; i < count; i += 1) {
      const previous = track.centerline[(i - 1 + count) % count]
      const current = track.centerline[i]
      const next = track.centerline[(i + 1) % count]
      const dx = next.x - previous.x
      const dy = next.y - previous.y
      const length = Math.hypot(dx, dy) || 1
      const nx = -dy / length
      const ny = dx / length
      const stripe = Math.floor(i / 2) % 2 === 0 ? kerbRed : kerbWhite
      for (const side of [1, -1]) {
        const width = side === 1 ? current.width_left_m : current.width_right_m
        const edge = side * width
        const inner = edge - side * 0.75
        const outer = edge + side * 0.05
        kerbPositions.push(
          current.x + nx * inner, 0.045, -(current.y + ny * inner),
          current.x + nx * outer, 0.045, -(current.y + ny * outer),
        )
        for (let j = 0; j < 2; j += 1) kerbColors.push(stripe.r, stripe.g, stripe.b)
      }
    }
    for (let i = 0; i < count; i += 1) {
      for (let side = 0; side < 2; side += 1) {
        const base = (i * 2 + side) * 2
        const next = (((i + 1) % count) * 2 + side) * 2
        kerbIndices.push(base, next, base + 1, base + 1, next, next + 1)
      }
    }
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(kerbPositions, 3))
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(kerbColors, 3))
    geometry.setIndex(kerbIndices)
    geometry.computeVertexNormals()
    return geometry
  }
  for (let i = 0; i < count; i += 1) {
    const a = i * 2
    const b = ((i + 1) % count) * 2
    indices.push(a, b, a + 1, a + 1, b, b + 1)
    if (edgeOnly) {
      const color = edgeColor
      colors.push(color.r, color.g, color.b, color.r, color.g, color.b)
    }
  }
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  if (edgeOnly) geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  return geometry
}

function createKerbStrip(track: TrackPayload) {
  return createRibbon(track, 0, 0.055, false, true)
}

function createBoundary(track: TrackPayload, side: Side) {
  const count = track.centerline.length
  return track.centerline.map((point, index) => {
    const previous = track.centerline[(index - 1 + count) % count]
    const next = track.centerline[(index + 1) % count]
    const dx = next.x - previous.x
    const dy = next.y - previous.y
    const length = Math.hypot(dx, dy) || 1
    const sign = side === 'left' ? 1 : -1
    const width = side === 'left' ? point.width_left_m : point.width_right_m
    const nx = -dy / length
    const ny = dx / length
    return new THREE.Vector3(point.x + sign * width * nx, 0.08, -(point.y + sign * width * ny))
  })
}

function createBarrier(track: TrackPayload, side: Side, height: number) {
  const points = track.centerline.map((_, index) => {
    const data = track.centerline[index]
    const extra = 7.5
    const width = side === 'left' ? data.width_left_m : data.width_right_m
    const sign = side === 'left' ? 1 : -1
    const before = track.centerline[(index - 1 + track.centerline.length) % track.centerline.length]
    const after = track.centerline[(index + 1) % track.centerline.length]
    const dx = after.x - before.x
    const dy = after.y - before.y
    const length = Math.hypot(dx, dy) || 1
    const nx = -dy / length
    const ny = dx / length
    return new THREE.Vector3(data.x + sign * (width + extra) * nx, height, -(data.y + sign * (width + extra) * ny))
  })
  const curve = new THREE.CatmullRomCurve3(points, true, 'centripetal')
  return new THREE.TubeGeometry(curve, points.length * 2, 0.28, 5, true)
}

function linesFromXY(points: XY[], height: number) {
  return points.map((point) => new THREE.Vector3(point.x, height, -point.y))
}

function SimplifiedGrandstand({ track }: { track: TrackPayload }) {
  const positions = useMemo(() => [0.03, 0.16, 0.27, 0.53, 0.79].map((fraction) => {
    const point = track.centerline[Math.floor(track.centerline.length * fraction)]
    const next = track.centerline[(Math.floor(track.centerline.length * fraction) + 1) % track.centerline.length]
    const dx = next.x - point.x
    const dy = next.y - point.y
    const heading = Math.atan2(dy, dx)
    const nx = -dy / (Math.hypot(dx, dy) || 1)
    const ny = dx / (Math.hypot(dx, dy) || 1)
    return { x: point.x + nx * 29, z: -(point.y + ny * 29), heading }
  }), [track])
  return <>
    {positions.map((position, index) => <group key={index} position={[position.x, 0, position.z]} rotation={[0, Math.PI / 2 + position.heading, 0]}>
      <mesh position={[0, 1.2, 0]} castShadow>
        <boxGeometry args={[28, 2.4, 7]} />
        <meshStandardMaterial color="#222a2d" roughness={0.75} />
      </mesh>
      {Array.from({ length: 6 }, (_, row) => <mesh key={row} position={[0, 2.1 + row * 0.8, -2.8 + row * 0.66]} castShadow>
        <boxGeometry args={[28, 0.72, 1.3]} />
        <meshStandardMaterial color={row % 2 ? '#596367' : '#303a3e'} roughness={0.72} />
      </mesh>)}
      <mesh position={[0, 5, -1]}>
        <boxGeometry args={[29, 0.15, 0.15]} />
        <meshStandardMaterial color="#b9c8c2" emissive="#43504d" />
      </mesh>
    </group>)}
  </>
}

export const Track = memo(function Track({ track }: { track: TrackPayload }) {
  const road = useMemo(() => createRibbon(track, 0, 0.025), [track])
  const runoff = useMemo(() => createRibbon(track, 11, -0.035), [track])
  const kerbs = useMemo(() => createKerbStrip(track), [track])
  const leftBoundary = useMemo(() => createBoundary(track, 'left'), [track])
  const rightBoundary = useMemo(() => createBoundary(track, 'right'), [track])
  const leftBarrierLower = useMemo(() => createBarrier(track, 'left', 0.5), [track])
  const rightBarrierLower = useMemo(() => createBarrier(track, 'right', 0.5), [track])
  const leftBarrierUpper = useMemo(() => createBarrier(track, 'left', 1.1), [track])
  const rightBarrierUpper = useMemo(() => createBarrier(track, 'right', 1.1), [track])
  const referenceLine = useMemo(() => linesFromXY(track.race_line, 0.11), [track])
  const ground = useMemo(() => {
    const xs = track.centerline.map((point) => point.x)
    const ys = track.centerline.map((point) => point.y)
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)
    return { x: (minX + maxX) / 2, z: -(minY + maxY) / 2, width: maxX - minX + 700, depth: maxY - minY + 700 }
  }, [track])
  return <group>
    <mesh position={[ground.x, -0.35, ground.z]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
      <planeGeometry args={[ground.width, ground.depth]} />
      <meshStandardMaterial color="#344237" roughness={1} />
    </mesh>
    <mesh geometry={runoff} receiveShadow>
      <meshStandardMaterial color="#8a7e62" roughness={0.97} side={THREE.DoubleSide} />
    </mesh>
    <mesh geometry={road} receiveShadow>
      <meshStandardMaterial color="#303438" roughness={0.9} side={THREE.DoubleSide} />
    </mesh>
    <mesh geometry={kerbs}>
      <meshStandardMaterial vertexColors roughness={0.9} side={THREE.DoubleSide} />
    </mesh>
    <Line points={leftBoundary} color="#b2c1b4" lineWidth={1.1} transparent opacity={0.72} />
    <Line points={rightBoundary} color="#b2c1b4" lineWidth={1.1} transparent opacity={0.72} />
    {[leftBarrierLower, rightBarrierLower].map((geometry, index) => <mesh key={`rail-low-${index}`} geometry={geometry}>
      <meshStandardMaterial color="#3d4745" metalness={0.46} roughness={0.58} />
    </mesh>)}
    {[leftBarrierUpper, rightBarrierUpper].map((geometry, index) => <mesh key={`rail-high-${index}`} geometry={geometry}>
      <meshStandardMaterial color="#727a74" metalness={0.64} roughness={0.46} />
    </mesh>)}
    <Line points={referenceLine} color="#9da2a3" lineWidth={1.25} transparent opacity={0.78} />
    <GrandstandAsset track={track} fallback={<SimplifiedGrandstand track={track} />} />
  </group>
})
