import React, { Suspense, useMemo } from 'react'
import { useGLTF } from '@react-three/drei'
import * as THREE from 'three'
import type { TrackPayload } from '../types'

class AssetBoundary extends React.Component<{ fallback: React.ReactNode; children: React.ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}

function StandModels({ track }: { track: TrackPayload }) {
  const gltf = useGLTF('/assets/six-tier-grandstand.glb')
  const scene = useMemo(() => {
    const clone = gltf.scene.clone(true)
    clone.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        object.castShadow = true
        object.receiveShadow = true
      }
    })
    return clone
  }, [gltf.scene])
  const positions = useMemo(() => [0.03, 0.16, 0.27, 0.53, 0.79].map((fraction) => {
    const index = Math.floor(track.centerline.length * fraction)
    const point = track.centerline[index]
    const next = track.centerline[(index + 1) % track.centerline.length]
    const dx = next.x - point.x
    const dy = next.y - point.y
    const length = Math.hypot(dx, dy) || 1
    const nx = -dy / length
    const ny = dx / length
    return { x: point.x + nx * 28, z: -(point.y + ny * 28), heading: Math.atan2(dy, dx) }
  }), [track])
  return <>
    {positions.map((position, index) => <group key={index} position={[position.x, 0, position.z]} rotation={[0, Math.PI / 2 + position.heading, 0]} scale={1.3}>
      <primitive object={scene.clone(true)} />
    </group>)}
  </>
}

export function GrandstandAsset({ track, fallback }: { track: TrackPayload; fallback: React.ReactNode }) {
  return <AssetBoundary fallback={fallback}>
    <Suspense fallback={fallback}>
      <StandModels track={track} />
    </Suspense>
  </AssetBoundary>
}

useGLTF.preload('/assets/six-tier-grandstand.glb')
