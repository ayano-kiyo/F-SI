import React, { Suspense, useEffect, useMemo, useRef } from 'react'
import { useGLTF } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js'
import type { XY } from '../types'
import type { VisualRacePose } from '../utils/telemetryInterpolation'

class AssetBoundary extends React.Component<{ fallback: React.ReactNode; children: React.ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}

function ProceduralCar({ color = '#d8e5dd' }: { color?: string }) {
  return <group>
    <mesh position={[0, 0.47, 0.02]} castShadow>
      <boxGeometry args={[0.82, 0.48, 2.35]} />
      <meshStandardMaterial color={color} metalness={0.28} roughness={0.32} />
    </mesh>
    <mesh position={[0, 0.48, 1.46]} castShadow>
      <boxGeometry args={[0.36, 0.24, 1.45]} />
      <meshStandardMaterial color="#d8e5dd" metalness={0.2} roughness={0.34} />
    </mesh>
    <mesh position={[0, 0.39, 2.04]} castShadow>
      <boxGeometry args={[3.25, 0.13, 0.32]} />
      <meshStandardMaterial color="#edf1e7" metalness={0.25} roughness={0.35} />
    </mesh>
    <mesh position={[0, 0.77, -1.32]} castShadow>
      <boxGeometry args={[2.15, 0.12, 0.26]} />
      <meshStandardMaterial color="#dae1dc" metalness={0.22} roughness={0.34} />
    </mesh>
    <mesh position={[0, 0.68, -1.05]} castShadow>
      <boxGeometry args={[0.11, 0.55, 0.12]} />
      <meshStandardMaterial color="#a5b3ad" metalness={0.45} />
    </mesh>
    <mesh position={[0, 0.82, -0.15]} castShadow>
      <sphereGeometry args={[0.42, 18, 12]} />
      <meshStandardMaterial color="#a96b48" roughness={0.72} />
    </mesh>
    {[-1, 1].flatMap((side) => [-1, 1].map((axle) => <mesh key={`${side}-${axle}`} position={[side * 0.97, 0.38, axle * 1.23]} rotation={[0, 0, Math.PI / 2]} castShadow>
      <cylinderGeometry args={[0.38, 0.38, 0.3, 24]} />
      <meshStandardMaterial color="#171a1b" roughness={0.96} />
    </mesh>))}
  </group>
}

function LoadedCar() {
  const gltf = useGLTF('/assets/f1-car.glb')
  const scene = useMemo(() => {
    const clone = gltf.scene.clone(true)
    clone.updateMatrixWorld(true)
    const worldToRoot = clone.matrixWorld.clone().invert()
    const batches = new Map<string, { material: THREE.Material; geometries: THREE.BufferGeometry[] }>()
    const unbatched: THREE.Mesh[] = []

    clone.traverse((object) => {
      if (!(object instanceof THREE.Mesh) || object instanceof THREE.SkinnedMesh) return

      const material = object.material
      const geometry = object.geometry.clone()
      const localTransform = worldToRoot.clone().multiply(object.matrixWorld)
      geometry.applyMatrix4(localTransform)

      if (Array.isArray(material)) {
        unbatched.push(new THREE.Mesh(geometry, material))
        return
      }

      const attributeLayout = Object.entries(geometry.attributes)
        .map(([name, attribute]) => {
          const bufferAttribute = attribute as THREE.BufferAttribute
          return `${name}:${bufferAttribute.itemSize}:${bufferAttribute.normalized}:${bufferAttribute.array.constructor.name}`
        })
        .sort()
        .join('|')
      const key = `${material.uuid}:${geometry.index ? 'indexed' : 'unindexed'}:${attributeLayout}`
      const batch: { material: THREE.Material; geometries: THREE.BufferGeometry[] } = batches.get(key) ?? { material, geometries: [] }
      batch.geometries.push(geometry)
      batches.set(key, batch)
    })

    const optimizedMeshes: THREE.Mesh[] = []
    for (const { material, geometries } of batches.values()) {
      const merged = geometries.length === 1 ? geometries[0] : mergeGeometries(geometries)
      if (merged) {
        if (merged !== geometries[0]) geometries.forEach((geometry) => geometry.dispose())
        merged.computeBoundingBox()
        merged.computeBoundingSphere()
        optimizedMeshes.push(new THREE.Mesh(merged, material))
      } else {
        optimizedMeshes.push(...geometries.map((geometry) => new THREE.Mesh(geometry, material)))
      }
    }

    clone.clear()
    for (const mesh of [...optimizedMeshes, ...unbatched]) {
      mesh.castShadow = true
      mesh.receiveShadow = true
      clone.add(mesh)
    }
    return clone
  }, [gltf.scene])
  return <primitive object={scene} />
}

useGLTF.preload('/assets/f1-car.glb')

export const Car = React.memo(function Car({ visualPoses, vehicleId, markerColor, fallbackPosition }: {
  visualPoses: { current: VisualRacePose }
  vehicleId: string
  markerColor: string
  fallbackPosition: XY
}) {
  const group = useRef<THREE.Group>(null)
  const shadowTexture = useMemo(() => {
    const canvas = document.createElement('canvas')
    canvas.width = 64
    canvas.height = 64
    const context = canvas.getContext('2d')
    if (!context) return null
    const gradient = context.createRadialGradient(32, 32, 4, 32, 32, 32)
    gradient.addColorStop(0, 'rgba(0, 0, 0, 0.46)')
    gradient.addColorStop(0.48, 'rgba(0, 0, 0, 0.2)')
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)')
    context.fillStyle = gradient
    context.fillRect(0, 0, 64, 64)
    const texture = new THREE.CanvasTexture(canvas)
    texture.minFilter = THREE.LinearFilter
    texture.generateMipmaps = false
    return texture
  }, [])
  useEffect(() => () => shadowTexture?.dispose(), [shadowTexture])
  useFrame(() => {
    if (!group.current) return
    const pose = visualPoses.current[vehicleId]
    if (!pose) return
    group.current.position.set(pose.x, 0.04, -pose.y)
    group.current.rotation.y = Math.PI / 2 + pose.heading_rad
  })
  return <group ref={group} position={[fallbackPosition.x, 0.04, -fallbackPosition.y]}>
    <mesh position={[0, 0.004, 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[0.72, 1.65, 1]} renderOrder={1}>
      <planeGeometry args={[3.8, 3.8]} />
      <meshBasicMaterial map={shadowTexture ?? undefined} transparent opacity={0.68} depthWrite={false} toneMapped={false} />
    </mesh>
    <AssetBoundary fallback={<ProceduralCar color={markerColor} />}>
      <Suspense fallback={<ProceduralCar color={markerColor} />}>
        <LoadedCar />
      </Suspense>
    </AssetBoundary>
    <mesh position={[0, 1.45, 0]} castShadow>
      <sphereGeometry args={[0.22, 12, 8]} />
      <meshBasicMaterial color={markerColor} toneMapped={false} />
    </mesh>
  </group>
})

