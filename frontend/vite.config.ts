import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const backend = process.env.VITE_BACKEND_ORIGIN ?? 'http://127.0.0.1:8000'
const proxy = {
  '/api': backend,
  '/ws': { target: backend, ws: true },
}

export default defineConfig({
  plugins: [react()],
  server: {
    proxy,
  },
  preview: {
    proxy,
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('/node_modules/three-stdlib/')) return 'three-stdlib-vendor'
          if (id.includes('/node_modules/three/')) return 'three-vendor'
          if (id.includes('/node_modules/@react-three/')) return 'r3f-vendor'
          if (id.includes('/node_modules/lucide-react/')) return 'icons-vendor'
        },
      },
    },
  },
})
