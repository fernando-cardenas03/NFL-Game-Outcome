import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy /api calls to Flask during development
      // so you don't need to worry about CORS in dev
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      }
    }
  }
})