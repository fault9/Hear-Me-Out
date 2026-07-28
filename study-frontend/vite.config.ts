import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, __dirname, '')
  const apiUrl = env.VITE_API_URL || 'http://localhost:5001'

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
        '@shared': path.resolve(__dirname, '../packages/shared/src'),
      },
    },
    server: {
      port: 3100,
      proxy: {
        '/api': { target: apiUrl, changeOrigin: true, secure: false },
        '/recordings': { target: apiUrl, changeOrigin: true, secure: false },
      },
    },
  }
})
