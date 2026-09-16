/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  // vitest 单测（jsdom 环境跑组件测试；生产构建不受影响）
  test: {
    environment: 'jsdom',
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      // 注意：必须用 127.0.0.1 显式 IPv4——uvicorn 只绑 IPv4，
      // 而 Node 解析 localhost 可能优先 ::1 导致 proxy ECONNREFUSED
      '/api': {
        target: 'http://127.0.0.1:8099',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://127.0.0.1:8099',
        ws: true,
      },
      '/thumbnails': {
        target: 'http://127.0.0.1:8099',
        changeOrigin: true,
      },
    },
  },
})
