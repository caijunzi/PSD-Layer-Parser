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
    // 显式绑 IPv4 回环：默认 localhost 在 Node 上可能只绑 ::1，
    // 导致本机预览/自动化按 127.0.0.1 探测时不可达（2026-09-17）。
    host: "127.0.0.1",
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
