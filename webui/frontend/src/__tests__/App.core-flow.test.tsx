// -*- coding: utf-8 -*-
/**
 * 核心交互单测（vitest + Testing Library，jsdom）：
 *   上传 → AI 推荐 → 提交
 *
 * 反造假口径：**被测的是 App 组件的真实逻辑**（状态流转 / 表单组装 / 推荐自动选中 /
 * 提交 payload 组装 / 吸底栏可用性）。仅在**网络边界**（fetch / WebSocket）打桩——
 * 这是组件单测的标准做法，不伪造任何被测行为。
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from '../App'

/* ---------- 测试夹具 ---------- */

const PRESETS = {
  data: [
    {
      name: 'chinese_ink_landscape_ai',
      display_name: '中国传统水墨山水画与屏风',
      semantic_classes: 11,
      output_modes: ['design'],
    },
    {
      name: 'textile_damask_photo',
      display_name: '壁布/面料实物样品照',
      semantic_classes: 2,
      output_modes: ['plate'],
    },
  ],
}

const UPLOAD_INFO = {
  data: {
    file_id: 'file_test_0001',
    filename: 'sample.jpeg',
    size: 6398400,
    dimensions: { width: 2848, height: 1600 },
    thumbnail_url: null,
    recommended_preset: 'textile_damask_photo',
    confidence: 0.99,
  },
}

/* ---------- 网络边界打桩（统一工厂，避免各用例桩形状漂移） ---------- */

const calls: Array<{ url: string; method: string; body: any }> = []

function jsonResponse(data: unknown, ok = true) {
  return { ok, status: ok ? 200 : 400, json: async () => data } as Response
}

function makeFetch(uploadOverride?: unknown) {
  return vi.fn(async (input: any, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.url
    const method = (init?.method ?? 'GET').toUpperCase()
    calls.push({ url, method, body: init?.body })
    if (url.startsWith('/api/health')) return jsonResponse({ status: 'healthy' })
    if (url.startsWith('/api/presets')) return jsonResponse(PRESETS)
    if (url.startsWith('/api/upload'))
      return jsonResponse(uploadOverride ?? UPLOAD_INFO)
    if (url.startsWith('/api/adaptive/suggest-auto-tune')) return jsonResponse({ data: {} })
    if (url.startsWith('/api/adaptive/pending-feedbacks'))
      return jsonResponse({ data: { items: [] } })
    if (url.startsWith('/api/history')) return jsonResponse({ data: { total: 0, items: [] } })
    if (url.startsWith('/api/process'))
      return jsonResponse({ data: { task_id: 'task_test_0001' } })
    return jsonResponse({ data: {} })
  })
}

function installStubs(uploadOverride?: unknown) {
  vi.stubGlobal('fetch', makeFetch(uploadOverride))
  // WebSocket 桩：只记录 URL，不发任何消息（组件应停在 processing 态等待进度）
  class FakeWebSocket {
    static last: any = null
    url: string
    onopen: (() => void) | null = null
    onmessage: ((e: any) => void) | null = null
    onclose: (() => void) | null = null
    onerror: ((e: any) => void) | null = null
    constructor(url: string) {
      this.url = url
      ;(FakeWebSocket as any).last = this
    }
    close() {}
    send() {}
  }
  vi.stubGlobal('WebSocket', FakeWebSocket as unknown as typeof WebSocket)
}

const findCall = (prefix: string) => calls.find((c) => c.url.startsWith(prefix))

const submitButton = () =>
  screen
    .getAllByRole('button')
    .find((b) => b.textContent?.includes('开始处理')) as HTMLButtonElement

async function uploadAndReachReady(file = new File(['fake-image-bytes'], 'sample.jpeg', { type: 'image/jpeg' })) {
  const { container } = render(<App />)
  const input = container.querySelector('#file-input') as HTMLInputElement
  fireEvent.change(input, { target: { files: [file] } })
  // 上传完成：文件名渲染（UploadInfo 真实进入组件状态）
  await screen.findByText(file.name)
  return container
}

beforeEach(() => {
  calls.length = 0
  installStubs()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

/* ---------- 用例 ---------- */

describe('App 核心交互：上传 → AI 推荐 → 提交', () => {
  it('初始加载：后端连通提示 + /api/presets 与 /api/history 真实拉取', async () => {
    render(<App />)
    // 健康检查成功 → 页面出现「后端已连接」
    expect(await screen.findByText(/后端已连接/)).toBeTruthy()
    // preset 列表与历史面板的数据源都已拉取（列表 UI 在上传后才渲染）
    expect(findCall('/api/presets')).toBeTruthy()
    expect(findCall('/api/history')).toBeTruthy()
  })

  it('上传真实文件 → AI 推荐自动选中 preset，提交按钮可用', async () => {
    await uploadAndReachReady()
    // preset 列表此刻渲染（config 区依赖 uploadInfo）
    expect(screen.getByText(/壁布\/面料实物样品照/)).toBeTruthy()
    // AI 推荐徽标出现（上传卡片置信度条 + Auto-Tune 面板，多处文案含该词）
    expect(screen.getAllByText(/AI 推荐/).length).toBeGreaterThanOrEqual(1)
    // 吸底操作栏的提交按钮可用（selectedPreset 已被推荐值自动选中）
    const submit = submitButton()
    expect(submit).toBeTruthy()
    expect(submit.disabled).toBe(false)
    // /api/upload 必须以 POST + FormData（含 file 字段）调用
    const up = findCall('/api/upload')
    expect(up).toBeTruthy()
    expect(up!.method).toBe('POST')
    expect(up!.body).toBeInstanceOf(FormData)
    expect((up!.body as FormData).get('file')).toBeTruthy()
  })

  it('点击「开始处理」→ POST /api/process payload 正确并进入处理态', async () => {
    await uploadAndReachReady()
    const submit = submitButton()
    expect(submit.disabled).toBe(false)
    fireEvent.click(submit)

    // 等待 /api/process 真实发出
    await waitFor(() => expect(findCall('/api/process')).toBeTruthy())
    const proc = findCall('/api/process')!
    expect(proc.method).toBe('POST')
    const payload = JSON.parse(proc.body)
    // payload 组装自组件真实状态：默认 mode=both / scale=4 / dpi=150 / seed=42
    expect(payload).toEqual({
      file_id: 'file_test_0001',
      preset: 'textile_damask_photo',
      mode: 'both',
      scale: 4,
      dpi: 150,
      seed: 42,
      profile: 'robust_performance',
    })
    // 提交后建立 WebSocket 进度通道
    const WS = (globalThis as any).WebSocket
    expect(WS.last?.url).toContain('/ws/progress/task_test_0001')
    // UI 进入处理态
    expect(await screen.findByText(/正在处理/)).toBeTruthy()
  })

  it('推荐驱动的自动选中：上传响应给什么推荐，提交就用什么 preset（防假自动选中）', async () => {
    calls.length = 0
    vi.stubGlobal(
      'fetch',
      makeFetch({
        data: {
          ...UPLOAD_INFO.data,
          filename: 'ink.jpeg',
          recommended_preset: 'chinese_ink_landscape_ai',
        },
      }),
    )
    await uploadAndReachReady(
      new File(['x'], 'ink.jpeg', { type: 'image/jpeg' }),
    )
    const submit = submitButton()
    expect(submit.disabled).toBe(false)
    fireEvent.click(submit)
    await waitFor(() => expect(findCall('/api/process')).toBeTruthy())
    const payload = JSON.parse(findCall('/api/process')!.body)
    expect(payload.preset).toBe('chinese_ink_landscape_ai')
  })
})
