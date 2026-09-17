import { useCallback, useEffect, useRef, useState } from 'react'

// Stage 5.3：Auto-Tune 卡片与人审弹窗抽为独立组件
import AutoTuneCard, { type AutoTuneSuggestionData } from './components/AutoTuneSuggestion'
import CategoryFeedbackModal, {
  type FeedbackRequest,
  type FeedbackAction,
} from './components/CategoryFeedbackModal'

/* ===== 类型 ===== */
type PresetInfo = {
  name: string
  display_name: string
  description: string
  semantic_classes: number
  output_modes: string[]
}

type UploadInfo = {
  file_id: string
  filename: string
  size: number
  dimensions: { width: number; height: number } | null
  thumbnail_url: string | null
  recommended_preset: string
  confidence: number
  /** 2026-09-17：材质判别是否真正命中（false = 按宽高比/兜底推测，需用户确认品类） */
  matched?: boolean
  recommend_reason?: string
  material_family?: string | null
  family_conf?: number | null
}

type OutputFile = {
  type: string
  filename: string
  size: number
  download_url: string
}

type WsMessage = {
  type: 'progress' | 'completed' | 'error' | 'audit'
  task_id: string
  stage?: string | null
  progress?: number
  message?: string
  elapsed?: number
  replay?: boolean
  elapsed_time?: number
  output_files?: OutputFile[]
  manifest?: Record<string, unknown>
  error?: { code: string; message: string }
}

type Phase = 'idle' | 'uploading' | 'ready' | 'processing' | 'done' | 'failed'

type LogLine = { message: string; stage: string | null; progress: number }

/* ===== 工具 ===== */
const fmtSize = (n: number) => {
  if (n < 0) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`
  return `${(n / 1024 ** 3).toFixed(2)} GB`
}

const MODE_LABELS: Record<string, string> = {
  design: '设计线 RGB',
  plate: '制版线 CMYK',
  both: '双线（两份都出）',
}

/** 输出模式的新手向说明：一句话讲清「产出什么、给谁用」（避免 only 英文缩写） */
const MODE_HELP: Record<string, { title: string; tag: string; desc: string }> = {
  design: {
    title: '设计线 RGB',
    tag: '屏幕 / 喷墨 / 办公打印',
    desc: '产出常规 RGB 彩色文件，交给设计稿、效果图、普通彩打使用。',
  },
  plate: {
    title: '制版线 CMYK',
    tag: '印刷厂印前制版',
    desc: '产出 CMYK 印刷分色文件（含黑版，已做总墨量合规），直接交印刷厂。',
  },
  both: {
    title: '双线（两份都出）',
    tag: '一次出两份 · 耗时≈两者之和',
    desc: '同时产出设计线与制版线两份文件，适合既要看效果又要送印的场景。',
  },
}

/** 算力档位（性能 vs 稳定性）——依据 engine/schemas/profile_config.py 的真实语义改写为人话 */
const PROFILE_META: Record<string, { label: string; hint: string; desc: string }> = {
  robust_performance: {
    label: '稳健（推荐）',
    hint: '独显主力 + 核显兜底',
    desc: '独立显卡主算，核显的大显存做安全垫，并有超时自愈：速度与稳定性兼顾，日常首选。',
  },
  '5070': {
    label: '5070 独显',
    hint: '最快',
    desc: '强制全部用独立显卡（RTX 5070）计算，速度最快；超大画幅时有显存不足的风险。',
  },
  arc: {
    label: 'arc 核显',
    hint: '大显存最不易崩',
    desc: '使用核显共享的 16GB 内存，超大图最不容易因显存不足中断；速度中等。',
  },
  cpu: {
    label: 'cpu 纯 CPU',
    hint: '最兼容 · 最慢',
    desc: '全部用 CPU 计算，不依赖显卡驱动，兼容性最好、结果最保守；耗时明显更长。',
  },
}

/** 参数行的小问号（点击展开白话说明）——不引入任何新依赖 */
function HelpTip({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false)
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-label="查看参数说明"
        aria-expanded={open}
        className="ml-1 inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-neutral-600 align-middle text-[10px] leading-none text-neutral-400 transition-colors hover:border-blue-400 hover:text-blue-300"
      >
        ?
      </button>
      {open && (
        <span className="mt-1.5 block rounded-lg border border-neutral-700 bg-neutral-900/70 px-3 py-2 text-[11px] font-normal leading-relaxed text-neutral-300">
          {children}
        </span>
      )}
    </>
  )
}

/** /api/history 条目（此前为孤儿端点：后端可用、前端零调用 → 刷新后看不到历史任务） */
interface HistoryItem {
  task_id: string
  filename: string
  preset: string
  mode: string
  status: string
  created_at: string
  completed_at?: string
  elapsed_time?: number
}

export default function App() {
  /* ===== 状态 ===== */
  const [phase, setPhase] = useState<Phase>('idle')
  const [uploadInfo, setUploadInfo] = useState<UploadInfo | null>(null)
  const [presets, setPresets] = useState<PresetInfo[]>([])
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null)
  const [mode, setMode] = useState<'design' | 'plate' | 'both'>('both')
  const [scale, setScale] = useState(4.0)
  const [dpi, setDpi] = useState(150)
  const [seed, setSeed] = useState('42')
  const [profile, setProfile] = useState('robust_performance')

  const [progress, setProgress] = useState(0)
  const [stage, setStage] = useState('—')
  const [logs, setLogs] = useState<LogLine[]>([])
  const [taskId, setTaskId] = useState<string | null>(null)
  const [outputs, setOutputs] = useState<OutputFile[]>([])
  const [manifest, setManifest] = useState<Record<string, any>>({})
  const [audit, setAudit] = useState<Record<string, any> | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const [dragging, setDragging] = useState(false)
  const [backendUp, setBackendUp] = useState<boolean | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const logBoxRef = useRef<HTMLDivElement>(null)

  // Auto-Tune 建议状态
  const [autoTuneSuggestion, setAutoTuneSuggestion] = useState<AutoTuneSuggestionData | null>(null)
  const [autoTuneLoading, setAutoTuneLoading] = useState(false)
  const [autoTuneError, setAutoTuneError] = useState<string | null>(null)
  const [autoTuneCollapsed, setAutoTuneCollapsed] = useState(false)
  const [applyingAutoTune, setApplyingAutoTune] = useState(false)

  // 历史记录（/api/history，此前为孤儿端点：刷新后看不到历史任务）
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(false)
  const loadHistory = useCallback(() => {
    setHistoryLoading(true)
    fetch('/api/history?limit=10')
      .then((r) => r.json())
      .then((b) => setHistory(b?.data?.items ?? []))
      .catch(() => setHistory([]))
      .finally(() => setHistoryLoading(false))
  }, [])
  useEffect(() => {
    loadHistory()
  }, [loadHistory])
  // 任务完成后刷新历史（audit 到手即 phase='done'）
  useEffect(() => {
    if (phase === 'done') loadHistory()
  }, [phase, loadHistory])

  /* ===== 后端真实健康检查（每 5 秒） ===== */
  useEffect(() => {
    const check = () =>
      fetch('/api/health')
        .then((r) => setBackendUp(r.ok))
        .catch(() => setBackendUp(false))
    check()
    const t = setInterval(check, 5000)
    return () => clearInterval(t)
  }, [])

  /* ===== 拉取 preset 列表 ===== */
  useEffect(() => {
    fetch('/api/presets')
      .then((r) => r.json())
      .then((d) => setPresets(d.data ?? []))
      .catch(() => setPresets([]))
  }, [])

  /* ===== 日志自动滚动 ===== */
  useEffect(() => {
    logBoxRef.current?.scrollTo({ top: logBoxRef.current.scrollHeight })
  }, [logs])

  /* ===== 上传 ===== */
  const doUpload = useCallback(async (file: File) => {
    setPhase('uploading')
    setError(null)
    setAutoTuneSuggestion(null)
    setAutoTuneError(null)
    const fd = new FormData()
    fd.append('file', file)
    try {
      const r = await fetch('/api/upload', { method: 'POST', body: fd })
      const body = await r.json()
      if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`)
      const info: UploadInfo = body.data
      setUploadInfo(info)
      setSelectedPreset(info.recommended_preset)
      setPhase('ready')

      // 自动触发 Auto-Tune 建议
      void fetchAutoTuneSuggestion(file, info.recommended_preset)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase('idle')
    }
  }, [])

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragging(false)
      const f = e.dataTransfer.files?.[0]
      if (f) void doUpload(f)
    },
    [doUpload],
  )

  /* ===== Auto-Tune 建议获取 ===== */
  const fetchAutoTuneSuggestion = useCallback(async (file: File, presetId: string) => {
    setAutoTuneLoading(true)
    setAutoTuneError(null)
    const fd = new FormData()
    fd.append('file', file)
    fd.append('preset_id', presetId)
    try {
      const r = await fetch('/api/adaptive/suggest-auto-tune', { method: 'POST', body: fd })
      const body = await r.json()
      if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`)
      setAutoTuneSuggestion(body)
    } catch (e) {
      setAutoTuneError(e instanceof Error ? e.message : String(e))
    } finally {
      setAutoTuneLoading(false)
    }
  }, [])

  /* ===== 采纳 Auto-Tune 建议 ===== */
  const applyAutoTune = useCallback(async () => {
    if (!autoTuneSuggestion || !selectedPreset) return
    setApplyingAutoTune(true)
    try {
      const r = await fetch('/api/adaptive/apply-auto-tune', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          preset_id: selectedPreset,
          suggestions: JSON.stringify({
            regions: autoTuneSuggestion.regions,
            density_bands: autoTuneSuggestion.density_bands,
          }),
        }),
      })
      const body = await r.json()
      if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`)
      // 成功：折叠建议卡片并提示
      setAutoTuneCollapsed(true)
      alert('✅ Auto-Tune 建议已采纳并写入 preset')
    } catch (e) {
      alert(`❌ 采纳失败: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setApplyingAutoTune(false)
    }
  }, [autoTuneSuggestion, selectedPreset])

  /* ===== Stage 5.2 主动学习：轮询待审类目（3 秒） ===== */
  const [pendingRequest, setPendingRequest] = useState<FeedbackRequest | null>(null)
  const [showFeedbackModal, setShowFeedbackModal] = useState(false)
  const [submittingFeedback, setSubmittingFeedback] = useState(false)

  useEffect(() => {
    let cancelled = false
    const timer = setInterval(async () => {
      // 弹窗打开时暂停轮询，避免打断用户操作
      if (showFeedbackModal) return
      try {
        const r = await fetch('/api/adaptive/pending-feedbacks?limit=1')
        if (!r.ok) return
        const body = await r.json()
        const list: FeedbackRequest[] = body.pending_feedbacks ?? []
        if (!cancelled && list.length > 0) {
          setPendingRequest(list[0])
          setShowFeedbackModal(true)
        }
      } catch {
        // 轮询失败静默处理（后端未启用或无待审数据），不打扰用户
      }
    }, 3000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [showFeedbackModal])

  /* ===== 提交人审反馈 ===== */
  const submitCategoryFeedback = useCallback(
    async (categoryId: string, action: FeedbackAction, userData?: Record<string, unknown>) => {
      if (!pendingRequest) return
      setSubmittingFeedback(true)
      try {
        const fd = new URLSearchParams({
          feedback_request_id: pendingRequest.feedback_request_id,
          user_action: action,
        })
        if (userData) fd.append('user_data', JSON.stringify(userData))

        const r = await fetch(
          `/api/adaptive/categories/${encodeURIComponent(categoryId)}/feedback`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: fd,
          },
        )
        const body = await r.json().catch(() => ({}))
        if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`)

        // 该类目已裁决：从待审列表移除；全部处理完才关闭弹窗
        const rest = (pendingRequest.uncertain_categories ?? []).filter(
          (c) => c.category_id !== categoryId,
        )
        if (rest.length === 0) {
          setShowFeedbackModal(false)
          setPendingRequest(null)
        } else {
          setPendingRequest({ ...pendingRequest, uncertain_categories: rest })
        }
      } catch (e) {
        alert(`❌ 提交反馈失败: ${e instanceof Error ? e.message : String(e)}`)
      } finally {
        setSubmittingFeedback(false)
      }
    },
    [pendingRequest],
  )

  /* ===== 提交处理 + WebSocket ===== */
  const startProcess = useCallback(async () => {
    if (!uploadInfo || !selectedPreset) return
    setError(null)
    setLogs([])
    setProgress(0)
    setStage('提交任务')
    try {
      const r = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_id: uploadInfo.file_id,
          preset: selectedPreset,
          mode,
          scale: scale || null,
          dpi: dpi > 0 ? dpi : 150,
          seed: seed ? parseInt(seed, 10) : null,
          profile,
        }),
      })
      const body = await r.json()
      if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`)
      const tid: string = body.data.task_id
      setTaskId(tid)
      setPhase('processing')

      // 建立 WebSocket（vite 代理 /ws → 后端）
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/ws/progress/${tid}`)
      wsRef.current = ws
      ws.onmessage = (ev) => {
        const msg: WsMessage = JSON.parse(ev.data as string)
        if (msg.type === 'progress') {
          if (typeof msg.progress === 'number') setProgress(msg.progress)
          if (msg.stage) setStage(msg.stage)
          if (msg.message)
            setLogs((prev) => [
              ...prev,
              { message: msg.message ?? '', stage: msg.stage ?? null, progress: msg.progress ?? 0 },
            ])
          if (typeof msg.elapsed === 'number') setElapsed(msg.elapsed)
        } else if (msg.type === 'completed') {
          setProgress(100)
          setStage('完成')
          setOutputs(msg.output_files ?? [])
          setManifest(msg.manifest ?? {})
        } else if (msg.type === 'audit') {
          // 交付前 8 维审计门结果（任务完成后异步推送）
          setAudit(msg)
          setElapsed(msg.elapsed_time ?? 0)
          setPhase('done')
          ws.close()
        } else if (msg.type === 'error') {
          setError(msg.error?.message ?? '引擎执行失败')
          setPhase('failed')
          ws.close()
        }
      }
      ws.onerror = () => {
        setError('WebSocket 连接中断')
        setPhase('failed')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase('ready')
    }
  }, [uploadInfo, selectedPreset, mode, scale, dpi, seed, profile])

  const resetAll = useCallback(() => {
    wsRef.current?.close()
    setPhase('idle')
    setUploadInfo(null)
    setSelectedPreset(null)
    setOutputs([])
    setLogs([])
    setProgress(0)
    setError(null)
    setTaskId(null)
  }, [])

  /* ===== 渲染 ===== */
  const ready = phase === 'ready'
  const busy = phase === 'processing'

  return (
    <div className="min-h-screen bg-neutral-900 text-neutral-100">
      {/* Header */}
      <header className="sticky top-0 z-10 border-b border-neutral-800 bg-neutral-900/95 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-6xl items-center px-6">
          <div className="mr-3 flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-blue-700 text-sm font-extrabold shadow-lg shadow-blue-500/30">
            UL
          </div>
          <h1 className="text-base font-semibold">
            Universal Layer Studio <span className="text-blue-400">PRO</span>
          </h1>
          <div className="ml-auto flex items-center gap-3 text-xs text-neutral-400">
            {backendUp === null && <span>后端检查中…</span>}
            {backendUp === true && (
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-emerald-500 shadow shadow-emerald-500/60" />
                后端已连接
              </span>
            )}
            {backendUp === false && (
              <span className="flex items-center gap-1.5 text-red-400">
                <span className="h-2 w-2 rounded-full bg-red-500" />
                后端离线（请启动 webui/backend）
              </span>
            )}
          </div>
        </div>
      </header>

      <main className={"mx-auto max-w-6xl space-y-6 px-6 py-8" + (ready && !busy ? " pb-28" : "")}>
        {/* 步骤条 */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {[
            ['1', '上传图片', phase !== 'idle' && phase !== 'uploading'],
            ['2', '选择 Preset', ready || busy || phase === 'done' || phase === 'failed'],
            ['3', '配置参数', ready || busy || phase === 'done' || phase === 'failed'],
            ['4', '处理与交付', phase === 'done'],
          ].map(([num, label, done], i) => (
            <div key={i} className="flex items-center gap-2">
              {i > 0 && <span className="text-neutral-600">→</span>}
              <span
                className={
                  'flex items-center gap-1.5 rounded-full border px-3 py-1 ' +
                  (done
                    ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                    : 'border-neutral-800 text-neutral-500')
                }
              >
                <span
                  className={
                    'flex h-4 w-4 items-center justify-center rounded-full text-[9px] ' +
                    (done ? 'bg-emerald-500 text-white' : 'bg-neutral-800')
                  }
                >
                  {done ? '✓' : num}
                </span>
                {label}
              </span>
            </div>
          ))}
        </div>

        {/* 错误提示 */}
        {error && (
          <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
            ⚠ {error}
          </div>
        )}

        {/* 上传区 / 已上传预览 */}
        {(phase === 'idle' || phase === 'uploading') && (
          <div
            onDrop={onDrop}
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onClick={() => document.getElementById('file-input')?.click()}
            className={
              'cursor-pointer rounded-2xl border-2 border-dashed p-12 text-center transition-all ' +
              (dragging
                ? 'border-blue-500 bg-blue-500/10'
                : 'border-neutral-700 bg-neutral-800/40 hover:border-blue-500 hover:bg-blue-500/5')
            }
          >
            <input
              id="file-input"
              type="file"
              accept=".jpg,.jpeg,.png,.psd,.psb"
              className="sr-only"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) void doUpload(f)
                e.target.value = ''
              }}
            />
            <div className="mx-auto mb-4 flex h-14 w-14 animate-bounce items-center justify-center rounded-2xl border border-slate-700 bg-gradient-to-br from-slate-800 to-blue-950">
              <svg className="h-7 w-7 text-blue-300" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
                <path d="M12 16V4m0 0L8 8m4-4l4 4" />
                <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
              </svg>
            </div>
            <h3 className="text-base font-semibold">
              {phase === 'uploading' ? '上传中…' : '拖拽图片到此处，或点击选择文件'}
            </h3>
            <p className="mt-1.5 text-sm text-neutral-400">JPG / PNG / PSD / PSB，≤ 100 MB，全程本地处理</p>
          </div>
        )}

        {uploadInfo && phase !== 'idle' && phase !== 'uploading' && (
          <div className="flex gap-6 rounded-2xl border border-neutral-800 bg-neutral-800/40 p-5">
            {uploadInfo.thumbnail_url ? (
              <img
                src={uploadInfo.thumbnail_url}
                alt="thumbnail"
                className="h-40 w-64 flex-shrink-0 rounded-xl border border-neutral-700 object-cover"
              />
            ) : (
              <div className="flex h-40 w-64 flex-shrink-0 items-center justify-center rounded-xl border border-neutral-700 bg-neutral-800 text-sm text-neutral-500">
                PSD/PSB 无预览
              </div>
            )}
            <div className="min-w-0 flex-1">
              <h3 className="font-semibold">{uploadInfo.filename}</h3>
              <p className="mt-0.5 font-mono text-xs text-neutral-500">
                {uploadInfo.file_id} · {fmtSize(uploadInfo.size)}
                {uploadInfo.dimensions ? ` · ${uploadInfo.dimensions.width}×${uploadInfo.dimensions.height}` : ''}
              </p>
              {/* 2026-09-17：推荐置信度分级展示（未匹配 = 材质判别没认出来，需用户确认品类） */}
              {(() => {
                const unmatched = uploadInfo.matched === false
                const conf = uploadInfo.confidence
                const tier: 'high' | 'mid' | 'unknown' =
                  unmatched || conf < 0.6 ? 'unknown' : conf >= 0.8 ? 'high' : 'mid'
                if (tier === 'unknown') {
                  return (
                    <div className="mt-3 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
                      <b>未识别品类</b>——材质判别置信度不足，当前 preset 是按画面比例推测的，
                      已自动选中（置信度 {conf.toFixed(2)}）。
                      <span className="mt-1 block text-amber-200/80">
                        内容不会丢失（未识别部分会完整进入「未分类墨迹残层」），
                        但语义分层可能不准确——建议从下方「品类 Preset」手动确认。
                      </span>
                      <button
                        onClick={() =>
                          document
                            .getElementById('preset-section')
                            ?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                        }
                        className="mt-2 rounded-lg border border-amber-400/50 px-3 py-1 text-[11px] text-amber-200 transition-colors hover:bg-amber-500/20"
                      >
                        去选择品类 ↓
                      </button>
                    </div>
                  )
                }
                const cls =
                  tier === 'high'
                    ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
                    : 'border-amber-500/40 bg-amber-500/10 text-amber-200'
                const label = tier === 'high' ? 'AI 推荐 · 可信' : 'AI 推荐 · 仅供参考'
                return (
                  <div className={`mt-3 rounded-lg border px-3 py-2 text-xs ${cls}`}>
                    {label}：<b>{uploadInfo.recommended_preset}</b>
                    （置信度 {conf.toFixed(2)}，已自动选中）
                    {tier === 'mid' && (
                      <span className="ml-1 text-amber-200/80">建议核对此品类是否符合预期</span>
                    )}
                  </div>
                )
              })()}
              {phase !== 'processing' && phase !== 'done' && (
                <button
                  onClick={resetAll}
                  className="mt-3 rounded-lg border border-neutral-700 px-4 py-1.5 text-xs text-neutral-400 transition-colors hover:border-neutral-500 hover:text-neutral-200"
                >
                  重新上传
                </button>
              )}
            </div>
          </div>
        )}

        {/* 配置区（已上传且未在处理中） */}
        {ready && (
          <>
            {/* Auto-Tune 建议卡片（Stage 5.3 抽出为组件） */}
            {(autoTuneLoading || autoTuneSuggestion || autoTuneError) && !autoTuneCollapsed && (
              <AutoTuneCard
                loading={autoTuneLoading}
                suggestion={autoTuneSuggestion}
                error={autoTuneError}
                applying={applyingAutoTune}
                onClose={() => setAutoTuneCollapsed(true)}
                onApply={() => void applyAutoTune()}
              />
            )}

            <div className="grid gap-4 md:grid-cols-2">
            {/* Preset 选择 */}
            <div id="preset-section" className="rounded-2xl border border-neutral-800 bg-neutral-800/40 p-5">
              <h4 className="mb-3 text-sm font-semibold text-neutral-300">
                品类 Preset（{presets.length} 类可用）
              </h4>
              <div className="space-y-1">
                {presets.map((p) => {
                  const sel = selectedPreset === p.name
                  return (
                    <button
                      key={p.name}
                      onClick={() => setSelectedPreset(p.name)}
                      className={
                        'flex w-full items-center gap-3 rounded-xl border px-3 py-2 text-left transition-all ' +
                        (sel
                          ? 'border-blue-500/50 bg-blue-500/10'
                          : 'border-transparent hover:bg-neutral-800')
                      }
                    >
                      <span
                        className={
                          'h-4 w-4 flex-shrink-0 rounded-full border-2 ' +
                          (sel ? 'border-blue-500 bg-blue-500 shadow-[inset_0_0_0_2.5px_#171717]' : 'border-neutral-600')
                        }
                      />
                      <span className="min-w-0 flex-1">
                        <span className="flex items-center gap-2 text-sm font-semibold">
                          {p.display_name.slice(0, 22)}
                          {p.name === uploadInfo?.recommended_preset && (
                            <span className="rounded-full border border-emerald-500/40 bg-emerald-500/15 px-1.5 py-px text-[9px] font-bold text-emerald-300">
                              AI 推荐
                            </span>
                          )}
                        </span>
                        <span className="block truncate text-[11px] text-neutral-500">
                          {p.semantic_classes} 类语义层 · {p.output_modes.join('/')}
                        </span>
                      </span>
                    </button>
                  )
                })}
              </div>
            </div>

            {/* 参数 */}
            <div className="rounded-2xl border border-neutral-800 bg-neutral-800/40 p-5">
              <h4 className="mb-1 text-sm font-semibold text-neutral-300">输出参数</h4>
              <p className="mb-4 text-[11px] leading-relaxed text-neutral-500">
                下面 5 项决定成品文件的<b className="text-neutral-400">用途、清晰度与大小</b>。
                不确定时保持默认即可；点某项后的 <span className="rounded-full border border-neutral-600 px-1 text-[10px]">?</span> 看白话解释。
              </p>

              {/* 1. 输出模式 */}
              <div className="mb-4">
                <label className="mb-1.5 block text-xs text-neutral-400">
                  <b className="text-neutral-200">输出模式</b>——要出哪种成品文件
                  <HelpTip>
                    <b className="text-neutral-200">RGB 与 CMYK 的区别</b>：
                    RGB（设计线）是屏幕与普通彩色打印用的颜色；CMYK（制版线）是印刷厂
                    分色制版用的颜色，含"黑版"，并已把总墨量控制在印刷可承受范围内。
                    拿不准时选<b className="text-neutral-200">双线</b>，两份都会给你。
                  </HelpTip>
                </label>
                <div className="space-y-1.5">
                  {(['design', 'plate', 'both'] as const).map((m) => {
                    const H = MODE_HELP[m]
                    const on = mode === m
                    return (
                      <button
                        key={m}
                        onClick={() => setMode(m)}
                        className={
                          'w-full rounded-lg border px-3 py-2 text-left transition-all ' +
                          (on
                            ? 'border-blue-500 bg-blue-500/15 shadow shadow-blue-500/20'
                            : 'border-neutral-700 bg-neutral-900 hover:border-neutral-600')
                        }
                      >
                        <span className="flex items-center gap-2">
                          <span
                            className={
                              'text-xs font-semibold ' + (on ? 'text-blue-200' : 'text-neutral-300')
                            }
                          >
                            {H.title}
                          </span>
                          <span
                            className={
                              'rounded-full border px-1.5 py-0.5 text-[10px] ' +
                              (on
                                ? 'border-blue-400/40 bg-blue-500/10 text-blue-200'
                                : 'border-neutral-700 text-neutral-500')
                            }
                          >
                            {H.tag}
                          </span>
                        </span>
                        <span className="mt-0.5 block text-[11px] leading-relaxed text-neutral-500">
                          {H.desc}
                        </span>
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* 2. 放大倍率 */}
              <div className="mb-4">
                <label className="mb-1.5 block text-xs text-neutral-400">
                  <b className="text-neutral-200">放大倍率</b> · {scale.toFixed(1)}×——成品比原图大多少
                  <HelpTip>
                    成品像素尺寸 = 原图尺寸 × 倍率。倍率越高，细节越多、可印尺寸越大，
                    但文件体积与处理时间也随之上升。倍率超过原图能提供的细节上限时，
                    多出来的细节由算法推算（放大越猛，推算成分越多）。
                  </HelpTip>
                </label>
                <input
                  type="range"
                  min="1"
                  max="8"
                  step="0.5"
                  value={scale}
                  onChange={(e) => setScale(parseFloat(e.target.value))}
                  className="w-full accent-blue-500"
                />
                <p className="mt-1 text-[11px] text-neutral-500">
                  {uploadInfo?.dimensions
                    ? <>成品约 <b className="text-neutral-300">{(uploadInfo.dimensions.width * scale).toFixed(0)} × {(uploadInfo.dimensions.height * scale).toFixed(0)}</b> 像素
                      <span className="text-neutral-600">（原图 {uploadInfo.dimensions.width} × {uploadInfo.dimensions.height}）</span></>
                    : '上传图片后这里会显示成品的实际像素尺寸'}
                  <span className="text-neutral-600"> · 倍率越高越慢、文件越大</span>
                </p>
              </div>

              {/* 3. 印刷精度 */}
              <div className="mb-4">
                <label className="mb-1.5 block text-xs text-neutral-400">
                  <b className="text-neutral-200">印刷精度（DPI）</b> · {dpi} · 成品宽{' '}
                  {/* 2026-09-17 修正单位口径：原式算得的是**米**（宽px / DPI × 25.4 / 1000），
                      却标注为 mm（差 1000 倍，如 11392px@150DPI 实为 1.93 米，旧 UI 显示 "2 mm"）。 */}
                  {(() => {
                    if (!uploadInfo?.dimensions || dpi <= 0) return '—'
                    const mm = (uploadInfo.dimensions.width * scale / dpi) * 25.4
                    return mm >= 1000 ? `${(mm / 1000).toFixed(2)} 米` : `${mm.toFixed(0)} 毫米`
                  })()}
                  <HelpTip>
                    DPI（每英寸点数）表示"印出来一英寸里有多少个像素点"，决定打印精细度：
                    <b className="text-neutral-200">150</b> 适合常规印刷与海报；
                    <b className="text-neutral-200">300</b> 适合高清画册、近距离观看的成品。
                    DPI 只影响"印多大"，不改变像素总数。
                  </HelpTip>
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={dpi}
                    onChange={(e) => {
                      const v = parseInt(e.target.value.replace(/[^0-9]/g, ''), 10)
                      setDpi(Number.isNaN(v) ? 0 : Math.min(600, v))
                    }}
                    className="w-28 rounded-lg border border-neutral-700 bg-neutral-900 px-3 py-2 text-center font-mono text-sm outline-none focus:border-blue-500"
                  />
                  {[150, 300].map((v) => (
                    <button
                      key={v}
                      onClick={() => setDpi(v)}
                      className={
                        'rounded-lg border px-3 py-1.5 text-xs transition-all ' +
                        (dpi === v
                          ? 'border-blue-500 bg-blue-500/15 text-blue-300 font-semibold'
                          : 'border-neutral-700 text-neutral-400 hover:text-neutral-200')
                      }
                    >
                      {v === 150 ? '150 常规印刷' : '300 高清画册'}
                    </button>
                  ))}
                </div>
                <p className="mt-1.5 text-[11px] leading-relaxed text-neutral-500">
                  按当前设置，成品每英寸的像素由原图的 <b className="text-neutral-400">{dpi > 0 ? Math.round(dpi / scale) : '—'}</b> 个点构成
                  {dpi > 0 && dpi / scale < 150
                    ? <>：低于 150 → <span className="text-amber-300/80">多出的细节由算法推算</span>（放大倍率调低可提升原生细节）</>
                    : <>：达到常规印刷所需的细节水平</>}
                </p>
              </div>

              <div className="mb-4">
                <label className="mb-1.5 block text-xs text-neutral-400">
                  <b className="text-neutral-200">随机种子</b>——复现编号
                  <HelpTip>
                    引擎在放大与补全时含少量随机性。填同一个编号 + 同样参数，两次出图会
                    <b className="text-neutral-200">完全一致</b>（便于对比与追溯）；换个编号，细节会变。
                    <b className="text-neutral-200">留空</b>则每次都随机。
                  </HelpTip>
                </label>
                <input
                  type="text"
                  value={seed}
                  placeholder="留空 = 每次随机"
                  onChange={(e) => setSeed(e.target.value.replace(/[^0-9]/g, ''))}
                  className="w-full rounded-lg border border-neutral-700 bg-neutral-900 px-3 py-2 font-mono text-sm outline-none placeholder:font-sans placeholder:text-neutral-600 focus:border-blue-500"
                />
                <p className="mt-1 text-[11px] text-neutral-500">
                  想复现某次结果，就填它当时用的编号；随手换编号即可得到不同细节版本。
                </p>
              </div>

              <div>
                <label className="mb-1.5 block text-xs text-neutral-400">
                  <b className="text-neutral-200">算力档位</b>——用哪种计算资源（速度与稳定性的取舍）
                  <HelpTip>
                    本机有三类算力：独立显卡（最快）、核显（显存最大）、CPU（最兼容）。
                    档位只影响<b className="text-neutral-200">处理速度与稳定性</b>，
                    不改变成品的颜色与图层结构。日常用「稳健」即可。
                  </HelpTip>
                </label>
                <div className="space-y-1.5">
                  {(['robust_performance', '5070', 'arc', 'cpu'] as const).map((pf) => {
                    const P = PROFILE_META[pf]
                    const on = profile === pf
                    return (
                      <button
                        key={pf}
                        onClick={() => setProfile(pf)}
                        className={
                          'w-full rounded-lg border px-3 py-1.5 text-left transition-all ' +
                          (on
                            ? 'border-blue-500 bg-blue-500/15 shadow shadow-blue-500/20'
                            : 'border-neutral-700 bg-neutral-900 hover:border-neutral-600')
                        }
                      >
                        <span className="flex items-center gap-2">
                          <span className={'text-xs font-semibold ' + (on ? 'text-blue-200' : 'text-neutral-300')}>
                            {P.label}
                          </span>
                          <span className={
                            'rounded-full border px-1.5 py-0.5 text-[10px] ' +
                            (on ? 'border-blue-400/40 bg-blue-500/10 text-blue-200' : 'border-neutral-700 text-neutral-500')
                          }>
                            {P.hint}
                          </span>
                        </span>
                        <span className="mt-0.5 block text-[11px] leading-relaxed text-neutral-500">{P.desc}</span>
                      </button>
                    )
                  })}
                </div>
              </div>
            </div>
          </div>
          </>
        )}

        {/* 开始处理：吸底操作栏（2026-09-16）——短视口下按钮此前在首屏外
            （视口 568px vs 页面 1735px），需滚动才能提交；改为 fixed 底栏后恒可达 */}
        {ready && !busy && (
          <div className="fixed inset-x-0 bottom-0 z-40 border-t border-neutral-800 bg-neutral-900/90 backdrop-blur">
            <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 py-3">
              <span className="truncate text-xs text-neutral-400">
                {selectedPreset
                  ? <>Preset：<b className="text-neutral-200">{selectedPreset}</b>
                    · {MODE_LABELS[mode]} · {scale.toFixed(1)}× · {dpi} PPI</>
                  : "请先选择 Preset"}
              </span>
              <button
                onClick={() => void startProcess()}
                disabled={!selectedPreset}
                className="shrink-0 rounded-xl bg-gradient-to-br from-blue-500 to-blue-700 px-8 py-2.5 text-sm font-bold text-white shadow-lg shadow-blue-500/30 transition-all hover:-translate-y-0.5 hover:shadow-blue-500/50 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:translate-y-0"
              >
                开始处理 →
              </button>
            </div>
          </div>
        )}

        {/* 处理中：进度 + 日志 */}
        {busy && (
          <div className="rounded-2xl border border-neutral-800 bg-neutral-800/40 p-6">
            <div className="mb-4 flex items-center gap-3">
              <span className="h-5 w-5 animate-spin rounded-full border-[3px] border-blue-500/20 border-t-blue-500" />
              <h3 className="font-semibold">正在处理 · 任务 {taskId?.slice(-10)}</h3>
              <span className="rounded-full border border-blue-500/30 bg-blue-500/10 px-3 py-0.5 text-xs text-blue-300">
                {stage}
              </span>
            </div>
            <div className="mb-2 flex justify-between text-xs text-neutral-400">
              <span>实时进度（按预估时间推算）</span>
              <span>
                <b className="text-neutral-100">{progress}</b>% · 已用 {elapsed.toFixed(0)}s
              </span>
            </div>
            <div className="mb-4 h-2 overflow-hidden rounded-full bg-neutral-800">
              <div
                className="h-full rounded-full bg-gradient-to-r from-blue-600 to-blue-400 shadow shadow-blue-500/50 transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div
              ref={logBoxRef}
              className="h-72 overflow-y-auto rounded-xl border border-neutral-800 bg-[#0d0d0d] p-4 font-mono text-xs leading-relaxed text-neutral-400"
            >
              {logs.map((l, i) => (
                <div key={i} className="animate-pulse">
                  {l.stage && <span className="mr-2 text-blue-400/70">[{l.stage}]</span>}
                  {l.message}
                </div>
              ))}
              {logs.length === 0 && <div className="text-neutral-600">等待引擎输出…</div>}
            </div>
            <p className="mt-3 text-xs text-neutral-500">
              MVP 串行约束：处理期间不能提交新任务 · 引擎日志实时推送 · 完成后自动出下载区
            </p>
          </div>
        )}

        {/* 失败 */}
        {phase === 'failed' && (
          <div className="rounded-2xl border border-red-500/30 bg-red-500/10 p-6 text-center">
            <h3 className="font-semibold text-red-300">处理失败</h3>
            <p className="mt-1 text-sm text-neutral-400">{error}</p>
            <button
              onClick={resetAll}
              className="mt-4 rounded-lg border border-neutral-600 px-5 py-2 text-sm text-neutral-300 hover:border-neutral-400"
            >
              返回重来
            </button>
          </div>
        )}

        {/* 完成：下载区 */}
        {phase === 'done' && (
          <>
            <div className="flex items-center gap-4 rounded-2xl border border-emerald-500/40 bg-emerald-500/10 p-5">
              <span className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-emerald-500 text-white shadow-lg shadow-emerald-500/40">
                <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="3" viewBox="0 0 24 24">
                  <path d="M20 6L9 17l-5-5" />
                </svg>
              </span>
              <div>
                <h3 className="font-semibold text-emerald-300">处理完成 · 双产品线交付就绪</h3>
                <p className="text-xs text-neutral-400">
                  总耗时 {elapsed.toFixed(1)}s · 任务 {taskId}
                </p>
              </div>
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              {outputs.map((f) => (
                <a
                  key={f.type}
                  href={f.download_url}
                  className="group flex items-center gap-4 rounded-2xl border border-neutral-800 bg-neutral-800/40 p-4 transition-all hover:-translate-y-0.5 hover:border-blue-500/50"
                >
                  <span className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl border border-blue-500/30 bg-blue-500/10">
                    <svg className="h-5 w-5 text-blue-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" />
                    </svg>
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm font-bold">{f.filename}</span>
                    <span className="block font-mono text-[11px] text-neutral-500">{fmtSize(f.size)}</span>
                  </span>
                  <span className="rounded-lg bg-blue-500 px-4 py-1.5 text-xs font-bold text-white transition-colors group-hover:bg-blue-600">
                    下载
                  </span>
                </a>
              ))}
            </div>

            {/* 分层报告：semantic_coverage（配置了什么/产出什么/什么被拒/为什么） */}
            {(() => {
              const sc = (manifest as any)?.totals?.semantic_coverage
              if (!sc) return null
              const SRC_LABEL: Record<string, string> = {
                sam: 'SAM 语义', density_refined: '密度精修', density_band: '密度带',
              }
              const SRC_COLOR: Record<string, string> = {
                sam: 'border-blue-500/40 bg-blue-500/10 text-blue-300',
                density_refined: 'border-amber-500/40 bg-amber-500/10 text-amber-300',
                density_band: 'border-cyan-500/40 bg-cyan-500/10 text-cyan-300',
              }
              return (
                <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5">
                  <h4 className="mb-3 text-sm font-semibold text-neutral-200">
                    分层报告 · 语义覆盖披露
                  </h4>
                  <div className="mb-3 text-xs text-neutral-400">
                    配置 {sc.configured?.length ?? 0} 类 · 产出{' '}
                    <b className="text-emerald-400">{sc.produced?.length ?? 0}</b> 层 · 被拒{' '}
                    <b className="text-red-400">{sc.rejected?.length ?? 0}</b> · 未命中{' '}
                    <b className="text-neutral-300">{sc.missing?.length ?? 0}</b>
                  </div>
                  <div className="space-y-1.5">
                    {(sc.produced_details ?? sc.produced ?? []).map((d: any, i: number) => {
                      const name = typeof d === 'string' ? d : d.name
                      const src = typeof d === 'string' ? 'sam' : d.source
                      return (
                        <div key={i} className="flex items-center gap-2 text-xs">
                          <span className="text-emerald-500">✓</span>
                          <span className="flex-1 truncate text-neutral-300">{name}</span>
                          <span className={`rounded border px-2 py-0.5 text-[10px] ${SRC_COLOR[src] ?? ''}`}>
                            {SRC_LABEL[src] ?? src}
                          </span>
                        </div>
                      )
                    })}
                    {(sc.rejected ?? []).map((r: any, i: number) => (
                      <div key={`r${i}`} className="flex items-start gap-2 text-xs">
                        <span className="text-red-500">✕</span>
                        <span className="flex-1 text-neutral-400">
                          <b className="text-neutral-300">{r.name}</b>
                          <span className="block text-[11px] text-neutral-600">{r.reason}</span>
                        </span>
                      </div>
                    ))}
                    {(sc.missing ?? []).map((m: any, i: number) => (
                      <div key={`m${i}`} className="flex items-center gap-2 text-xs">
                        <span className="text-neutral-600">−</span>
                        <span className="flex-1 truncate text-neutral-500">{m.name}</span>
                        <span className="text-[10px] text-neutral-600">{m.reason}</span>
                      </div>
                    ))}
                  </div>
                  <p className="mt-3 text-[11px] text-neutral-600">
                    被拒/未命中的层内容保留在底板上（合成不损失），仅缺独立可编辑图层——
                    完整记录见 manifest.json
                  </p>
                </div>
              )
            })()}

            {/* 交付审计（8 维门禁）：任务完成后异步推送 */}
            {audit && (
              <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5">
                <div className="mb-3 flex items-center gap-2">
                  <h4 className="text-sm font-semibold text-neutral-200">交付审计 · 8 维门禁</h4>
                  <span
                    className={
                      'rounded border px-2 py-0.5 text-[11px] ' +
                      (audit.passed
                        ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                        : audit.passed === false
                          ? 'border-red-500/40 bg-red-500/10 text-red-300'
                          : 'border-neutral-600 text-neutral-400')
                    }
                  >
                    {audit.passed ? '全部通过' : audit.passed === false ? '存在不通过维度' : audit.error ? '审计异常' : '审计中…'}
                  </span>
                  {audit.download_url && (
                    <a href={audit.download_url} className="ml-auto text-[11px] text-blue-400 hover:underline">
                      下载审计报告
                    </a>
                  )}
                </div>
                {audit.dims ? (
                  <div className="space-y-1">
                    {Object.entries(audit.dims).map(([k, v]: [string, any]) => (
                      <div key={k} className="flex items-start gap-2 text-xs">
                        <span>{v.passed ? '✅' : '❌'}</span>
                        <span className="w-32 flex-shrink-0 text-neutral-300">{k}</span>
                        <span className="flex-1 truncate font-mono text-[11px] text-neutral-500">
                          {JSON.stringify(v.metrics ?? {})}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : audit.error ? (
                  <p className="text-xs text-red-400">{audit.error}</p>
                ) : (
                  <p className="text-xs text-neutral-500">审计进行中（读取 16K 产物约需 1-2 分钟）…</p>
                )}
                {(audit.issues ?? []).length > 0 && (
                  <ul className="mt-3 space-y-1">
                    {audit.issues.map((it: string, i: number) => (
                      <li key={i} className="text-[11px] text-red-300">· {it}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            <div className="flex justify-end">
              <button
                onClick={resetAll}
                className="rounded-xl border border-neutral-700 px-6 py-2 text-sm text-neutral-300 transition-colors hover:border-neutral-500"
              >
                处理新图片
              </button>
            </div>
          </>
        )}

        {/* 历史记录（/api/history，此前为孤儿端点：刷新后看不到历史任务） */}
        <section className="mt-10 rounded-2xl border border-neutral-800 bg-neutral-900/60">
          <button
            onClick={() => setHistoryOpen((v) => !v)}
            className="flex w-full items-center justify-between px-5 py-3 text-sm text-neutral-300"
          >
            <span className="font-semibold">
              历史记录{history.length > 0 ? `（最近 ${history.length} 次）` : ''}
            </span>
            <span className="flex items-center gap-3 text-xs text-neutral-500">
              <span
                onClick={(e) => {
                  e.stopPropagation()
                  loadHistory()
                }}
                className="transition-colors hover:text-neutral-200"
              >
                {historyLoading ? '刷新中…' : '刷新'}
              </span>
              <span>{historyOpen ? '收起 ▲' : '展开 ▼'}</span>
            </span>
          </button>
          {historyOpen && (
            <div className="border-t border-neutral-800 px-5 py-3">
              {history.length === 0 ? (
                <p className="text-xs text-neutral-500">暂无历史任务</p>
              ) : (
                <ul className="divide-y divide-neutral-800/60 text-xs">
                  {history.map((h) => (
                    <li
                      key={h.task_id}
                      className="flex flex-wrap items-center gap-x-3 gap-y-1 py-2"
                    >
                      <span className="font-mono text-neutral-400">{h.task_id}</span>
                      <span className="text-neutral-500">{MODE_LABELS[h.mode] ?? h.mode}</span>
                      <span className="text-neutral-500">{h.preset}</span>
                      <span
                        className={
                          h.status === 'completed' ? 'text-emerald-400' : 'text-red-400'
                        }
                      >
                        {h.status}
                      </span>
                      {h.status === 'completed' && (
                        <span className="flex items-center gap-2">
                          {(h.mode === 'both' ? ['plate', 'design'] : [h.mode]).map((ft) => (
                            <a
                              key={ft}
                              href={`/api/download/${h.task_id}/${ft}`}
                              download
                              className="text-blue-400 transition-colors hover:text-blue-300 hover:underline"
                            >
                              PSB·{ft === 'plate' ? '制版' : '设计'}
                            </a>
                          ))}
                          <a
                            href={`/api/download/${h.task_id}/audit`}
                            download
                            className="text-neutral-400 transition-colors hover:text-neutral-200 hover:underline"
                          >
                            审计
                          </a>
                          <a
                            href={`/api/download/${h.task_id}/manifest`}
                            download
                            className="text-neutral-400 transition-colors hover:text-neutral-200 hover:underline"
                          >
                            清单
                          </a>
                        </span>
                      )}
                      {typeof h.elapsed_time === 'number' && (
                        <span className="text-neutral-600">{h.elapsed_time.toFixed(0)}s</span>
                      )}
                      <span className="ml-auto text-neutral-600">
                        {h.created_at?.replace('T', ' ').slice(0, 19)}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
              <p className="mt-2 text-[11px] text-neutral-600">
                产物位于 webui/data/outputs/&lt;task_id&gt;/（PSB + manifest + masks + 审计报告）
              </p>
            </div>
          )}
        </section>
      </main>

      {/* Stage 5.3 人审弹窗（有待审类目时显示） */}
      {showFeedbackModal && pendingRequest && (
        <CategoryFeedbackModal
          request={pendingRequest}
          submitting={submittingFeedback}
          onClose={() => setShowFeedbackModal(false)}
          onSubmit={submitCategoryFeedback}
        />
      )}
    </div>
  )
}
