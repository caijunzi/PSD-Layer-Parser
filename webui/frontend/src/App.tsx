import { useCallback, useEffect, useRef, useState } from 'react'

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
  both: '双线 both',
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
  }, [uploadInfo, selectedPreset, mode, scale, seed, profile])

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

      <main className="mx-auto max-w-6xl space-y-6 px-6 py-8">
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
              <div className="mt-3 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-300">
                AI 推荐：<b>{uploadInfo.recommended_preset}</b>（置信度 {uploadInfo.confidence.toFixed(2)}，已自动选中）
              </div>
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
          <div className="grid gap-4 md:grid-cols-2">
            {/* Preset 选择 */}
            <div className="rounded-2xl border border-neutral-800 bg-neutral-800/40 p-5">
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
              <h4 className="mb-4 text-sm font-semibold text-neutral-300">输出参数</h4>

              <div className="mb-4">
                <label className="mb-1.5 block text-xs text-neutral-400">
                  <b className="text-neutral-200">输出模式</b>
                </label>
                <div className="flex gap-1 rounded-lg border border-neutral-700 bg-neutral-900 p-1">
                  {(['design', 'plate', 'both'] as const).map((m) => (
                    <button
                      key={m}
                      onClick={() => setMode(m)}
                      className={
                        'flex-1 rounded-md px-2 py-1.5 text-xs transition-all ' +
                        (mode === m
                          ? 'bg-blue-500 font-semibold text-white shadow shadow-blue-500/40'
                          : 'text-neutral-400 hover:text-neutral-200')
                      }
                    >
                      {MODE_LABELS[m]}
                    </button>
                  ))}
                </div>
              </div>

              <div className="mb-4">
                <label className="mb-1.5 block text-xs text-neutral-400">
                  <b className="text-neutral-200">放大倍率</b> · {scale.toFixed(1)}×
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
              </div>

              <div className="mb-4">
                <label className="mb-1.5 block text-xs text-neutral-400">
                  <b className="text-neutral-200">输出分辨率 (DPI)</b> · {dpi} PPI · 物理宽{' '}
                  {(dpi > 0 ? (16000 * (scale / 4) / dpi * 25.4) / 1000 : 0).toFixed(0)} mm
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
                      {v === 150 ? '150 标准' : '300 精细'}
                    </button>
                  ))}
                </div>
                <p className="mt-1.5 text-[11px] text-neutral-600">
                  300 DPI 出产时源图有效分辨率 {dpi > 0 && uploadInfo?.dimensions ? Math.round(uploadInfo.dimensions.width / (16000 * (scale / 4) / dpi)) : '—'} PPI
                  （4× 放大极限，如实披露）
                </p>
              </div>

              <div className="mb-4">
                <label className="mb-1.5 block text-xs text-neutral-400">
                  <b className="text-neutral-200">随机种子</b> · 同 seed 逐像素可复现（RK-16）
                </label>
                <input
                  type="text"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value.replace(/[^0-9]/g, ''))}
                  className="w-full rounded-lg border border-neutral-700 bg-neutral-900 px-3 py-2 font-mono text-sm outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="mb-1.5 block text-xs text-neutral-400">
                  <b className="text-neutral-200">性能档位</b>
                </label>
                <div className="flex gap-1 rounded-lg border border-neutral-700 bg-neutral-900 p-1">
                  {['robust_performance', '5070', 'arc', 'cpu'].map((pf) => (
                    <button
                      key={pf}
                      onClick={() => setProfile(pf)}
                      className={
                        'flex-1 rounded-md px-1 py-1.5 text-[11px] transition-all ' +
                        (profile === pf
                          ? 'bg-blue-500 font-semibold text-white'
                          : 'text-neutral-400 hover:text-neutral-200')
                      }
                    >
                      {pf === 'robust_performance' ? '稳健' : pf}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 开始处理按钮 */}
        {ready && (
          <div className="flex items-center justify-end gap-3">
            {!selectedPreset && <span className="text-xs text-neutral-500">请先选择 Preset</span>}
            <button
              onClick={() => void startProcess()}
              disabled={!selectedPreset}
              className="rounded-xl bg-gradient-to-br from-blue-500 to-blue-700 px-8 py-2.5 text-sm font-bold text-white shadow-lg shadow-blue-500/30 transition-all hover:-translate-y-0.5 hover:shadow-blue-500/50 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:translate-y-0"
            >
              开始处理 →
            </button>
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
      </main>
    </div>
  )
}
