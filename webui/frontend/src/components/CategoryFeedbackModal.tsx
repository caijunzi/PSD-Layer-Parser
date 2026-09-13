/* Stage 5.3 主动学习人审弹窗
 *
 * 展示引擎识别出的「不确定类目」（confidence < 0.7），让用户逐个裁决：
 *   accept / rename / delete / merge
 *
 * 纯受控组件：请求数据与提交动作均由父级（App.tsx）提供。
 */
import { useState } from 'react'

export type UncertainCategory = {
  category_id: string
  confidence: number
  bbox?: [number, number, number, number] | null
  prompt?: string | null
}

export type FeedbackRequest = {
  feedback_request_id: string
  task_id: string
  image_path?: string | null
  uncertain_categories: UncertainCategory[]
  created_at?: number
}

export type FeedbackAction = 'accept' | 'rename' | 'delete' | 'merge'

type Props = {
  request: FeedbackRequest
  submitting: boolean
  onClose: () => void
  onSubmit: (
    categoryId: string,
    action: FeedbackAction,
    userData?: Record<string, unknown>,
  ) => Promise<void>
}

export default function CategoryFeedbackModal({
  request,
  submitting,
  onClose,
  onSubmit,
}: Props) {
  const [busyId, setBusyId] = useState<string | null>(null)

  // rename 面板：记录当前正在重命名的类目 + 输入值
  const [renameId, setRenameId] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')

  // merge 面板：记录当前正在合并的类目 + 目标类目 id 输入值
  // （后端 merge 目前为 TODO，故用文本输入目标 id，避免为未实现功能扩展 API）
  const [mergeId, setMergeId] = useState<string | null>(null)
  const [mergeValue, setMergeValue] = useState('')

  const cats = request.uncertain_categories ?? []

  async function run(categoryId: string, action: FeedbackAction, userData?: Record<string, unknown>) {
    setBusyId(categoryId)
    try {
      await onSubmit(categoryId, action, userData)
      // 提交成功后收起对应面板
      setRenameId(null)
      setMergeId(null)
    } finally {
      setBusyId(null)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded-2xl border border-neutral-700 bg-neutral-900 shadow-2xl">
        {/* 头部 */}
        <div className="sticky top-0 flex items-start justify-between gap-4 border-b border-neutral-800 bg-neutral-900 px-5 py-4">
          <div>
            <div className="flex items-center gap-2">
              <svg
                className="h-5 w-5 text-amber-400"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                viewBox="0 0 24 24"
              >
                <path d="M12 9v4m0 4h.01M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
              </svg>
              <h3 className="text-base font-semibold text-neutral-100">
                待确认类目（{cats.length} 个）
              </h3>
            </div>
            <p className="mt-1 text-xs text-neutral-500">
              以下类目置信度低于 0.7，请人工确认。任务：
              <span className="font-mono">{request.task_id}</span>
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg px-2 py-1 text-sm text-neutral-500 hover:text-neutral-300"
          >
            ✕
          </button>
        </div>

        {/* 类目列表 */}
        <div className="space-y-3 px-5 py-4">
          {cats.length === 0 && (
            <div className="py-6 text-center text-sm text-neutral-500">没有待确认的类目</div>
          )}

          {cats.map((c) => {
            const busy = busyId === c.category_id
            const pct = Math.round((c.confidence ?? 0) * 100)
            return (
              <div
                key={c.category_id}
                className="rounded-xl border border-neutral-700 bg-neutral-800/50 px-4 py-3"
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-semibold text-neutral-200">
                      {c.category_id}
                    </div>
                    {c.prompt && (
                      <div className="truncate text-xs text-neutral-500">prompt: {c.prompt}</div>
                    )}
                    {c.bbox && (
                      <div className="mt-0.5 font-mono text-[11px] text-neutral-600">
                        bbox=[{c.bbox.map((v) => v.toFixed(2)).join(', ')}]
                      </div>
                    )}
                  </div>
                  <div className="flex-shrink-0 text-right">
                    <div className="text-xs text-neutral-400">置信度</div>
                    <div className="text-sm font-semibold text-amber-400">
                      {(c.confidence ?? 0).toFixed(2)}
                    </div>
                  </div>
                </div>

                {/* 置信度进度条 */}
                <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-neutral-700">
                  <div
                    className="h-full rounded-full bg-amber-500"
                    style={{ width: `${Math.max(2, Math.min(100, pct))}%` }}
                  />
                </div>

                {/* 操作按钮 */}
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <button
                    disabled={busy || submitting}
                    onClick={() => void run(c.category_id, 'accept')}
                    className="rounded-lg bg-emerald-500/90 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-emerald-600 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    ✓ 采纳
                  </button>
                  <button
                    disabled={busy || submitting}
                    onClick={() => {
                      setRenameId(renameId === c.category_id ? null : c.category_id)
                      setRenameValue('')
                      setMergeId(null)
                    }}
                    className="rounded-lg border border-neutral-600 px-3 py-1.5 text-xs text-neutral-300 transition hover:bg-neutral-700 disabled:opacity-50"
                  >
                    ✎ 重命名
                  </button>
                  <button
                    disabled={busy || submitting}
                    onClick={() => {
                      setMergeId(mergeId === c.category_id ? null : c.category_id)
                      setMergeValue('')
                      setRenameId(null)
                    }}
                    className="rounded-lg border border-neutral-600 px-3 py-1.5 text-xs text-neutral-300 transition hover:bg-neutral-700 disabled:opacity-50"
                  >
                    ⇄ 合并到…
                  </button>
                  <button
                    disabled={busy || submitting}
                    onClick={() => void run(c.category_id, 'delete')}
                    className="rounded-lg border border-red-500/40 px-3 py-1.5 text-xs text-red-300 transition hover:bg-red-500/10 disabled:opacity-50"
                  >
                    🗑 删除
                  </button>
                </div>

                {/* rename 面板 */}
                {renameId === c.category_id && (
                  <div className="mt-3 flex items-center gap-2">
                    <input
                      value={renameValue}
                      onChange={(e) => setRenameValue(e.target.value)}
                      placeholder="输入新类目名"
                      className="flex-1 rounded-lg border border-neutral-600 bg-neutral-900 px-3 py-1.5 text-xs text-neutral-200 outline-none focus:border-blue-500"
                    />
                    <button
                      disabled={!renameValue.trim() || busy}
                      onClick={() => void run(c.category_id, 'rename', { new_name: renameValue.trim() })}
                      className="rounded-lg bg-blue-500 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-blue-600 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      提交
                    </button>
                  </div>
                )}

                {/* merge 面板 */}
                {mergeId === c.category_id && (
                  <div className="mt-3 flex items-center gap-2">
                    <input
                      value={mergeValue}
                      onChange={(e) => setMergeValue(e.target.value)}
                      placeholder="目标类目 id（如 mountains_cliffs）"
                      className="flex-1 rounded-lg border border-neutral-600 bg-neutral-900 px-3 py-1.5 text-xs text-neutral-200 outline-none focus:border-blue-500"
                    />
                    <button
                      disabled={!mergeValue.trim() || busy}
                      onClick={() =>
                        void run(c.category_id, 'merge', {
                          target_category_id: mergeValue.trim(),
                        })
                      }
                      className="rounded-lg bg-blue-500 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-blue-600 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      提交
                    </button>
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* 底部说明 */}
        <div className="border-t border-neutral-800 px-5 py-3 text-[11px] text-neutral-500">
          采纳将提升该类目提示词权重（×1.1）。重命名 / 删除 / 合并后端暂为 TODO，提交后仅记录反馈。
        </div>
      </div>
    </div>
  )
}
