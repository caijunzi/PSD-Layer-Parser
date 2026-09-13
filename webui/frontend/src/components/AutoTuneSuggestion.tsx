/* Stage 2 图级 Auto-Tune 建议卡片（Stage 5.3 从 App.tsx 抽出重构）
 *
 * 纯展示组件：状态全部由父级持有，本组件只负责渲染与回调。
 */
export type AutoTuneSuggestionData = {
  image_size: [number, number]
  global_percentiles: Record<string, number>
  regions: Array<{ region: [number, number, number, number]; score: number; blocks: number }>
  density_bands: Array<{
    label: string
    region?: [number, number, number, number]
    density_min: number
    density_max: number
    coverage: number
  }>
}

type Props = {
  loading: boolean
  suggestion: AutoTuneSuggestionData | null
  error: string | null
  applying: boolean
  onClose: () => void
  onApply: () => void
}

export default function AutoTuneCard({
  loading,
  suggestion,
  error,
  applying,
  onClose,
  onApply,
}: Props) {
  return (
    <div className="rounded-2xl border border-neutral-800 bg-neutral-800/40 p-5">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <svg
            className="h-5 w-5 text-blue-400"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            viewBox="0 0 24 24"
          >
            <path d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          <h4 className="text-sm font-semibold text-neutral-200">Auto-Tune 建议</h4>
          {loading && (
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-blue-500/20 border-t-blue-500" />
          )}
        </div>
        <button
          onClick={onClose}
          className="rounded-lg px-2 py-1 text-xs text-neutral-500 hover:text-neutral-300"
        >
          ✕ 关闭
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">
          ⚠ {error}
        </div>
      )}

      {suggestion && !loading && (
        <div className="space-y-4">
          {/* 墨迹热点区域 */}
          {suggestion.regions && suggestion.regions.length > 0 && (
            <div>
              <h5 className="mb-2 text-xs font-semibold text-neutral-300">
                墨迹热点区域（{suggestion.regions.length} 个）
              </h5>
              <div className="space-y-1.5">
                {suggestion.regions.slice(0, 3).map((r, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-2 rounded-lg border border-neutral-700 bg-neutral-900 px-3 py-2 text-xs"
                  >
                    <span className="text-neutral-500">#{i + 1}</span>
                    <span className="flex-1 font-mono text-neutral-400">
                      region=[{r.region.map((v) => v.toFixed(2)).join(', ')}]
                    </span>
                    <span className="text-neutral-500">
                      score={r.score.toFixed(2)} · {r.blocks} blocks
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 密度带推荐 */}
          {suggestion.density_bands && suggestion.density_bands.length > 0 && (
            <div>
              <h5 className="mb-2 text-xs font-semibold text-neutral-300">
                密度带推荐（{suggestion.density_bands.length} 条）
              </h5>
              <div className="space-y-1.5">
                {suggestion.density_bands.map((b, i) => (
                  <div
                    key={i}
                    className="rounded-lg border border-neutral-700 bg-neutral-900 px-3 py-2"
                  >
                    <div className="mb-1 flex items-center justify-between text-xs">
                      <span className="font-semibold text-neutral-300">{b.label}</span>
                      <span className="font-mono text-neutral-500">
                        [{b.density_min.toFixed(4)}, {b.density_max.toFixed(4)}]
                      </span>
                    </div>
                    <div className="flex items-center gap-2 text-[11px] text-neutral-500">
                      <span>覆盖率 {(b.coverage * 100).toFixed(1)}%</span>
                      {b.region && (
                        <span className="font-mono">
                          region=[{b.region.map((v) => v.toFixed(2)).join(', ')}]
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 采纳按钮 */}
          <div className="flex items-center justify-end gap-3">
            <button
              onClick={onApply}
              disabled={applying}
              className="rounded-lg bg-blue-500 px-4 py-2 text-sm font-semibold text-white shadow shadow-blue-500/40 transition-all hover:bg-blue-600 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {applying ? '采纳中…' : '✓ 采纳建议并写入 preset'}
            </button>
          </div>
        </div>
      )}

      {loading && !suggestion && (
        <div className="py-6 text-center text-sm text-neutral-500">正在分析图片密度特征…</div>
      )}
    </div>
  )
}
