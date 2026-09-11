import { useState } from 'react'

function App() {
  const [isDark] = useState(true)

  return (
    <div className={isDark ? 'dark' : ''}>
      <div className="min-h-screen bg-neutral-900 text-neutral-100">
        {/* Header */}
        <header className="border-b border-neutral-800 bg-neutral-900/95 backdrop-blur supports-[backdrop-filter]:bg-neutral-900/60">
          <div className="container flex h-16 items-center px-4">
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-lg bg-primary-500"></div>
              <h1 className="text-xl font-semibold">
                Universal Layer Studio PRO
              </h1>
            </div>
            <div className="ml-auto flex items-center gap-4">
              <span className="text-sm text-neutral-400">v1.0.0</span>
            </div>
          </div>
        </header>

        {/* Main Layout */}
        <div className="flex">
          {/* Sidebar */}
          <aside className="w-60 border-r border-neutral-800 bg-neutral-900">
            <nav className="space-y-1 p-4">
              <a
                href="#"
                className="flex items-center gap-3 rounded-lg bg-primary-500/10 px-3 py-2 text-sm font-medium text-primary-500"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                新任务
              </a>
              <a
                href="#"
                className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-neutral-400 hover:bg-neutral-800 hover:text-neutral-100"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                历史记录
              </a>
              <a
                href="#"
                className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-neutral-400 hover:bg-neutral-800 hover:text-neutral-100"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                设置
              </a>
            </nav>
          </aside>

          {/* Main Content */}
          <main className="flex-1 p-8">
            <div className="mx-auto max-w-5xl space-y-8">
              {/* Upload Zone */}
              <div className="rounded-lg border-2 border-dashed border-neutral-700 bg-neutral-800/50 p-12 text-center transition-colors hover:border-primary-500 hover:bg-primary-500/5">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-neutral-700">
                  <svg className="h-8 w-8 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                </div>
                <h3 className="mt-4 text-lg font-medium text-neutral-100">
                  拖拽图片到此处，或点击选择文件
                </h3>
                <p className="mt-2 text-sm text-neutral-400">
                  支持 JPG / PNG / PSD，最大 100MB
                </p>
                <button className="mt-6 rounded-lg bg-primary-500 px-6 py-2 text-sm font-medium text-white transition-colors hover:bg-primary-600">
                  选择文件
                </button>
              </div>

              {/* Info Card */}
              <div className="rounded-lg border border-neutral-800 bg-neutral-800/30 p-6">
                <h2 className="text-lg font-semibold">开始使用</h2>
                <div className="mt-4 space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary-500 text-xs font-bold text-white">
                      1
                    </div>
                    <div>
                      <p className="font-medium">上传图片</p>
                      <p className="text-sm text-neutral-400">
                        支持屏风、壁布、水墨、烫金、油画五类图像
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-neutral-700 text-xs font-bold text-neutral-400">
                      2
                    </div>
                    <div>
                      <p className="font-medium text-neutral-400">选择 Preset 与参数</p>
                      <p className="text-sm text-neutral-500">
                        系统会自动推荐最适合的配置
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-neutral-700 text-xs font-bold text-neutral-400">
                      3
                    </div>
                    <div>
                      <p className="font-medium text-neutral-400">开始处理</p>
                      <p className="text-sm text-neutral-500">
                        实时查看进度，完成后下载 PSB 分层文件
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Status Banner */}
              <div className="rounded-lg border border-primary-500/20 bg-primary-500/10 p-4">
                <div className="flex items-center gap-3">
                  <svg className="h-5 w-5 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="text-sm text-primary-400">
                    <span className="font-medium">WebUI 开发中</span> — 当前为静态界面预览，完整功能正在实现中（预计 Week 1-2 完成）
                  </p>
                </div>
              </div>
            </div>
          </main>
        </div>
      </div>
    </div>
  )
}

export default App
