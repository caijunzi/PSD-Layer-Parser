"""任务管理器：创建、执行、查询处理任务。

策略（MVP）：
- 内存 dict 存储 task（单机单用户够用）
- 串行执行：同一时刻只允许一个引擎进程，避免 GPU 抢占
- 引擎以子进程异步执行，stdout 实时读取并推送 WebSocket
- 进度估算：按已用时间 / 预估总时间（封顶 95%），完成时 100%
"""
import asyncio
import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from core.connection_manager import manager as ws_manager
from core.file_handler import (
    UPLOADS_DIR, OUTPUTS_DIR, ENGINE_ENTRY, append_history,
)


def _resolve_python_bin() -> str:
    """解析引擎解释器：环境变量 ULS_PYTHON_BIN > 当前解释器 > 历史硬编码兜底。

    修复（2026-09-15）：此前硬编码本机绝对路径，换机器/换解释器即失效。
    优先级保证：显式配置可控、默认用运行本进程的解释器（最可能装齐依赖）、
    最终兜底历史路径以免行为突变。
    """
    env_bin = os.environ.get("ULS_PYTHON_BIN")
    if env_bin and os.path.isfile(env_bin):
        return env_bin
    if sys.executable and os.path.isfile(sys.executable):
        return sys.executable
    return r"C:/Users/CK/AppData/Local/Programs/Python/Python312/python.exe"


# 引擎 Python 解释器（可经环境变量 ULS_PYTHON_BIN 覆盖）
PYTHON_BIN = _resolve_python_bin()

# preset → 预估耗时（秒），用于进度条与 UI 提示
PRESET_ESTIMATE = {
    "japanese_screen_gold": 250,
    "textile_damask": 80,
    "chinese_ink_landscape_ai": 230,
    "western_oil_painting": 220,
    "traditional_chinese_ink": 230,
}

# 阶段关键词 → 阶段名（用于 WebSocket 推送，前端展示）
# 注意顺序敏感：更具体的阶段在前。"加载模型"不含 SAM2——
# 否则 "[SAM2] 级联分割执行中" 会被误标为加载模型（测试 test_match_stage 锁定）
STAGE_KEYWORDS = [
    (re.compile(r"初始化|引擎.*启动|seed", re.I), "初始化引擎"),
    (re.compile(r"CUDA|探活|加载完成|GroundingDINO", re.I), "加载模型"),
    (re.compile(r"分割|DINORadar|语义类", re.I), "神经分割"),
    (re.compile(r"超分|RealESRGAN|渐进", re.I), "超分重建"),
    (re.compile(r"补全|LaMa|遮挡", re.I), "遮挡补全"),
    (re.compile(r"ICC|分色|黑版|TAC|陷印|trapping", re.I), "制版分色"),
    (re.compile(r"写盘|PSDCompiler|写盘完成|\.psb", re.I), "写盘输出"),
    (re.compile(r"manifest|掩码|完成|RK-16", re.I), "收尾校验"),
]


def _match_stage(line: str) -> Optional[str]:
    for pat, name in STAGE_KEYWORDS:
        if pat.search(line):
            return name
    return None


class TaskManager:
    """单例任务管理器。"""

    def __init__(self):
        self.tasks: Dict[str, dict] = {}
        self._current_process: Optional[asyncio.subprocess.Process] = None
        self._lock = asyncio.Lock()

    @property
    def is_busy(self) -> bool:
        return self._current_process is not None

    def create_task(self, file_id: str, cfg: dict) -> tuple[str, int, str]:
        """创建任务（不立即执行）。返回 (task_id, estimated, ws_url)。"""
        preset = cfg.get("preset", "japanese_screen_gold")
        estimated = PRESET_ESTIMATE.get(preset, 240)
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.tasks[task_id] = {
            "task_id": task_id,
            "file_id": file_id,
            "config": cfg,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "progress": 0,
            "stage": None,
            "logs": [],
            "output_files": [],
            "elapsed_time": 0,
            "error": None,
        }
        ws_url = f"ws://localhost:8099/ws/progress/{task_id}"
        return task_id, estimated, ws_url

    def get_task(self, task_id: str) -> Optional[dict]:
        return self.tasks.get(task_id)

    async def execute(self, task_id: str) -> None:
        """异步执行引擎子进程，实时推送进度。

        注意：调用方负责并发互斥（同一时刻只跑一个）。
        """
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return

            task["status"] = "processing"
            task["started_at"] = datetime.now().isoformat()
            start = asyncio.get_event_loop().time()

            # 定位输入文件
            file_id = task["file_id"]
            input_path = self._resolve_input(file_id)
            if not input_path:
                task["status"] = "failed"
                task["error"] = f"找不到上传文件: {file_id}"
                await ws_manager.broadcast(task_id, {
                    "type": "error", "task_id": task_id,
                    "error": {"code": "FILE_NOT_FOUND",
                              "message": task["error"]},
                })
                return

            # 输出目录 = outputs/{task_id}/，引擎 --output 指向其下 result.psb
            out_dir = OUTPUTS_DIR / task_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_psb = out_dir / "result.psb"

            cfg = task["config"]
            cmd = [
                PYTHON_BIN, "-u", str(ENGINE_ENTRY),
                "--input", str(input_path),
                "--output", str(out_psb),
                "--preset", cfg.get("preset", "japanese_screen_gold"),
                "--mode", cfg.get("mode", "both"),
                "--dpi", str(cfg.get("dpi", 150.0)),
                "--profile", cfg.get("profile", "robust_performance"),
            ]
            if cfg.get("scale") is not None:
                cmd += ["--scale", str(cfg["scale"])]
            if cfg.get("seed") is not None:
                cmd += ["--seed", str(cfg["seed"])]

            estimated = PRESET_ESTIMATE.get(cfg.get("preset"), 240)

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=str(ENGINE_ENTRY.parent),
                )
                self._current_process = proc
            except Exception as e:
                task["status"] = "failed"
                task["error"] = f"引擎启动失败: {type(e).__name__}: {e}"
                await ws_manager.broadcast(task_id, {
                    "type": "error", "task_id": task_id,
                    "error": {"code": "ENGINE_START_FAILED",
                              "message": task["error"]},
                })
                return

            # 实时读 stdout
            assert proc.stdout is not None

            # 心跳任务：引擎长计算阶段（如 CMYK 分色/1.5GB 写盘）可能数分钟
            # 无 stdout 行，若无心跳前端会误判卡死（本次全流程验证实测 5 分钟静默）
            async def heartbeat():
                while True:
                    await asyncio.sleep(10)
                    elapsed = asyncio.get_event_loop().time() - start
                    prog = min(95, int(elapsed / max(1, estimated) * 100))
                    await ws_manager.broadcast(task_id, {
                        "type": "progress", "task_id": task_id,
                        "stage": task.get("stage") or "计算中",
                        "progress": max(task.get("progress", 0), prog),
                        "message": f"…引擎计算中（长阶段无逐行日志）已 {elapsed:.0f}s",
                        "elapsed": round(elapsed, 1),
                        "heartbeat": True,
                    })

            hb_task = asyncio.create_task(heartbeat())
            try:
                async for raw in proc.stdout:
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    if not line:
                        continue
                    ts = datetime.now().isoformat()
                    task["logs"].append({"timestamp": ts, "message": line})

                    stage = _match_stage(line)
                    elapsed = asyncio.get_event_loop().time() - start
                    # 进度估算：时间比例，封顶 95%
                    progress = min(95, int(elapsed / max(1, estimated) * 100))
                    task["progress"] = progress
                    if stage:
                        task["stage"] = stage

                    await ws_manager.broadcast(task_id, {
                        "type": "progress", "task_id": task_id,
                        "stage": stage or task.get("stage") or "处理中",
                        "progress": progress,
                        "message": line,
                        "elapsed": round(elapsed, 1),
                        "timestamp": ts,
                    })

                await proc.wait()
            finally:
                hb_task.cancel()

            self._current_process = None

            elapsed_total = round(asyncio.get_event_loop().time() - start, 1)
            task["elapsed_time"] = elapsed_total

            if proc.returncode == 0:
                task["status"] = "completed"
                task["completed_at"] = datetime.now().isoformat()
                task["progress"] = 100
                files = self._collect_outputs(
                    out_dir, task_id, cfg.get("mode", "both")
                )
                task["output_files"] = files
                manifest = self._read_manifest(out_dir)
                await ws_manager.broadcast(task_id, {
                    "type": "completed", "task_id": task_id,
                    "elapsed_time": elapsed_total,
                    "output_files": files,
                    "manifest": manifest,
                })
                # 写历史
                append_history({
                    "task_id": task_id,
                    "filename": task.get("filename", ""),
                    "preset": cfg.get("preset"),
                    "mode": cfg.get("mode"),
                    "status": "completed",
                    "created_at": task["created_at"],
                    "completed_at": task["completed_at"],
                    "elapsed_time": elapsed_total,
                })
                # 交付前 8 维审计门（2026-09-12 接入，异步不阻塞完成消息）：
                # 层属性/分辨率/实例重复/内容承载/合成等价性/底板纯净度/plate 合规/
                # manifest 一致性 → 结果落盘 result.audit.json 并 WS 推送。
                # 审计不通过**不改变任务状态**（产物仍可下载），但显式告警供人工复核。
                asyncio.create_task(self._run_delivery_audit(task_id, out_dir, cfg))
            else:
                task["status"] = "failed"
                task["error"] = f"引擎退出码 {proc.returncode}"
                await ws_manager.broadcast(task_id, {
                    "type": "error", "task_id": task_id,
                    "error": {"code": "ENGINE_ERROR",
                              "message": task["error"]},
                })

    async def _run_delivery_audit(self, task_id: str, out_dir: Path, cfg: dict) -> None:
        """交付前 8 维审计门（异步）：落盘 result.audit.json + WS 推送摘要。

        审计门见 tools/audit_psb.py；阈值已按 V2 参照产物校准。
        """
        try:
            import sys
            _root = Path(__file__).resolve().parents[3]      # → 项目根
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            from tools.audit_psb import audit as run_audit

            psb = next((p for p in (out_dir / "result.plate.psb", out_dir / "result.psb",
                                    out_dir / "result.design.psb") if p.is_file()), None)
            if psb is None:
                return
            manifest = next((p for p in (out_dir / "result.plate.manifest.json",
                                         out_dir / "result.manifest.json",
                                         out_dir / "result.design.manifest.json") if p.is_file()), None)
            src = self._resolve_input(cfg.get("file_id", ""))
            res = await asyncio.to_thread(
                run_audit, str(psb),
                str(manifest) if manifest else None,
                str(src) if src else None,
                False,
            )
            (out_dir / "result.audit.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

            # 审计完成后：回填审计结果到 episode 并重建 CBR 索引。
            # 修复（2026-09-15）：引擎归档发生在审计之前（审计此时尚不存在），
            # 导致 episode 的 audit_passed 恒为 False、CBR 检索永远筛空。
            # 此处从任务日志里取 episode_id（引擎归档时打印），回填后重建索引。
            self._backfill_episode_audit(task_id, out_dir)

            if isinstance(res, dict):
                res = dict(res)
                res["has_report"] = True
            await ws_manager.broadcast(task_id, {
                "type": "audit", "task_id": task_id,
                "passed": res.get("passed"),
                "dims": {k: {"passed": v.get("passed"), "metrics": v.get("metrics")}
                         for k, v in res.get("dims", {}).items()},
                "issues": res.get("issues", []),
                "download_url": f"/api/download/{task_id}/audit",
            })
        except Exception as e:  # 审计异常不得影响任务本身
            try:
                await ws_manager.broadcast(task_id, {
                    "type": "audit", "task_id": task_id, "passed": None,
                    "error": f"{type(e).__name__}: {str(e)[:160]}",
                })
            except Exception:
                pass

    def _backfill_episode_audit(self, task_id: str, out_dir: Path) -> None:
        """把本次任务的审计 8 维回填到对应 episode，并重建 CBR 指纹索引。

        全程 try 包裹：CBR 属旁路能力，失败不得影响任务与产物。
        """
        try:
            task = self.tasks.get(task_id) or {}
            ep_id = None
            for lg in reversed(task.get("logs", [])):
                m = re.search(r"episode 已归档:\s*(\S+)", lg.get("message", ""))
                if m:
                    ep_id = m.group(1)
                    break
            if not ep_id:
                return

            import sys as _sys
            _root = Path(__file__).resolve().parents[3]
            if str(_root) not in _sys.path:
                _sys.path.insert(0, str(_root))

            from engine.adaptive.episode_archiver import (
                update_episode_audit,
                sync_index_from_log,
            )
            from engine.adaptive.regression_tester import extract_audit_8d

            audit_json = out_dir / "result.audit.json"
            if not audit_json.is_file():
                return
            audit_8d = extract_audit_8d(str(audit_json))
            if update_episode_audit(ep_id, audit_8d):
                n = sync_index_from_log()
                print(f"[TaskManager] CBR 索引已按真实审计重建：{n} 条（episode={ep_id}）")
        except Exception as e:  # 旁路失败静默
            print(f"[TaskManager] 审计回填 episode 失败（不影响任务）: {e}")

    def _resolve_input(self, file_id: str) -> Optional[Path]:
        """根据 file_id 找上传文件（扩展名未知，glob）。"""
        for p in UPLOADS_DIR.iterdir():
            if p.stem == file_id:
                return p
        return None

    def _collect_outputs(self, out_dir: Path, task_id: str, mode: str = "both") -> list[dict]:
        """收集产物文件，构造下载 URL 列表。

        引擎命名规则（run_universal_engine.py 实测）：
        - both 模式：result.plate.psb + result.design.psb（加后缀防覆盖）
        - 单模式：产物就是 --output 本身，即 result.psb（不加后缀！）
        - manifest：both 模式每线一份 result.{plate,design}.manifest.json；
          单模式为 result.{mode}.manifest.json 或旧版 result.manifest.json
        - masks：both 模式 result.{plate,design}.masks/；单模式 result.masks/
        """
        files = []
        if mode == "both":
            candidates = [("result.plate.psb", "plate"), ("result.design.psb", "design")]
            manifests = ["result.plate.manifest.json", "result.design.manifest.json"]
            masks_dir = out_dir / "result.plate.masks"
            masks_tag = "plate"
        else:
            # 单模式：result.psb 即该模式产物
            candidates = [("result.psb", mode)]
            manifests = [f"result.{mode}.manifest.json", "result.manifest.json"]
            masks_dir = out_dir / "result.masks"
            masks_tag = mode
        for fname, ftype in candidates:
            p = out_dir / fname
            if p.exists():
                files.append({
                    "type": ftype,
                    "filename": fname,
                    "size": p.stat().st_size,
                    "download_url": f"/api/download/{task_id}/{ftype}",
                })
        # manifest（按候选顺序取第一份存在的；both 模式优先 plate 版=印前审计凭据）
        for mf_name in manifests:
            mf = out_dir / mf_name
            if mf.exists():
                files.append({
                    "type": "manifest",
                    "filename": mf_name,
                    "size": mf.stat().st_size,
                    "download_url": f"/api/download/{task_id}/manifest",
                })
                break
        # 掩码目录打包为 zip（前端按需下载）
        if masks_dir.exists() and any(masks_dir.iterdir()):
            files.append({
                "type": "masks",
                "filename": f"masks_{masks_tag}.zip",
                "size": -1,
                "download_url": f"/api/download/{task_id}/masks",
            })
        return files

    def _read_manifest(self, out_dir: Path) -> dict:
        """读 manifest（both 模式优先 plate 版审计凭据）。"""
        for name in ("result.plate.manifest.json", "result.design.manifest.json",
                     "result.manifest.json", "manifest.json"):
            p = out_dir / name
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
        return {}


# 模块级单例
task_manager = TaskManager()
