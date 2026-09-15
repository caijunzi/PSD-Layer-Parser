"""
后台学习任务封装（Background Learner）—— Stage 3（反馈闭环 + 影子进化）

职责：
- 封装 Starlette BackgroundTasks，提供异步学习接口
- 实现学习任务队列管理（避免重复学习、控制并发）
- 提供学习任务状态查询接口
- 异步执行（不阻塞主流程）

设计原则：
- 学习器不直接修改 preset 文件，而是生成"推荐调整"JSON 供人工审核
- 学习任务队列使用线程安全的 deque，避免重复学习
- 学习任务状态持久化到 webui/data/learning_tasks.jsonl（JSONL 格式，便于追加）
- 提供 REST API：POST /api/adaptive/trigger-learning（手动触发）+ GET /api/adaptive/learning-status（查询状态）
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import deque

from engine.adaptive.learner import SemanticLearner
from engine.adaptive.episode_archiver import load_episodes


class BackgroundLearner:
    """
    后台学习器：封装异步学习任务，提供队列管理和状态查询。
    
    职责：
    1. 封装 SemanticLearner，提供异步学习接口
    2. 实现学习任务队列管理（避免重复学习、控制并发）
    3. 提供学习任务状态查询接口
    4. 异步执行（不阻塞主流程）
    
    不修改 preset 文件，只生成推荐调整 JSON 供人工审核。
    """
    
    def __init__(
        self,
        episode_path: str = "webui/data/adaptive_episodes.jsonl",
        baseline_path: str = "tests/baseline_audit_8d.json",
        learning_tasks_path: str = "webui/data/learning_tasks.jsonl"
    ):
        """
        初始化后台学习器。
        
        Args:
            episode_path: episode 归档路径
            baseline_path: 回归基线数据路径
            learning_tasks_path: 学习任务状态持久化路径
        """
        self.episode_path = episode_path
        self.baseline_path = baseline_path
        self.learning_tasks_path = learning_tasks_path
        
        # 学习器实例
        self.learner = SemanticLearner(baseline_path=baseline_path)
        
        # 学习任务队列（线程安全）
        self.task_queue: deque = deque()
        self.task_lock = threading.Lock()
        
        # 学习任务状态（episode_id -> task_status）
        self.task_status: Dict[str, Dict[str, Any]] = {}
        self.status_lock = threading.Lock()
        
        # 加载已有的学习任务状态
        self._load_task_status()
    
    def _load_task_status(self):
        """从持久化文件加载学习任务状态"""
        path = Path(self.learning_tasks_path)
        if not path.exists():
            return
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                task = json.loads(line)
                episode_id = task.get("episode_id")
                if episode_id:
                    self.task_status[episode_id] = task
    
    def _save_task_status(self, task: Dict[str, Any]):
        """追加保存学习任务状态到持久化文件"""
        path = Path(self.learning_tasks_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    
    def enqueue_learning_task(
        self,
        episode_id: str,
        trigger: str = "auto"
    ) -> Dict[str, Any]:
        """
        入队学习任务（异步执行）。
        
        Args:
            episode_id: episode ID
            trigger: 触发方式（"auto" | "manual"）
            
        Returns:
            {
                "status": "queued" | "running" | "completed" | "failed" | "duplicate",
                "episode_id": str,
                "message": str
            }
        """
        # 检查是否已存在
        with self.status_lock:
            if episode_id in self.task_status:
                existing_status = self.task_status[episode_id].get("status")
                if existing_status in ("queued", "running", "completed"):
                    return {
                        "status": "duplicate",
                        "episode_id": episode_id,
                        "message": f"学习任务已存在（状态：{existing_status}）"
                    }
        
        # 入队
        with self.task_lock:
            self.task_queue.append({
                "episode_id": episode_id,
                "trigger": trigger,
                "enqueued_at": datetime.now().isoformat()
            })
        
        # 更新状态
        task_status = {
            "episode_id": episode_id,
            "status": "queued",
            "trigger": trigger,
            "enqueued_at": datetime.now().isoformat(),
            "message": "学习任务已入队"
        }
        
        with self.status_lock:
            self.task_status[episode_id] = task_status
        
        self._save_task_status(task_status)
        
        return task_status
    
    def process_learning_task(self, episode_id: str):
        """
        处理学习任务（后台线程执行）。
        
        Args:
            episode_id: episode ID
        """
        # 更新状态为 running
        task_status = {
            "episode_id": episode_id,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "message": "学习任务执行中"
        }
        
        with self.status_lock:
            self.task_status[episode_id] = task_status
        
        self._save_task_status(task_status)
        
        try:
            # 1. 加载 episode
            episodes = load_episodes(
                episode_path=self.episode_path,
                limit=1000
            )
            
            # 找到目标 episode
            target_episode = None
            for ep in episodes:
                if ep.get("episode_id") == episode_id:
                    target_episode = ep
                    break
            
            if not target_episode:
                raise ValueError(f"未找到 episode: {episode_id}")
            
            # 2. 从 episode 中学习
            learning_result = self.learner.learn_from_episode(target_episode)
            
            # 3. 更新状态为 completed
            task_status = {
                "episode_id": episode_id,
                "status": "completed",
                "started_at": task_status["started_at"],
                "completed_at": datetime.now().isoformat(),
                "message": "学习任务完成",
                "learning_result": learning_result
            }
            
            with self.status_lock:
                self.task_status[episode_id] = task_status
            
            self._save_task_status(task_status)
        
        except Exception as e:
            # 更新状态为 failed
            task_status = {
                "episode_id": episode_id,
                "status": "failed",
                "started_at": task_status.get("started_at"),
                "failed_at": datetime.now().isoformat(),
                "message": f"学习任务失败: {str(e)}",
                "error": str(e)
            }
            
            with self.status_lock:
                self.task_status[episode_id] = task_status
            
            self._save_task_status(task_status)
    
    def get_task_status(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """
        查询学习任务状态。
        
        Args:
            episode_id: episode ID
            
        Returns:
            {
                "episode_id": str,
                "status": "queued" | "running" | "completed" | "failed",
                "enqueued_at": str,
                "started_at": str,
                "completed_at": str,
                "message": str,
                "learning_result": {...}  # 仅 completed 状态有
            }
        """
        with self.status_lock:
            return self.task_status.get(episode_id)
    
    def get_all_task_status(
        self,
        limit: int = 100,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        查询所有学习任务状态。
        
        Args:
            limit: 最大返回数量
            status_filter: 状态过滤（"queued" | "running" | "completed" | "failed"）
            
        Returns:
            [
                {
                    "episode_id": str,
                    "status": str,
                    "enqueued_at": str,
                    ...
                },
                ...
            ]
        """
        with self.status_lock:
            tasks = list(self.task_status.values())
        
        # 过滤状态
        if status_filter:
            tasks = [t for t in tasks if t.get("status") == status_filter]
        
        # 按入队时间倒序排序
        tasks.sort(key=lambda t: t.get("enqueued_at", ""), reverse=True)
        
        return tasks[:limit]

    # ---------------- 后台消费者（修复：此前 task_queue 无消费者，任务永不执行） ----------------

    def drain_once(self) -> Optional[str]:
        """从队列取出一个任务并执行；队列空返回 None（供测试同步调用）。"""
        with self.task_lock:
            if not self.task_queue:
                return None
            item = self.task_queue.popleft()
        self.process_learning_task(item["episode_id"])
        return item["episode_id"]

    def start_worker(self, poll_interval: float = 1.0) -> threading.Thread:
        """启动后台守护线程消费学习任务队列（幂等）。

        修复（2026-09-15）：enqueue_learning_task 只入队，从未有消费者，
        导致学习任务永远停在 queued。此处补上守护线程。
        """
        if getattr(self, "_worker", None) is not None:
            return self._worker

        def _loop():
            while True:
                try:
                    processed = self.drain_once()
                except Exception as e:  # 单任务异常不应杀死 worker
                    print(f"[BackgroundLearner] worker 处理异常: {e}")
                    processed = None
                if processed is None:
                    time.sleep(poll_interval)

        self._worker = threading.Thread(target=_loop, name="bg-learner", daemon=True)
        self._worker.start()
        return self._worker


# 全局单例（便于 API 路由调用）
_background_learner: Optional[BackgroundLearner] = None


def get_background_learner() -> BackgroundLearner:
    """获取全局后台学习器实例（单例），并确保后台消费线程已启动。"""
    global _background_learner
    if _background_learner is None:
        _background_learner = BackgroundLearner()
        _background_learner.start_worker()
    return _background_learner
