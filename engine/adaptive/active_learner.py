"""主动学习模块（Stage 5.2）

识别不确定类目（confidence < threshold），触发人审反馈，最小化人工介入。
采用轮询方案：后端维护待审队列（JSONL），前端定期轮询 GET /api/adaptive/pending-feedbacks。

典型流程：
1. 引擎检出结果 → identify_uncertain_categories() 识别低置信类目
2. request_human_feedback() 写入待审队列（webui/data/pending_feedbacks.jsonl）
3. 前端 3 秒轮询 GET /api/adaptive/pending-feedbacks，读取队列
4. 用户操作（accept / rename / delete / merge）→ POST /api/categories/{id}/feedback
5. 反馈写回 episode + 更新 category_prompts 权重
"""
import json
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

# 线程安全写
_lock = threading.Lock()

DEFAULT_PENDING_PATH = "webui/data/pending_feedbacks.jsonl"


def identify_uncertain_categories(
    detection_result: Dict[str, Any],
    confidence_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """识别不确定类目（confidence < threshold）
    
    Args:
        detection_result: 引擎检出结果，包含 categories / detections / confidence 等
        confidence_threshold: 置信度阈值（默认 0.7，低于此值视为不确定）
    
    Returns:
        不确定类目列表 [{category_id, confidence, bbox, prompt, ...}, ...]
        
    示例输入：
        {
            "categories": ["distant_mountains", "water_ripples"],
            "detections": [
                {"category_id": "distant_mountains", "confidence": 0.85, "bbox": [...]},
                {"category_id": "water_ripples", "confidence": 0.62, "bbox": [...]}
            ]
        }
    
    示例输出：
        [{"category_id": "water_ripples", "confidence": 0.62, "bbox": [...]}]
    """
    uncertain = []
    
    # 从 detections 中提取低置信类目
    detections = detection_result.get("detections", [])
    for det in detections:
        confidence = det.get("confidence", 0.0)
        if confidence < confidence_threshold:
            uncertain.append({
                "category_id": det.get("category_id"),
                "confidence": confidence,
                "bbox": det.get("bbox"),
                "prompt": det.get("prompt"),
                "logit": det.get("logit"),
            })
    
    return uncertain


def request_human_feedback(
    uncertain_categories: List[Dict[str, Any]],
    task_id: str,
    image_path: Optional[str] = None,
    pending_path: str = DEFAULT_PENDING_PATH
) -> str:
    """触发人审请求：写入待审队列（JSONL）
    
    Args:
        uncertain_categories: 不确定类目列表（identify_uncertain_categories 返回值）
        task_id: 任务 ID（用于关联 episode）
        image_path: 图像路径（可选，用于前端展示缩略图）
        pending_path: 待审队列文件路径
    
    Returns:
        feedback_request_id: 反馈请求 ID（用于后续跟踪）
    
    数据格式（JSONL 每行一个请求）：
        {
            "feedback_request_id": "fb_20260913_234500_abc123",
            "task_id": "task_20260913_120555_b2b831",
            "image_path": "/path/to/image.png",
            "uncertain_categories": [
                {"category_id": "water_ripples", "confidence": 0.62, "bbox": [...]}
            ],
            "status": "pending",  // pending / reviewed / expired
            "created_at": 1789276800,
            "reviewed_at": null
        }
    """
    if not uncertain_categories:
        return ""  # 无不确定类目，无需人审
    
    # 生成反馈请求 ID
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    import hashlib
    digest = hashlib.sha1(f"{task_id}|{ts}".encode()).hexdigest()[:8]
    feedback_request_id = f"fb_{ts}_{digest}"
    
    # 构造请求数据
    request_data = {
        "feedback_request_id": feedback_request_id,
        "task_id": task_id,
        "image_path": image_path,
        "uncertain_categories": uncertain_categories,
        "status": "pending",
        "created_at": int(datetime.now(timezone.utc).timestamp()),
        "reviewed_at": None,
    }
    
    # 追加写入 JSONL（线程安全）
    pending_file = Path(pending_path)
    pending_file.parent.mkdir(parents=True, exist_ok=True)
    
    with _lock:
        with pending_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(request_data, ensure_ascii=False) + "\n")
    
    return feedback_request_id


def get_pending_feedbacks(
    pending_path: str = DEFAULT_PENDING_PATH,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """读取待审队列（前端轮询调用）
    
    Args:
        pending_path: 待审队列文件路径
        limit: 最多返回多少条（默认 10，避免前端过载）
    
    Returns:
        待审请求列表（status="pending" 的前 N 条）
    """
    pending_file = Path(pending_path)
    if not pending_file.exists():
        return []
    
    pending_requests = []
    with pending_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                if req.get("status") == "pending":
                    pending_requests.append(req)
                    if len(pending_requests) >= limit:
                        break
            except json.JSONDecodeError:
                continue  # 跳过损坏行
    
    return pending_requests


def mark_feedback_reviewed(
    feedback_request_id: str,
    user_action: str,
    user_data: Optional[Dict[str, Any]] = None,
    pending_path: str = DEFAULT_PENDING_PATH
) -> bool:
    """标记反馈请求已审核（用户提交反馈后调用）
    
    Args:
        feedback_request_id: 反馈请求 ID
        user_action: 用户操作（accept / rename / delete / merge）
        user_data: 用户操作附加数据（如 rename 的新名称、merge 的目标类目）
        pending_path: 待审队列文件路径
    
    Returns:
        是否成功标记
    
    实现：读取 JSONL → 找到对应请求 → 更新 status="reviewed" + reviewed_at + user_action → 写回
    """
    pending_file = Path(pending_path)
    if not pending_file.exists():
        return False
    
    # 读取全部请求
    requests = []
    with pending_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                requests.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    # 查找并更新目标请求
    found = False
    for req in requests:
        if req.get("feedback_request_id") == feedback_request_id:
            req["status"] = "reviewed"
            req["reviewed_at"] = int(datetime.now(timezone.utc).timestamp())
            req["user_action"] = user_action
            req["user_data"] = user_data or {}
            found = True
            break
    
    if not found:
        return False
    
    # 写回 JSONL（线程安全）
    with _lock:
        with pending_file.open("w", encoding="utf-8") as f:
            for req in requests:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
    
    return True


def apply_feedback_to_category(
    category_id: str,
    user_action: str,
    user_data: Optional[Dict[str, Any]] = None,
    db_path: str = "webui/data/adaptive_semantics.db"
):
    """应用用户反馈到类目（更新 category_prompts 权重或执行 rename/delete/merge）
    
    Args:
        category_id: 类目 ID
        user_action: 用户操作（accept / rename / delete / merge）
        user_data: 操作附加数据
        db_path: 数据库路径
    
    操作逻辑：
        - accept: 提升该类目的所有 prompt 权重 1.1×（贝叶斯正反馈）
        - rename: 更新 categories.name_zh / name_en
        - delete: 标记类目为已删除（soft delete，不真删）
        - merge: 将该类目的 prompt 合并到目标类目
    
    ⚠️ TODO：当前仅实现 accept 逻辑（权重更新），rename/delete/merge 留待后续补充
    """
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    try:
        if user_action == "accept":
            # 提升权重 1.1×（贝叶斯正反馈，与 Stage 3 learner 一致）
            conn.execute(
                """
                UPDATE category_prompts
                SET weight = weight * 1.1, n_accept = n_accept + 1
                WHERE category_id = ?
                """,
                (category_id,)
            )
            conn.commit()
            print(f"[ActiveLearner] accept: {category_id} 权重提升 1.1×")
        
        elif user_action == "rename":
            # TODO：更新 categories 表的 name_zh / name_en
            print(f"[ActiveLearner] rename: {category_id} → {user_data} (TODO)")
        
        elif user_action == "delete":
            # TODO：soft delete（添加 deleted_at 字段或标记 status="deleted"）
            print(f"[ActiveLearner] delete: {category_id} (TODO)")
        
        elif user_action == "merge":
            # TODO：合并 prompt 到目标类目
            target = user_data.get("target_category_id") if user_data else None
            print(f"[ActiveLearner] merge: {category_id} → {target} (TODO)")
        
        else:
            print(f"[ActiveLearner] 未知操作：{user_action}")
    
    finally:
        conn.close()
