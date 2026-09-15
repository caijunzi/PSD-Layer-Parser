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
import os
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

# 线程安全写
_lock = threading.Lock()

# 待审队列默认路径（保留常量供外部引用）
DEFAULT_PENDING_PATH = "webui/data/pending_feedbacks.jsonl"


def default_pending_path() -> str:
    """解析待审队列路径（**调用时**读环境变量，便于测试隔离）。"""
    return os.environ.get("ADAPTIVE_PENDING_PATH", DEFAULT_PENDING_PATH)


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

    # 兼容两种输入：
    #   ① 显式 detections（含 category_id / confidence / bbox）—— 文档示例格式
    #   ② grounded_sam_provider.dino_detections（含 layer_name / logits / boxes）—— 生产真实格式
    detections = detection_result.get("detections", [])
    if not detections:
        detections = detection_result.get("dino_detections", [])

    for det in detections:
        # 类目：显式 category_id 优先，回退 dino_detections 的 layer_name
        category_id = det.get("category_id") or det.get("layer_name")

        # 置信度：显式 confidence 优先；否则以 logits 最大值作为确定性代理
        confidence = det.get("confidence")
        max_logit = _max_logit(det.get("logits") if det.get("logits") is not None else det.get("logit"))
        if confidence is None:
            confidence = max_logit if max_logit is not None else 0.0

        # bbox：显式 bbox 优先，回退 boxes 的第一个
        bbox = det.get("bbox")
        if bbox is None:
            boxes = det.get("boxes")
            try:
                if boxes is not None and len(boxes) > 0:
                    first = boxes[0]
                    bbox = [float(v) for v in first] if hasattr(first, "__iter__") else None
            except Exception:
                bbox = None

        if float(confidence) < confidence_threshold:
            uncertain.append({
                "category_id": category_id,
                "confidence": float(confidence),
                "bbox": bbox,
                "prompt": det.get("prompt"),
                "logit": det.get("logit") if det.get("logit") is not None else max_logit,
            })

    return uncertain


def _max_logit(logits) -> Optional[float]:
    """从 logits（list / ndarray / torch.Tensor）中取最大值；无法解析时返回 None。"""
    if logits is None:
        return None
    try:
        import numpy as _np
        arr = _np.asarray(logits, dtype=float).ravel()
        return float(arr.max()) if arr.size else None
    except Exception:
        try:
            return float(max(logits))
        except Exception:
            return None


def request_human_feedback(
    uncertain_categories: List[Dict[str, Any]],
    task_id: str,
    image_path: Optional[str] = None,
    pending_path: Optional[str] = None
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
    pending_file = Path(pending_path or default_pending_path())
    pending_file.parent.mkdir(parents=True, exist_ok=True)
    
    with _lock:
        with pending_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(request_data, ensure_ascii=False) + "\n")
    
    return feedback_request_id


def get_pending_feedbacks(
    pending_path: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """读取待审队列（前端轮询调用）
    
    Args:
        pending_path: 待审队列文件路径
        limit: 最多返回多少条（默认 10，避免前端过载）
    
    Returns:
        待审请求列表（status="pending" 的前 N 条）
    """
    pending_file = Path(pending_path or default_pending_path())
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
    pending_path: Optional[str] = None
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
    pending_file = Path(pending_path or default_pending_path())
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
        - rename: 更新 categories.name_zh / name_en / name_template
        - delete: 软删除（置 categories.deleted_at 时间戳），保留历史引用
        - merge:  源类目 prompts 迁移到目标类目 + 软删源类目

    Returns:
        dict: 操作结果 {"ok": bool, "action": str, "detail": str}
    """
    import sqlite3
    from datetime import datetime, timezone

    now = int(datetime.now(timezone.utc).timestamp())
    conn = sqlite3.connect(db_path)
    result = {"ok": False, "action": user_action, "detail": ""}

    def _fail(msg: str):
        result["detail"] = msg
        print(f"[ActiveLearner] {user_action}: {category_id} 失败 —— {msg}")
        return result

    def _category_exists(cur, cid: str):
        row = cur.execute("SELECT id FROM categories WHERE id = ?", (cid,)).fetchone()
        return row is not None

    try:
        cur = conn.cursor()

        if user_action == "accept":
            cur.execute(
                """
                UPDATE category_prompts
                SET weight = weight * 1.1, n_accept = n_accept + 1
                WHERE category_id = ?
                """,
                (category_id,),
            )
            conn.commit()
            result["ok"] = True
            result["detail"] = f"权重 ×1.1（影响 {cur.rowcount} 条 prompt）"
            print(f"[ActiveLearner] accept: {category_id} {result['detail']}")

        elif user_action == "rename":
            new_name = (user_data or {}).get("new_name") if user_data else None
            if not new_name or not str(new_name).strip():
                return _fail("缺少 new_name（user_data.new_name）")
            new_name = str(new_name).strip()
            new_name_en = (user_data or {}).get("new_name_en")

            row = cur.execute(
                "SELECT name_en, name_template FROM categories WHERE id = ?", (category_id,)
            ).fetchone()
            if not row:
                return _fail(f"类目不存在：{category_id}")

            old_en, old_template = row[0], row[1]
            name_en = str(new_name_en).strip() if new_name_en else (old_en or "")

            # 重建 name_template：保留原序号前缀（如 "NN_"），其余按 "{name_zh}_{name_en}"
            prefix = ""
            if old_template and "_" in old_template:
                head = old_template.split("_")[0]
                if head and len(head) <= 4 and head.isalnum():
                    prefix = f"{head}_"
            new_template = f"{prefix}{new_name}_{name_en}" if name_en else f"{prefix}{new_name}"

            cur.execute(
                """
                UPDATE categories
                SET name_zh = ?, name_en = ?, name_template = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_name, name_en or None, new_template, now, category_id),
            )
            conn.commit()
            result["ok"] = True
            result["detail"] = f"{old_template} → {new_template}"
            print(f"[ActiveLearner] rename: {category_id} {result['detail']}")

        elif user_action == "delete":
            if not _category_exists(cur, category_id):
                return _fail(f"类目不存在：{category_id}")

            # 软删除：置 deleted_at；查询层用 deleted_at IS NULL 过滤
            cur.execute(
                "UPDATE categories SET deleted_at = ?, updated_at = ? WHERE id = ?",
                (now, now, category_id),
            )
            conn.commit()
            result["ok"] = True
            result["detail"] = f"软删除（deleted_at={now}），保留 prompt 与历史引用"
            print(f"[ActiveLearner] delete: {category_id} {result['detail']}")

        elif user_action == "merge":
            target = (user_data or {}).get("target_category_id") if user_data else None
            if not target or not str(target).strip():
                return _fail("缺少 target_category_id（user_data.target_category_id）")
            target = str(target).strip()

            if target == category_id:
                return _fail("目标类目不能与源类目相同")

            if not _category_exists(cur, category_id):
                return _fail(f"源类目不存在：{category_id}")
            if not _category_exists(cur, target):
                return _fail(f"目标类目不存在：{target}")

            # 迁移 prompts：目标已有同名 prompt → 取权重较大者；否则改挂到目标
            src_prompts = cur.execute(
                """
                SELECT id, prompt, weight FROM category_prompts
                WHERE category_id = ?
                """,
                (category_id,),
            ).fetchall()

            migrated, merged, kept = 0, 0, 0
            for pid, prompt, weight in src_prompts:
                dup = cur.execute(
                    "SELECT id, weight FROM category_prompts WHERE category_id = ? AND prompt = ?",
                    (target, prompt),
                ).fetchone()
                if dup:
                    # 同名 prompt：保留权重较大者，删除源行
                    if (weight or 0) > (dup[1] or 0):
                        cur.execute(
                            "UPDATE category_prompts SET weight = ? WHERE id = ?",
                            (weight, dup[0]),
                        )
                        merged += 1
                    else:
                        kept += 1
                    cur.execute("DELETE FROM category_prompts WHERE id = ?", (pid,))
                else:
                    cur.execute(
                        "UPDATE category_prompts SET category_id = ? WHERE id = ?",
                        (target, pid),
                    )
                    migrated += 1

            # 源类目软删除
            cur.execute(
                "UPDATE categories SET deleted_at = ?, updated_at = ? WHERE id = ?",
                (now, now, category_id),
            )
            conn.commit()
            result["ok"] = True
            result["detail"] = (
                f"→ {target}：迁移 {migrated} 条、权重合并 {merged} 条、"
                f"目标已更优跳过 {kept} 条；源类目软删除"
            )
            print(f"[ActiveLearner] merge: {category_id} {result['detail']}")

        else:
            return _fail(f"未知操作：{user_action}")

        # ---- 版本链登记（B4 接线，2026-09-15）----
        # 人审改动落一个版本节点（Git 式链），便于回溯与回滚。
        # 登记失败**不影响**反馈本身（词库已 commit），仅回传 version_note。
        try:
            from .db_manager import DBManager

            mgr = DBManager(db_path)
            vid = "v" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            ev = ""
            if user_data:
                ev = json.dumps(
                    {k: user_data[k] for k in ("feedback_request_id",) if k in user_data},
                    ensure_ascii=False,
                )
            ok_ver = mgr.create_version(
                version_id=vid,
                changeset=f"human_feedback:{user_action}:{category_id} | {result['detail']}",
                evidence=ev,
                snapshot={
                    "action": user_action,
                    "category_id": category_id,
                    "detail": result["detail"],
                    "user_data": user_data or {},
                },
                regression_status="passed",
                activate=True,
            )
            mgr.close()
            if ok_ver:
                result["version_id"] = vid
            else:
                result["version_note"] = "版本登记失败（反馈已生效）"
        except Exception as _ve:
            result["version_note"] = f"版本登记异常（反馈已生效）: {_ve}"

        return result

    finally:
        conn.close()
