"""Adaptive Semantics API - 自适应语义统计与配置端点

提供自适应语义机制的当前状态、版本信息、类目统计等查询接口，
以及 Stage 2 的图级 Auto-Tune 建议接口。
"""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from typing import Optional
import os
from pathlib import Path

import cv2
import numpy as np

router = APIRouter()


# 数据库默认路径：<project_root>/webui/data/adaptive_semantics.db
# adaptive.py 位于 webui/backend/api/，向上 4 级到 project_root 再进 webui/data
DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "webui" / "data" / "adaptive_semantics.db"
)


@router.get("/adaptive/stats")
async def get_adaptive_stats():
    """获取自适应语义统计信息

    Returns:
        {
            "enabled": bool,           # 是否启用自适应语义
            "mode": str,               # 当前模式（locked/auto/hybrid）
            "version": str,            # 当前激活版本号
            "categories": int,         # 类目总数
            "prompts": int,            # 提示词总数
            "material_families": int,  # 材质家族数
            "db_path": str,            # 数据库路径
            "db_exists": bool,         # 数据库是否存在
        }
    """
    from engine.schemas.presets import get_adaptive_mode

    # 读取 ADAPTIVE_MODE 环境变量（实际模式由 preset.mode 决定，这里仅作展示）
    adaptive_mode = os.getenv("ADAPTIVE_MODE", "off")
    enabled = adaptive_mode != "off"

    # 数据库路径（优先环境变量，否则用默认）
    db_path = os.getenv("ADAPTIVE_DB_PATH", str(DEFAULT_DB_PATH))
    db_exists = Path(db_path).exists()

    # 统计信息
    stats = {
        "enabled": enabled,
        "mode": adaptive_mode,
        "version": None,
        "categories": 0,
        "prompts": 0,
        "material_families": 5,  # 固定 5 个材质家族
        "db_path": db_path,
        "db_exists": db_exists,
    }

    if db_exists:
        try:
            from engine.adaptive.db_manager import create_db_manager

            db_mgr = create_db_manager(db_path)

            # 查询当前激活版本
            version_row = db_mgr.execute(
                "SELECT version_id FROM active_version LIMIT 1"
            ).fetchone()
            if version_row:
                stats["version"] = version_row["version_id"]

            # 查询类目总数
            cat_row = db_mgr.execute(
                "SELECT COUNT(*) AS n FROM categories"
            ).fetchone()
            stats["categories"] = cat_row["n"]

            # 查询提示词总数
            prompt_row = db_mgr.execute(
                "SELECT COUNT(*) AS n FROM category_prompts"
            ).fetchone()
            stats["prompts"] = prompt_row["n"]

            db_mgr.close()
        except Exception as e:
            # 数据库读取失败，返回基础信息
            stats["error"] = str(e)

    return stats


@router.get("/adaptive/categories")
async def get_adaptive_categories(material_family: Optional[str] = None):
    """获取自适应语义类目列表

    Args:
        material_family: 可选的材质家族过滤（金地屏风/宣纸水墨/绢本工笔/油画布/其他）

    Returns:
        {
            "categories": [
                {
                    "name": str,
                    "confidence": float,
                    "prompts": [str],
                }
            ]
        }
    """
    db_path = os.getenv("ADAPTIVE_DB_PATH", str(DEFAULT_DB_PATH))

    if not Path(db_path).exists():
        raise HTTPException(status_code=404, detail="自适应语义数据库不存在")

    try:
        from engine.adaptive.db_manager import create_db_manager

        db_mgr = create_db_manager(db_path)

        # 查询当前激活版本
        version_row = db_mgr.execute(
            "SELECT version_id FROM active_version LIMIT 1"
        ).fetchone()
        if not version_row:
            raise HTTPException(status_code=404, detail="未找到激活版本")

        # 查询类目列表
        if material_family:
            rows = db_mgr.execute(
                """
                SELECT c.id, c.name_zh, c.name_en, a.affinity
                FROM categories c
                JOIN category_material_affinity a ON c.id = a.category_id
                WHERE a.material_family = ?
                ORDER BY a.affinity DESC
                """,
                (material_family,),
            ).fetchall()
        else:
            rows = db_mgr.execute(
                """
                SELECT c.id, c.name_zh, c.name_en, c.confidence
                FROM categories c
                ORDER BY c.name_zh
                """
            ).fetchall()

        # 构建返回结果
        result = []
        for row in rows:
            name_zh = row["name_zh"]
            name_en = row["name_en"]
            display_name = f"{name_zh}_{name_en}" if name_en else name_zh
            confidence = row["affinity"] if material_family else row["confidence"]

            # 查询该类目的提示词（权重最高前 5）
            prompt_rows = db_mgr.execute(
                """
                SELECT prompt FROM category_prompts
                WHERE category_id = ?
                  AND source != 'shadow'
                  AND weight >= 0.3
                ORDER BY weight DESC
                LIMIT 5
                """,
                (row["id"],),
            ).fetchall()

            result.append({
                "name": display_name,
                "confidence": float(confidence),
                "prompts": [p["prompt"] for p in prompt_rows],
            })

        db_mgr.close()
        return {"categories": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.post("/adaptive/suggest-auto-tune")
async def suggest_auto_tune_endpoint(
    file: UploadFile = File(...),
    preset_id: Optional[str] = Form(None),
):
    """图级 Auto-Tune（Stage 2）：上传图片，返回 region + 密度带建议。

    Args:
        file: 上传的图像（multipart）
        preset_id: 可选。preset 名称或 JSON 路径；提供则对 preset 中带 region
                   的语义类逐类推荐密度带。

    Returns:
        {
          "image_size": [w, h],
          "global_percentiles": {"p50","p75","p90","p95"},
          "regions": [{"region": [x0,y0,x1,y1], "score", "blocks"}, ...],
          "density_bands": [{"label","region","density_min","density_max","coverage"}, ...],
        }
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")

    # Unicode 安全：bytes → imdecode（不落盘，避免中文路径问题）
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码图像（格式不支持或损坏）")

    classes = None
    if preset_id:
        try:
            from engine.schemas.presets import load_preset

            preset = load_preset(preset_id)
            classes = preset.get("ai_semantic_classes")
        except Exception as e:
            # preset 加载失败不致命：退化为通用九宫格建议
            print(f"[suggest-auto-tune] preset 加载失败，退化为通用建议: {e}")

    try:
        from engine.adaptive.auto_tune import suggest_auto_tune

        result = suggest_auto_tune(img, classes=classes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-Tune 失败: {str(e)}")

    h, w = img.shape[:2]
    result["image_size"] = [int(w), int(h)]
    return result


@router.post("/adaptive/apply-auto-tune")
async def apply_auto_tune_endpoint(
    preset_id: str = Form(...),
    suggestions: str = Form(...),
):
    """采纳 Auto-Tune 建议并写回 preset（Stage 2 闭环）。

    Args:
        preset_id: preset 名称或 JSON 路径
        suggestions: JSON 字符串，格式 {"regions": [...], "density_bands": [...]}

    Returns:
        {"status": "ok", "preset_path": "..."}
    """
    from pathlib import Path
    import json

    # 解析建议 JSON
    try:
        sugg = json.loads(suggestions)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"suggestions JSON 格式错误: {e}")

    if not isinstance(sugg, dict):
        raise HTTPException(status_code=400, detail="suggestions 必须是对象")

    regions = sugg.get("regions", [])
    density_bands = sugg.get("density_bands", [])

    # 加载 preset
    try:
        from engine.schemas.presets import load_preset
        preset = load_preset(preset_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"preset 加载失败: {e}")

    # 确定 preset 文件路径（load_preset 返回 dict，需反查路径）
    preset_dir = Path(__file__).parent.parent.parent.parent / "presets"
    preset_path = preset_dir / f"{preset_id}.json"
    if not preset_path.exists():
        # 尝试 preset_id 本身是相对/绝对路径
        preset_path = Path(preset_id)
        if not preset_path.exists():
            raise HTTPException(status_code=404, detail=f"preset 文件不存在: {preset_id}")

    # 合并建议到 preset（原地修改）
    # regions: 写入顶层 "auto_tune_regions" 字段（新增，供未来消费）
    # density_bands: 追加/更新 "density_band_classes"
    if regions:
        preset["auto_tune_regions"] = regions

    if density_bands:
        # 已有 density_band_classes 的合并策略：按 label 匹配更新，或追加
        existing = {b.get("name"): b for b in preset.get("density_band_classes", [])}
        for band in density_bands:
            label = band.get("label")
            if not label:
                continue
            # Auto-Tune 产出的 band 可能无 name，用 label 作 name
            entry = {
                "name": label,
                "density_min": band.get("density_min"),
                "density_max": band.get("density_max"),
                "region": band.get("region"),
                "note": f"Auto-Tune 推荐（coverage={band.get('coverage', 0):.2f}）",
            }
            # 过滤 None
            entry = {k: v for k, v in entry.items() if v is not None}
            existing[label] = entry
        preset["density_band_classes"] = list(existing.values())

    # 写回文件
    try:
        preset_path.write_text(json.dumps(preset, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"写入 preset 失败: {e}")

    return {"status": "ok", "preset_path": str(preset_path)}


# ========== Stage 5.2：主动学习（人审反馈）端点 ==========


@router.get("/adaptive/pending-feedbacks")
async def get_pending_feedbacks_endpoint(limit: int = 10):
    """获取待审类目列表（前端轮询调用，Stage 5.2）
    
    Args:
        limit: 最多返回多少条（默认 10）
    
    Returns:
        {
            "pending_feedbacks": [
                {
                    "feedback_request_id": "fb_20260913_234500_abc123",
                    "task_id": "task_20260913_120555_b2b831",
                    "image_path": "/path/to/image.png",
                    "uncertain_categories": [
                        {"category_id": "water_ripples", "confidence": 0.62, "bbox": [...]}
                    ],
                    "created_at": 1789276800
                },
                ...
            ]
        }
    """
    try:
        from engine.adaptive.active_learner import get_pending_feedbacks
        
        pending = get_pending_feedbacks(limit=limit)
        return {"pending_feedbacks": pending}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取待审队列失败: {str(e)}")


@router.post("/adaptive/categories/{category_id}/feedback")
async def submit_category_feedback_endpoint(
    category_id: str,
    feedback_request_id: str = Form(...),
    user_action: str = Form(...),  # accept / rename / delete / merge
    user_data: Optional[str] = Form(None),  # JSON 字符串，如 {"new_name": "..."} 或 {"target_category_id": "..."}
):
    """提交类目人审反馈（Stage 5.2）
    
    Args:
        category_id: 类目 ID
        feedback_request_id: 反馈请求 ID（关联待审队列）
        user_action: 用户操作（accept / rename / delete / merge）
        user_data: 操作附加数据（JSON 字符串，可选）
    
    Returns:
        {"status": "ok", "message": "反馈已应用"}
    """
    import json
    from engine.adaptive.active_learner import (
        mark_feedback_reviewed,
        apply_feedback_to_category,
        get_feedback_request,
    )
    
    # 解析 user_data（JSON 字符串 → dict）
    user_data_dict = None
    if user_data:
        try:
            user_data_dict = json.loads(user_data)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"user_data JSON 格式错误: {e}")
    
    # 验证 user_action
    valid_actions = {"accept", "rename", "delete", "merge"}
    if user_action not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"无效的 user_action: {user_action}，必须是 {valid_actions}"
        )
    
    try:
        # 0. 幂等：同一反馈请求已审核（已应用）则跳过重复应用，避免污染
        existing = get_feedback_request(feedback_request_id)
        if existing and existing.get("status") == "reviewed":
            return {
                "status": "ok",
                "message": f"反馈已应用（重复请求，已忽略）：{user_action} on {category_id}",
                "detail": "duplicate_skipped",
            }

        # 1. 先成功应用反馈到类目（accept 更新权重 / rename / delete / merge）
        #    库路径支持环境变量覆盖（与 /adaptive/stats 一致，便于测试隔离）
        db_path = os.getenv("ADAPTIVE_DB_PATH", str(DEFAULT_DB_PATH))
        outcome = apply_feedback_to_category(
            category_id=category_id,
            user_action=user_action,
            user_data=user_data_dict,
            db_path=db_path,
        )

        # 操作本身失败（如缺 user_data、目标类目不存在、rowcount 零）→ 明确报 400
        if not outcome.get("ok"):
            raise HTTPException(
                status_code=400,
                detail=f"反馈操作失败：{outcome.get('detail', '未知原因')}",
            )

        # 2. 应用成功后再标记反馈请求已审核（先应用后标 reviewed，防止伪完成）
        marked = mark_feedback_reviewed(
            feedback_request_id=feedback_request_id,
            user_action=user_action,
            user_data=user_data_dict,
        )
        if not marked:
            raise HTTPException(
                status_code=500,
                detail=f"反馈已应用但标记审核失败：{feedback_request_id}",
            )

        return {
            "status": "ok",
            "message": f"反馈已应用：{user_action} on {category_id}",
            "detail": outcome.get("detail", ""),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"应用反馈失败: {str(e)}")


# ========== Stage 3：后台学习任务端点（此前仅 docstring 声明，从未注册） ==========


def _ensure_project_root_on_path() -> None:
    """确保项目根在 sys.path（engine 包可被 webui 后端导入）。"""
    import sys
    from pathlib import Path as _P
    root = str(_P(__file__).resolve().parent.parent.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


@router.post("/adaptive/trigger-learning")
async def trigger_learning_endpoint(episode_id: Optional[str] = Form(None)):
    """手动触发一次后台学习任务（Stage 3）。

    Args:
        episode_id: 可选。缺省时对最近一个已归档 episode 触发。

    Returns:
        {"status", "episode_id", "message"}
    """
    _ensure_project_root_on_path()
    try:
        from core.background_learner import get_background_learner

        if not episode_id:
            from engine.adaptive.episode_archiver import load_episodes

            recent = load_episodes(limit=1)
            if not recent:
                raise HTTPException(status_code=404, detail="无可用 episode（归档为空）")
            episode_id = recent[0].get("episode_id")

        learner = get_background_learner()
        return learner.enqueue_learning_task(episode_id, trigger="manual")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"触发学习任务失败: {str(e)}")


@router.get("/adaptive/learning-status")
async def learning_status_endpoint(limit: int = 100, status: Optional[str] = None):
    """查询后台学习任务状态（Stage 3）。

    Args:
        limit: 最多返回条数（默认 100）
        status: 可选状态过滤（queued / running / completed / failed）

    Returns:
        {"tasks": [ {...}, ... ]}
    """
    _ensure_project_root_on_path()
    try:
        from core.background_learner import get_background_learner

        learner = get_background_learner()
        return {"tasks": learner.get_all_task_status(limit=limit, status_filter=status)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询学习状态失败: {str(e)}")

