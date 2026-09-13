"""Adaptive Semantics API - 自适应语义统计与配置端点

提供自适应语义机制的当前状态、版本信息、类目统计等查询接口。
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
import os
from pathlib import Path

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
