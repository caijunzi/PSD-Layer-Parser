"""文件操作：上传保存、缩略图、历史记录。

所有运行时数据落在 webui/data/ 下，与开发目录分离。
"""
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image

# 数据根目录（backend/.. /data）
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
THUMBS_DIR = DATA_DIR / "thumbnails"
HISTORY_FILE = DATA_DIR / "history.json"

# 项目根（用于调用引擎与读 presets）
PROJECT_ROOT = DATA_DIR.parent.parent
PRESETS_DIR = PROJECT_ROOT / "presets"
ENGINE_ENTRY = PROJECT_ROOT / "run_universal_engine.py"

# 允许的图片格式与大小
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".psd", ".psb"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


def ensure_dirs() -> None:
    """确保运行时数据目录存在（main.py 启动时也会调用一次）。"""
    for d in (UPLOADS_DIR, OUTPUTS_DIR, THUMBS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def is_allowed_file(filename: str, size: int) -> bool:
    """前端基础校验：扩展名 + 大小。"""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTS and size <= MAX_FILE_SIZE


def save_upload(src_path: str, original_name: str) -> dict:
    """保存上传文件，生成 file_id 与缩略图，返回上传响应。

    返回: {file_id, filename, size, dimensions, thumbnail_url, recommended_preset, confidence}
    """
    ensure_dirs()
    ext = Path(original_name).suffix.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"
    stored_name = f"{file_id}{ext}"
    stored_path = UPLOADS_DIR / stored_name
    shutil.move(src_path, stored_path)

    size = stored_path.stat().st_size

    # 读尺寸（PSD/PSB 用 PIL 可解析；失败时 dimensions=None）
    dimensions = None
    try:
        with Image.open(stored_path) as im:
            dimensions = {"width": im.width, "height": im.height}
    except Exception:
        # PSD 可能 Pillow 无法直接 open，尝试前几字节判断
        dimensions = None

    # 生成缩略图（仅光栅图，PSD 跳过）
    thumbnail_url = None
    if ext in (".jpg", ".jpeg", ".png"):
        thumbnail_url = _make_thumbnail(stored_path, file_id)

    # 智能 preset 推荐（MVP：按宽高比粗判）
    rec_preset, confidence = _recommend_preset(dimensions)

    return {
        "file_id": file_id,
        "filename": original_name,
        "size": size,
        "dimensions": dimensions,
        "thumbnail_url": thumbnail_url,
        "recommended_preset": rec_preset,
        "confidence": confidence,
    }


def _make_thumbnail(src_path: Path, file_id: str) -> str:
    """生成 800px 宽 JPEG 缩略图，返回相对 URL。"""
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w > 800:
                nh = int(h * 800 / w)
                im = im.resize((800, nh), Image.LANCZOS)
            thumb_name = f"{file_id}.jpg"
            im.save(THUMBS_DIR / thumb_name, "JPEG", quality=85)
        return f"/thumbnails/{thumb_name}"
    except Exception:
        return None


def _recommend_preset(dimensions: Optional[dict]) -> tuple[str, float]:
    """MVP 智能推荐：按宽高比粗判品类。

    - 宽高比 ≈ 2:1（横向长卷/屏风）→ japanese_screen_gold
    - 接近 1:1（方阵纹样）→ textile_damask
    - 其余 → japanese_screen_gold（默认）
    返回 (preset_name, confidence)
    """
    if not dimensions:
        return "japanese_screen_gold", 0.5
    w, h = dimensions["width"], dimensions["height"]
    ratio = w / h if h else 1.0
    if 1.6 <= ratio <= 2.4:
        return "japanese_screen_gold", 0.78
    if 0.85 <= ratio <= 1.18:
        return "textile_damask", 0.72
    return "japanese_screen_gold", 0.55


def append_history(item: dict) -> None:
    """追加一条历史记录到 history.json。"""
    ensure_dirs()
    history = read_history()
    history.append(item)
    HISTORY_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_history() -> list:
    """读取全部历史记录（按创建时间倒序）。"""
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return sorted(data, key=lambda x: x.get("created_at", ""), reverse=True)
    except Exception:
        return []


def list_presets() -> list[dict]:
    """扫描 presets/ 目录，返回 preset 清单。"""
    if not PRESETS_DIR.exists():
        return []
    result = []
    for p in sorted(PRESETS_DIR.glob("*.json")):
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        classes = cfg.get("ai_semantic_classes", [])
        mode = (cfg.get("output") or {}).get("mode", "design")
        result.append({
            "name": cfg.get("preset_name", p.stem),
            "display_name": cfg.get("display_name", cfg.get("description", p.stem)),
            "description": cfg.get("description", ""),
            "semantic_classes": len(classes) if isinstance(classes, list) else 0,
            "output_modes": [mode] if mode != "both" else ["design", "plate"],
        })
    return result
