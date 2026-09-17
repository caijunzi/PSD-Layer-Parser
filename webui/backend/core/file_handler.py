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
# 未匹配样本待标注队列（短期 2）：材质判别没认出来的上传图，逐条 JSONL 追加
UNKNOWN_QUEUE = DATA_DIR / "unknown_samples.jsonl"

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

    # 智能 preset 推荐（材质家族优先，宽高比仅兜底）——返回结构化结果：
    #   matched=True  → 材质/样块真正命中，confidence 可信
    #   matched=False → 按宽高比或兜底推测（confidence 封顶 0.55），需用户确认品类
    rec = _recommend_preset(dimensions, str(stored_path))

    # 短期 2：未匹配样本写入「待标注队列」，供后续人工命名/训练（失败不影响上传）
    if not rec.get("matched"):
        _append_unknown_queue({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "file_id": file_id,
            "filename": original_name,
            "stored_path": str(stored_path),
            "dimensions": dimensions,
            "material_family": rec.get("material_family"),
            "family_conf": rec.get("family_conf"),
            "reason": rec.get("reason"),
            "recommended_preset": rec.get("preset"),
            "confidence": rec.get("confidence"),
        })

    return {
        "file_id": file_id,
        "filename": original_name,
        "size": size,
        "dimensions": dimensions,
        "thumbnail_url": thumbnail_url,
        "recommended_preset": rec["preset"],
        "confidence": rec["confidence"],
        # 2026-09-17：推荐可信度结构化披露（前端据此做分级提示）
        "matched": rec["matched"],
        "recommend_reason": rec["reason"],
        "material_family": rec.get("material_family"),
        "family_conf": rec.get("family_conf"),
    }


def _append_unknown_queue(record: dict) -> None:
    """未匹配样本入队（JSONL 追加；任何失败都不影响上传主流程）。"""
    try:
        UNKNOWN_QUEUE.parent.mkdir(parents=True, exist_ok=True)
        with open(UNKNOWN_QUEUE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:  # pragma: no cover
        print(f"[file_handler] 未匹配样本入队失败（不影响上传）: {e}")


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


#: 材质家族 → 品类 preset（**材质判别远比宽高比可靠**，故优先）
FAMILY_TO_PRESET = {
    "金地屏风": "japanese_screen_gold",
    "绢本工笔": "chinese_ink_landscape_ai",   # README：绢本工笔必须用该 preset（勿用 textile_damask）
    "宣纸水墨": "chinese_ink_landscape_ai",
    "油画布": "western_oil_painting",
    "织物壁布": "textile_damask_photo",       # 壁布/面料实物照（非可平铺数码纹样）
}

#: 家族判别的置信度门槛（`classify_material_family` 的“其他”兜底为 0.3）
_FAMILY_MIN_CONF = 0.6

#: **纺织类**家族：绢本（绢）与织物壁布（布）同属纺织品，仅靠材质判别难以再分。
#: 在此范围内，**「检出凸起实体样块」= 实物样品照**，可据此改判为壁布样品照 preset。
#: 放在这个窄范围里做，才不会把带绫边外框的金地屏风也误判成样品照。
_TEXTILE_FAMILIES = {"绢本工笔", "织物壁布"}


def _recommend_preset(dimensions: Optional[dict],
                      image_path: Optional[str] = None) -> dict:
    """智能推荐：**材质家族优先**，宽高比仅作兜底。

    判据顺序（2026-09-16 修正，顺序本身是关键）：
      1. **材质判别（指纹）→ 家族 → preset**（`FAMILY_TO_PRESET`，需 conf ≥ 0.6）；
      2. 纺织类再问「有没有凸起实体样块」→ 命中则 `textile_damask_photo`；
      3. 材质没认出来 / 检测异常 → 宽高比粗判。

    2026-09-17（短期 1：拒识标记）：返回值从 (preset, confidence) 升级为结构化 dict——
      {
        "preset": str, "confidence": float,
        "matched": bool,          # True=材质/样块真正命中；False=按宽高比或兜底推测
        "reason": str,            # material | panel | material_low_conf | aspect | no_dimensions | material_error
        "material_family": str|None, "family_conf": float|None,
      }
    未命中时 confidence **封顶 0.55**，让前端能以「未识别品类」醒目提示并引导
    用户手动确认，而不是把猜测值伪装成可信推荐。推荐本身仍会给出（不阻断
    上传/处理流程）；是否采信由用户决定。

    ⚠️ 曾踩的坑（勿回退）：早期把「**样块检测**」放在最前并单独决定品类 —— 而
    带绫边外框的**金地屏风**的四条直边恰好满足"长直边持续性"判据，被误判为
    「实物样块」→ 推荐成壁布 preset（实测 `inputs/金地屏风_江户芦雁寒林六曲_绫边装裱.jpg` 中招）。
    故现在样块检测**只用于织物族的置信度增强**，不单独决定品类。
    """
    result = {"preset": None, "confidence": 0.0, "matched": False,
              "reason": "aspect", "material_family": None, "family_conf": None}

    def _set(preset: str, conf: float, matched: bool, reason: str,
             family=None, fconf=None) -> dict:
        result["preset"] = preset
        # 未命中时封顶 0.55：保证前端「未识别」分级必然触发，不伪装可信
        result["confidence"] = (round(float(conf), 2) if matched
                                else min(round(float(conf), 2), 0.55))
        result["matched"] = matched
        result["reason"] = reason
        if family is not None:
            result["material_family"] = family
        if fconf is not None:
            result["family_conf"] = round(float(fconf), 2)
        return result

    def _aspect() -> tuple[str, float]:
        """宽高比粗判（兜底，永远给结果）。"""
        if not dimensions:
            return "japanese_screen_gold", 0.5
        w, h = dimensions["width"], dimensions["height"]
        ratio = w / h if h else 1.0
        if 1.6 <= ratio <= 2.4:
            return "japanese_screen_gold", 0.78
        if 0.85 <= ratio <= 1.18:
            return "textile_damask", 0.72
        return "japanese_screen_gold", 0.55

    family, fconf = None, None
    if image_path:
        try:
            import sys as _sys
            from pathlib import Path as _Path

            root = str(_Path(__file__).resolve().parents[3])   # 项目根
            if root not in _sys.path:
                _sys.path.insert(0, root)
            import cv2 as _cv2
            from engine.core.io_utils import imread_unicode
            from engine.core.sample_panel import detect_sample_panel_bbox
            from engine.adaptive.fingerprint import extract_fingerprint
            from engine.adaptive.material_classifier import classify_material_family

            # 统一走 Unicode 安全读图（中文路径下 cv2.imread 会静默返回 None）
            img = imread_unicode(str(image_path))
            if img is not None:
                family, fconf = classify_material_family(extract_fingerprint(img))
                if (family in FAMILY_TO_PRESET) and float(fconf) >= _FAMILY_MIN_CONF:
                    # 纺织类：再问一句「有没有凸起实体样块」——有则说明是实物样品照
                    if family in _TEXTILE_FAMILIES:
                        gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
                        if detect_sample_panel_bbox(gray) is not None:
                            return _set("textile_damask_photo", 0.85, True, "panel",
                                        family, fconf)
                    return _set(FAMILY_TO_PRESET[family], fconf, True, "material",
                                family, fconf)
                # 材质跑了但置信度不足 / 家族不在映射表（含「其他」）→ 宽高比兜底，
                # 但如实标记未匹配（短期 1 的核心：不再把猜测伪装成可信推荐）
                p, c = _aspect()
                return _set(p, c, False, "material_low_conf", family, fconf)
        except Exception as _e:
            # 检测不可用/失败 → 降级为宽高比判断，绝不影响上传；但**披露**原因，
            # 避免"材质没被识别"这种退化无声无息（P1-1 静默降级）。
            print(f"[file_handler] 材质/样块检测跳过，降级为宽高比推荐：{type(_e).__name__}: {_e}")
            p, c = _aspect()
            return _set(p, c, False, "material_error")

    p, c = _aspect()
    return _set(p, c, False, "no_dimensions" if not dimensions else "aspect")


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
