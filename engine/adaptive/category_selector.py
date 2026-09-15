"""
类目选择器模块

根据材质家族、指纹特征、preset 模式，从词库选出适配的语义类目清单
"""
import re
import numpy as np
from typing import List, Dict, Optional, Any
from .db_manager import DBManager


def select_categories(
    fingerprint: Dict,
    material_family: str,
    db_path: str,
    mode: str = "auto",
    preset_categories: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    从词库按材质亲和度 Top-K 选类目
    
    Args:
        fingerprint: extract_fingerprint 返回的字典
        material_family: 材质家族（如"宣纸水墨"）
        db_path: 数据库路径
        mode: 选择模式
            - "locked": 直接使用 preset_categories，不查库
            - "auto": 完全由材质亲和度决定（Top-50）
            - "hybrid": preset_categories + 自动补充 Top-20
        preset_categories: 预设类目列表（mode=locked/hybrid 时必需）
    
    Returns:
        类目列表，每项包含：
        {
            "category_id": str,
            "name_zh": str,
            "name_en": str,
            "name_template": str,
            "prompts": [{"text": str, "weight": float, "lang": str}, ...],
            "area_budget": (float, float) | None,
            "confidence": float,
            "affinity": float,
        }
    """
    # 模式验证
    if mode not in ("locked", "auto", "hybrid"):
        raise ValueError(f"无效的 mode: {mode}，仅支持 locked/auto/hybrid")
    
    # locked 模式：直接返回 preset
    if mode == "locked":
        if preset_categories is None:
            raise ValueError("locked 模式需要提供 preset_categories")
        return _build_categories_from_preset(preset_categories, db_path)
    
    # 连接数据库
    db = DBManager(db_path)
    
    # auto 模式：按材质亲和度 Top-50
    if mode == "auto":
        selected_ids = _select_by_affinity(db, material_family, limit=50)
    
    # hybrid 模式：preset + 自动补充 Top-20
    elif mode == "hybrid":
        if preset_categories is None:
            raise ValueError("hybrid 模式需要提供 preset_categories")
        
        # 先用 preset
        preset_ids = set(preset_categories)
        
        # 补充高亲和度类目（去重）
        auto_ids = _select_by_affinity(db, material_family, limit=20)
        selected_ids = list(preset_ids) + [cid for cid in auto_ids if cid not in preset_ids]
    
    # 强制保留核心类目（印章/题跋/修复痕迹）
    selected_ids = _ensure_core_categories(selected_ids)
    
    # 构建完整类目信息
    categories = []
    for cat_id in selected_ids:
        cat_info = _build_category_info(db, cat_id)
        if cat_info:
            categories.append(cat_info)
    
    db.close()
    return categories


def _select_by_affinity(db: DBManager, material_family: str, limit: int) -> List[str]:
    """
    按材质亲和度选择类目 ID（Top-K）
    """
    rows = db.get_categories_by_material(material_family, limit=limit, min_affinity=0.3)
    return [row["category_id"] for row in rows]


def _ensure_core_categories(category_ids: List[str]) -> List[str]:
    """
    强制保留核心类目（印章/题跋/修复痕迹）
    """
    core = ["seal", "calligraphy", "repair_marks"]
    result = list(category_ids)
    
    for cid in core:
        if cid not in result:
            result.insert(0, cid)  # 插入开头（优先检测）
    
    return result


def resolve_canonical_category_id(
    db: Optional[DBManager],
    name: str,
) -> Optional[str]:
    """把 preset 类目名解析为 DB 规范 category_id（仅当存在真实映射，绝不猜不存在的 ID）。

    解析优先级（2026-09-15 修复「DB 规范 ID 与层名混淆 → 学习零建议」）：
      1) ``category_preset_aliases.preset_name`` 精确匹配（完整名 / 去数字前缀后的基名）；
      2) ``categories.name_zh / name_en / name_template`` 精确匹配（完整名 / 基名）；
      3) 整词匹配：把基名按 ``_`` 拆词，命中某 category 的 name_zh / name_en 整词
         （词频打分取最高，≥1 才采纳）。

    该解析只返回**真实存在的** category_id；任何情况下都不会臆造 ID。

    Args:
        db:   DBManager 实例（为 None 时直接返回 None）
        name: preset 类目名（如 "08_芦雁群禽_Geese_Flock"）

    Returns:
        规范 category_id 字符串，或 None（无映射）
    """
    if not name or db is None:
        return None
    base = re.sub(r"^\d+[A-Da-d]?_", "", str(name))

    # 1) 别名表精确匹配（完整名 / 基名）
    for key in (name, base):
        try:
            row = db.execute(
                "SELECT category_id FROM category_preset_aliases WHERE preset_name = ?",
                (key,),
            ).fetchone()
        except Exception:
            row = None
        if row:
            return row[0]

    # 2) categories 的 name_zh / name_en / name_template 精确匹配
    for key in (name, base):
        try:
            row = db.execute(
                "SELECT id FROM categories WHERE name_zh = ? OR name_en = ? OR name_template = ?",
                (key, key, key),
            ).fetchone()
        except Exception:
            row = None
        if row:
            return row[0]

    # 3) 整词匹配（name_zh / name_en 拆词后与基名词集交集打分）
    tokens = {t for t in re.split(r"[_\s]+", base) if t}
    if not tokens:
        return None
    try:
        cur = db.execute("SELECT id, name_zh, name_en FROM categories")
        rows = cur.fetchall()
    except Exception:
        return None
    best_cid = None
    best_score = 0
    for cid, nz, ne in rows:
        score = 0
        if nz and nz in tokens:
            score += 1
        if ne:
            ne_tokens = {t for t in re.split(r"[_\s]+", ne) if t}
            score += len(ne_tokens & tokens)
        if score > best_score:
            best_score = score
            best_cid = cid
    return best_cid if best_score >= 1 else None


def _attach_canonical_id(cls: Dict[str, Any], db: Optional[DBManager]) -> Dict[str, Any]:
    """给单个 preset 类目字典补规范 category_id，并采用已学习的 DB prompt 选择。

    - 若 cls 已有 category_id 则保留；否则经 ``resolve_canonical_category_id`` 解析。
    - 若解析到规范 ID 且 DB 中存在真实提示词（排除 shadow），采用**权重最高**的
      DB 提示词作为该类目的 prompt（「采用已学习的 DB prompt 选择」）。
    - 保留 preset 原有的 region / gate / instance_split 等元数据（仅补充 ID 与 prompt，
      不覆盖检测元数据）。

    不臆造 ID：解析失败则 category_id 保持缺省（不写入假 ID）。
    """
    cid = cls.get("category_id")
    if not cid and db is not None:
        cid = resolve_canonical_category_id(
            db,
            cls.get("name") or cls.get("layer_name") or cls.get("label_cn", ""),
        )
        if cid:
            cls["category_id"] = cid

    # 采用已学习的 DB prompt 选择（仅当 DB 确有真实提示词）
    if cid and db is not None:
        try:
            prompts = db.get_category_prompts(cid, min_weight=0.0)
            real = [p for p in prompts if p.get("source") != "shadow"]
        except Exception:
            real = []
        if real:
            learned = db.execute(
                "SELECT prompt, weight FROM category_prompts WHERE category_id = ? "
                "AND lang = 'en' AND (n_accept > 0 OR n_reject > 0) ORDER BY weight DESC, prompt",
                (cid,),
            ).fetchall()
            if learned:
                cls["prompt"] = learned[0]["prompt"]
            cls["db_prompts"] = [
                {"text": p["prompt"], "weight": p["weight"], "lang": p["lang"]}
                for p in real
            ]
    return cls


def _build_category_info(db: DBManager, category_id: str) -> Optional[Dict[str, Any]]:
    """
    构建单个类目的完整信息（含 prompts / area_budget / confidence）
    """
    # 查询类目元信息
    cur = db.execute(
        "SELECT name_zh, name_en, name_template, confidence FROM categories WHERE id = ?",
        (category_id,)
    )
    row = cur.fetchone()
    if row is None:
        return None
    
    # 查询提示词
    prompt_rows = db.get_category_prompts(category_id, min_weight=0.3)
    prompts = [
        {"text": p["prompt"], "weight": p["weight"], "lang": p["lang"]}
        for p in prompt_rows
    ]
    
    # 查询形状先验
    priors = db.get_category_priors(category_id)
    area_budget = None
    if priors and priors["area_budget_min"] is not None:
        area_budget = (priors["area_budget_min"], priors["area_budget_max"])
    
    # 查询材质亲和度（用于排序/过滤）
    cur = db.execute(
        "SELECT affinity FROM category_material_affinity WHERE category_id = ? LIMIT 1",
        (category_id,)
    )
    affinity_row = cur.fetchone()
    affinity = affinity_row["affinity"] if affinity_row else 0.5
    
    return {
        "category_id": category_id,
        "name_zh": row["name_zh"],
        "name_en": row["name_en"],
        "name_template": row["name_template"],
        "prompts": prompts,
        "area_budget": area_budget,
        "confidence": row["confidence"],
        "affinity": affinity,
    }


def _build_categories_from_preset(preset_ids: List[str], db_path: str) -> List[Dict[str, Any]]:
    """
    从 preset 类目 ID 构建完整类目信息（locked 模式专用）
    """
    db = DBManager(db_path)
    categories = []
    
    for cat_id in preset_ids:
        cat_info = _build_category_info(db, cat_id)
        if cat_info:
            categories.append(cat_info)
    
    db.close()
    return categories


def merge_semantic_classes(
    preset_classes: List[str],
    selected_categories: List[Dict[str, Any]],
    mode: str
) -> List[str]:
    """
    合并 preset 与自动选出的类目（供主流程调用）
    
    Args:
        preset_classes: 预设类目 ID 列表
        selected_categories: select_categories 返回的类目列表
        mode: locked/auto/hybrid
    
    Returns:
        合并后的类目 ID 列表
    """
    if mode == "locked":
        return preset_classes
    elif mode == "auto":
        return [cat["category_id"] for cat in selected_categories]
    elif mode == "hybrid":
        # hybrid 已在 select_categories 内合并，直接返回
        return [cat["category_id"] for cat in selected_categories]
    else:
        raise ValueError(f"无效的 mode: {mode}")


def db_category_to_preset_class(cat: Dict[str, Any], db: Optional[DBManager] = None) -> Dict[str, Any]:
    """
    将 DB 类目（select_categories 返回格式）转换为 preset ai_semantic_classes 兼容格式。

    preset 的 ai_semantic_classes 是 SSOT，下游 grounded_sam_provider.segment_objects
    按 `name` / `prompt` 键做图层命名与检测；DB 类目使用 name_zh/name_en/prompts，需桥接。
    
    Stage 2：优先查询 category_preset_aliases 表，获取符合 preset 约定的别名。
    """
    # Stage 2：查询 preset 别名（优先级最高）
    name = None
    if db and cat.get("category_id"):
        try:
            result = db.execute(
                "SELECT preset_name FROM category_preset_aliases WHERE category_id = ? ORDER BY priority ASC LIMIT 1",
                (cat["category_id"],)
            )
            row = result.fetchone()
            if row:
                name = row[0]
        except Exception:
            pass  # 降级到 name_zh_name_en
    
    # 降级：name_zh_name_en 拼接
    if not name:
        name_zh = cat.get("name_zh") or ""
        name_en = cat.get("name_en") or ""
        name = f"{name_zh}_{name_en}" if name_en else (name_zh or cat.get("category_id") or "")

    prompts = cat.get("prompts") or []
    prompt = prompts[0].get("text", "") if prompts else ""

    out: Dict[str, Any] = {
        "name": name,
        "layer_name": name,
        "prompt": prompt,
        "source": "adaptive_db",
    }
    if cat.get("category_id") is not None:
        out["category_id"] = cat["category_id"]
    if cat.get("affinity") is not None:
        out["affinity"] = cat["affinity"]
    if cat.get("confidence") is not None:
        out["confidence"] = cat["confidence"]
    return out


def merge_into_preset_format(
    preset_classes: List[Dict[str, Any]],
    selected_db: List[Dict[str, Any]],
    mode: str,
    db: Optional[DBManager] = None,
    min_affinity: float = 0.6,
    max_supplement: int = 3,
    preset_full: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    将 DB 选出的类目合并进 preset 的 ai_semantic_classes（保持 preset 为 SSOT）。

    - mode="auto":   用 DB 类目整体替换（preset 视为未配置自适应类目）
    - mode="hybrid": 保留 preset 原有类目（含 region/instance_split/blob_instances 等
                      元数据），仅**限量**补充 DB 中未覆盖的类目（按 name_zh 子串去重）
    - mode="locked": 调用方不应进入此函数（默认不启用自适应，preset 原样保留）

    hybrid 护栏（2026-09-15，端到端实测后加固）：
      ① 去重在**别名解析之后**进行（比对最终落盘名，而非 name_zh）；
      ② 英文词元重叠视为语义重复（*_cliffs / *_trees / *_brocade_outer_frame…）；
      ③ 提供 preset_full 时，去重语料扩展到**整个 preset 文本**，覆盖 10A/10B
         等不在 ai_semantic_classes 中的支撑/装饰层；
      ④ 亲和度门槛 min_affinity + 数量上限 max_supplement。
    实测教训：未加固前 hybrid 曾把 10 个词库泛类灌入旗舰 preset
    （其中 calligraphy/brocade_frame 与 preset 已有层重复），DINO 提示词翻倍、
    Step1 从 ~102s 涨到 225s。

    Args:
        preset_classes: 原始 preset ai_semantic_classes（list of dict）
        selected_db:    select_categories(..., mode="auto") 返回的 DB 类目
        mode:           "auto" / "hybrid"
        db:             DBManager 实例（用于查询 preset 别名）
        min_affinity:   hybrid 补充类目的最低亲和度门槛
        max_supplement: hybrid 最多补充的类目数
        preset_full:    完整 preset dict（把去重语料扩展到全部图层名/映射）

    Returns:
        list of dict（preset 格式：至少含 name / prompt）
    """
    if mode == "auto":
        return [db_category_to_preset_class(c, db) for c in selected_db]

    # hybrid：保留 preset，限量补充未覆盖的高亲和度 DB 类目
    preset_names_lower = [(pc.get("name") or "").lower() for pc in preset_classes]
    preset_tokens: set = set()
    for pn in preset_names_lower:
        preset_tokens |= _english_tokens(pn)
    # 去重语料扩展到整个 preset 文本（覆盖 10A/10B 等支撑/装饰层）
    corpus_lower = ""
    if preset_full:
        import json as _json

        corpus_lower = _json.dumps(preset_full, ensure_ascii=False).lower()
        preset_tokens |= _english_tokens(corpus_lower)

    # 先给 preset 原有类目补规范 category_id（并采用已学习的 DB prompt 选择），
    # 保留 region / gate / instance_split 等元数据。
    result = []
    for pc in preset_classes:
        # preset 的 SSOT 历史上同时存在 name/layer_name/label_cn 三种命名键；
        # 下游统一使用 name，保留原 layer_name 作为检测追溯字段。
        enriched = _attach_canonical_id(pc, db)
        if not enriched.get("name"):
            enriched["name"] = (
                enriched.get("layer_name")
                or enriched.get("label_cn")
                or enriched.get("category_id")
                or ""
            )
        result.append(enriched)
    supplemented = 0
    for c in selected_db:
        # 先把 DB 类目解析成 preset 命名（含别名表），再做去重——
        # 修复（2026-09-15）：原实现用 name_zh 去重，但最终落盘名来自别名表，
        # 导致别名与 preset 同名的类目（如 calligraphy→"题跋落款墨书_Calligraphy_Inscription"
        # 与 preset 09B 同名）漏网，注入重复层。
        resolved = db_category_to_preset_class(c, db)
        rname = (resolved.get("name") or "").lower()
        nz = (c.get("name_zh") or "").lower()

        duplicate = bool(rname and corpus_lower and rname in corpus_lower)
        if not duplicate:
            for pn in preset_names_lower:
                if not pn:
                    continue
                if rname and (rname in pn or pn in rname):
                    duplicate = True
                    break
                if nz and nz in pn:
                    duplicate = True
                    break
        # 英文词元重叠（如同为 *_cliffs / *_trees）：视为语义重复，跳过
        if not duplicate and (preset_tokens & _english_tokens(rname)):
            duplicate = True
        if duplicate:
            continue  # 已被 preset 覆盖，跳过避免重复检测

        aff = c.get("affinity")
        if aff is not None and float(aff) < min_affinity:
            continue  # 亲和不达标（仅在亲和度已知时过滤），不补充
        if supplemented >= max_supplement:
            break
        result.append(resolved)
        supplemented += 1
    return result


def _english_tokens(name: str) -> set:
    """提取名称中的英文词元（去掉序号与中文），用于语义去重。

    例："04a_前景墨岩峭壁_foreground_dark_cliffs" → {"foreground","dark","cliffs"}
        "山石崖壁_mountains_cliffs"               → {"mountains","cliffs"}
    """
    import re as _re

    tokens = set()
    for seg in _re.findall(r"[A-Za-z_]{3,}", str(name).lower()):
        for t in seg.split("_"):
            if len(t) >= 4:
                tokens.add(t)
    return tokens
