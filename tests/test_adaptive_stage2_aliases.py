"""
Stage 2 单元测试：preset 命名空间桥接
验证 category_preset_aliases 表与转换函数正确性

注：DBManager 使用线程本地连接，必须显式 close()，否则 Windows 上
临时库文件被锁、teardown 删除失败。故用 fixture 统一登记并关闭。
"""
import sqlite3
from pathlib import Path

import pytest

from engine.adaptive.db_manager import DBManager
from engine.adaptive.category_selector import (
    select_categories,
    db_category_to_preset_class,
    merge_into_preset_format,
)

ROOT = Path(__file__).resolve().parents[1]


def _init_db(db_path: Path) -> None:
    """执行 migration 001 + 002 的 SQL，建出完整测试库。"""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    for rel in [
        "engine/adaptive/schema.sql",
        "engine/adaptive/seed_data.sql",
        "engine/adaptive/migrations/migration_002_preset_aliases.sql",
        "engine/adaptive/seed_data_002_aliases.sql",
    ]:
        conn.executescript((ROOT / rel).read_text(encoding="utf-8"))
    conn.commit()
    conn.close()


@pytest.fixture
def db_env(tmp_path):
    """返回 (db_path, open_db 工厂)；teardown 时关闭所有连接。"""
    db_path = tmp_path / "test.db"
    _init_db(db_path)
    opened = []

    def open_db() -> DBManager:
        d = DBManager(str(db_path))
        opened.append(d)
        return d

    yield str(db_path), open_db

    for d in opened:
        try:
            d.close()
        except Exception:
            pass


def test_alias_table_integrity(db_env):
    """验证别名表结构与数据完整性"""
    _, open_db = db_env
    db = open_db()

    # 表存在性
    result = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='category_preset_aliases'")
    assert result.fetchone() is not None, "category_preset_aliases 表不存在"

    # 别名数量（20 个类目，部分多别名）
    result = db.execute("SELECT COUNT(*) FROM category_preset_aliases")
    count = result.fetchone()[0]
    assert count >= 20, f"别名数量不足：{count} < 20"

    # 抽样：seal → 印章_Cinnabar_Seal
    result = db.execute(
        "SELECT preset_name FROM category_preset_aliases WHERE category_id = 'seal' ORDER BY priority ASC LIMIT 1"
    )
    row = result.fetchone()
    assert row is not None, "seal 类目无别名"
    assert "印章" in row[0] and "Seal" in row[0], f"seal 别名格式错误：{row[0]}"


def test_view_category_with_aliases(db_env):
    """验证 category_with_aliases 视图"""
    _, open_db = db_env
    db = open_db()

    result = db.execute("""
        SELECT id, name_zh, primary_preset_name
        FROM category_with_aliases
        WHERE primary_preset_name IS NOT NULL
        LIMIT 5
    """)
    rows = result.fetchall()
    assert len(rows) > 0, "视图无数据"

    for cat_id, name_zh, preset_name in rows:
        assert preset_name, f"{cat_id} 无 primary_preset_name"
        assert "_" in preset_name, f"{cat_id} preset_name 格式错误：{preset_name}"


def test_db_category_to_preset_class_with_alias(db_env):
    """验证转换函数优先使用别名"""
    _, open_db = db_env
    db = open_db()

    cat = {
        "category_id": "seal",
        "name_zh": "印章",
        "name_en": "Cinnabar_Seal",
        "prompts": [{"text": "red square seal", "weight": 0.9}],
    }
    result = db_category_to_preset_class(cat, db=db)

    assert result["name"] == "印章_Cinnabar_Seal", f"未使用别名：{result['name']}"
    assert result["prompt"] == "red square seal"
    assert result["category_id"] == "seal"


def test_db_category_to_preset_class_fallback():
    """验证无 db 时降级为 name_zh_name_en"""
    cat = {
        "category_id": "unknown_cat",
        "name_zh": "未知类目",
        "name_en": "Unknown_Category",
        "prompts": [{"text": "test prompt", "weight": 0.8}],
    }
    result = db_category_to_preset_class(cat, db=None)
    assert result["name"] == "未知类目_Unknown_Category", f"降级失败：{result['name']}"


def test_merge_auto_mode_uses_aliases(db_env):
    """验证 auto 模式合并时使用别名"""
    _, open_db = db_env
    db = open_db()

    selected_db = [
        {"category_id": "seal", "name_zh": "印章", "name_en": "Cinnabar_Seal", "prompts": [{"text": "red seal", "weight": 0.9}]},
        {"category_id": "mountains_cliffs", "name_zh": "山石崖壁", "name_en": "Mountains_Cliffs", "prompts": [{"text": "rocky cliff", "weight": 0.85}]},
    ]
    merged = merge_into_preset_format([], selected_db, mode="auto", db=db)

    assert len(merged) == 2
    assert merged[0]["name"] == "印章_Cinnabar_Seal"
    assert merged[1]["name"] == "山石崖壁_Mountains_Cliffs"


def test_merge_hybrid_mode_normalizes_layer_name_and_resolves_id(db_env):
    """preset 只有 layer_name 时，仍应补 name 与规范 category_id。"""
    _, open_db = db_env
    db = open_db()

    preset_classes = [
        {
            "label_cn": "印章",
            "layer_name": "印章_Cinnabar_Seal",
            "prompt": "red seal",
            "region": [0.8, 0.8, 0.95, 0.95],
        },
    ]
    merged = merge_into_preset_format(preset_classes, [], mode="hybrid", db=db)

    assert merged[0]["name"] == "印章_Cinnabar_Seal"
    assert merged[0]["layer_name"] == "印章_Cinnabar_Seal"
    assert merged[0]["category_id"] == "seal"
    assert "region" in merged[0]


def test_merge_hybrid_mode_preserves_preset_metadata(db_env):
    """验证 hybrid 模式保留 preset 元数据"""
    _, open_db = db_env
    db = open_db()

    preset_classes = [
        {"name": "印章_Cinnabar_Seal", "prompt": "red seal", "region": [0.8, 0.8, 0.95, 0.95]},
    ]
    selected_db = [
        {"category_id": "seal", "name_zh": "印章", "name_en": "Cinnabar_Seal", "prompts": [{"text": "red seal", "weight": 0.9}]},
        {"category_id": "mountains_cliffs", "name_zh": "山石崖壁", "name_en": "Mountains_Cliffs", "prompts": [{"text": "rocky cliff", "weight": 0.85}]},
    ]
    merged = merge_into_preset_format(preset_classes, selected_db, mode="hybrid", db=db)

    assert len(merged) == 2
    assert merged[0]["name"] == "印章_Cinnabar_Seal"
    assert "region" in merged[0], "preset 元数据丢失"
    assert merged[1]["name"] == "山石崖壁_Mountains_Cliffs"


def test_e2e_auto_mode_produces_valid_names(db_env):
    """端到端：auto 模式产出的 name 可被下游识别"""
    db_path, open_db = db_env

    fingerprint = {"global_features": [0.5] * 55, "texture_features": [0.3] * 24, "structure_features": [0.2] * 5}
    selected = select_categories(fingerprint, "金地屏风", db_path, mode="auto", preset_categories=None)
    assert len(selected) > 0, "auto 模式未选出类目"

    db = open_db()
    merged = merge_into_preset_format([], selected, mode="auto", db=db)

    for c in merged:
        assert "_" in c["name"], f"name 格式错误：{c['name']}"
        assert c.get("prompt"), f"{c['name']} 缺 prompt"
