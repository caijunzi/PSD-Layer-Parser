"""
自进化语义匹配机制 - 引擎端模块入口

提供：
- 图级指纹提取（fingerprint.py）
- 材质家族判别（material_classifier.py）
- 类目选择器（category_selector.py）
- 数据库管理（db_manager.py）
- episode 归档（episode_archiver.py）

环境变量控制：
  preset["mode"]=locked|auto|hybrid 决定本机制是否参与主流程（默认 locked = 不启用）
  ADAPTIVE_DB_PATH=webui/data/adaptive_semantics.db（默认词库路径）
  注：早期设想的 ADAPTIVE_MODE=stage1..stage5 已被 preset.mode 取代，保留环境变量仅作展示。
"""

__version__ = "1.0.0-stage1"

from .fingerprint import extract_fingerprint
from .material_classifier import classify_material_family
from .category_selector import select_categories
from .db_manager import DBManager
from .episode_archiver import archive_episode

__all__ = [
    "extract_fingerprint",
    "classify_material_family",
    "select_categories",
    "DBManager",
    "archive_episode",
]
