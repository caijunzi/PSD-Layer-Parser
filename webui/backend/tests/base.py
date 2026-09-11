"""WebUI 后端测试公共基类。

职责：
1. 把 webui/backend 加入 sys.path（保证 core/api/ws/main 可导入）
2. 每个测试把数据目录 patch 到临时目录，零污染真实 webui/data
3. 提供 TestClient 单例

注意：绝不真跑引擎——process 正常路径测试 patch 掉 task_manager.execute。
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402


class BaseWebUITest(unittest.TestCase):
    """所有 API 测试的基类。"""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        # 临时数据目录（测试结束自动清理）
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)

        overrides = {
            "core.file_handler.DATA_DIR": tmp,
            "core.file_handler.UPLOADS_DIR": tmp / "uploads",
            "core.file_handler.OUTPUTS_DIR": tmp / "outputs",
            "core.file_handler.THUMBS_DIR": tmp / "thumbnails",
            "core.file_handler.HISTORY_FILE": tmp / "history.json",
            # task_manager 用 from import 绑定了这些名字，需单独 patch
            "core.task_manager.UPLOADS_DIR": tmp / "uploads",
            "core.task_manager.OUTPUTS_DIR": tmp / "outputs",
        }
        for target, value in overrides.items():
            patcher = mock.patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        for sub in ("uploads", "outputs", "thumbnails"):
            (tmp / sub).mkdir(parents=True, exist_ok=True)
