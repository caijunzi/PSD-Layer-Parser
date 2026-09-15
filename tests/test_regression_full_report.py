"""regression_tester 兼容完整 8 维报告（保留原始 passed / 各维判定）。

- extract_audit_8d 不再丢失原始报告：返回 passed 与 dims_passed。
- run_regression_test：基线通过→当前不通过 视为退化；逐维度通过→失败 视为退化。
"""
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.adaptive.regression_tester import extract_audit_8d, run_regression_test  # noqa: E402


def _make_audit(passed_dims=8, failed_dim_index=None, color_mode="cmyk"):
    dims = {}
    keys = ["① 层属性", "② 分辨率真实性", "③ 实例重复", "④ 内容承载",
            "⑤ 合成等价性", "⑥ 底板纯净度", "⑦ plate 合规", "⑧ manifest 一致"]
    for i, k in enumerate(keys):
        ok = (i < passed_dims) and (failed_dim_index is None or i != failed_dim_index)
        dims[k] = {"passed": ok, "evaluated": True, "na": False,
                   "metrics": {}}
    dims["④ 内容承载"]["metrics"] = {"lost_ratio": 0.01}
    dims["⑤ 合成等价性"]["metrics"] = {"rmse_raw": 20.0, "rmse_lowfreq": 10.0}
    dims["⑦ plate 合规"]["metrics"] = {"plate_purity_ok": True, "tac_max_pct": 300.0}
    dims["① 层属性"]["metrics"] = {"layer_count": 42}
    return {
        "psb": "x.psb", "color_mode": color_mode,
        "passed": failed_dim_index is None,
        "dims": dims,
    }


class TestExtractFullReport(unittest.TestCase):
    def test_extract_preserves_passed_and_dims(self):
        report = _make_audit()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False))
            path = f.name
        try:
            out = extract_audit_8d(path)
            self.assertTrue(out["passed"])
            self.assertEqual(out["lost_ratio"], 0.01)
            self.assertEqual(out["rmse_lowfreq"], 10.0)
            self.assertTrue(out["plate_purity_ok"])
            self.assertEqual(out["n_layers"], 42)
            # 保留原始 8 维判定
            self.assertIn("④ 内容承载", out["dims_passed"])
            self.assertTrue(out["dims_passed"]["④ 内容承载"])
            self.assertTrue(out["dims_passed"]["⑦ plate 合规"])
        finally:
            Path(path).unlink()


class TestRegressionPassed(unittest.TestCase):
    def _run(self, base_audit, cur_audit):
        with tempfile.TemporaryDirectory() as d:
            bp = Path(d) / "base.json"
            cp = Path(d) / "cur.json"
            bp.write_text(json.dumps(base_audit, ensure_ascii=False), encoding="utf-8")
            cp.write_text(json.dumps(cur_audit, ensure_ascii=False), encoding="utf-8")
            # run_regression_test 期望 baseline 文件为 {task_id: 抽取后 audit_8d}（顶层指标）
            base_extracted = extract_audit_8d(str(bp))
            baseline_file = Path(d) / "baseline.json"
            baseline_file.write_text(json.dumps({"task_x": base_extracted}, ensure_ascii=False),
                                    encoding="utf-8")
            return run_regression_test(str(cp), "task_x", str(baseline_file))

    def test_passed_regression_detected(self):
        cur = _make_audit(failed_dim_index=2)  # 某一维失败 → 整体不通过
        res = self._run(_make_audit(), cur)
        self.assertFalse(res["passed"])
        self.assertTrue(any("由通过退化为不通过" in i for i in res["issues"]))

    def test_no_regression_when_both_pass(self):
        res = self._run(_make_audit(), _make_audit())
        self.assertTrue(res["passed"])


if __name__ == "__main__":
    unittest.main()
