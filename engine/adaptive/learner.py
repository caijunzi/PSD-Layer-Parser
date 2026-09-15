"""
学习器核心逻辑（Semantic Learner）—— Stage 3（反馈闭环 + 影子进化）

职责：
- 从 episode 归档中提取归因信息（调用 attribution.py）
- 对 reject 的 prompt 进行贝叶斯权重更新（降权）
- 对 accept 的 prompt 进行权重增强（升权）
- 跑回归测试验证 8 维指标不退化（调用 regression_tester.py）
- 异步执行（不阻塞主流程）

设计原则：
- 贝叶斯更新公式：P(good|reject) = P(reject|good) * P(good) / P(reject)
- 权重更新策略：reject → 降权 0.8×，accept → 升权 1.1×
- 回归测试阈值：核心指标（lost_ratio, rmse_lowfreq）退化 > 5% 视为失败
- 学习器不直接修改 preset 文件，而是生成"推荐调整"JSON 供人工审核
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from engine.adaptive.attribution import attribute_reject_to_prompt, extract_accept_reject_signals
from engine.adaptive.regression_tester import run_regression_test


class SemanticLearner:
    """
    语义学习器：从 episode 归档中学习，调整 prompt 权重，生成推荐调整。

    职责：
    1. 从 episode 归档中提取归因信息
    2. 对 reject 的 prompt 进行贝叶斯权重更新（降权）
    3. 对 accept 的 prompt 进行权重增强（升权）
    4. 跑回归测试验证 8 维指标不退化
    5. 生成推荐调整 JSON 供人工审核

    不修改 preset 文件，只生成推荐。
    """

    def __init__(
        self,
        baseline_path: str = "tests/baseline_audit_8d.json",
        reject_penalty: float = 0.8,
        accept_boost: float = 1.1,
        regression_threshold: float = 0.05,
        weight_floor: float = 0.05,
        weight_ceiling: float = 5.0,
        db_path: Optional[str] = None,
    ):
        """
        初始化学习器。

        Args:
            baseline_path: 回归基线数据路径
            reject_penalty: reject 时的降权因子（默认 0.8，即降权 20%）
            accept_boost: accept 时的升权因子（默认 1.1，即升权 10%）
            regression_threshold: 回归测试阈值（默认 5%）
            weight_floor: 权重下限（默认 0.05，防止指数衰减到 0）
            weight_ceiling: 权重上限（默认 5.0，防止指数放大失控）
            db_path: 自适应语义库路径（用于读取真实旧权重）。为 None 时按
                     ADAPTIVE_DB_PATH 环境变量解析；仍为空则回退到默认权重 1.0。
        """
        self.baseline_path = baseline_path
        self.reject_penalty = reject_penalty
        self.accept_boost = accept_boost
        self.regression_threshold = regression_threshold
        self.weight_floor = weight_floor
        self.weight_ceiling = weight_ceiling
        self.db_path = db_path

    def _clamp(self, weight: float) -> float:
        """把权重限制在 [floor, ceiling]，避免指数更新溢出。"""
        return max(self.weight_floor, min(self.weight_ceiling, weight))

    def learn_from_episode(
        self,
        episode: Dict[str, Any],
        db_mgr: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        从单个 episode 中学习，生成 prompt 权重调整建议（影子建议，不自动应用）。

        设计原则：
        - 仅产出「推荐调整」JSON，绝不修改 preset / 自动激活模型（影子建议）。
        - 读取**真实 DB 旧权重**作为贝叶斯更新基线；若该 (category_id, prompt)
          在库中不存在（未知检测类目），跳过，不写假成功。

        Args:
            episode: episode 归档 JSON（包含 dino_detections 字段）
            db_mgr: 可选 DBManager 实例（由调用方复用，避免重复建连）。
                    为 None 时按 self.db_path / ADAPTIVE_DB_PATH 解析，仍为空则回退 1.0。

        Returns:
            {
                "episode_id": str,
                "timestamp": str,
                "attributions": [(prompt, category_id, signal, reason), ...],
                "weight_adjustments": {category_id: {prompt: {...}}},
                "unknown_detections": [str, ...]  # 库中不存在、被跳过的不明检测类目
            }
        """
        episode_id = episode.get("episode_id", "unknown")
        timestamp = datetime.now().isoformat()

        # 1. 提取归因信息
        attributions = attribute_reject_to_prompt(episode)

        # 2. 提取 accept/reject 信号统计
        signals = extract_accept_reject_signals(episode)

        # 3. 解析 DB（用于读取真实旧权重 / 识别未知类目）
        own_conn = False
        if db_mgr is None and self.db_path:
            db_path = self.db_path
        elif db_mgr is None:
            db_path = os.environ.get("ADAPTIVE_DB_PATH")
        else:
            db_path = None

        if db_mgr is None and db_path:
            try:
                from engine.adaptive.db_manager import create_db_manager
                db_mgr = create_db_manager(db_path)
                own_conn = True
            except Exception:
                db_mgr = None

        try:
            # 4. 计算权重调整建议（影子建议，不写库）
            weight_adjustments = {}
            unknown_detections = []

            for category_id, cat_signals in signals.items():
                for prompt, prompt_signals in cat_signals.get("prompts", {}).items():
                    n_accept = prompt_signals.get("n_accept", 0)
                    n_quality_reject = prompt_signals.get("n_quality_reject", 0)
                    n_diffuse_reject = prompt_signals.get("n_diffuse_reject", 0)
                    n_total_reject = n_quality_reject + n_diffuse_reject

                    if n_total_reject == 0 and n_accept == 0:
                        continue

                    # 读取真实 DB 旧权重；未知 (category_id, prompt) → 跳过，不写假成功
                    if db_mgr is not None:
                        old_weight = db_mgr.get_prompt_weight(category_id, prompt)
                        if old_weight is None:
                            unknown_key = f"{category_id}:{prompt}"
                            if unknown_key not in unknown_detections:
                                unknown_detections.append(unknown_key)
                            continue
                    else:
                        # 无 DB 可用（如离线单元测试）：回退到默认基线权重 1.0
                        old_weight = 1.0

                    # 贝叶斯权重更新：reject 降权，accept 升权
                    new_weight = old_weight
                    if n_total_reject > 0:
                        # 降权：每次 reject 乘以 reject_penalty
                        new_weight *= (self.reject_penalty ** n_total_reject)
                        signal = "reject"
                        reason = f"{n_quality_reject} 质量门拒绝 + {n_diffuse_reject} 弥散门拒绝"
                    else:
                        # 升权：每次 accept 乘以 accept_boost
                        new_weight *= (self.accept_boost ** n_accept)
                        signal = "accept"
                        reason = f"{n_accept} 次成功检测"

                    new_weight = self._clamp(new_weight)

                    # 仅在确有建议时登记该 (category_id, prompt)，避免留下空类目键
                    weight_adjustments.setdefault(category_id, {})[prompt] = {
                        "old_weight": old_weight,
                        "new_weight": new_weight,
                        "signal": signal,
                        "reason": reason
                    }
        finally:
            if own_conn and db_mgr is not None:
                db_mgr.close()

        return {
            "episode_id": episode_id,
            "timestamp": timestamp,
            "attributions": [
                {
                    "prompt": attr[0],
                    "category_id": attr[1],
                    "signal": attr[2],
                    "reason": attr[3]
                }
                for attr in attributions
            ],
            "weight_adjustments": weight_adjustments,
            "unknown_detections": unknown_detections,
        }

    def learn_from_episodes(
        self,
        episodes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        从多个 episode 中批量学习，聚合权重调整建议。

        Args:
            episodes: episode 归档 JSON 列表

        Returns:
            {
                "timestamp": str,
                "n_episodes": int,
                "aggregated_adjustments": {
                    "category_id": {
                        "prompt": {
                            "old_weight": float,
                            "new_weight": float,
                            "n_accept": int,
                            "n_reject": int
                        }
                    }
                }
            }
        """
        timestamp = datetime.now().isoformat()

        # 聚合多个 episode 的权重调整（复用单一 DB 连接读取真实旧权重）
        db_mgr = None
        if self.db_path or os.environ.get("ADAPTIVE_DB_PATH"):
            try:
                from engine.adaptive.db_manager import create_db_manager
                db_path = self.db_path or os.environ.get("ADAPTIVE_DB_PATH")
                db_mgr = create_db_manager(db_path)
            except Exception:
                db_mgr = None

        try:
            aggregated = {}

            for episode in episodes:
                learning_result = self.learn_from_episode(episode, db_mgr=db_mgr)

                for category_id, prompts in learning_result["weight_adjustments"].items():
                    if category_id not in aggregated:
                        aggregated[category_id] = {}

                    for prompt, adjustment in prompts.items():
                        if prompt not in aggregated[category_id]:
                            aggregated[category_id][prompt] = {
                                "old_weight": adjustment["old_weight"],
                                "new_weight": adjustment["new_weight"],
                                "n_accept": 0,
                                "n_reject": 0
                            }
                        else:
                            # 累乘权重调整（并 clamp，防止跨 episode 累积溢出）
                            aggregated[category_id][prompt]["new_weight"] = self._clamp(
                                aggregated[category_id][prompt]["new_weight"]
                                * (adjustment["new_weight"] / adjustment["old_weight"])
                            )

                        # 统计 accept/reject 次数
                        if adjustment["signal"] == "accept":
                            aggregated[category_id][prompt]["n_accept"] += 1
                        elif adjustment["signal"] == "reject":
                            aggregated[category_id][prompt]["n_reject"] += 1
        finally:
            if db_mgr is not None:
                db_mgr.close()

        return {
            "timestamp": timestamp,
            "n_episodes": len(episodes),
            "aggregated_adjustments": aggregated
        }

    def run_regression_and_learn(
        self,
        current_audit_path: str,
        task_id: str,
        episode: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        跑回归测试 + 学习闭环：验证当前任务的审计 8 维不退化，然后从 episode 中学习。

        Args:
            current_audit_path: 当前任务的 result.audit.json 路径
            task_id: 任务 ID（用于查找基线）
            episode: episode 归档 JSON

        Returns:
            {
                "regression_passed": bool,
                "regression_issues": [str, ...],
                "learning_result": {...},
                "recommendation": str
            }
        """
        # 1. 跑回归测试
        regression_result = run_regression_test(
            current_audit_path=current_audit_path,
            task_id=task_id,
            baseline_path=self.baseline_path,
            threshold=self.regression_threshold
        )

        # 2. 从 episode 中学习
        learning_result = self.learn_from_episode(episode)

        # 3. 生成推荐
        if regression_result["passed"]:
            recommendation = "回归测试通过，可以采纳学习器的权重调整建议"
        else:
            recommendation = f"回归测试失败（{len(regression_result['issues'])} 个退化项），建议人工审核后再采纳"

        return {
            "regression_passed": regression_result["passed"],
            "regression_issues": regression_result["issues"],
            "learning_result": learning_result,
            "recommendation": recommendation
        }
