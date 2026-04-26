"""
指标收集器

提供线程安全的指标收集、存储、聚合功能。
"""

import json
import os
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Union

from .models import (
    IntentionMetric,
    ToolCallMetric,
    HallucinationMetric,
    LoopEfficiencyMetric,
    EvaluationResult,
    EvaluationReport,
    MetricCategory,
)


class MetricCollector:
    """指标收集器

    线程安全的指标收集、存储和聚合类。
    支持按会话、按类别聚合指标。
    """

    _instance: Optional["MetricCollector"] = None
    _lock = threading.Lock()

    # 存储路径
    DEFAULT_STORAGE_DIR = "usr/evaluation/metrics"

    def __new__(cls, storage_dir: str = None):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, storage_dir: str = None):
        if self._initialized:
            return

        self.storage_dir = Path(storage_dir or self.DEFAULT_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # 内存缓冲区
        self._metrics_buffer: dict[str, list] = {
            MetricCategory.INTENTION.value: [],
            MetricCategory.TOOL_CALL.value: [],
            MetricCategory.HALLUCINATION.value: [],
            MetricCategory.LOOP_EFFICIENCY.value: [],
        }
        self._buffer_lock = threading.Lock()

        # 按会话索引
        self._chat_metrics: dict[str, dict] = defaultdict(lambda: {
            MetricCategory.INTENTION.value: [],
            MetricCategory.TOOL_CALL.value: [],
            MetricCategory.HALLUCINATION.value: [],
            MetricCategory.LOOP_EFFICIENCY.value: [],
        })

        self._initialized = True

    def save(self, metric: Union[IntentionMetric, ToolCallMetric, HallucinationMetric, LoopEfficiencyMetric]) -> None:
        """保存单个指标"""
        category = self._get_category(metric)
        metric_dict = metric.to_dict()

        with self._buffer_lock:
            self._metrics_buffer[category].append(metric_dict)

            # 按会话索引
            chat_id = metric_dict.get("chat_id", "unknown")
            self._chat_metrics[chat_id][category].append(metric_dict)

        # 持久化
        self._persist_metric(metric_dict, category)

    def save_intention(self, metric: IntentionMetric) -> None:
        """保存意图识别指标"""
        self.save(metric)

    def save_tool_call(self, metric: ToolCallMetric) -> None:
        """保存工具调用指标"""
        self.save(metric)

    def save_hallucination(self, metric: HallucinationMetric) -> None:
        """保存幻觉检测指标"""
        self.save(metric)

    def save_loop_efficiency(self, metric: LoopEfficiencyMetric) -> None:
        """保存循环效率指标"""
        self.save(metric)

    def get_metrics_by_chat(self, chat_id: str) -> dict:
        """获取指定会话的所有指标"""
        with self._buffer_lock:
            return dict(self._chat_metrics.get(chat_id, {}))

    def get_metrics_by_category(self, category: MetricCategory) -> list:
        """获取指定类别的所有指标"""
        with self._buffer_lock:
            return list(self._metrics_buffer.get(category.value, []))

    def aggregate_tool_call_metrics(self) -> dict:
        """聚合工具调用指标"""
        metrics = self.get_metrics_by_category(MetricCategory.TOOL_CALL)

        if not metrics:
            return {
                "total_calls": 0,
                "success_rate": 0.0,
                "parse_success_rate": 0.0,
                "error_breakdown": {},
            }

        total = len(metrics)
        success_count = sum(1 for m in metrics if m.get("execution_success", False))
        parse_success_count = sum(1 for m in metrics if m.get("parse_success", False))

        # 错误类型统计
        error_breakdown = defaultdict(int)
        for m in metrics:
            if not m.get("execution_success") and m.get("error_type"):
                error_breakdown[m["error_type"]] += 1

        return {
            "total_calls": total,
            "success_rate": success_count / total,
            "parse_success_rate": parse_success_count / total,
            "error_breakdown": dict(error_breakdown),
        }

    def aggregate_loop_efficiency_metrics(self) -> dict:
        """聚合循环效率指标"""
        metrics = self.get_metrics_by_category(MetricCategory.LOOP_EFFICIENCY)

        if not metrics:
            return {
                "total_chats": 0,
                "avg_iterations": 0.0,
                "avg_efficiency_score": 0.0,
                "stuck_rate": 0.0,
                "avg_repeat_rate": 0.0,
            }

        total = len(metrics)
        avg_iterations = sum(m.get("total_iterations", 0) for m in metrics) / total
        avg_efficiency = sum(m.get("efficiency_score", 0) for m in metrics) / total
        stuck_count = sum(1 for m in metrics if m.get("loop_stuck", False))
        avg_repeat_rate = sum(m.get("repeat_rate", 0) for m in metrics) / total

        return {
            "total_chats": total,
            "avg_iterations": avg_iterations,
            "avg_efficiency_score": avg_efficiency,
            "stuck_rate": stuck_count / total,
            "avg_repeat_rate": avg_repeat_rate,
        }

    def aggregate_intention_metrics(self) -> dict:
        """聚合意图识别指标"""
        metrics = self.get_metrics_by_category(MetricCategory.INTENTION)

        if not metrics:
            return {
                "total_evaluations": 0,
                "accuracy_rate": 0.0,
                "avg_confidence": 0.0,
            }

        # 只统计已评估的
        evaluated = [m for m in metrics if m.get("is_correct") is not None]
        if not evaluated:
            return {
                "total_evaluations": 0,
                "accuracy_rate": 0.0,
                "avg_confidence": 0.0,
            }

        total = len(evaluated)
        correct_count = sum(1 for m in evaluated if m.get("is_correct", False))
        avg_confidence = sum(m.get("confidence", 0) for m in evaluated) / total

        return {
            "total_evaluations": total,
            "accuracy_rate": correct_count / total,
            "avg_confidence": avg_confidence,
        }

    def aggregate_hallucination_metrics(self) -> dict:
        """聚合幻觉检测指标"""
        metrics = self.get_metrics_by_category(MetricCategory.HALLUCINATION)

        if not metrics:
            return {
                "total_checks": 0,
                "hallucination_rate": 0.0,
                "severity_breakdown": {},
            }

        total = len(metrics)
        hallucination_count = sum(1 for m in metrics if m.get("has_hallucination", False))

        # 按严重程度统计
        severity_breakdown = defaultdict(int)
        for m in metrics:
            if m.get("has_hallucination") and m.get("severity"):
                severity_breakdown[m["severity"]] += 1

        return {
            "total_checks": total,
            "hallucination_rate": hallucination_count / total,
            "severity_breakdown": dict(severity_breakdown),
        }

    def generate_summary_report(self) -> dict:
        """生成汇总报告"""
        return {
            "timestamp": datetime.now().isoformat(),
            "tool_call": self.aggregate_tool_call_metrics(),
            "loop_efficiency": self.aggregate_loop_efficiency_metrics(),
            "intention": self.aggregate_intention_metrics(),
            "hallucination": self.aggregate_hallucination_metrics(),
        }

    def clear_buffer(self) -> None:
        """清空内存缓冲区"""
        with self._buffer_lock:
            self._metrics_buffer = {
                MetricCategory.INTENTION.value: [],
                MetricCategory.TOOL_CALL.value: [],
                MetricCategory.HALLUCINATION.value: [],
                MetricCategory.LOOP_EFFICIENCY.value: [],
            }
            self._chat_metrics = defaultdict(lambda: {
                MetricCategory.INTENTION.value: [],
                MetricCategory.TOOL_CALL.value: [],
                MetricCategory.HALLUCINATION.value: [],
                MetricCategory.LOOP_EFFICIENCY.value: [],
            })

    def save_report(self, report: EvaluationReport) -> str:
        """保存评估报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        report_dir = self.storage_dir.parent / "results" / timestamp
        report_dir.mkdir(parents=True, exist_ok=True)

        # 保存汇总
        summary_path = report_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        # 保存 Markdown 报告
        md_path = report_dir / "report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())

        # 更新 latest 符号链接（或复制）
        latest_dir = self.storage_dir.parent / "results" / "latest"
        if latest_dir.exists():
            # Windows 不支持 symlink，直接复制
            import shutil
            shutil.rmtree(latest_dir)
        import shutil
        shutil.copytree(report_dir, latest_dir)

        return str(summary_path)

    def _get_category(self, metric: Any) -> str:
        """获取指标类别"""
        if isinstance(metric, IntentionMetric):
            return MetricCategory.INTENTION.value
        elif isinstance(metric, ToolCallMetric):
            return MetricCategory.TOOL_CALL.value
        elif isinstance(metric, HallucinationMetric):
            return MetricCategory.HALLUCINATION.value
        elif isinstance(metric, LoopEfficiencyMetric):
            return MetricCategory.LOOP_EFFICIENCY.value
        else:
            raise ValueError(f"Unknown metric type: {type(metric)}")

    def _persist_metric(self, metric_dict: dict, category: str) -> None:
        """持久化单个指标到文件"""
        chat_id = metric_dict.get("chat_id", "unknown")
        timestamp = metric_dict.get("timestamp", datetime.now().isoformat())

        # 按日期组织存储
        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = self.storage_dir / date_str / f"{category}.json"

        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 读取现有数据并追加
        existing = []
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing = []

        existing.append(metric_dict)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)


# 全局单例
collector = MetricCollector()


def get_collector() -> MetricCollector:
    """获取全局指标收集器"""
    return collector