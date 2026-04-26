"""
四维指标数据结构定义

定义意图识别、工具调用、幻觉检测、循环效率四类指标的数据结构。
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from datetime import datetime
from enum import Enum


class MetricCategory(Enum):
    """指标类别"""
    INTENTION = "intention"
    TOOL_CALL = "tool_call"
    HALLUCINATION = "hallucination"
    LOOP_EFFICIENCY = "loop_efficiency"


class HallucinationType(Enum):
    """幻觉类型"""
    FACTUAL = "factual"           # 事实性幻觉：编造不存在的事实
    TOOL_RESULT = "tool_result"   # 工具结果幻觉：歪曲工具返回结果
    INFERENCE = "inference"       # 推理性幻觉：错误的推理结论


class Severity(Enum):
    """严重程度"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class IntentionMetric:
    """意图识别指标

    记录 Agent 对用户意图的理解和工具选择的准确性。
    """
    timestamp: str
    chat_id: str
    iteration: int
    user_message: str              # 用户原始输入
    parsed_intention: str          # Agent 解析的意图（从 thoughts 字段提取）
    selected_tool: str            # 选择的工具
    tool_args: dict               # 工具参数
    is_correct: Optional[bool] = None     # 是否正确（LLM 判断或人工标注）
    confidence: float = 0.0        # 评估置信度
    reasoning: str = ""            # 评估理由

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IntentionMetric":
        return cls(**data)


@dataclass
class ToolCallMetric:
    """工具调用指标

    记录工具调用的格式正确性、执行成功率、参数正确率。
    """
    timestamp: str
    chat_id: str
    iteration: int
    tool_name: str
    tool_args: dict
    parse_success: bool            # JSON 解析成功
    execution_success: bool        # 执行成功
    error_type: Optional[str] = None       # 错误类型
    error_message: Optional[str] = None    # 错误信息
    retry_count: int = 0           # 重试次数
    execution_time_ms: float = 0.0 # 执行耗时

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCallMetric":
        return cls(**data)


@dataclass
class HallucinationMetric:
    """幻觉检测指标

    记录 Agent 输出中虚假、错误或无法验证信息的检测结果。
    """
    timestamp: str
    chat_id: str
    message_id: str
    content: str                           # Agent 输出内容片段
    hallucination_type: str                  # factual/tool_result/inference
    severity: str                           # high/medium/low
    detected_by: str                        # llm_judge/rule/tool_mismatch
    details: str = ""
    confidence: float = 0.0
    has_hallucination: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "HallucinationMetric":
        return cls(**data)


@dataclass
class LoopEfficiencyMetric:
    """循环效率指标

    记录任务完成所需的循环次数、效率、卡死检测。
    """
    timestamp: str
    chat_id: str
    total_iterations: int
    unique_tools_used: list                  # 使用的唯一工具列表
    repeat_actions: int = 0                 # 重复操作次数
    loop_stuck: bool = False                # 是否卡死（>20 次循环）
    task_completed: bool = False
    task_completed_time_seconds: float = 0.0
    first_response_time_seconds: float = 0.0

    # 效率计算
    @property
    def efficiency_score(self) -> float:
        """效率得分 = min(1, 基准循环次数 / 实际循环次数)"""
        baseline_iterations = 5  # 基准循环次数
        if self.total_iterations <= 0:
            return 0.0
        return min(1.0, baseline_iterations / self.total_iterations)

    @property
    def repeat_rate(self) -> float:
        """重复率 = 重复操作次数 / 总操作次数"""
        total_actions = self.total_iterations
        if total_actions <= 0:
            return 0.0
        return self.repeat_actions / total_actions

    def to_dict(self) -> dict:
        d = asdict(self)
        d["efficiency_score"] = self.efficiency_score
        d["repeat_rate"] = self.repeat_rate
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "LoopEfficiencyMetric":
        # 移除计算属性
        data.pop("efficiency_score", None)
        data.pop("repeat_rate", None)
        return cls(**data)


@dataclass
class EvaluationResult:
    """评估结果

    单个测试用例的评估结果。
    """
    test_case_id: str
    passed: bool
    confidence: float
    needs_review: bool           # confidence < 0.7 时为 True
    reasoning: str

    # 四维指标评估结果
    intention_correct: Optional[bool] = None
    intention_confidence: float = 0.0
    tool_call_success: bool = True
    tool_call_confidence: float = 1.0
    has_hallucination: bool = False
    hallucination_confidence: float = 0.0
    loop_efficiency_score: float = 1.0

    # 执行详情
    execution_time_seconds: float = 0.0
    total_iterations: int = 0
    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationResult":
        return cls(**data)


@dataclass
class EvaluationReport:
    """评估报告

    一次评估运行的完整报告。
    """
    run_id: str
    timestamp: str
    git_commit: str
    git_branch: str

    # 总体统计
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    needs_review_count: int = 0

    @property
    def pass_rate(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.passed / self.total_cases

    # 四维指标汇总
    intention_accuracy: float = 0.0     # 意图识别准确率
    tool_call_success_rate: float = 0.0 # 工具调用成功率
    hallucination_rate: float = 0.0      # 幻觉率
    avg_iterations: float = 0.0          # 平均循环次数
    avg_efficiency_score: float = 0.0    # 平均效率得分

    # 详细结果
    case_results: list = field(default_factory=list)

    # Badcase 分析
    badcases: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pass_rate"] = self.pass_rate
        return d

    def to_markdown(self) -> str:
        """生成 Markdown 格式报告"""
        lines = [
            "# Agent-Zero 评测报告",
            "",
            "## 运行信息",
            f"- **运行时间**: {self.timestamp}",
            f"- **Git 提交**: {self.git_commit}",
            f"- **分支**: {self.git_branch}",
            "",
            "## 总体结果",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 通过率 | {self.pass_rate:.1%} |",
            f"| 通过/总数 | {self.passed}/{self.total_cases} |",
            f"| 需复核 | {self.needs_review_count} |",
            "",
            "## 四维指标",
            "| 指标 | 值 | 阈值 | 状态 |",
            "|------|-----|------|------|",
            f"| 意图识别准确率 | {self.intention_accuracy:.1%} | >=85% | {'PASS' if self.intention_accuracy >= 0.85 else 'FAIL'} |",
            f"| 工具调用成功率 | {self.tool_call_success_rate:.1%} | >=90% | {'PASS' if self.tool_call_success_rate >= 0.90 else 'FAIL'} |",
            f"| 幻觉率 | {self.hallucination_rate:.1%} | <=5% | {'PASS' if self.hallucination_rate <= 0.05 else 'FAIL'} |",
            f"| 平均循环次数 | {self.avg_iterations:.1f} | <=10 | {'PASS' if self.avg_iterations <= 10 else 'FAIL'} |",
            "",
        ]

        if self.badcases:
            lines.extend([
                "## Badcase 分析",
                "",
            ])
            for bc in self.badcases:
                lines.append(f"- **{bc.get('test_case_id', 'unknown')}**: {bc.get('reason', 'N/A')}")

        return "\n".join(lines)
