"""
回归测试用例定义

定义测试用例的数据结构、加载和验证逻辑。
"""

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any
import yaml


@dataclass
class RegressionTestCase:
    """回归测试用例

    定义一个完整的测试用例，包括输入、期望输出和评估配置。
    """

    id: str                              # 唯一标识，如 tc_001_fix_simple_bug
    name: str                            # 用例名称
    category: str                        # 分类: code/research/tool_use/analysis
    difficulty: str = "medium"           # 难度: easy/medium/hard

    # 输入
    user_message: str = ""               # 用户输入
    context: dict = field(default_factory=dict)  # 上下文（文件、数据等）

    # 期望输出
    expected_tools: list = field(default_factory=list)      # 期望使用的工具
    expected_keywords: list = field(default_factory=list)   # 响应应包含的关键词
    expected_not_keywords: list = field(default_factory=list)  # 响应不应包含的内容
    success_criteria: str = ""           # 成功标准描述

    # 评估配置
    eval_intention: bool = True          # 是否评估意图识别
    eval_hallucination: bool = True      # 是否评估幻觉
    eval_completion: bool = True         # 是否评估任务完成度
    max_iterations: int = 20             # 最大循环次数阈值（超过视为卡死）
    timeout_seconds: int = 300          # 超时时间（秒）

    # 元数据
    tags: list = field(default_factory=list)   # 标签，便于筛选
    description: str = ""                # 详细描述

    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        # 确保列表字段不是 None
        for key in ["expected_tools", "expected_keywords", "expected_not_keywords", "tags"]:
            if data[key] is None:
                data[key] = []
        return data

    def to_yaml(self) -> str:
        """转换为 YAML 字符串"""
        return yaml.dump(self.to_dict(), allow_unicode=True, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict) -> "RegressionTestCase":
        """从字典创建"""
        # 处理缺失字段
        defaults = {
            "id": "unknown",
            "name": "Unknown Test Case",
            "category": "general",
            "difficulty": "medium",
            "user_message": "",
            "context": {},
            "expected_tools": [],
            "expected_keywords": [],
            "expected_not_keywords": [],
            "success_criteria": "",
            "eval_intention": True,
            "eval_hallucination": True,
            "eval_completion": True,
            "max_iterations": 20,
            "timeout_seconds": 300,
            "tags": [],
            "description": "",
        }
        # 合并默认值
        for key, default in defaults.items():
            if key not in data:
                data[key] = default

        return cls(**data)

    @classmethod
    def from_yaml_file(cls, path: str) -> "RegressionTestCase":
        """从 YAML 文件加载"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Test case file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        test_case = cls.from_dict(data)
        test_case._source_file = str(path)
        return test_case

    def validate(self) -> list[str]:
        """验证测试用例，返回错误列表"""
        errors = []

        if not self.id:
            errors.append("id is required")
        elif not re.match(r'^[a-z0-9_]+$', self.id):
            errors.append(f"invalid id format: {self.id}")

        if not self.name:
            errors.append("name is required")

        if not self.user_message:
            errors.append("user_message is required")

        if self.category not in ["code", "research", "tool_use", "analysis", "general"]:
            errors.append(f"invalid category: {self.category}")

        if self.difficulty not in ["easy", "medium", "hard"]:
            errors.append(f"invalid difficulty: {self.difficulty}")

        if self.max_iterations <= 0:
            errors.append("max_iterations must be positive")

        if self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")

        return errors

    def is_valid(self) -> bool:
        """检查是否有效"""
        return len(self.validate()) == 0


class TestCaseLoader:
    """测试用例加载器"""

    def __init__(self, test_cases_dir: str):
        self.test_cases_dir = Path(test_cases_dir)

    def load_all(self) -> list[RegressionTestCase]:
        """加载所有测试用例"""
        test_cases = []

        if not self.test_cases_dir.exists():
            return test_cases

        # 递归加载所有 .yaml 文件
        for yaml_file in self.test_cases_dir.rglob("*.yaml"):
            try:
                test_case = RegressionTestCase.from_yaml_file(str(yaml_file))
                if test_case.is_valid():
                    test_cases.append(test_case)
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file}: {e}")

        return test_cases

    def load_by_category(self, category: str) -> list[RegressionTestCase]:
        """按类别加载测试用例"""
        all_cases = self.load_all()
        return [tc for tc in all_cases if tc.category == category]

    def load_by_ids(self, ids: list[str]) -> list[RegressionTestCase]:
        """按 ID 加载测试用例"""
        all_cases = self.load_all()
        return [tc for tc in all_cases if tc.id in ids]

    def load_by_tags(self, tags: list[str]) -> list[RegressionTestCase]:
        """按标签筛选测试用例"""
        all_cases = self.load_all()
        return [
            tc for tc in all_cases
            if any(tag in tc.tags for tag in tags)
        ]
