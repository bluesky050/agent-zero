"""
LLM-as-a-Judge 评估器

提供基于 LLM 的自动化评估能力，支持意图识别、幻觉检测、任务完成度评估。
"""

import json
import re
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

from evaluation.metrics.models import EvaluationResult


# 评估 prompt 模板目录
PROMPTS_DIR = Path(__file__).parent / "prompts"

# 配置文件路径
CONFIG_FILE = Path("usr/evaluation/eval_model_config.yaml")


def load_config() -> dict:
    """加载评估模型配置"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_prompt(name: str) -> str:
    """加载评估 prompt 模板"""
    prompt_path = PROMPTS_DIR / f"{name}.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


class LiteLLMDirectWrapper:
    """直接使用 litellm 的简单封装（绕过项目复杂依赖）"""

    def __init__(self, config: dict):
        self.config = config
        import os
        if config.get("api_key"):
            os.environ["OPENAI_API_KEY"] = config["api_key"]
        if config.get("api_base"):
            os.environ["OPENAI_API_BASE"] = config["api_base"]

    async def unified_call(self, system_message: str, user_message: str, **kwargs):
        """调用 LLM API"""
        import litellm
        model_name = self.config.get("name", "gpt-4o")
        provider = self.config.get("provider", "openai")
        litellm_model = f"{provider}/{model_name}"

        response = await litellm.acompletion(
            model=litellm_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
            **kwargs
        )
        return response.choices[0].message.content, ""


class JudgeEvaluator(ABC):
    """评估器基类

    提供 LLM-as-a-Judge 评估能力。
    子类需实现 build_prompt 和 parse_response 方法。
    """

    DEFAULT_MODEL_CONFIG = {
        "provider": "anthropic",
        "name": "claude-sonnet-4-20250514",
        "api_key": "",
        "api_base": "",
        "ctx_length": 32000,
        "limit_requests": 100,
        "limit_input": 20000,
        "limit_output": 4096,
    }

    def __init__(self, model_config: dict = None):
        if model_config:
            self.model_config = model_config
        else:
            file_config = load_config()
            if file_config:
                self.model_config = file_config
            else:
                self.model_config = self.DEFAULT_MODEL_CONFIG.copy()
        self._model = None

    @property
    def model(self):
        """懒加载模型实例"""
        if self._model is None:
            try:
                from models import LiteLLMChatWrapper
                config = self.model_config.copy()
                model_name = config.get("name", config.get("model", ""))
                provider = config.get("provider", "openai")
                kwargs = {}
                if config.get("api_base"):
                    kwargs["api_base"] = config["api_base"]
                if config.get("api_key"):
                    kwargs["api_key"] = config["api_key"]
                self._model = LiteLLMChatWrapper(
                    model=model_name,
                    provider=provider,
                    **kwargs
                )
            except Exception:
                self._model = LiteLLMDirectWrapper(self.model_config)
        return self._model

    async def evaluate(self, context: dict) -> EvaluationResult:
        """执行评估"""
        test_case_id = context.get("test_case_id", "unknown")

        if self.model is None:
            return self._mock_evaluate(test_case_id, context)

        prompt = self.build_prompt(context)
        system_prompt = self.get_system_prompt()
        response, _ = await self.model.unified_call(
            system_message=system_prompt,
            user_message=prompt,
        )
        result = self.parse_response(response)
        result.test_case_id = test_case_id
        return result

    def _mock_evaluate(self, test_case_id: str, context: dict) -> EvaluationResult:
        """模拟评估（当模型不可用时）"""
        import random
        passed = random.random() > 0.3
        confidence = random.uniform(0.6, 0.95)
        return EvaluationResult(
            test_case_id=test_case_id,
            passed=passed,
            confidence=confidence,
            needs_review=confidence < 0.7,
            reasoning="[模拟模式] 模型未配置",
            intention_correct=passed,
            intention_confidence=confidence,
            has_hallucination=not passed,
            hallucination_confidence=confidence,
        )

    @abstractmethod
    def build_prompt(self, context: dict) -> str:
        pass

    @abstractmethod
    def parse_response(self, response: str) -> EvaluationResult:
        pass

    def get_system_prompt(self) -> str:
        return "你是一个专业的 AI Agent 评估专家。请按照用户要求进行客观、准确的评估。"


class IntentionJudge(JudgeEvaluator):
    """意图识别评估器"""

    def get_system_prompt(self) -> str:
        return load_prompt("intention_eval") or """你是一个评估 Agent 意图识别准确性的专家。

请客观评估 Agent 是否正确理解了用户意图，并选择了合适的工具来完成任务。

评估标准：
1. Agent 是否准确识别了用户的核心需求
2. 选择的工具是否是完成该任务的最佳选择
3. 工具参数是否合理

请输出 JSON 格式的评估结果。"""

    def build_prompt(self, context: dict) -> str:
        test_case = context.get("test_case", {})
        agent_result = context.get("agent_result", {})
        return f"""## 用户输入
{test_case.get('user_message', 'N/A')}

## 期望使用的工具
{test_case.get('expected_tools', '无特定要求')}

## Agent 实际行为
- 选择的工具: {agent_result.get('selected_tool', 'N/A')}
- 工具参数: {json.dumps(agent_result.get('tool_args', {}), ensure_ascii=False, indent=2)}

## 工具执行结果
{agent_result.get('tool_result', 'N/A')[:500]}

请评估 Agent 的意图识别是否正确。输出 JSON：
{{
  "is_correct": true/false,
  "correct_intention": "如果不正确，正确的意图应该是什么",
  "correct_tool": "如果不正确，应该选择什么工具",
  "confidence": 0.0-1.0,
  "reasoning": "简短解释"
}}"""

    def parse_response(self, response: str) -> EvaluationResult:
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            data = json.loads(json_match.group()) if json_match else {}
            confidence = float(data.get("confidence", 0.5))
            is_correct = data.get("is_correct", True)
            return EvaluationResult(
                test_case_id="",
                passed=is_correct,
                confidence=confidence,
                needs_review=confidence < 0.7,
                reasoning=data.get("reasoning", ""),
                intention_correct=is_correct,
                intention_confidence=confidence,
            )
        except (json.JSONDecodeError, ValueError) as e:
            return EvaluationResult(
                test_case_id="",
                passed=False,
                confidence=0.0,
                needs_review=True,
                reasoning=f"解析错误: {str(e)}",
                intention_correct=False,
                intention_confidence=0.0,
            )


class HallucinationJudge(JudgeEvaluator):
    """幻觉检测评估器"""

    def get_system_prompt(self) -> str:
        return load_prompt("hallucination_eval") or """你是一个检测 Agent 输出幻觉的专家。

请仔细检查 Agent 的响应是否存在幻觉：
1. 事实性幻觉：编造不存在的事实
2. 工具结果幻觉：歪曲工具返回结果
3. 推理性幻觉：错误推理

请输出 JSON 格式的评估结果。"""

    def build_prompt(self, context: dict) -> str:
        test_case = context.get("test_case", {})
        agent_result = context.get("agent_result", {})
        return f"""## 用户任务
{test_case.get('user_message', 'N/A')}

## 工具调用记录
{json.dumps(agent_result.get('tool_calls', []), ensure_ascii=False, indent=2)[:2000]}

## Agent 最终响应
{agent_result.get('final_response', 'N/A')[:2000]}

请检测响应中是否存在幻觉。输出 JSON：
{{
  "has_hallucination": true/false,
  "hallucination_type": "factual|tool_result|inference",
  "severity": "high|medium|low",
  "details": "具体描述",
  "confidence": 0.0-1.0
}}"""

    def parse_response(self, response: str) -> EvaluationResult:
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            data = json.loads(json_match.group()) if json_match else {}
            has_hallucination = data.get("has_hallucination", False)
            confidence = float(data.get("confidence", 0.5))
            return EvaluationResult(
                test_case_id="",
                passed=not has_hallucination,
                confidence=confidence,
                needs_review=confidence < 0.7 and has_hallucination,
                reasoning=data.get("details", ""),
                has_hallucination=has_hallucination,
                hallucination_confidence=confidence,
            )
        except (json.JSONDecodeError, ValueError) as e:
            return EvaluationResult(
                test_case_id="",
                passed=True,
                confidence=0.0,
                needs_review=True,
                reasoning=f"解析错误: {str(e)}",
                has_hallucination=False,
                hallucination_confidence=0.0,
            )


class TaskCompletionJudge(JudgeEvaluator):
    """任务完成度评估器"""

    def get_system_prompt(self) -> str:
        return load_prompt("task_completion_eval") or """你是一个评估 Agent 任务完成度的专家。

请综合评估 Agent 是否成功完成了用户的任务：
1. 任务目标是否达成
2. 输出是否符合用户预期
3. 过程是否高效

请输出 JSON 格式的评估结果。"""

    def build_prompt(self, context: dict) -> str:
        test_case = context.get("test_case", {})
        agent_result = context.get("agent_result", {})
        metrics = context.get("metrics", {})
        loop_metrics = metrics.get("loop_efficiency", {})
        return f"""## 用户任务
{test_case.get('user_message', 'N/A')}

## 成功标准
{test_case.get('success_criteria', '任务完成')}

## Agent 最终响应
{agent_result.get('final_response', 'N/A')[:2000]}

## 执行指标
- 总循环次数: {loop_metrics.get('total_iterations', 0)}
- 使用的工具: {loop_metrics.get('unique_tools_used', [])}
- 是否标记完成: {agent_result.get('task_completed', False)}

请评估任务是否完成。输出 JSON：
{{
  "task_completed": true/false,
  "completion_level": "full|partial|failed",
  "confidence": 0.0-1.0,
  "reasoning": "简短解释",
  "issues": []
}}"""

    def parse_response(self, response: str) -> EvaluationResult:
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            data = json.loads(json_match.group()) if json_match else {}
            task_completed = data.get("task_completed", False)
            confidence = float(data.get("confidence", 0.5))
            return EvaluationResult(
                test_case_id="",
                passed=task_completed,
                confidence=confidence,
                needs_review=confidence < 0.7,
                reasoning=data.get("reasoning", ""),
                loop_efficiency_score=1.0 if task_completed else 0.5,
            )
        except (json.JSONDecodeError, ValueError) as e:
            return EvaluationResult(
                test_case_id="",
                passed=False,
                confidence=0.0,
                needs_review=True,
                reasoning=f"解析错误: {str(e)}",
                loop_efficiency_score=0.0,
            )


class CompositeJudge(JudgeEvaluator):
    """组合评估器 - 整合意图、幻觉、任务完成三维度评估"""

    def __init__(self, model_config: dict = None):
        super().__init__(model_config)
        self.intention_judge = IntentionJudge(model_config)
        self.hallucination_judge = HallucinationJudge(model_config)
        self.completion_judge = TaskCompletionJudge(model_config)

    async def evaluate(self, context: dict) -> EvaluationResult:
        import asyncio
        intention_result, hallucination_result, completion_result = await asyncio.gather(
            self.intention_judge.evaluate(context),
            self.hallucination_judge.evaluate(context),
            self.completion_judge.evaluate(context),
        )

        passed = (
            intention_result.intention_correct
            and not hallucination_result.has_hallucination
            and completion_result.passed
        )
        confidence = (
            intention_result.intention_confidence
            + hallucination_result.hallucination_confidence
            + completion_result.confidence
        ) / 3

        return EvaluationResult(
            test_case_id=context.get("test_case_id", ""),
            passed=passed,
            confidence=confidence,
            needs_review=confidence < 0.7,
            reasoning=self._build_reasoning(intention_result, hallucination_result, completion_result),
            intention_correct=intention_result.intention_correct,
            intention_confidence=intention_result.intention_confidence,
            tool_call_success=True,
            has_hallucination=hallucination_result.has_hallucination,
            hallucination_confidence=hallucination_result.hallucination_confidence,
            loop_efficiency_score=completion_result.loop_efficiency_score,
            total_iterations=completion_result.total_iterations,
        )

    def build_prompt(self, context: dict) -> str:
        return ""

    def parse_response(self, response: str) -> EvaluationResult:
        return EvaluationResult(test_case_id="", passed=False, confidence=0.0, needs_review=True, reasoning="")

    def _build_reasoning(self, intention, hallucination, completion) -> str:
        parts = []
        if not intention.intention_correct:
            parts.append(f"意图识别错误: {intention.reasoning}")
        if hallucination.has_hallucination:
            parts.append(f"存在幻觉: {hallucination.reasoning}")
        if not completion.passed:
            parts.append(f"任务未完成: {completion.reasoning}")
        return "; ".join(parts) if parts else "评估通过"
