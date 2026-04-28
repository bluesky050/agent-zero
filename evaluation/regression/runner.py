"""
测试执行器

运行回归测试，收集指标，执行评估。
"""

import os
import sys
from pathlib import Path

# 设置 HuggingFace 镜像（解决中国大陆网络问题）
# 必须在导入其他模块之前设置
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 手动加载 .env 文件中的关键环境变量
def _load_env_file():
    """手动加载 .env 文件中的关键环境变量"""
    env_path = Path(__file__).parent.parent.parent / "usr" / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key and key not in os.environ:
                        os.environ[key] = value

_load_env_file()

# 保存原始 sys.argv
_original_argv = sys.argv.copy()

# 设置 dockerized 模式参数（必须在 runtime.initialize 之前）
# 在 dockerized 模式下，函数会直接调用而不是通过 HTTP RFC
sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if not arg.startswith("--dockerized")]
sys.argv.append("--dockerized=true")

import asyncio
import json
import subprocess
import tempfile
import time
from datetime import datetime
from typing import Optional, Any

# 初始化 runtime 并强制设置 dockerized 模式
from helpers import runtime as _runtime
_runtime.initialize()
# 强制设置 dockerized 模式（覆盖参数解析结果）
_runtime.args["dockerized"] = True

# 恢复原始 sys.argv（让 Click 正常解析）
sys.argv = _original_argv

from .test_case import RegressionTestCase, TestCaseLoader
from evaluation.metrics.models import EvaluationResult, EvaluationReport
from evaluation.metrics.collector import MetricCollector, get_collector
from evaluation.judge.evaluator import CompositeJudge, IntentionJudge, HallucinationJudge, TaskCompletionJudge


class RegressionRunner:
    """回归测试执行器

    加载测试用例，执行 Agent，收集指标，生成报告。
    """

    def __init__(
        self,
        test_cases_dir: str,
        results_dir: str,
        agent_config: dict = None,
        judge_config: dict = None,
    ):
        self.test_cases_dir = Path(test_cases_dir)
        self.results_dir = Path(results_dir)
        self.agent_config = agent_config or {}
        self.judge_config = judge_config or {}

        self.loader = TestCaseLoader(str(test_cases_dir))
        self.collector = get_collector()
        self.judge = CompositeJudge(judge_config)

        # 创建结果目录
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def run_single(self, test_case: RegressionTestCase) -> EvaluationResult:
        """运行单个测试用例"""
        start_time = time.time()

        # 1. 准备上下文（创建临时文件等）
        context_files = await self._prepare_context(test_case)

        # 2. 执行 Agent
        agent_result = await self._execute_agent(test_case, context_files)

        # 3. 收集指标
        metrics = self._collect_metrics(agent_result)

        # 4. LLM-as-a-Judge 评估
        eval_context = {
            "test_case_id": test_case.id,
            "test_case": test_case.to_dict(),
            "agent_result": agent_result,
            "metrics": metrics,
        }

        # 根据配置决定评估维度
        if test_case.eval_intention and test_case.eval_hallucination and test_case.eval_completion:
            result = await self.judge.evaluate(eval_context)
        else:
            result = await self._partial_evaluate(test_case, eval_context)

        # 5. 补充执行信息
        result.execution_time_seconds = time.time() - start_time
        result.total_iterations = metrics.get("loop_efficiency", {}).get("total_iterations", 0)

        # 6. 清理临时文件
        await self._cleanup_context(context_files)

        return result

    async def run_all(
        self,
        categories: list[str] = None,
        ids: list[str] = None,
        tags: list[str] = None,
        parallel: int = 1,
    ) -> EvaluationReport:
        """运行所有测试用例"""
        # 加载测试用例
        if ids:
            test_cases = self.loader.load_by_ids(ids)
        elif categories:
            test_cases = []
            for cat in categories:
                test_cases.extend(self.loader.load_by_category(cat))
        elif tags:
            test_cases = self.loader.load_by_tags(tags)
        else:
            test_cases = self.loader.load_all()

        if not test_cases:
            print("Warning: No test cases found")
            return self._create_empty_report()

        # 运行测试
        results = []
        if parallel > 1:
            # 并行执行
            semaphore = asyncio.Semaphore(parallel)
            async def run_with_semaphore(tc):
                async with semaphore:
                    return await self.run_single(tc)
            results = await asyncio.gather(*[run_with_semaphore(tc) for tc in test_cases])
        else:
            # 串行执行
            for tc in test_cases:
                print(f"Running test case: {tc.id} - {tc.name}")
                result = await self.run_single(tc)
                results.append(result)
                print(f"  Result: {'PASS' if result.passed else 'FAIL'} (confidence: {result.confidence:.2f})")

        # 生成报告
        report = self._generate_report(test_cases, results)

        # 保存报告
        report_path = self.collector.save_report(report)
        print(f"Report saved to: {report_path}")

        return report

    async def _prepare_context(self, test_case: RegressionTestCase) -> dict:
        """准备测试上下文"""
        context_files = {}

        if "files" in test_case.context:
            for file_spec in test_case.context["files"]:
                path = file_spec.get("path", "")
                content = file_spec.get("content", "")

                # 创建临时文件
                temp_dir = tempfile.mkdtemp()
                full_path = Path(temp_dir) / Path(path).name

                full_path.write_text(content, encoding="utf-8")
                context_files[path] = str(full_path)

        return context_files

    async def _execute_agent(self, test_case: RegressionTestCase, context_files: dict) -> dict:
        """执行 Agent - 真正调用 Agent 系统"""
        agent_result = {
            "selected_tool": "",
            "tool_args": {},
            "tool_calls": [],
            "tool_results": [],
            "final_response": "",
            "task_completed": False,
        }

        try:
            from agent import AgentContext, AgentConfig, AgentContextType, UserMessage
            from helpers import settings
            import asyncio

            # 获取当前设置
            current_settings = settings.get_settings()

            # 覆盖配置（如果有）
            if self.agent_config:
                current_settings = settings.merge_settings(current_settings, self.agent_config)

            # 创建 Agent 配置
            config = AgentConfig(
                profile=current_settings.get("agent_profile", "developer"),
                knowledge_subdirs=[current_settings.get("agent_knowledge_subdir", "default"), "default"],
                mcp_servers=current_settings.get("mcp_servers", ""),
            )

            # 创建新的独立上下文（不与现有对话冲突）
            context_id = f"eval_{test_case.id}_{int(time.time())}"
            context = AgentContext(
                config=config,
                id=context_id,
                name=f"[Test] {test_case.name}",
                type=AgentContextType.TASK,
            )

            # 设置超时
            timeout_seconds = test_case.timeout_seconds or 300

            # 准备用户消息
            user_msg = test_case.user_message
            attachments = []

            # 添加上下文文件作为附件
            for path, temp_path in context_files.items():
                attachments.append(temp_path)

            # 发送消息并等待完成
            task = context.communicate(UserMessage(message=user_msg, attachments=attachments))

            try:
                result = await asyncio.wait_for(task.result(), timeout=timeout_seconds)
                agent_result["final_response"] = result or ""
                agent_result["task_completed"] = True
            except asyncio.TimeoutError:
                agent_result["final_response"] = f"[超时] 执行时间超过 {timeout_seconds} 秒"
                agent_result["task_completed"] = False
            except Exception as e:
                agent_result["final_response"] = f"[执行错误] {str(e)}"
                agent_result["task_completed"] = False

            # 提取执行历史
            agent = context.get_agent()
            if agent:
                # 获取循环数据
                loop_data = getattr(agent, "loop_data", None)
                if loop_data:
                    agent_result["total_iterations"] = loop_data.iteration

                # 从历史记录中提取工具调用信息
                history = getattr(agent, "history", None)
                if history:
                    tool_calls = []
                    messages = []
                    for msg in history.output():
                        content = msg.get("content", "")
                        # 检查是否是工具结果消息
                        if isinstance(content, dict) and "tool_name" in content:
                            tool_calls.append({
                                "tool_name": content.get("tool_name", ""),
                                "tool_result": str(content.get("tool_result", ""))[:500],
                            })
                        messages.append({
                            "ai": msg.get("ai", False),
                            "content": str(content)[:500]
                        })
                    agent_result["tool_calls"] = tool_calls
                    agent_result["messages"] = messages

            # 清理测试上下文
            AgentContext.remove(context.id)

        except ImportError as e:
            agent_result["final_response"] = f"[导入错误] {str(e)}"
            agent_result["task_completed"] = False
        except Exception as e:
            agent_result["final_response"] = f"[执行错误] {str(e)}"
            agent_result["task_completed"] = False

        return agent_result

    def _collect_metrics(self, agent_result: dict) -> dict:
        """收集指标"""
        metrics = {
            "tool_call": {},
            "loop_efficiency": {},
            "intention": {},
            "hallucination": {},
        }

        # 从 collector 获取聚合指标
        metrics["tool_call"] = self.collector.aggregate_tool_call_metrics()
        metrics["loop_efficiency"] = self.collector.aggregate_loop_efficiency_metrics()

        return metrics

    async def _partial_evaluate(self, test_case: RegressionTestCase, context: dict) -> EvaluationResult:
        """部分维度评估"""
        results = []

        if test_case.eval_intention:
            intention_judge = IntentionJudge(self.judge_config)
            results.append(await intention_judge.evaluate(context))

        if test_case.eval_hallucination:
            hallucination_judge = HallucinationJudge(self.judge_config)
            results.append(await hallucination_judge.evaluate(context))

        if test_case.eval_completion:
            completion_judge = TaskCompletionJudge(self.judge_config)
            results.append(await completion_judge.evaluate(context))

        # 合并结果
        passed = all(r.passed for r in results) if results else True
        confidence = sum(r.confidence for r in results) / len(results) if results else 1.0

        return EvaluationResult(
            test_case_id=test_case.id,
            passed=passed,
            confidence=confidence,
            needs_review=confidence < 0.7,
            reasoning="; ".join(r.reasoning for r in results if r.reasoning),
        )

    async def _cleanup_context(self, context_files: dict) -> None:
        """清理测试上下文"""
        for path, temp_path in context_files.items():
            try:
                os.remove(temp_path)
                # 尝试删除临时目录
                temp_dir = Path(temp_path).parent
                if temp_dir.exists() and not any(temp_dir.iterdir()):
                    os.rmdir(temp_dir)
            except OSError:
                pass

    def _generate_report(self, test_cases: list, results: list) -> EvaluationReport:
        """生成评估报告"""
        # 获取 git 信息
        git_commit = self._get_git_commit()
        git_branch = self._get_git_branch()

        report = EvaluationReport(
            run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            timestamp=datetime.now().isoformat(),
            git_commit=git_commit,
            git_branch=git_branch,
            total_cases=len(results),
            passed=sum(1 for r in results if r.passed),
            failed=sum(1 for r in results if not r.passed),
            skipped=0,
            needs_review_count=sum(1 for r in results if r.needs_review),
        )

        # 计算四维指标
        intention_results = [r for r in results if r.intention_correct is not None]
        if intention_results:
            report.intention_accuracy = sum(1 for r in intention_results if r.intention_correct) / len(intention_results)

        # 计算工具调用成功率
        # tool_call_confidence 存储的是实际成功率 (0.0-1.0)
        if results:
            report.tool_call_success_rate = sum(r.tool_call_confidence for r in results) / len(results)

        hallucination_results = [r for r in results if r.has_hallucination is not None or r.hallucination_confidence > 0]
        if hallucination_results:
            report.hallucination_rate = sum(1 for r in hallucination_results if r.has_hallucination) / len(hallucination_results)

        # 计算平均循环次数和效率得分
        report.avg_iterations = sum(r.total_iterations for r in results) / len(results) if results else 0
        report.avg_efficiency_score = sum(r.loop_efficiency_score for r in results) / len(results) if results else 0

        # 详细结果
        report.case_results = [r.to_dict() for r in results]

        # Badcase 分析
        badcases = []
        for tc, r in zip(test_cases, results):
            if not r.passed:
                badcases.append({
                    "test_case_id": tc.id,
                    "reason": r.reasoning,
                    "confidence": r.confidence,
                })
        report.badcases = badcases

        return report

    def _create_empty_report(self) -> EvaluationReport:
        """创建空报告"""
        return EvaluationReport(
            run_id="empty",
            timestamp=datetime.now().isoformat(),
            git_commit=self._get_git_commit(),
            git_branch=self._get_git_branch(),
        )

    def _get_git_commit(self) -> str:
        """获取当前 git commit"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    def _get_git_branch(self) -> str:
        """获取当前 git branch"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"


async def run_regression_tests(
    test_cases_dir: str = "usr/evaluation/test_cases",
    results_dir: str = "usr/evaluation/results",
    categories: list[str] = None,
    parallel: int = 1,
) -> EvaluationReport:
    """运行回归测试的便捷函数"""
    runner = RegressionRunner(test_cases_dir, results_dir)
    return await runner.run_all(categories=categories, parallel=parallel)