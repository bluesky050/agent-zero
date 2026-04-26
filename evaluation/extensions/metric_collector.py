"""
指标收集扩展

集成到 Agent 执行流程中，实时收集四维指标。
"""

import time
from datetime import datetime
from typing import Any

from evaluation.metrics.models import (
    IntentionMetric,
    ToolCallMetric,
    HallucinationMetric,
    LoopEfficiencyMetric,
)
from evaluation.metrics.collector import get_collector


class MetricCollectorExtension:
    """指标收集扩展

    注册到 Agent 的扩展点，实时收集指标。
    """

    def __init__(self):
        self.collector = get_collector()
        self._start_times: dict[str, float] = {}  # chat_id -> start_time

    async def execute(self, data: dict, **kwargs) -> dict:
        """扩展执行入口"""
        return data

    # === 注册到扩展点的方法 ===

    async def on_monologue_start(self, data: dict) -> None:
        """主循环开始时"""
        agent = data.get("agent")
        if agent:
            chat_id = agent.context.id if hasattr(agent, "context") else "unknown"
            self._start_times[chat_id] = time.time()

    async def on_message_loop_end(self, data: dict) -> None:
        """消息循环结束时"""
        agent = data.get("agent")
        if not agent:
            return

        loop_data = getattr(agent, "loop_data", None)
        if not loop_data:
            return

        chat_id = agent.context.id if hasattr(agent, "context") else "unknown"

        # 收集循环效率指标
        metric = LoopEfficiencyMetric(
            timestamp=datetime.now().isoformat(),
            chat_id=chat_id,
            total_iterations=loop_data.iteration,
            unique_tools_used=list(getattr(loop_data, "tools_used", set())),
            repeat_actions=getattr(loop_data, "repeat_count", 0),
            loop_stuck=loop_data.iteration > 20,
            task_completed=getattr(loop_data, "task_completed", False),
        )

        # 计算完成时间
        if chat_id in self._start_times and metric.task_completed:
            metric.task_completed_time_seconds = time.time() - self._start_times[chat_id]

        self.collector.save_loop_efficiency(metric)

    async def on_tool_execute_before(self, data: dict) -> None:
        """工具执行前"""
        tool = data.get("tool")
        if tool:
            chat_id = tool.agent.context.id if hasattr(tool, "agent") else "unknown"
            iteration = getattr(tool.agent, "loop_data", None)
            iteration = iteration.iteration if iteration else 0

            # 记录开始时间
            key = f"{chat_id}_{tool.name}_{iteration}"
            self._start_times[key] = time.time()

    async def on_tool_execute_after(self, data: dict) -> None:
        """工具执行后"""
        tool = data.get("tool")
        response = data.get("response")

        if not tool:
            return

        agent = getattr(tool, "agent", None)
        chat_id = agent.context.id if agent and hasattr(agent, "context") else "unknown"

        loop_data = getattr(agent, "loop_data", None)
        iteration = loop_data.iteration if loop_data else 0

        # 计算执行时间
        key = f"{chat_id}_{tool.name}_{iteration}"
        execution_time = 0.0
        if key in self._start_times:
            execution_time = (time.time() - self._start_times[key]) * 1000  # ms
            del self._start_times[key]

        # 判断执行是否成功
        execution_success = True
        error_type = None
        error_message = None

        if response:
            response_message = getattr(response, "message", "")
            if response_message and ("error" in response_message.lower() or "exception" in response_message.lower()):
                execution_success = False
                error_message = response_message[:500]

        # 收集工具调用指标
        metric = ToolCallMetric(
            timestamp=datetime.now().isoformat(),
            chat_id=chat_id,
            iteration=iteration,
            tool_name=tool.name,
            tool_args=tool.args if hasattr(tool, "args") else {},
            parse_success=True,  # 能执行到这里说明解析成功
            execution_success=execution_success,
            error_type=error_type,
            error_message=error_message,
            execution_time_ms=execution_time,
        )

        self.collector.save_tool_call(metric)

    async def on_response_stream_end(self, data: dict) -> None:
        """响应流结束时"""
        agent = data.get("agent")
        response = data.get("response", "")

        if not agent:
            return

        chat_id = agent.context.id if hasattr(agent, "context") else "unknown"

        # 提取意图信息
        loop_data = getattr(agent, "loop_data", None)
        if loop_data:
            last_user_message = getattr(loop_data, "last_user_message", "")
            selected_tool = ""
            tool_args = {}
            thoughts = ""

            # 尝试解析响应中的工具调用
            if response:
                try:
                    import json
                    # 简单提取
                    if "tool_name" in response:
                        parsed = json.loads(response) if response.startswith("{") else {}
                        selected_tool = parsed.get("tool_name", "")
                        tool_args = parsed.get("tool_args", {})
                        thoughts = parsed.get("thoughts", "")
                except (json.JSONDecodeError, TypeError):
                    pass

            # 收集意图指标（待 LLM 评估）
            metric = IntentionMetric(
                timestamp=datetime.now().isoformat(),
                chat_id=chat_id,
                iteration=loop_data.iteration if loop_data else 0,
                user_message=last_user_message,
                parsed_intention=thoughts,
                selected_tool=selected_tool,
                tool_args=tool_args,
                is_correct=None,  # 待 LLM 评估
                confidence=0.0,
            )

            self.collector.save_intention(metric)


class EvaluationTriggerExtension:
    """评估触发扩展

    在特定事件后触发评估。
    """

    def __init__(self, eval_config: dict = None):
        self.eval_config = eval_config or {}
        self._pending_evaluations = []

    async def execute(self, data: dict, **kwargs) -> dict:
        return data

    async def on_monologue_end(self, data: dict) -> None:
        """主循环结束时触发评估"""
        agent = data.get("agent")
        if not agent:
            return

        chat_id = agent.context.id if hasattr(agent, "context") else "unknown"

        # 检查是否需要评估
        if self.eval_config.get("auto_evaluate", False):
            # 触发异步评估
            # 实际实现中可以放入队列，由后台任务处理
            pass


def register_extensions():
    """注册所有扩展到 Agent 系统"""
    from helpers import extension

    metric_collector = MetricCollectorExtension()
    eval_trigger = EvaluationTriggerExtension()

    # 注册扩展点
    # 注意：实际的扩展注册需要根据 Agent-Zero 的扩展系统实现

    # 示例注册方式（需要根据实际系统调整）
    # extension.register("monologue_start", metric_collector.on_monologue_start)
    # extension.register("message_loop_end", metric_collector.on_message_loop_end)
    # extension.register("tool_execute_before", metric_collector.on_tool_execute_before)
    # extension.register("tool_execute_after", metric_collector.on_tool_execute_after)
    # extension.register("response_stream_end", metric_collector.on_response_stream_end)
    # extension.register("monologue_end", eval_trigger.on_monologue_end)

    return metric_collector, eval_trigger
