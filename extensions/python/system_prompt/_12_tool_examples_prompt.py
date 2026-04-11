import os
from typing import Any

from helpers.extension import Extension
from helpers.print_style import PrintStyle
from agent import Agent, LoopData


class ToolExamplesPrompt(Extension):

    async def execute(
        self,
        system_prompt: list[str] = [],
        loop_data: LoopData = LoopData(),
        **kwargs: Any,
    ):
        if not self.agent:
            return
        try:
            # Use agent's root_path to locate the knowledge directory
            root = getattr(self.agent, "root_path", None) or os.getcwd()
            examples_path = os.path.join(
                root, "knowledge", "main", "tool_call_reference_examples.md"
            )
            if os.path.exists(examples_path):
                with open(examples_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if content and content.strip():
                    system_prompt.append(
                        "\n## Tool Call Format Reference\n" + content
                    )
            else:
                PrintStyle().error(f"Tool examples file not found at: {examples_path}")
        except Exception as e:
            PrintStyle().error(f"Error loading tool examples: {e}")
