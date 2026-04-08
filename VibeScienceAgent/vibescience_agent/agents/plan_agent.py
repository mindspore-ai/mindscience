# Copyright 2026 Huawei Technologies Co., Ltd
# Copyright 2025 Biomni
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Plan Agent for VibeScienceAgent

Handles task decomposition and structured plan generation. Wraps the original
planner() function into a BaseAgent subclass for unified agent management.
"""

from typing import Dict

from vibescience_agent.agents.base_agent import BaseAgent
from vibescience_agent.utils.prompts import generate_prompt
from vibescience_agent.utils.utils import serialize_agent_messages
from vibescience_agent.utils.message import create_assistant_msg
from vibescience_agent.utils import logger
from vibescience_agent.config.agent_config import AgentConfig
from vibescience_agent.config.tool_config import ToolConfig

_PLAN_BASE_PROMPT = """
You are a helpful assistant assigned with the task of problem-solving.
To achieve this, you will be using a Execute Agent equipped with a variety of tool functions, data, and softwares to assist you throughout the process.

Given a task, make a plan first. The plan should be a numbered list of steps that you will take to solve the task. Be specific and detailed.
Format your plan as a checklist with empty checkboxes like this:
1. [ ] First step
2. [ ] Second step
3. [ ] Third step

Follow the plan step by step. After completing each step, update the checklist by replacing the empty checkbox with a checkmark:
1. [✓] First step (completed)
2. [ ] Second step
3. [ ] Third step

If a step fails or needs modification, mark it with an X and explain why:
1. [✓] First step (completed)
2. [✗] Second step (failed because...)
3. [ ] Modified second step
4. [ ] Third step

Always show the updated plan after each step so the user can track progress.

At each turn, you should first provide your thinking and reasoning given the conversation history.
After that, you have two options:

1) Interact with a Execute Agent and receive the corresponding output within <observation></observation>. Your code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>. IMPORTANT: You must end the code block with </execute> tag.
   - For Python code (default): <execute> print("Hello World!") </execute>

2) When you think it is ready, directly provide a solution that adheres to the required format for the given task to the user. Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>. IMPORTANT: You must end the solution block with </solution> tag.

You have many chances to interact with the code agent to receive the observation. So you can decompose your code into multiple steps.
Don't overcomplicate the code. Keep it simple and easy to understand.
When writing the code, please print out the steps and results in a clear and concise manner, like a research log.
When calling the existing python functions in the function dictionary, YOU MUST SAVE THE OUTPUT and PRINT OUT the result.
For example, result = understand_scRNA(XXX) print(result)
Otherwise the system will not be able to know what has been done.

In each response, you must include EITHER <execute> or <solution> tag. Not both at the same time. Do not respond with messages without any tags. No empty messages.
"""

class PlanAgent(BaseAgent):
    """Plan Agent: analyses tasks and creates structured execution plans.

    Tool and library context is built inside this agent (see ``TOOL_DESCRIPTION_MODULES``
    and optional ``custom_*`` keys). When ``use_tool_retriever`` is enabled,
    the first ``execute`` runs :func:`run_tool_retrieval_once_if_enabled` and
    updates :meth:`update_selected_resources`.
    """

    #: Subset of ``read_module2api`` fields, e.g. ``frozenset({"literature", "support_tools"})``.
    #: ``None`` means all built-in tool description modules.
    TOOL_DESCRIPTION_MODULES: frozenset[str] | None = None

    def __init__(self, model, config: AgentConfig, tool_config: Dict[str, ToolConfig] = None, **kwargs):
        super().__init__(model, config, tool_config)
        self.sciencedata_with_desc = kwargs.get("sciencedata_with_desc", [])
        self.sciencedata_path = kwargs.get("sciencedata_path", "")

        self.ctx = self._build_agent_tool_context(self.TOOL_DESCRIPTION_MODULES)

    async def execute(self, messages, **params):
        """Generate a plan based on message history.

        Args:
            context: Must contain ``messages`` (list[BaseMessage]).
            params: May contain ``survey_results``.

        Returns:
            Dict with ``content`` (str) – the raw plan text from the LLM.
        """
        survey_results = params.get("survey_results", None)
        user_query = messages[0]["content"]

        self._build_system_prompt(user_query, survey_results)

        logger.debug(f"PlanAgent call model inputs:\n{serialize_agent_messages(messages)}")

        content = await self._call_model(prompt=messages, system_prompt=self.system_prompt)

        logger.debug("PlanAgent call model output:\n" + content)

        return self._process_output(content)

    def _process_output(self, content):
        if "<execute>" in content and "</execute>" not in content:
            content += "</execute>"
        if "<solution>" in content and "</solution>" not in content:
            content += "</solution>"
        if "<think>" in content and "</think>" not in content:
            content += "</think>"
        return create_assistant_msg(content)

    def _build_system_prompt(self, user_query, survey_results):
        """Build the planner system prompt with environment resources."""
        if self.system_prompt:
            return
        self.run_tool_retrieval_if_enabled(user_query)

        self.system_prompt = generate_prompt(
            base_prompt=_PLAN_BASE_PROMPT,
            tool_desc=self.ctx["tool_desc"],
            library_content_list=self.ctx["library_content_list"],
            use_tool_retriever=self.use_tool_retriever,
            custom_tools=self.ctx["custom_tools"],
            custom_data=self.ctx["custom_data"],
            custom_software=self.ctx["custom_software"],
            survey_results=survey_results,
            sciencedata_path=self.sciencedata_path,
            sciencedata_content=self.sciencedata_with_desc,
            skill_path=self.skill_path
        ) + (
                "\n\nIMPORTANT FOR GPT MODELS: You MUST use XML tags <think> or "
                "<solution> in EVERY response. Do not use markdown code blocks "
                "(```) - use <execute> tags instead."
            )

        logger.debug("PlanAgent system prompt:\n" + self.system_prompt)
