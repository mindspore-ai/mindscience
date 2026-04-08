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
"""Critic Agent for VibeScienceAgent

Provides constructive feedback on plans and execution results.
Wraps the original critic() function into a BaseAgent subclass.
"""
from typing import Dict

from vibescience_agent.agents.base_agent import BaseAgent, AgentExecutionError
from vibescience_agent.utils.utils import serialize_agent_messages
from vibescience_agent.utils.message import create_user_msg
from vibescience_agent.utils import logger
from vibescience_agent.config.agent_config import AgentConfig
from vibescience_agent.config.tool_config import ToolConfig


class CriticAgent(BaseAgent):
    """Critic Agent: reviews plans and provides improvement feedback.

    Receives the full conversation history via ``context["messages"]``
    and returns structured feedback.
    """

    def __init__(self, model, config: AgentConfig, tool_config: Dict[str, ToolConfig] = None):
        super().__init__(model, config, tool_config)

    async def execute(self, messages, **params):
        """Generate critical feedback on the current plan.

        Args:
            context: Must contain ``messages`` (list[BaseMessage]) – the conversation
                     history including the original user request and plan.

        Returns:
            Dict with ``content`` (str) – the critic feedback text.
        """
        if not messages:
            raise AgentExecutionError("CriticAgent requires non-empty message history")

        user_request = messages[0]["content"] if messages else ""

        feedback_prompt = (
            f"Here is a reminder of what the user requested: {user_request}\n"
            "Examine the previous executions, reasoning, and solutions.\n"
            "Critic harshly on what could be improved?\n"
            "Be specific and constructive.\n"
            "Think hard what are missing to solve the task.\n"
            "No question asked, just feedbacks."
        )

        messages.append(create_user_msg(feedback_prompt))

        logger.debug("CriticAgent call model inputs:\n", serialize_agent_messages(messages))
        content = await self._call_model(
            prompt=messages
        )
        logger.debug("CriticAgent call model output:\n" + content)

        return self._process_output(content)

    def _process_output(self, content):
        return create_user_msg(
            f"Wait... this is not enough to solve the task. "
            f"Here are some feedbacks for improvement:\n{content}"
        )
