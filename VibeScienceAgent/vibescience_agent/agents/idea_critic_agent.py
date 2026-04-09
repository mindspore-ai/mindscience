# Copyright 2026 Huawei Technologies Co., Ltd
# Copyright 2025 InternAgent
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
"""
Idea Critic Agent for VibeScienceAgent

This module implements the IdeaCriticAgent, which analyzes existing ideas and
enhances the most promising one into a comprehensive proposal. It scores ideas
based on novelty, feasibility, and completeness, then provides detailed enhancement.
"""

from typing import Dict, Any

from vibescience_agent.config.agent_config import AgentConfig
from vibescience_agent.config.tool_config import ToolConfig
from vibescience_agent.agents.base_agent import BaseAgent, AgentExecutionError
from vibescience_agent.utils import logger
from vibescience_agent.utils.message import create_assistant_msg


_IDEA_CRITIC_SYSTEM_PROMPT = """You are a scientific critic working with a researcher. Your task is to provide
constructive criticism of multiple research ideas.

## Evaluation Criteria

- **Scientific rigor**: Is the idea grounded in solid scientific principles?
- **Logical consistency**: Are the arguments coherent and well-structured?
- **Alignment**: Does the idea align with the research goals?
- **Feasibility**: Can the idea be implemented with available resources?

## Feedback Guidelines

1. **Be objective**: Evaluate fairly, neither too harsh nor too lenient
2. **Identify strengths**: What works well in each idea?
3. **Point out weaknesses**: What are the scientific or logical gaps?
4. **Provide actionable suggestions**: How can each weakness be addressed?
5. **Be constructive**: Encourage refinement, not dismissal

Remember: Your goal is to help strengthen the ideas through iterative refinement.
Scientific progress comes from identifying and fixing weaknesses."""


class IdeaCriticAgent(BaseAgent):
    """
    Idea Critic Agent for analyzing and criticizing ideas into comprehensive proposals.

    This agent provides constructive criticism of research ideas by evaluating them against
    scientific rigor, logical consistency, alignment with research goals, and feasibility.

    Args:
        model (BaseModel): Language model for idea analysis and enhancement.
        config (AgentConfig): Agent configuration.
        tool_config (Dict[str, ToolConfig], optional): Tool configurations for the agent.

    Inputs:
        - messages: List of message dicts containing conversation history
        - params: Additional parameters including

    Outputs:
        - Dict containing criticism for previous generated ideas.
    """

    def __init__(self, model, config: AgentConfig,
                 tool_config: Dict[str, ToolConfig] = None):
        """Initialize IdeaCriticAgent with model and config."""
        super().__init__(model, config, tool_config)
        self.system_prompt = _IDEA_CRITIC_SYSTEM_PROMPT
        logger.debug("IdeaCriticAgent initialized with system prompt")

    async def execute(self, messages, **params) -> Dict[str, Any]:
        """Execute the idea analysis and enhancement task."""
        if not messages:
            raise AgentExecutionError("IdeaCriticAgent requires non-empty message history")

        logger.debug(f"IdeaCriticAgent call model inputs:\n{messages}")

        # Call the model
        content = await self._call_model(
            prompt=messages,
            system_prompt=self.system_prompt
        )

        logger.debug("IdeaCriticAgent call model output:\n" + content)

        # Process and return the output
        return create_assistant_msg(content)

    def _process_output(self, content):
        """Format the critic feedback with header."""
        feedback = "# Previous Feedback from Idea Critic Agent\n" + content
        return create_assistant_msg(feedback)
