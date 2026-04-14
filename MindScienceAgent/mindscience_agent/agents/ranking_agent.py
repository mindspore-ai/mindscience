# Copyright 2026 Huawei Technologies Co., Ltd
# Copyright 2025 AI-Researcher
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
Idea Ranking Agent for MindScienceAgent

This module implements the RankingAgent, which evaluates and selects the best
idea from a set of enhanced proposals. It provides comprehensive evaluation based
on technical innovation, potential impact, feasibility, and completeness.
"""

from typing import Dict, Any

from mindscience_agent.config.agent_config import AgentConfig
from mindscience_agent.config.tool_config import ToolConfig
from mindscience_agent.agents.base_agent import BaseAgent, AgentExecutionError
from mindscience_agent.utils import logger
from mindscience_agent.utils.message import create_assistant_msg


_RANKING_SYSTEM_PROMPT = """You are a scientific idea evaluator. Your task is to objectively evaluate
and rank research ideas based on specific criteria.

## Scoring Criteria (0.0 - 10.0)

1. **Technical Innovation**: How novel is the idea? Does it have breakthrough potential?
2. **Potential Impact**: How significant is the problem it solves?
3. **Feasibility**: How practical is the methodology? Can it be implemented?
4. **Completeness**: How deep and rigorous is the technical design?

## Evaluation Guidelines

- Apply the same standards consistently across all ideas
- Provide clear, specific rationales for each score
- Consider both strengths and weaknesses
- Focus on scientific merit and testability, not writing style
- Identify the most promising idea to pursue

Your goal is to help the researcher select the best idea based on scientific value."""


class RankingAgent(BaseAgent):
    """
    Ranking Agent for evaluating and selecting the best idea from enhanced proposals.

    This agent objectively evaluates and ranks research ideas based on specific criteria
    including technical innovation, potential impact, feasibility, and completeness.

    Args:
        model (BaseModel): Language model for idea evaluation.
        config (AgentConfig): Agent configuration.
        tool_config (Dict[str, ToolConfig], optional): Tool configurations for the agent.

    Inputs:
        - messages: List of message dicts containing conversation history
        - params: Additional parameters including

    Outputs:
        - Dict containing the selected best idea with justification.
    """
    def __init__(self, model, config: AgentConfig,
                 tool_config: Dict[str, ToolConfig] = None):
        super().__init__(model, config, tool_config)
        self.system_prompt = _RANKING_SYSTEM_PROMPT
        logger.debug("RankingAgent initialized with system prompt")

    async def execute(self, messages, **params) -> Dict[str, Any]:
        """ Execute the idea ranking and selection task. """
        if not messages:
            raise AgentExecutionError("RankingAgent requires non-empty message history")

        logger.debug(f"RankingAgent call model inputs:\n{messages}")

        # Call the model
        content = await self._call_model(
            prompt=messages,
            system_prompt=self.system_prompt
        )

        logger.debug("RankingAgent call model output:\n" + content)

        # Process and return the output
        return create_assistant_msg(content)
