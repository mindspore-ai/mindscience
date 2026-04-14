# Copyright 2026 Huawei Technologies Co., Ltd
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
Idea Generation Agent for MindScienceAgent

This module implements the Idea Agent, which generates innovative scientific ideas
by analyzing academic papers and identifying research gaps. It performs thorough
literature review and generates comprehensive ideas with detailed technical solutions.
"""

from typing import Dict, TypedDict

from langchain_core.globals import set_debug
from langchain_core.messages import AIMessage
from langgraph.graph import START, StateGraph

from mindscience_agent.config.agent_config import IdeaAgentConfig
from mindscience_agent.config.tool_config import ToolConfig
from mindscience_agent.agents.base_agent import BaseAgent, AgentExecutionError
from mindscience_agent.utils import logger
from mindscience_agent.utils.message import create_assistant_msg
from mindscience_agent.utils.prompts import generate_prompt


class _IdeaSubgraphState(TypedDict):
    messages: list


_IDEA_GENERATION_SYSTEM_PROMPT = """
You are a creative scientific idea generator. Your task is to generate at least {minimal_ideas} scientific ideas based on the user's research goal and the provided literature.

## Requirements for Each Idea

1. **Novelty**: The idea must be new and not obvious from existing literature. Ground it in solid scientific principles.

2. **Mechanism**: Propose a specific mechanism or pathway. Include a brief explanation of how it works.

3. **Testability**: Make the idea concrete and testable through experiments.

4. **Relevance**: Directly address the research goal and have clear scientific significance.

5. **Feasibility**: Consider practical constraints and domain-specific knowledge.

## Output Guidelines

- Be creative but scientifically rigorous
- Make unexpected connections between concepts or apply principles from one field to another
- Each idea must be distinct and different from the others
- Write clearly enough for a domain expert to understand, but avoid requiring specialized knowledge outside the field
"""


class IdeaAgent(BaseAgent):
    """
    Idea Agent for generating innovative scientific ideas from paper analysis.

    This agent analyzes research papers and generates novel scientific ideas based on the
    survey results. It propose testable hypotheses that address specific research goals.

    Args:
        model (Model): Language model for generating ideas.
        config (IdeaAgentConfig): Agent configuration containing parameters like minimal_ideas.
        tool_config (Dict[str, ToolConfig], optional): Tool configurations for the agent.

    Inputs:
        - messages: List of message dicts containing conversation history
        - params: Additional parameters including

    Outputs:
        - List of messages containing the generated ideas in a structured format,
          ready for further analysis and refinement.
    """
    def __init__(self, model, config: IdeaAgentConfig,
                 tool_config: Dict[str, ToolConfig] = None):
        super().__init__(model, config, tool_config)

        self.minimal_ideas = config.minimal_ideas
        self.debug = logger.LOG_LEVEL == logger.LOG_LEVEL_MAP["DEBUG"]
        self._compiled_subgraph = self._build_idea_subgraph()
        self.ctx = self._build_agent_tool_context()
        self.ctx["skills"] = []     # skills already passed to deep agent

    def _build_idea_subgraph(self):
        """ Build the LangGraph workflow for idea generation. """
        deep_agent = self._build_deep_agent()
        workflow = StateGraph(_IdeaSubgraphState)
        workflow.add_node("idea_agent", deep_agent)
        workflow.add_edge(START, "idea_agent")
        return workflow.compile()

    async def execute(self, messages, **params) -> list:
        """ Execute the idea generation task. """
        if not messages:
            raise AgentExecutionError("IdeaAgent requires non-empty message history")

        user_query = messages[0]["content"]
        enable_idea_critic = params.get("enable_idea_critic", False)
        survey_results = params.get("survey_results", None)

        self._build_system_prompt(user_query, survey_results, enable_idea_critic)

        processed_input = self._process_input(messages, enable_idea_critic)
        logger.debug(f"IdeaAgent call model inputs:\n{processed_input}")
        input_msg = [("system", self.system_prompt), ("user", processed_input)]
        remaining_retries = self.config.max_retries

        if self.debug:
            set_debug(self.debug)

        while True:
            try:
                final_state = await self._invoke_subgraph(input_msg)
                break
            except Exception as e:
                remaining_retries -= 1
                logger.warning(f"IdeaAgent execution error: {e}. Retries left: {remaining_retries}")

                if remaining_retries <= 0:
                    logger.warning(f"IdeaAgent failed after max retries. Error: {e}")

                    if self.debug:
                        set_debug(False)

                    return {"messages": []}

        if self.debug:
            set_debug(False)

        # Process and return the output
        return self._process_output(final_state)

    def _build_system_prompt(self, user_query, survey_results, enable_idea_critic):
        """ Build the system prompt with survey results and critic feedback instructions. """
        if self.system_prompt:
            return

        self.run_tool_retrieval_if_enabled(user_query)

        base_prompt = _IDEA_GENERATION_SYSTEM_PROMPT.format(minimal_ideas=self.minimal_ideas)
        if enable_idea_critic:
            base_prompt += (
                "You may or may not receive feedbacks from an idea critic agent. "
                "If so, address the feedbacks by revising your ideas or "
                "coming up with some new ideas."
            )

        self.system_prompt = generate_prompt(
            base_prompt=base_prompt,
            tool_desc=self.ctx["tool_desc"],
            use_tool_retriever=self.use_tool_retriever,
            survey_results=survey_results
        )

        logger.debug("IdeaAgent system prompt:\n" + self.system_prompt)

    def _process_input(self, messages, enable_idea_critic):
        """ Build the prompt for idea generation. """
        # Start with the goal
        user_query = messages[0]["content"]
        prompt = f"# Research Goal\n{user_query}\n"
        prompt += "Please give novel ideas based on the research goal and the literature"
        if enable_idea_critic:
            prompt += ", or revise generated ideas based on the critic feadbacks."

        return prompt

    def _process_output(self, final_state: dict) -> list:
        """ Extract and format the generated ideas from the final state. """
        messages = final_state.get("messages", [])
        if len(messages) > 0:
            outputs = []
            for message in messages:
                if isinstance(message, AIMessage) and message.content.strip():
                    outputs.append(create_assistant_msg(message.content))
            return outputs
        return [create_assistant_msg("No ideas were generated.")]
