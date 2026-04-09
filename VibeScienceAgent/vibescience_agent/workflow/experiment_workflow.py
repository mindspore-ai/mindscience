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
"""
Experiment workflow for VibeScienceAgent

Orchestrates the multi-agent pipeline: plan -> [critic] -> execute.
All agents are created through AgentFactory with unified configuration from VibeScienceConfig.
"""

import re
import uuid
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from vibescience_agent.config.vibescience_config import VibeScienceConfig
from vibescience_agent.utils.message import create_user_msg, create_assistant_msg
from vibescience_agent.utils.utils import extract_between_regex
from vibescience_agent.utils import logger
from vibescience_agent.workflow.base_workflow import BaseWorkflow
from vibescience_agent.context.simple_context import SimpleContext


class AgentState(TypedDict):
    context: SimpleContext
    next_step: str | None


class ExperimentWorkflow(BaseWorkflow):
    """Orchestrator for the VibeScienceAgent experiment pipeline.

    All configuration comes from a single ``VibeScienceConfig`` instance.
    Constructor parameters serve as highest-priority overrides.
    """
    AGENT_TYPES = ("plan", "critic", "execute")
    REQUIRED_MODEL_AGENT_TYPES = ("plan", "execute")

    def __init__(
        self,
        config: VibeScienceConfig,
        enable_critic: bool = False,
        test_time_scale_round: int = 1,
    ):
        super().__init__(config=config)
        logger.info(f"ExperimentWorkflow inputs: "
                    f"enable_critic={enable_critic}, test_time_scale_round={test_time_scale_round}")
        self.enable_critic = enable_critic
        self.test_time_scale_round = test_time_scale_round
        if self.enable_critic:
            self.critic_count = 0

        self._init_model()
        self._create_agents()
        self._create_workflow()

    # =========================================================================
    # Workflow
    # =========================================================================

    def _create_workflow(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("plan", self.plan)
        workflow.add_node("execute", self.execute)

        workflow.add_edge(START, "plan")

        if self.enable_critic:
            workflow.add_node("critic", self.critic)

            workflow.add_conditional_edges(
                "plan",
                self.routing_function,
                path_map={"plan": "plan", "critic": "critic", "execute": "execute", "end": END},
            )
            workflow.add_edge("critic", "plan")
            workflow.add_edge("execute", "plan")
        else:
            workflow.add_conditional_edges(
                "plan",
                self.routing_function,
                path_map={"plan": "plan", "execute": "execute", "end": END},
            )
            workflow.add_edge("execute", "plan")

        self.app = workflow.compile()
        self.checkpointer = MemorySaver()
        self.app.checkpointer = self.checkpointer

    # =========================================================================
    # Workflow Nodes
    # =========================================================================

    async def plan(self, state: AgentState) -> AgentState:
        logger.info("Planning...")
        result = await self.plan_agent.execute(     # pylint: disable=E1101
            messages=state["context"].get_context("messages"),
        )

        state["context"].add_context("messages", result)
        self._print_message(result, "PLAN")

        msg = result["content"]

        execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL | re.IGNORECASE)
        think_match = re.search(r"<think>(.*?)</think>", msg, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL | re.IGNORECASE)

        if answer_match:
            state["next_step"] = "end"
        elif execute_match:
            state["next_step"] = "execute"
        elif think_match:
            critic_enabled = self.enable_critic and self.critic_count < self.test_time_scale_round
            state["next_step"] = "critic" if critic_enabled else "plan"
        else:
            logger.warning("Plan output parsing error: no <think>, <execute> or <solution> tags found")
            # Count prior tag-remediation user turns — not AIMessages. Failed plan outputs (e.g. <tool_call>...)
            # never contain the phrase "There are no tags", so the old AIMessage-based count stayed 0 forever.
            _remediation_marker = "But there are no tags in the current"
            error_count = sum(
                1 for m in state["context"].get_context("messages")
                if m["role"] == "user" and _remediation_marker in m["content"]
            )
            if error_count >= 2:
                logger.error("Detected repeated parsing errors, ending conversation")
                state["next_step"] = "end"
                state["context"].add_context(
                    "messages",
                    create_assistant_msg("Execution terminated due to repeated parsing errors.")
                )
            else:
                state["context"].add_context(
                    "messages",
                    create_user_msg(
                        "Each response must include <think>, <execute> or <solution> tag. "
                        "But there are no tags in the current response. "
                        "Please follow the instruction, fix and regenerate."
                    )
                )
                state["next_step"] = "plan"
        return state

    async def critic(self, state: AgentState) -> AgentState:
        logger.info("Criticing...")
        result = await self.critic_agent.execute(       # pylint: disable=E1101
            messages=state["context"].get_context("messages")
        )
        state["context"].add_context("messages", result)
        self._print_message(result, "CRITIC")
        self.critic_count += 1
        state["next_step"] = "plan"
        return state

    async def execute(self, state: AgentState) -> AgentState:
        logger.info("Executing...")
        result = await self.execute_agent.execute(     # pylint: disable=E1101
            messages=state["context"].get_context("messages")
        )
        state["context"].add_context("messages", result)
        self._print_message(result, "EXECUTE")
        state["next_step"] = "plan"
        return state

    # =========================================================================
    # Routing
    # =========================================================================

    def routing_function(
        self, state: AgentState,
    ) -> Literal["plan", "execute", "end", "critic"]:
        next_step = state.get("next_step")
        valid = {"plan", "execute", "end", "critic"}
        if next_step in valid:
            return next_step
        raise ValueError(f"Unexpected next_step: {next_step}")

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    async def run(self, prompt):
        """Execute the agent pipeline with the given prompt."""
        logger.info("Workflow run started with user query:\n" + prompt)

        if self.enable_critic:
            self.critic_count = 0

        inputs = {"context": SimpleContext(), "next_step": None}
        inputs["context"].add_context("messages", create_user_msg(prompt))
        config = {
            "recursion_limit": 500,
            "configurable": {"thread_id": uuid.uuid4().hex},
        }

        message = None
        async for s in self.app.astream(inputs, stream_mode="values", config=config):
            message = s["context"].get_context("messages")[-1]

        solution = extract_between_regex(message["content"], "<solution>", "</solution>")

        logger.info(f"Workflow run finished with solution:\n{solution}")
        return solution
