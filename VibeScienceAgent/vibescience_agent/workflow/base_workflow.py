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
Base workflow for VibeScienceAgent.

Contains shared initialization and runtime utilities that can be reused by
different workflow implementations.
"""

from vibescience_agent.config.vibescience_config import VibeScienceConfig
from vibescience_agent.agents.agent_factory import AgentFactory
from vibescience_agent.model.model_factory import ModelFactory
from vibescience_agent.utils import logger


class BaseWorkflow:
    """Shared base workflow with common setup and helper methods."""
    AGENT_TYPES: tuple[str, ...] = ()
    REQUIRED_MODEL_AGENT_TYPES: tuple[str, ...] = ()

    def __init__(
        self,
        config: VibeScienceConfig,
    ):
        # Single config source — constructor params override config values
        self.config = config

    # =========================================================================
    # Initialization
    # =========================================================================

    def _init_model(self):
        """Initialize shared model factory for agent construction."""
        self.model_factory = ModelFactory()     # pylint: disable=W0201

    def _print_message(self, result, process):
        if result['role'] == "user":
            msg_type = " User Message "
        elif result['role'] == "assistant":
            msg_type = " Assistant Message "
        else:
            msg_type = "Unknown Message"
        msg_ending = "\n" + "=" * 60
        msg_title = "=" * 20 + msg_type + "=" * 20 + "\n"
        logger.info(f"message added to context from {process}:\n" + msg_title + result['content'] + msg_ending)

    # =========================================================================
    # Agent Creation (unified via AgentFactory)
    # =========================================================================

    def _create_agents(self):
        """Create all agents via AgentFactory."""
        for agent_type in self.AGENT_TYPES:
            agent = AgentFactory.create_agent(
                agent_type=agent_type,
                config=self.config.get_agent_config(agent_type),
                tool_config=self.config.tools,
                model_factory=self.model_factory,
            )
            setattr(self, f"{agent_type}_agent", agent)

    # =========================================================================
    # Extension Points
    # =========================================================================

    def _create_workflow(self):
        raise NotImplementedError
