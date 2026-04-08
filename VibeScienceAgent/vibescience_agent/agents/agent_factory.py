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
Agent Factory for VibeScienceAgent

Provides a unified registry and factory for creating all agent types.
All agents (survey, plan, critic, execute) are created through
this factory using consistent configuration and model injection.
"""

import importlib

from typing import Dict, Type

from vibescience_agent.model.model_factory import ModelFactory
from vibescience_agent.agents.base_agent import BaseAgent
from vibescience_agent.utils import logger
from vibescience_agent.config.agent_config import AgentConfig
from vibescience_agent.config.tool_config import ToolConfig


class AgentFactory:
    """Factory for creating agent instances based on configuration.

    Maintains a registry of available agent types and creates properly
    configured instances, with model injection via ModelFactory or
    direct model passing.
    """

    @classmethod
    def create_agent(
        cls,
        agent_type: str,
        config: AgentConfig,
        tool_config: Dict[str, ToolConfig] = None,
        model_factory: "ModelFactory" = None,
        model = None,
        **kwargs
    ) -> BaseAgent:
        """Create an agent instance of the specified type.

        The model can be supplied in two ways:
        1. Via ``model`` parameter – uses the given model directly.
        2. Via ``model_factory`` – creates a model through ModelFactory.

        Args:
            agent_type: Type identifier (e.g. "plan", "execute").
            config: Agent configuration dict.
            model_factory: Optional ModelFactory for creating models.
            model: Optional pre-built model instance.

        Returns:
            Configured agent instance.
        """

        if model is None and model_factory is not None:
            model = model_factory.create_model(config.model_config)
        elif model is None:
            raise ValueError(
                f"Either model or model_factory must be provided for agent {agent_type}"
            )

        agent_class = cls._import_agent_class(agent_type)

        agent = agent_class(model, config, tool_config, **kwargs)
        logger.info(f"Created agent instance: {agent_type} ({agent_class.__name__})")
        return agent

    @staticmethod
    def _import_agent_class(agent_type: str) -> Type[BaseAgent]:
        """Dynamically import the agent class based on type.

        Args:
            agent_type: Type identifier (e.g. "plan", "execute").

        Returns:
            The agent class.

        Raises:
            ImportError: If the agent class cannot be imported.
        """
        module_name = f"vibescience_agent.agents.{agent_type}_agent"
        class_name = f"{agent_type.capitalize()}Agent"

        module = importlib.import_module(module_name)
        return getattr(module, class_name)
