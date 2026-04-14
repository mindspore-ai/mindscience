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
"""Agent factory for creating agent instances based on configuration."""
import importlib

from typing import Dict, Type

from mindscience_agent.model.model import Model
from mindscience_agent.agents.base_agent import BaseAgent
from mindscience_agent.utils import logger
from mindscience_agent.config.agent_config import AgentConfig
from mindscience_agent.config.tool_config import ToolConfig


class AgentManager:
    """Factory for creating agent instances based on configuration.

    Args:
        agent_type (str): Type identifier (e.g. "plan", "execute").
        config (AgentConfig): Agent configuration dict.
        tool_config (Dict[str, ToolConfig]): Tool configuration dict.
        model (Optional[Model]): Optional pre-built model instance.
    """
    @classmethod
    def create_agent(
        cls,
        agent_type: str,
        config: AgentConfig,
        tool_config: Dict[str, ToolConfig] = None,
        model: Model = None,
        **kwargs
    ) -> BaseAgent:
        """Create an agent instance of the specified type."""
        if model is None:
            model = Model(config.model_config)

        agent_class = cls._import_agent_class(agent_type)

        agent = agent_class(model, config, tool_config, **kwargs)
        logger.info(f"Created agent instance: {agent_type} ({agent_class.__name__})")
        return agent

    @staticmethod
    def _import_agent_class(agent_type: str) -> Type[BaseAgent]:
        """Dynamically import agent class based on type."""
        module_name = f"mindscience_agent.agents.{agent_type}_agent"
        class_name = f"{agent_type.capitalize()}Agent"

        module = importlib.import_module(module_name)
        return getattr(module, class_name)
