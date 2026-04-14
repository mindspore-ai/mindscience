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
"""MindScienceAgent configuration management."""
from pathlib import Path
from typing import Dict, Optional

import yaml

from mindscience_agent.config.base_config import BaseConfig
from mindscience_agent.config.model_config import ModelConfig
from mindscience_agent.config.agent_config import AgentConfig
from mindscience_agent.config.tool_config import TOOL_CONFIG_CLASSES, ToolConfig
from mindscience_agent.config.log_config import LogConfig
from mindscience_agent.utils import logger


PROJECT_ROOT = Path(__file__).parent.parent.parent


class MindScienceConfig(BaseConfig):
    """
    Main configuration class for MindScienceAgent that aggregates all
    sub-configurations and provides initialization from YAML file.

    Args:
        model_defaults (Optional[ModelConfig], optional): Default model settings. Defaults to None.
        agents (Optional[Dict[str, AgentConfig]], optional): Dictionary of agent configurations. Defaults to None.
        tools (Optional[Dict[str, ToolConfig]], optional): Dictionary of tool configurations. Defaults to None.
        logging_config (Optional[LogConfig], optional): Logging configuration. Defaults to None.
    """
    def __init__(
        self,
        model_defaults: Optional[ModelConfig] = None,
        agents: Optional[Dict[str, AgentConfig]] = None,
        tools: Optional[Dict[str, ToolConfig]] = None,
        logging_config: Optional[LogConfig] = None,
    ):
        super().__init__(config_name="mindscience_agent")

        self.model_defaults = model_defaults
        self.agents = agents or {}
        self.tools = tools or {}
        self.logging_config = logging_config

    @classmethod
    def init_config_from_yaml(
        cls,
        yaml_path: str,
    ):
        """Initialize configuration from YAML file."""
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise ValueError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        config = cls._parse_config_data(config_data)

        config.print_config()

        return config

    @classmethod
    def _parse_config_data(
        cls,
        config_data: dict,
    ):
        """Parse configuration data from dictionary."""
        # Parse logging
        logging_data = config_data.get("logging", {})
        logging_config = cls._parse_log_config(logging_data)
        # init logger
        logger.init_logger(logging_config)

        # Parse model defaults
        model_defaults_data = config_data.get("model_defaults", {})
        model_defaults = cls._parse_model_config(model_defaults_data)

        # Parse agents
        agents_data = config_data.get("agents", {})
        agents = {}
        for agent_type, agent_data in agents_data.items():
            agents[agent_type] = cls._parse_agent_config(
                agent_type, agent_data, model_defaults
            )

        # Parse tools
        tools_data = config_data.get("tools", {})
        tools = {}
        for tool_name, tool_data in tools_data.items():
            tools[tool_name] = cls._parse_tool_config(tool_name, tool_data)

        config = cls(
            model_defaults=model_defaults,
            agents=agents,
            tools=tools,
            logging_config=logging_config,
        )

        return config

    @classmethod
    def _parse_model_config(cls, data: dict) -> ModelConfig:
        """Parse model configuration from data."""
        return ModelConfig(**data)

    @classmethod
    def _parse_agent_config(
        cls,
        agent_type: str,
        data: dict,
        model_defaults: ModelConfig,
    ) -> AgentConfig:
        """Parse agent configuration from data."""
        # Extract model config from 'model' sub-key
        model_data = data.get("model", {})
        # Extract agent config from 'agent' sub-key
        agent_data = data.get("agent", {})

        # Fill in missing model config fields from model_defaults
        if model_defaults:
            defaults_dict = vars(model_defaults)
            for key, value in defaults_dict.items():
                if key not in model_data or model_data.get(key) is None:
                    model_data[key] = value

        model_config = ModelConfig(**model_data)

        return AgentConfig(
            agent_type=agent_type,
            model_config=model_config,
            **agent_data,
        )

    @classmethod
    def _parse_tool_config(cls, tool_name: str, data: dict) -> ToolConfig:
        """Parse tool configuration from data."""
        tool_class = TOOL_CONFIG_CLASSES.get(tool_name, ToolConfig)

        if tool_class is ToolConfig:
            return ToolConfig(tool_name=tool_name, **data)

        return tool_class(**data)

    @classmethod
    def _parse_log_config(cls, data: dict) -> LogConfig:
        """Parse logging configuration from data."""
        return LogConfig(**data)

    def get_agent_config(self, agent_type: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent type."""
        return self.agents.get(agent_type)

    def _format_config_tree(self, data: dict, prefix: str = "") -> list:
        """Format config data as a tree structure for logging."""
        lines = []
        items = list(data.items())
        for i, (key, value) in enumerate(items):
            is_item_last = i == len(items) - 1
            connector = "└─ " if is_item_last else "├─ "
            branch = "    " if is_item_last else "│   "

            if isinstance(value, dict):
                lines.append(f"{prefix}{connector}{key}:")
                lines.extend(self._format_config_tree(value, prefix + branch))
            else:
                lines.append(f"{prefix}{connector}{key}: {value}")
        return lines

    def _get_config_dict(self, obj, depth: int = 0, max_depth: int = 3) -> dict:
        """Recursively get config object's attributes as dictionary."""
        if depth >= max_depth:
            return {"<max_depth>": "..."}

        if obj is None:
            return None

        if not isinstance(obj, dict):
            obj = vars(obj)

        result = {}
        for key, value in obj.items():
            # Skip private attributes
            if key.startswith("_"):
                continue

            # Mask sensitive fields
            if key == "api_key" and value:
                result[key] = "***"
            elif isinstance(value, BaseConfig):
                result[key] = self._get_config_dict(value, depth + 1, max_depth)
            elif hasattr(value, "__dict__"):
                result[key] = self._get_config_dict(value, depth + 1, max_depth)
            else:
                result[key] = value

        return result

    def print_config(self):
        """Log the current configuration in a tree format."""
        # Build config dict dynamically
        config_dict = {
            "agents": self._get_config_dict(self.agents) if self.agents else {},
            "tools": self._get_config_dict(self.tools) if self.tools else {},
            "logging_config": self._get_config_dict(self.logging_config) if self.logging_config else {},
        }

        lines = [
            "=" * 50,
            "MINDSCIENCEAGENT CONFIGURATION",
            "=" * 50,
        ]

        lines.extend(self._format_config_tree(config_dict))

        lines.append("=" * 50)

        text = "\n" + "\n".join(lines)
        logger.info(text)
