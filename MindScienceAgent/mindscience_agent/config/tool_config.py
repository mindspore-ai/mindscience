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
"""Tool configuration classes for MindScienceAgent."""
from mindscience_agent.config.base_config import BaseConfig


class ToolConfig(BaseConfig):
    """Base configuration class for tool settings.

    Args:
        tool_name (str): Name identifier for the tool.
        **kwargs: Additional tool-specific configuration parameters.
    """
    def __init__(
        self,
        tool_name,
        **kwargs,
    ):
        super().__init__(config_name=f"{tool_name}_tool")

        self._tool_name = tool_name

        self.update_attrs(**kwargs)

    def get_tool_name(self):
        """Get tool name."""
        return self._tool_name


# Tool config class registry for extensibility
TOOL_CONFIG_CLASSES = {
}
