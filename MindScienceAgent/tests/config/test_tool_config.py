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
"""Tests for tool_config module."""
import pytest

from mindscience_agent.config.tool_config import ToolConfig


class TestToolConfig:
    """Test cases for ToolConfig."""

    def test_creation_with_tool_name(self):
        """Test creating ToolConfig with tool_name."""
        config = ToolConfig(tool_name="test_tool")
        assert config.get_tool_name() == "test_tool"
        assert config.get_config_name() == "test_tool_tool"

    def test_creation_with_kwargs(self):
        """Test creating ToolConfig with additional kwargs."""
        config = ToolConfig(tool_name="test_tool", param1="value1", param2=42)
        assert config.param1 == "value1"
        assert config.param2 == 42
