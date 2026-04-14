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
Unit tests for ExecuteAgent.

Tests:: initialization, and tool_retriever toggle.
"""

from unittest.mock import Mock, AsyncMock, patch
import pytest

from mindscience_agent.agents import ExecuteAgent
from mindscience_agent.config import ModelConfig, AgentConfig
from mindscience_agent.utils import create_user_msg, create_assistant_msg


def _create_execute_config(use_tool_retriever=False):
    """Helper to create minimal ExecuteAgent config."""
    model_config = ModelConfig(model_name="test-model", base_url="http://test-model-api.com", api_key="test-api-key")
    return AgentConfig(agent_type="execute", model_config=model_config, use_tool_retriever=use_tool_retriever)


# =============================================================================
# Initialization Tests
# =============================================================================

class TestExecuteAgentInitialization:
    """Test suite for ExecuteAgent initialization."""

    @pytest.mark.unit
    def test_init(self, mock_base_model):
        """Test ExecuteAgent initialization."""
        with patch('mindscience_agent.agents.execute_agent.ExecuteAgent._build_execute_subgraph'):
            agent = ExecuteAgent(mock_base_model, _create_execute_config())

            assert agent.agent_type == "execute"
            assert hasattr(agent, '_compiled_subgraph')


# =============================================================================
# Execute Tests
# =============================================================================

class TestExecuteAgentExecute:
    """Test suite for ExecuteAgent execute method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute(self, mock_base_model):
        """Test ExecuteAgent execute method."""
        mock_final_state = {
            "messages": [
                Mock(type="tool", name="python_executor", content="Success")
            ]
        }

        with patch('mindscience_agent.agents.execute_agent.ExecuteAgent._build_execute_subgraph'):
            agent = ExecuteAgent(mock_base_model, _create_execute_config())
            agent._invoke_subgraph = AsyncMock(return_value=mock_final_state)   # pylint: disable=W0212

            messages = [
                create_user_msg("Execute code"),
                create_assistant_msg("<execute>print('test')</execute>")
            ]
            result = await agent.execute(messages)

            assert result["role"] == "assistant"
            assert "<observation>" in result["content"]


# =============================================================================
# Tool Retriever Toggle Tests
# =============================================================================

class TestExecuteAgentToolRetrieverToggle:
    """Test suite for ExecuteAgent tool retriever toggle."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_tool_retriever_enabled(self, mock_base_model, mock_tool_registry, mock_tool_retriever):
        """Test ExecuteAgent with tool retriever enabled."""
        mock_final_state = {
            "messages": [
                Mock(type="tool", name="python_executor", content="Success")
            ]
        }

        with patch('mindscience_agent.agents.execute_agent.ExecuteAgent._build_execute_subgraph'):
            agent = ExecuteAgent(mock_base_model, _create_execute_config(use_tool_retriever=True))
            agent.tool_registry = mock_tool_registry
            agent.retriever = mock_tool_retriever
            agent._invoke_subgraph = AsyncMock(return_value=mock_final_state)   # pylint: disable=W0212

            messages = [
                create_user_msg("Execute code"),
                create_assistant_msg("<execute>print('test')</execute>")
            ]
            result = await agent.execute(messages)

            assert result["role"] == "assistant"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_tool_retriever_disabled(self, mock_base_model):
        """Test ExecuteAgent with tool retriever disabled."""
        mock_final_state = {
            "messages": [
                Mock(type="tool", name="python_executor", content="Success")
            ]
        }

        with patch('mindscience_agent.agents.execute_agent.ExecuteAgent._build_execute_subgraph'):
            agent = ExecuteAgent(mock_base_model, _create_execute_config(use_tool_retriever=False))
            agent._invoke_subgraph = AsyncMock(return_value=mock_final_state)   # pylint: disable=W0212

            messages = [
                create_user_msg("Execute code"),
                create_assistant_msg("<execute>print('test')</execute>")
            ]
            result = await agent.execute(messages)

            assert result["role"] == "assistant"
