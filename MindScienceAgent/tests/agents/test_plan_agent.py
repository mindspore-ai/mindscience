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
Unit tests for PlanAgent.

Tests:: initialization, and tool_retriever toggle.
"""

from unittest.mock import Mock, AsyncMock
import pytest

from mindscience_agent.agents import PlanAgent
from mindscience_agent.config import AgentConfig, ModelConfig
from mindscience_agent.utils import create_user_msg


def _create_plan_config(use_tool_retriever=False):
    """Helper to create minimal PlanAgent config."""
    model_config = ModelConfig(model_name="test-model", base_url="http://test-model-api.com", api_key="test-api-key")
    return AgentConfig(agent_type="plan", model_config=model_config, use_tool_retriever=use_tool_retriever)


# =============================================================================
# Initialization Tests
# =============================================================================

class TestPlanAgentInitialization:
    """Test suite for PlanAgent initialization."""

    @pytest.mark.unit
    def test_init(self, mock_base_model):
        """Test PlanAgent initialization."""
        agent = PlanAgent(mock_base_model, _create_plan_config())

        assert agent.agent_type == "plan"
        assert hasattr(agent, 'ctx')


# =============================================================================
# Execute Tests
# =============================================================================

class TestPlanAgentExecute:
    """Test suite for PlanAgent execute method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute(self, mock_base_model):
        """Test PlanAgent execute method."""
        mock_base_model.generate = AsyncMock(
            return_value="1. [ ] Step 1\n<execute>print('test')</execute>"
        )

        agent = PlanAgent(mock_base_model, _create_plan_config())
        messages = [create_user_msg("Test task")]
        result = await agent.execute(messages)

        assert result["role"] == "assistant"
        assert "<execute>" in result["content"]


# =============================================================================
# Tool Retriever Toggle Tests
# =============================================================================

class TestPlanAgentToolRetrieverToggle:
    """Test suite for PlanAgent tool retriever toggle."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_tool_retriever_enabled(self, mock_base_model, mock_tool_retriever):
        """Test execute with tool retriever enabled."""
        mock_base_model.generate = AsyncMock(
            return_value="1. [ ] Step\n<execute>print('test')</execute>"
        )

        agent = PlanAgent(mock_base_model, _create_plan_config(use_tool_retriever=True))
        agent.retriever = mock_tool_retriever
        mock_tool_retriever.prompt_based_retrieval = Mock(return_value={
            "skills": [], "tools": []
        })

        messages = [create_user_msg("Test task")]
        await agent.execute(messages)

        mock_tool_retriever.prompt_based_retrieval.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_tool_retriever_disabled(self, mock_base_model):
        """Test execute with tool retriever disabled."""
        mock_base_model.generate = AsyncMock(
            return_value="1. [ ] Step\n<execute>print('test')</execute>"
        )

        agent = PlanAgent(mock_base_model, _create_plan_config(use_tool_retriever=False))
        messages = [create_user_msg("Test task")]
        await agent.execute(messages)

        assert not hasattr(agent, 'retriever') or agent.retriever is None
