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
Unit tests for CriticAgent.

Tests:: initialization and execute.
"""

from unittest.mock import AsyncMock
import pytest

from vibescience_agent.agents import CriticAgent
from vibescience_agent.config import AgentConfig, ModelConfig
from vibescience_agent.utils import create_user_msg, create_assistant_msg


def _create_critic_config():
    """Helper to create minimal CriticAgent config."""
    model_config = ModelConfig(model_name="test-model", base_url="http://test-model-api.com", api_key="test-api-key")
    return AgentConfig(agent_type="critic", model_config=model_config)


# =============================================================================
# Initialization Tests
# =============================================================================

class TestCriticAgentInitialization:
    """Test suite for CriticAgent initialization."""

    @pytest.mark.unit
    def test_init(self, mock_base_model):
        """Test CriticAgent initialization."""
        agent = CriticAgent(mock_base_model, _create_critic_config())

        assert agent.agent_type == "critic"


# =============================================================================
# Execute Tests
# =============================================================================

class TestCriticAgentExecute:
    """Test suite for CriticAgent execute method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute(self, mock_base_model):
        """Test CriticAgent execute method."""
        mock_base_model.generate = AsyncMock(
            return_value="The plan could be improved by adding more detailed error handling."
        )

        agent = CriticAgent(mock_base_model, _create_critic_config())

        messages = [
            create_user_msg("Solve this problem"),
            create_assistant_msg("Here's my plan:\n1. Step 1\n2. Step 2"),
        ]
        result = await agent.execute(messages)

        assert result["role"] == "user"
        assert "feedbacks" in result["content"].lower()
