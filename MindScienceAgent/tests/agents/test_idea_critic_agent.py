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
Unit tests for IdeaCriticAgent.

Tests:: initialization and execute.
"""

from unittest.mock import AsyncMock
import pytest

from mindscience_agent.agents.idea_critic_agent import IdeaCriticAgent
from mindscience_agent.config import AgentConfig, ModelConfig
from mindscience_agent.utils import create_user_msg, create_assistant_msg


def _create_critic_config():
    """Helper to create minimal IdeaCriticAgent config."""
    model_config = ModelConfig(model_name="test-model", base_url="http://test-model-api.com", api_key="test-api-key")
    return AgentConfig(agent_type="idea_critic", model_config=model_config)


# =============================================================================
# Initialization Tests
# =============================================================================

class TestIdeaCriticAgentInitialization:
    """Test suite for IdeaCriticAgent initialization."""

    @pytest.mark.unit
    def test_init(self, mock_base_model):
        """Test IdeaCriticAgent initialization."""
        agent = IdeaCriticAgent(mock_base_model, _create_critic_config())

        assert agent.agent_type == "idea_critic"
        assert agent.system_prompt is not None
        assert "scientific critic" in agent.system_prompt.lower()

    @pytest.mark.unit
    def test_system_prompt_contains_evaluation_criteria(self, mock_base_model):
        """Test IdeaCriticAgent system prompt contains evaluation criteria."""
        agent = IdeaCriticAgent(mock_base_model, _create_critic_config())

        prompt_lower = agent.system_prompt.lower()
        assert "scientific rigor" in prompt_lower
        assert "logical consistency" in prompt_lower
        assert "alignment" in prompt_lower
        assert "feasibility" in prompt_lower


# =============================================================================
# Execute Tests
# =============================================================================

class TestIdeaCriticAgentExecute:
    """Test suite for IdeaCriticAgent execute method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_basic(self, mock_base_model):
        """Test IdeaCriticAgent execute method with basic input."""
        mock_base_model.generate = AsyncMock(
            return_value="This idea has good novelty but needs more detail on the methodology."
        )

        agent = IdeaCriticAgent(mock_base_model, _create_critic_config())

        messages = [
            create_user_msg("Research goal: Improve AI reasoning"),
            create_assistant_msg("Idea 1: Use transformer architecture for reasoning")
        ]
        result = await agent.execute(messages)

        assert result["role"] == "assistant"
        assert "content" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_multiple_ideas(self, mock_base_model):
        """Test IdeaCriticAgent execute with multiple ideas."""
        mock_base_model.generate = AsyncMock(
            return_value="Feedback on Idea 1: Good novelty but needs feasibility check. "
                        "Feedback on Idea 2: Strong methodology but lacks innovation."
        )

        agent = IdeaCriticAgent(mock_base_model, _create_critic_config())

        messages = [
            create_user_msg("Research goal: Improve AI reasoning"),
            create_assistant_msg("Idea 1: Use transformer architecture\nIdea 2: Use reinforcement learning")
        ]
        result = await agent.execute(messages)

        assert result["role"] == "assistant"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_empty_messages_raises_error(self, mock_base_model):
        """Test IdeaCriticAgent execute raises error with empty messages."""
        agent = IdeaCriticAgent(mock_base_model, _create_critic_config())

        with pytest.raises(Exception):
            await agent.execute([])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_returns_critic_feedback(self, mock_base_model):
        """Test IdeaCriticAgent returns structured feedback."""
        mock_base_model.generate = AsyncMock(
            return_value="Strengths: Novel approach. Weaknesses: Implementation unclear."
        )

        agent = IdeaCriticAgent(mock_base_model, _create_critic_config())

        messages = [create_user_msg("Evaluate these ideas")]
        result = await agent.execute(messages)

        assert result["role"] == "assistant"
        assert len(result["content"]) > 0


# =============================================================================
# Internal Method Tests
# =============================================================================

class TestIdeaCriticAgentInternalMethods:
    """Test suite for IdeaCriticAgent internal methods."""

    @pytest.mark.unit
    def test_process_output(self, mock_base_model):
        """Test IdeaCriticAgent output processing."""
        agent = IdeaCriticAgent(mock_base_model, _create_critic_config())

        content = "This idea needs more methodological detail."
        result = agent._process_output(content)

        assert result["role"] == "assistant"
        assert "Previous Feedback" in result["content"]
        assert content in result["content"]
