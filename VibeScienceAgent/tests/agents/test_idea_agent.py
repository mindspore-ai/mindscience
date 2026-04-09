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
Unit tests for IdeaAgent.

Tests:: initialization and execute.
"""

from unittest.mock import Mock, AsyncMock, patch
import pytest

from vibescience_agent.agents.idea_agent import IdeaAgent
from vibescience_agent.config import AgentConfig, ModelConfig
from vibescience_agent.config.agent_config import IdeaAgentConfig
from vibescience_agent.utils import create_user_msg, create_assistant_msg


def _create_idea_agent_config(minimal_ideas=3, max_retries=3):
    """Helper to create minimal IdeaAgent config."""
    model_config = ModelConfig(model_name="test-model", base_url="http://test-model-api.com", api_key="test-api-key")
    return IdeaAgentConfig(
        model_config=model_config,
        minimal_ideas=minimal_ideas,
        max_retries=max_retries
    )


# =============================================================================
# Initialization Tests
# =============================================================================

class TestIdeaAgentInitialization:
    """Test suite for IdeaAgent initialization."""

    @pytest.mark.unit
    def test_init(self, mock_base_model):
        """Test IdeaAgent initialization."""
        with patch('vibescience_agent.agents.idea_agent.IdeaAgent._build_idea_subgraph'):
            agent = IdeaAgent(mock_base_model, _create_idea_agent_config())

            assert agent.agent_type == "idea"
            assert agent.minimal_ideas == 3
            assert hasattr(agent, '_compiled_subgraph')

    @pytest.mark.unit
    def test_init_with_custom_minimal_ideas(self, mock_base_model):
        """Test IdeaAgent initialization with custom minimal_ideas."""
        with patch('vibescience_agent.agents.idea_agent.IdeaAgent._build_idea_subgraph'):
            agent = IdeaAgent(mock_base_model, _create_idea_agent_config(minimal_ideas=5))

            assert agent.minimal_ideas == 5


# =============================================================================
# Execute Tests
# =============================================================================

class TestIdeaAgentExecute:
    """Test suite for IdeaAgent execute method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_basic(self, mock_base_model):
        """Test IdeaAgent execute method with basic input."""
        # Mock the _invoke_subgraph method to return a state with messages
        mock_message = Mock()
        mock_message.content = "Here are some novel scientific ideas..."

        mock_base_model.generate = AsyncMock(
            return_value="Generated scientific ideas about the research topic."
        )
        with patch('vibescience_agent.agents.idea_agent.IdeaAgent._build_idea_subgraph'):
            agent = IdeaAgent(mock_base_model, _create_idea_agent_config())

            messages = [create_user_msg("Research goal: Improve AI reasoning")]

            with patch.object(agent, '_invoke_subgraph', return_value={"messages": [mock_message]}):
                result = await agent.execute(messages)

            assert result["role"] == "assistant"
            assert "content" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_survey_results(self, mock_base_model):
        """Test IdeaAgent execute with survey results."""
        mock_message = Mock()
        mock_message.content = "Based on literature review..."

        with patch('vibescience_agent.agents.idea_agent.IdeaAgent._build_idea_subgraph'):
            agent = IdeaAgent(mock_base_model, _create_idea_agent_config())

            messages = [create_user_msg("Research goal: Improve AI reasoning")]
            survey_results = {"papers": [{"title": "Paper 1", "abstract": "Abstract 1"}]}

            with patch.object(agent, '_invoke_subgraph', return_value={"messages": [mock_message]}):
                result = await agent.execute(messages, survey_results=survey_results)

            assert result["role"] == "assistant"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_empty_messages_raises_error(self, mock_base_model):
        """Test IdeaAgent execute raises error with empty messages."""
        with patch('vibescience_agent.agents.idea_agent.IdeaAgent._build_idea_subgraph'):
            agent = IdeaAgent(mock_base_model, _create_idea_agent_config())

            with pytest.raises(Exception):
                await agent.execute([])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_idea_critic_enabled(self, mock_base_model):
        """Test IdeaAgent execute with idea critic enabled."""
        mock_message = Mock()
        mock_message.content = "Revised ideas based on critic feedback..."

        with patch('vibescience_agent.agents.idea_agent.IdeaAgent._build_idea_subgraph'):
            agent = IdeaAgent(mock_base_model, _create_idea_agent_config())

            messages = [create_user_msg("Research goal: Improve AI reasoning")]

            with patch.object(agent, '_invoke_subgraph', return_value={"messages": [mock_message]}):
                result = await agent.execute(messages, enable_idea_critic=True)

            assert result["role"] == "assistant"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_retry_on_error(self, mock_base_model):
        """Test IdeaAgent retry mechanism on error."""
        call_count = 0

        async def mock_invoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Simulated error")
            mock_message = Mock()
            mock_message.content = "Success after retry"
            return {"messages": [mock_message]}

        with patch('vibescience_agent.agents.idea_agent.IdeaAgent._build_idea_subgraph'):
            agent = IdeaAgent(mock_base_model, _create_idea_agent_config(max_retries=3))

            messages = [create_user_msg("Research goal: Improve AI reasoning")]

            with patch.object(agent, '_invoke_subgraph', side_effect=mock_invoke):
                result = await agent.execute(messages)

            assert call_count == 2
            assert result["role"] == "assistant"


# =============================================================================
# Internal Method Tests
# =============================================================================

class TestIdeaAgentInternalMethods:
    """Test suite for IdeaAgent internal methods."""

    @pytest.mark.unit
    def test_build_system_prompt(self, mock_base_model):
        """Test IdeaAgent system prompt building."""
        with patch('vibescience_agent.agents.idea_agent.IdeaAgent._build_idea_subgraph'):
            agent = IdeaAgent(mock_base_model, _create_idea_agent_config())

            agent._build_system_prompt(survey_results=None, enable_idea_critic=False)

            assert agent.system_prompt is not None
            assert "scientific idea generator" in agent.system_prompt.lower()

    @pytest.mark.unit
    def test_build_system_prompt_with_critic(self, mock_base_model):
        """Test IdeaAgent system prompt with critic enabled."""
        with patch('vibescience_agent.agents.idea_agent.IdeaAgent._build_idea_subgraph'):
            agent = IdeaAgent(mock_base_model, _create_idea_agent_config())

            agent._build_system_prompt(survey_results=None, enable_idea_critic=True)

            assert agent.system_prompt is not None

    @pytest.mark.unit
    def test_process_input(self, mock_base_model):
        """Test IdeaAgent input processing."""
        with patch('vibescience_agent.agents.idea_agent.IdeaAgent._build_idea_subgraph'):
            agent = IdeaAgent(mock_base_model, _create_idea_agent_config())

            messages = [create_user_msg("Research goal: Improve AI reasoning")]
            prompt = agent._process_input(messages, enable_idea_critic=False)

            assert "Research Goal" in prompt
            assert "Improve AI reasoning" in prompt

    @pytest.mark.unit
    def test_process_input_with_critic(self, mock_base_model):
        """Test IdeaAgent input processing with critic enabled."""
        with patch('vibescience_agent.agents.idea_agent.IdeaAgent._build_idea_subgraph'):
            agent = IdeaAgent(mock_base_model, _create_idea_agent_config())

            messages = [create_user_msg("Research goal: Improve AI reasoning")]
            prompt = agent._process_input(messages, enable_idea_critic=True)

            assert "Research Goal" in prompt
            assert "critic" in prompt.lower()

    @pytest.mark.unit
    def test_process_output_with_messages(self, mock_base_model):
        """Test IdeaAgent output processing with valid messages."""
        with patch('vibescience_agent.agents.idea_agent.IdeaAgent._build_idea_subgraph'):
            agent = IdeaAgent(mock_base_model, _create_idea_agent_config())

            final_state = {
                "messages": [Mock(content="Idea 1: Something\nIdea 2: Something else")]
            }

            result = agent._process_output(final_state)

            assert result["role"] == "assistant"

    @pytest.mark.unit
    def test_process_output_empty_messages(self, mock_base_model):
        """Test IdeaAgent output processing with empty messages."""
        with patch('vibescience_agent.agents.idea_agent.IdeaAgent._build_idea_subgraph'):
            agent = IdeaAgent(mock_base_model, _create_idea_agent_config())

            final_state = {"messages": []}

            result = agent._process_output(final_state)

            assert result["role"] == "assistant"
            assert "No ideas were generated" in result["content"]