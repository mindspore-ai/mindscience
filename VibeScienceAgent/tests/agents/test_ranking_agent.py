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
Unit tests for RankingAgent.

Tests:: initialization and execute.
"""

from unittest.mock import AsyncMock
import pytest

from vibescience_agent.agents.ranking_agent import RankingAgent
from vibescience_agent.config import AgentConfig, ModelConfig
from vibescience_agent.utils import create_user_msg, create_assistant_msg


def _create_ranking_config():
    """Helper to create minimal RankingAgent config."""
    model_config = ModelConfig(model_name="test-model", base_url="http://test-model-api.com", api_key="test-api-key")
    return AgentConfig(agent_type="ranking", model_config=model_config)


# =============================================================================
# Initialization Tests
# =============================================================================

class TestRankingAgentInitialization:
    """Test suite for RankingAgent initialization."""

    @pytest.mark.unit
    def test_init(self, mock_base_model):
        """Test RankingAgent initialization."""
        agent = RankingAgent(mock_base_model, _create_ranking_config())

        assert agent.agent_type == "ranking"
        assert agent.system_prompt is not None
        assert "scientific idea evaluator" in agent.system_prompt.lower()

    @pytest.mark.unit
    def test_system_prompt_contains_scoring_criteria(self, mock_base_model):
        """Test RankingAgent system prompt contains scoring criteria."""
        agent = RankingAgent(mock_base_model, _create_ranking_config())

        prompt_lower = agent.system_prompt.lower()
        assert "technical innovation" in prompt_lower
        assert "potential impact" in prompt_lower
        assert "feasibility" in prompt_lower
        assert "completeness" in prompt_lower


# =============================================================================
# Execute Tests
# =============================================================================

class TestRankingAgentExecute:
    """Test suite for RankingAgent execute method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_basic(self, mock_base_model):
        """Test RankingAgent execute method with basic input."""
        mock_base_model.generate = AsyncMock(
            return_value="Based on the evaluation criteria, Idea 1 scores highest due to its "
                        "innovative approach and strong methodology."
        )

        agent = RankingAgent(mock_base_model, _create_ranking_config())

        messages = [
            create_user_msg("Research goal: Improve AI reasoning"),
            create_assistant_msg("Idea 1: Novel transformer approach\nIdea 2: RL-based method")
        ]
        result = await agent.execute(messages)

        assert result["role"] == "assistant"
        assert "content" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_enhanced_ideas(self, mock_base_model):
        """Test RankingAgent execute with enhanced ideas."""
        mock_base_model.generate = AsyncMock(
            return_value="After comprehensive evaluation:\n"
                        "- Idea 1: Technical Innovation 8.5, Potential Impact 9.0, Feasibility 7.5\n"
                        "- Idea 2: Technical Innovation 7.0, Potential Impact 8.0, Feasibility 8.5\n"
                        "Recommendation: Idea 1 is the best choice."
        )

        agent = RankingAgent(mock_base_model, _create_ranking_config())

        messages = [
            create_user_msg("Select the best research idea"),
            create_assistant_msg("Enhanced Idea 1: Transformer-based reasoning\nEnhanced Idea 2: RL-based reasoning")
        ]
        result = await agent.execute(messages)

        assert result["role"] == "assistant"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_empty_messages_raises_error(self, mock_base_model):
        """Test RankingAgent execute raises error with empty messages."""
        agent = RankingAgent(mock_base_model, _create_ranking_config())

        with pytest.raises(Exception):
            await agent.execute([])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_returns_ranked_ideas(self, mock_base_model):
        """Test RankingAgent returns ranked ideas with scores."""
        mock_base_model.generate = AsyncMock(
            return_value="Ranked Ideas:\n"
                        "1. Idea A - Score: 8.5/10 (Best)\n"
                        "2. Idea B - Score: 7.0/10\n"
                        "3. Idea C - Score: 6.5/10"
        )

        agent = RankingAgent(mock_base_model, _create_ranking_config())

        messages = [create_user_msg("Rank these research ideas")]
        result = await agent.execute(messages)

        assert result["role"] == "assistant"
        assert len(result["content"]) > 0


# =============================================================================
# Evaluation Criteria Tests
# =============================================================================

class TestRankingAgentEvaluationCriteria:
    """Test suite for RankingAgent evaluation criteria."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_technical_innovation(self, mock_base_model):
        """Test RankingAgent evaluates technical innovation."""
        mock_base_model.generate = AsyncMock(
            return_value="Idea 1 demonstrates breakthrough potential with novel architecture."
        )

        agent = RankingAgent(mock_base_model, _create_ranking_config())

        messages = [create_user_msg("Evaluate technical innovation")]
        result = await agent.execute(messages)

        assert result["role"] == "assistant"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_potential_impact(self, mock_base_model):
        """Test RankingAgent evaluates potential impact."""
        mock_base_model.generate = AsyncMock(
            return_value="Idea 2 addresses a critical problem in the field with high significance."
        )

        agent = RankingAgent(mock_base_model, _create_ranking_config())

        messages = [create_user_msg("Evaluate potential impact")]
        result = await agent.execute(messages)

        assert result["role"] == "assistant"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_feasibility(self, mock_base_model):
        """Test RankingAgent evaluates feasibility."""
        mock_base_model.generate = AsyncMock(
            return_value="Idea 1 is highly feasible with available resources and clear methodology."
        )

        agent = RankingAgent(mock_base_model, _create_ranking_config())

        messages = [create_user_msg("Evaluate feasibility")]
        result = await agent.execute(messages)

        assert result["role"] == "assistant"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_completeness(self, mock_base_model):
        """Test RankingAgent evaluates completeness."""
        mock_base_model.generate = AsyncMock(
            return_value="Idea 3 has the most comprehensive technical design with detailed experiments."
        )

        agent = RankingAgent(mock_base_model, _create_ranking_config())

        messages = [create_user_msg("Evaluate completeness")]
        result = await agent.execute(messages)

        assert result["role"] == "assistant"
