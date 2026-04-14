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
Unit tests for SurveyAgent.

Tests:: initialization and execute.
"""

from unittest.mock import AsyncMock, patch
import pytest

from mindscience_agent.agents import SurveyAgent
from mindscience_agent.config import ModelConfig, SurveyAgentConfig, PaperSurveyConfig


def _create_survey_config():
    """Helper to create minimal SurveyAgent config."""
    model_config = ModelConfig(model_name="test-model", base_url="http://test-model-api.com", api_key="test-api-key")
    return SurveyAgentConfig(model_config=model_config, max_papers=5)

def _create_tool_config():
    return {"paper_survey": PaperSurveyConfig(max_results=10, sources=["pubmed", "arxiv", "semantic_scholar"])}


# =============================================================================
# Initialization Tests
# =============================================================================

class TestSurveyAgentInitialization:
    """Test suite for SurveyAgent initialization."""

    @pytest.mark.unit
    def test_init(self, mock_base_model):
        """Test SurveyAgent initialization."""
        agent = SurveyAgent(mock_base_model, _create_survey_config(), _create_tool_config())

        assert agent.agent_type == "survey"
        assert agent.max_papers == 5
        assert agent.paper_survey is not None
        assert agent.paper_survey.max_results == 10
        assert agent.paper_survey.sources == ["pubmed", "arxiv", "semantic_scholar"]


# =============================================================================
# Execute Tests
# =============================================================================

class TestSurveyAgentExecute:
    """Test suite for SurveyAgent execute method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute(self, mock_base_model):
        """Test SurveyAgent execute method."""
        mock_base_model.generate_json = AsyncMock(
            return_value={
                "description": "Machine learning for healthcare",
                "domain": "Machine Learning"
            }
        )

        agent = SurveyAgent(mock_base_model, _create_survey_config(), _create_tool_config())
        mock_paper_result = [
            {
                "id": 0,
                "title": "Test Paper",
                "abstract": "Test abstract",
                "source": "semantic_scholar",
                "score": 8
            }
        ]

        with patch.object(agent, 'advanced_query_paper', AsyncMock(return_value=mock_paper_result)):
            messages = [{"role": "user", "content": "ML in healthcare"}]
            result = await agent.execute(messages)

            assert result == mock_paper_result
