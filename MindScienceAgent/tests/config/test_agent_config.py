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
"""Tests for agent_config module."""
import pytest

from mindscience_agent.config.model_config import ModelConfig
from mindscience_agent.config.agent_config import AgentConfig, SurveyAgentConfig, IdeaAgentConfig


class TestAgentConfig:
    """Test cases for AgentConfig."""

    def _create_model_config(self):
        """Helper to create a ModelConfig for testing."""
        return ModelConfig(
            model_name="gpt-4",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )

    def test_creation_with_defaults(self):
        """Test creating AgentConfig with default values."""
        model_config = self._create_model_config()
        config = AgentConfig(
            agent_type="survey",
            model_config=model_config,
        )
        assert config.agent_type == "survey"
        assert config.model_config == model_config
        assert config.max_retries == 2
        assert config.use_tool_retriever is False
        assert config.skill_path == ""

    def test_creation_with_custom_values(self):
        """Test creating AgentConfig with custom values."""
        model_config = self._create_model_config()
        config = AgentConfig(
            agent_type="plan",
            model_config=model_config,
            max_retries=5,
            use_tool_retriever=True,
            skill_path="/path/to/skills",
        )
        assert config.agent_type == "plan"
        assert config.max_retries == 5
        assert config.use_tool_retriever is True
        assert config.skill_path == "/path/to/skills"

    def test_validate_agent_type_valid(self):
        """Test agent_type validation with valid value."""
        model_config = self._create_model_config()
        config = AgentConfig(agent_type="survey", model_config=model_config)
        assert config.agent_type == "survey"

    def test_validate_agent_type_invalid(self):
        """Test agent_type validation with invalid value."""
        model_config = self._create_model_config()
        with pytest.raises(ValueError, match="agent is not supported"):
            AgentConfig(agent_type="invalid_agent", model_config=model_config)

    def test_validate_model_config_valid(self):
        """Test model_config validation with valid config."""
        model_config = self._create_model_config()
        config = AgentConfig(agent_type="survey", model_config=model_config)
        assert config.model_config is not None

    def test_validate_max_retries_valid(self):
        """Test max_retries validation with valid value."""
        model_config = self._create_model_config()
        config = AgentConfig(agent_type="survey", model_config=model_config)
        config.max_retries = 5
        assert config.max_retries == 5

    def test_validate_max_retries_above_max(self):
        """Test max_retries validation with value above maximum."""
        model_config = self._create_model_config()
        config = AgentConfig(agent_type="survey", model_config=model_config)
        with pytest.raises(ValueError, match="is above maximum allowed value"):
            config.max_retries = 20

    def test_validate_max_retries_below_min(self):
        """Test max_retries validation with value below minimum."""
        model_config = self._create_model_config()
        config = AgentConfig(agent_type="survey", model_config=model_config)
        with pytest.raises(ValueError, match="is below minimum allowed value"):
            config.max_retries = -1

    def test_validate_use_tool_retriever_invalid_type(self):
        """Test use_tool_retriever validation with invalid type."""
        model_config = self._create_model_config()
        config = AgentConfig(agent_type="survey", model_config=model_config)
        with pytest.raises(ValueError, match="is not an instance of"):
            config.use_tool_retriever = "true"

    def test_validate_skill_path_invalid_type(self):
        """Test skill_path validation with invalid type."""
        model_config = self._create_model_config()
        config = AgentConfig(agent_type="survey", model_config=model_config)
        with pytest.raises(ValueError, match="is not an instance of"):
            config.skill_path = True

    def test_validate_model_config_missing_model_name(self):
        """Test model_config validation raises when model_name is None."""
        model_config = ModelConfig(
            model_name=None,
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )
        with pytest.raises(ValueError, match="model_name"):
            AgentConfig(agent_type="survey", model_config=model_config)

    def test_validate_model_config_missing_base_url(self):
        """Test model_config validation raises when base_url is None."""
        model_config = ModelConfig(
            model_name="gpt-4",
            api_key="test-key",
            base_url=None,
        )
        with pytest.raises(ValueError, match="base_url"):
            AgentConfig(agent_type="survey", model_config=model_config)

    def test_validate_model_config_missing_api_key(self):
        """Test model_config validation raises when api_key is None."""
        model_config = ModelConfig(
            model_name="gpt-4",
            api_key=None,
            base_url="https://api.openai.com/v1",
        )
        with pytest.raises(ValueError, match="api_key"):
            AgentConfig(agent_type="survey", model_config=model_config)


class TestSurveyAgentConfig:
    """Test cases for SurveyAgentConfig."""

    def _create_model_config(self):
        """Helper to create a ModelConfig for testing."""
        return ModelConfig(
            model_name="gpt-4",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )

    def test_creation_with_defaults(self):
        """Test creating SurveyAgentConfig with default values."""
        model_config = self._create_model_config()
        config = SurveyAgentConfig(
            model_config=model_config,
        )
        assert config.max_papers == 5

    def test_creation_with_custom_values(self):
        """Test creating SurveyAgentConfig with custom values."""
        model_config = self._create_model_config()
        config = SurveyAgentConfig(
            model_config=model_config,
            max_papers=10,
        )
        assert config.max_papers == 10

    def test_validate_max_papers_valid(self):
        """Test max_papers validation with valid value."""
        model_config = self._create_model_config()
        config = SurveyAgentConfig(model_config=model_config)
        config.max_papers = 8
        assert config.max_papers == 8

    def test_validate_max_papers_below_min(self):
        """Test max_papers validation with value below minimum."""
        model_config = self._create_model_config()
        config = SurveyAgentConfig(model_config=model_config)
        with pytest.raises(ValueError, match="is below minimum allowed value"):
            config.max_papers = 0

    def test_validate_max_papers_invalid_type(self):
        """Test max_papers validation with invalid type."""
        model_config = self._create_model_config()
        config = SurveyAgentConfig(model_config=model_config)
        with pytest.raises(ValueError, match="is not an instance of"):
            config.max_papers = "5"


class TestIdeaAgentConfig:
    """Test cases for IdeaAgentConfig."""

    def _create_model_config(self):
        """Helper to create a ModelConfig for testing."""
        return ModelConfig(
            model_name="gpt-4",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )

    def test_creation_with_defaults(self):
        """Test creating IdeaAgentConfig with default values."""
        model_config = self._create_model_config()
        config = IdeaAgentConfig(
            model_config=model_config,
        )
        assert config.agent_type == "idea"
        assert config.minimal_ideas == 5

    def test_creation_with_custom_values(self):
        """Test creating IdeaAgentConfig with custom values."""
        model_config = self._create_model_config()
        config = IdeaAgentConfig(
            model_config=model_config,
            minimal_ideas=10,
            max_retries=3,
        )
        assert config.minimal_ideas == 10
        assert config.max_retries == 3

    def test_validate_minimal_ideas_valid(self):
        """Test minimal_ideas validation with valid value."""
        model_config = self._create_model_config()
        config = IdeaAgentConfig(model_config=model_config)
        config.minimal_ideas = 8
        assert config.minimal_ideas == 8

    def test_validate_minimal_ideas_below_min(self):
        """Test minimal_ideas validation with value below minimum."""
        model_config = self._create_model_config()
        config = IdeaAgentConfig(model_config=model_config)
        with pytest.raises(ValueError, match="is below minimum allowed value"):
            config.minimal_ideas = 0

    def test_validate_minimal_ideas_invalid_type(self):
        """Test minimal_ideas validation with invalid type."""
        model_config = self._create_model_config()
        config = IdeaAgentConfig(model_config=model_config)
        with pytest.raises(ValueError, match="is not an instance of"):
            config.minimal_ideas = "5"

    def test_validate_minimal_ideas_equal_to_min(self):
        """Test minimal_ideas validation with minimum valid value."""
        model_config = self._create_model_config()
        config = IdeaAgentConfig(model_config=model_config, minimal_ideas=1)
        assert config.minimal_ideas == 1

    def test_inherits_from_agent_config(self):
        """Test IdeaAgentConfig inherits from AgentConfig."""
        model_config = self._create_model_config()
        config = IdeaAgentConfig(model_config=model_config)
        assert isinstance(config, AgentConfig)
        assert hasattr(config, 'max_retries')
        assert hasattr(config, 'use_tool_retriever')
        assert hasattr(config, 'skill_path')
