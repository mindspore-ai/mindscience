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

from vibescience_agent.config.tool_config import (
    ToolConfig,
    PaperSurveyConfig,
    SUPPORTED_PAPER_SOURCES,
)


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


class TestPaperSurveyConfig:
    """Test cases for PaperSurveyConfig."""

    def test_creation_with_defaults(self):
        """Test creating PaperSurveyConfig with default values."""
        config = PaperSurveyConfig()
        assert config.get_tool_name() == "paper_survey"
        assert config.max_results == 10
        assert config.sources == SUPPORTED_PAPER_SOURCES

    def test_creation_with_custom_values(self):
        """Test creating PaperSurveyConfig with custom values."""
        config = PaperSurveyConfig(
            max_results=15,
            sources=["pubmed", "arxiv"],
        )
        assert config.max_results == 15
        assert config.sources == ["pubmed", "arxiv"]

    def test_validate_max_results_valid(self):
        """Test max_results validation with valid value."""
        config = PaperSurveyConfig()
        config.max_results = 15
        assert config.max_results == 15

    def test_validate_max_results_boundary(self):
        """Test max_results validation with boundary values."""
        config = PaperSurveyConfig()
        config.max_results = 0
        assert config.max_results == 0
        config.max_results = 20
        assert config.max_results == 20

    def test_validate_max_results_below_min(self):
        """Test max_results validation with value below minimum."""
        config = PaperSurveyConfig()
        with pytest.raises(ValueError, match="is below minimum allowed value"):
            config.max_results = -1

    def test_validate_max_results_above_max(self):
        """Test max_results validation with value above maximum."""
        config = PaperSurveyConfig()
        with pytest.raises(ValueError, match="is above maximum allowed value"):
            config.max_results = 21

    def test_validate_max_results_invalid_type(self):
        """Test max_results validation with invalid type."""
        config = PaperSurveyConfig()
        with pytest.raises(ValueError, match="is not an instance of"):
            config.max_results = "10"

    def test_validate_sources_invalid(self):
        """Test sources validation with invalid source."""
        config = PaperSurveyConfig()
        with pytest.raises(ValueError, match="contains elements not in allowed list"):
            config.sources = ["pubmed", "invalid_source"]

    def test_validate_sources_invalid_type(self):
        """Test sources validation with invalid type."""
        config = PaperSurveyConfig()
        with pytest.raises(ValueError, match="is not an instance of"):
            config.sources = "pubmed"
