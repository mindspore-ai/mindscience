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
"""Tests for mindscience_config module."""
import pytest
import tempfile
import os

from mindscience_agent.config.mindscience_config import MindScienceConfig
from mindscience_agent.config.model_config import ModelConfig
from mindscience_agent.config.agent_config import AgentConfig
from mindscience_agent.config.tool_config import PaperSurveyConfig
from mindscience_agent.config.log_config import LogConfig


class TestMindScienceConfig:
    """Test cases for MindScienceConfig."""

    def test_creation_with_defaults(self):
        """Test creating MindScienceConfig with default values."""
        config = MindScienceConfig()
        assert config.model_defaults is None
        assert config.agents == {}
        assert config.tools == {}
        assert config.logging_config is None

    def test_creation_with_values(self):
        """Test creating MindScienceConfig with custom values."""
        model_defaults = ModelConfig(model_name="gpt-4")
        logging_config = LogConfig(level="DEBUG")
        config = MindScienceConfig(
            model_defaults=model_defaults,
            logging_config=logging_config,
        )
        assert config.model_defaults == model_defaults
        assert config.logging_config == logging_config

    def test_parse_config_data_full(self):
        """Test parsing full configuration data."""
        config_data = {
            "model_defaults": {
                "model_name": "gpt-4",
                "api_key": "test-key",
                "base_url": "https://api.openai.com/v1",
            },
            "agents": {
                "survey": {
                    "model": {
                        "model_name": "gpt-3.5-turbo",
                        "api_key": "survey-key",
                        "base_url": "https://api.openai.com/v1",
                        "temperature": 0.3,
                    },
                    "agent": {
                        "max_papers": 10,
                    }
                }
            },
            "tools": {
                "paper_survey": {
                    "max_results": 15,
                }
            },
            "logging": {
                "level": "DEBUG",
            },
        }
        config = MindScienceConfig._parse_config_data(config_data)

        assert config.model_defaults is not None
        assert config.model_defaults.model_name == "gpt-4"
        assert "survey" in config.agents
        assert config.agents["survey"].model_config.model_name == "gpt-3.5-turbo"
        assert config.tools["paper_survey"].max_results == 15
        assert config.logging_config is not None
        assert config.logging_config.level == "DEBUG"

    def test_parse_config_data_minimal(self):
        """Test parsing minimal configuration data."""
        config_data = {}
        config = MindScienceConfig._parse_config_data(config_data)

        assert isinstance(config.model_defaults, ModelConfig)
        assert config.agents == {}
        assert config.tools == {}
        assert isinstance(config.logging_config, LogConfig)

    def test_parse_model_config(self):
        """Test parsing model configuration."""
        data = {
            "model_name": "gpt-4",
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
        }
        model_config = MindScienceConfig._parse_model_config(data)

        assert model_config.model_name == "gpt-4"
        assert model_config.api_key == "test-key"
        assert model_config.base_url == "https://api.openai.com/v1"

    def test_parse_agent_config_with_defaults(self):
        """Test parsing agent configuration with model defaults."""
        model_defaults = ModelConfig(
            model_name="gpt-4",
            api_key="default-key",
            base_url="https://api.openai.com/v1",
        )
        agent_data = {
            "agent": {
                "max_retries": 10,
            },
        }
        agent_config = MindScienceConfig._parse_agent_config(
            "plan", agent_data, model_defaults
        )

        assert agent_config.agent_type == "plan"
        assert agent_config.model_config.model_name == "gpt-4"
        assert agent_config.model_config.api_key == "default-key"
        assert agent_config.max_retries == 10

    def test_parse_agent_config_override_defaults(self):
        """Test parsing agent configuration that overrides defaults."""
        model_defaults = ModelConfig(
            model_name="gpt-4",
            api_key="default-key",
            base_url="https://api.openai.com/v1",
        )
        agent_data = {
            "model": {
                "model_name": "gpt-3.5-turbo",
                "api_key": "agent-key",
            },
            "agent": {
                "max_retries": 10,
            }
        }
        agent_config = MindScienceConfig._parse_agent_config(
            "survey", agent_data, model_defaults
        )

        assert agent_config.model_config.model_name == "gpt-3.5-turbo"
        assert agent_config.model_config.api_key == "agent-key"
        assert agent_config.max_retries == 10

    def test_parse_tool_config_paper_survey(self):
        """Test parsing paper_survey tool configuration."""
        data = {
            "max_results": 10,
            "sources": ["pubmed", "arxiv"],
        }
        tool_config = MindScienceConfig._parse_tool_config("paper_survey", data)

        assert isinstance(tool_config, PaperSurveyConfig)
        assert tool_config.max_results == 10
        assert tool_config.sources == ["pubmed", "arxiv"]

    def test_parse_tool_config_unknown(self):
        """Test parsing unknown tool configuration."""
        data = {
            "param1": "value1",
        }
        tool_config = MindScienceConfig._parse_tool_config("unknown_tool", data)

        assert tool_config.get_tool_name() == "unknown_tool"
        assert tool_config.param1 == "value1"

    def test_parse_log_config(self):
        """Test parsing logging configuration."""
        data = {
            "level": "DEBUG",
        }
        log_config = MindScienceConfig._parse_log_config(data)

        assert log_config.level == "DEBUG"

    def test_get_agent_config_existing(self):
        """Test getting configuration for existing agent."""
        model_config = ModelConfig(
            model_name="gpt-4",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )
        agent_config = AgentConfig(
            agent_type="survey",
            model_config=model_config,
        )
        config = MindScienceConfig(agents={"survey": agent_config})

        result = config.get_agent_config("survey")
        assert result == agent_config

    def test_get_agent_config_nonexistent(self):
        """Test getting configuration for non-existent agent."""
        config = MindScienceConfig()
        result = config.get_agent_config("nonexistent")
        assert result is None

    def test_init_config_from_yaml(self):
        """Test initializing configuration from YAML file."""
        yaml_content = """
model_defaults:
  model_name: "gpt-4"
  api_key: "test-key"
  base_url: "https://api.openai.com/v1"
agents:
  survey:
    model:
      model_name: null
      base_url: null
      api_key: null
      temperature: 0.2
tools:
  paper_survey:
    max_results: 10
logging:
  level: "INFO"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            config = MindScienceConfig.init_config_from_yaml(yaml_path)
            assert config.model_defaults is not None
            assert config.model_defaults.model_name == "gpt-4"
            assert "survey" in config.agents
            assert "paper_survey" in config.tools
            assert config.logging_config is not None
        finally:
            os.unlink(yaml_path)

    def test_init_config_from_yaml_file_not_found(self):
        """Test initializing configuration from non-existent file."""
        with pytest.raises(ValueError, match="Config file not found"):
            config = MindScienceConfig.init_config_from_yaml("/non/existent/path.yaml")

    def test_format_config_tree(self):
        """Test formatting config as tree structure."""
        config = MindScienceConfig()
        data = {"key1": "value1", "key2": {"nested": "value"}}
        lines = config._format_config_tree(data)

        assert len(lines) > 0
        assert "├─ key1: value1" in lines

    def test_get_config_dict(self):
        """Test getting config as dictionary."""
        model_config = ModelConfig(model_name="gpt-4", api_key="test-key")
        config = MindScienceConfig(model_defaults=model_config)

        config_dict = config._get_config_dict(config)
        assert config_dict is not None

    def test_print_config(self):
        """Test print_config runs without error."""
        model_config = ModelConfig(model_name="gpt-4")
        config = MindScienceConfig(model_defaults=model_config)
        # Should not raise any exceptions
        config.print_config()
