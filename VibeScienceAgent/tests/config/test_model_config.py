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
# ============================================================================y
"""Tests for model_config module."""
import pytest

from vibescience_agent.config.model_config import ModelConfig


class TestModelConfig:
    """Test cases for ModelConfig."""

    def test_creation_with_defaults(self):
        """Test creating ModelConfig with default values."""
        config = ModelConfig()
        assert config.model_name is None
        assert config.api_key is None
        assert config.base_url is None
        assert config.provider == "openai"
        assert config.temperature == 0.2
        assert config.max_tokens == 4096
        assert config.timeout == 60
        assert config.max_retries == 2
        assert config.max_connections == 8

    def test_creation_with_custom_values(self):
        """Test creating ModelConfig with custom values."""
        config = ModelConfig(
            model_name="gpt-4",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            provider="openai",
            temperature=0.5,
            max_tokens=8192,
            timeout=120,
            max_retries=5,
            max_connections=4,
        )
        assert config.model_name == "gpt-4"
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.openai.com/v1"
        assert config.provider == "openai"
        assert config.temperature == 0.5
        assert config.max_tokens == 8192
        assert config.timeout == 120
        assert config.max_retries == 5
        assert config.max_connections == 4

    def test_validate_temperature_valid(self):
        """Test temperature validation with valid value."""
        config = ModelConfig()
        config.temperature = 0.5
        assert config.temperature == 0.5

    def test_validate_temperature_negative(self):
        """Test temperature validation with negative value."""
        config = ModelConfig()
        with pytest.raises(ValueError, match="must be non-negative"):
            config.temperature = -0.1

    def test_validate_temperature_invalid_type(self):
        """Test temperature validation with invalid type."""
        config = ModelConfig()
        with pytest.raises(ValueError, match="is not an instance of"):
            config.temperature = "0.5"

    def test_validate_max_tokens_valid(self):
        """Test max_tokens validation with valid value."""
        config = ModelConfig()
        config.max_tokens = 2048
        assert config.max_tokens == 2048

    def test_validate_max_tokens_negative(self):
        """Test max_tokens validation with negative value."""
        config = ModelConfig()
        with pytest.raises(ValueError, match="must be non-negative"):
            config.max_tokens = -1

    def test_validate_timeout_valid(self):
        """Test timeout validation with valid value."""
        config = ModelConfig()
        config.timeout = 120
        assert config.timeout == 120

    def test_validate_timeout_below_min(self):
        """Test timeout validation with value below minimum."""
        config = ModelConfig()
        with pytest.raises(ValueError, match="is below minimum allowed value"):
            config.timeout = 20

    def test_validate_timeout_above_max(self):
        """Test timeout validation with value above maximum."""
        config = ModelConfig()
        with pytest.raises(ValueError, match="is above maximum allowed value"):
            config.timeout = 400

    def test_validate_max_retries_valid(self):
        """Test max_retries validation with valid value."""
        config = ModelConfig()
        config.max_retries = 3
        assert config.max_retries == 3

    def test_validate_max_retries_above_max(self):
        """Test max_retries validation with value above maximum."""
        config = ModelConfig()
        with pytest.raises(ValueError, match="is above maximum allowed value"):
            config.max_retries = 20

    def test_validate_max_connections_valid(self):
        """Test max_connections validation with valid value."""
        config = ModelConfig()
        config.max_connections = 4
        assert config.max_connections == 4

    def test_validate_max_connections_above_max(self):
        """Test max_connections validation with value above maximum."""
        config = ModelConfig()
        with pytest.raises(ValueError, match="is above maximum allowed value"):
            config.max_connections = 20

    def test_validate_max_connections_below_min(self):
        """Test max_connections validation with value below minimum."""
        config = ModelConfig()
        with pytest.raises(ValueError, match="is below minimum allowed value"):
            config.max_connections = 0

    def test_update_attrs(self):
        """Test updating attributes via update_attrs."""
        config = ModelConfig()
        config.update_attrs(
            model_name="gpt-4",
            temperature=0.8,
            max_tokens=1024,
        )
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.8
        assert config.max_tokens == 1024

    def test_config_name(self):
        """Test that config_name is correctly set."""
        config = ModelConfig(model_name="test_model")
        assert config.get_config_name() == "test_model_model"

    def test_validate_provider_valid(self):
        """Test provider validation with valid value."""
        config = ModelConfig()
        config.provider = "openai"
        assert config.provider == "openai"
