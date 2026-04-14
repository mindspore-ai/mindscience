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
"""Tests for base_config module."""
import pytest

from mindscience_agent.config.base_config import BaseConfig


class ConcreteConfig(BaseConfig):
    """Concrete implementation of BaseConfig for testing."""

    def __init__(self, config_name, value=None):
        super().__init__(config_name)
        self.value = value


class TestBaseConfig:
    """Test cases for BaseConfig."""

    def test_concrete_config_creation(self):
        """Test creating a concrete config."""
        config = ConcreteConfig("test_config")
        assert config._config_name == "test_config"

    def test_update_attrs(self):
        """Test updating attributes via update_attrs."""
        config = ConcreteConfig("test_config")
        config.update_attrs(value=42, other_attr="test")
        assert config.value == 42
        assert config.other_attr == "test"

    def test_validator_decorator(self):
        """Test validator decorator registers validation function."""

        @ConcreteConfig.validator("test_field")
        def validate_test_field(self, value):
            if value < 0:
                raise ValueError("test_field must be non-negative")
            return value

        config = ConcreteConfig("test_config")
        config.test_field = 10  # Should not raise
        assert config.test_field == 10

    def test_validator_decorator_invalid(self):
        """Test validator decorator catches invalid values."""
        config = ConcreteConfig("test_config")
        with pytest.raises(ValueError, match="test_field must be non-negative"):
            config.test_field = -1

    def test_config_to_str(self):
        """Test config_to_str function produces correct output."""
        from mindscience_agent.config.base_config import config_to_str
        config = ConcreteConfig("test_config", value="test_value")
        result = config_to_str(config)
        assert "ConcreteConfig" in result
        assert "test_value" in result

    def test_str_method(self):
        """Test __str__ method returns correct format."""
        config = ConcreteConfig("test_config", value="hello")
        result = str(config)
        assert "ConcreteConfig" in result
        assert "hello" in result

    def test_get_config_name(self):
        """Test get_config_name returns correct name."""
        config = ConcreteConfig("my_config")
        assert config.get_config_name() == "my_config"
