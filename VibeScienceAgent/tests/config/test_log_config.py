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
"""Tests for log_config module."""
import pytest

from vibescience_agent.config.log_config import LogConfig


class TestLogConfig:
    """Test cases for LogConfig."""

    def test_creation_with_defaults(self):
        """Test creating LogConfig with default values."""
        config = LogConfig()
        assert config.level == "INFO"
        assert config.get_config_name() == "logging"

    def test_creation_with_custom_values(self):
        """Test creating LogConfig with custom values."""
        config = LogConfig(level="DEBUG")
        assert config.level == "DEBUG"

    def test_validate_level_valid_values(self):
        """Test level validation with valid values."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            config = LogConfig(level=level)
            assert config.level == level

    def test_validate_level_invalid_values(self):
        """Test level validation with invalid values."""
        with pytest.raises(ValueError, match="is not supported"):
            config = LogConfig(level="CRITICAL")

    def test_validate_level_invalid_type(self):
        """Test level validation with invalid type."""
        config = LogConfig()
        with pytest.raises(ValueError, match="is not an instance of"):
            config.level = 123
