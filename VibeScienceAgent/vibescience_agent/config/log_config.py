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
"""Logging configuration module for VibeScienceAgent."""

from vibescience_agent.config.base_config import BaseConfig
from vibescience_agent.config import validator
from vibescience_agent.utils import logger


class LogConfig(BaseConfig):
    """
    Manages logging configuration parameters, primarily log level, with dynamic attribute support.

    Args:
        level (str, optional): Logging level for the application. Defaults to "INFO".
            Valid values: "DEBUG", "INFO", "WARNING", "ERROR".
        **kwargs: Additional keyword arguments passed to `update_attrs` for dynamic configuration.
    """
    def __init__(
        self,
        level: str = "INFO",
        **kwargs,
    ):
        super().__init__(config_name="logging")

        self.level = level

        self.update_attrs(**kwargs)


@LogConfig.validator("level")
def validate_level(config_instance: LogConfig, level):  # pylint: disable=W0613
    """Validate level."""
    validator.check_type("level", value=level, expected_type=str)
    supported_levels = logger.LOG_LEVEL_MAP.keys()
    if level not in supported_levels:
        raise ValueError(f"log level {level} is not supported, "
                       f"supported levels are {supported_levels}")
    return level
