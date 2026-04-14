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
"""Model configuration classes for MindScienceAgent."""
from mindscience_agent.config.base_config import BaseConfig
from mindscience_agent.model.model_factory import ModelFactory
from mindscience_agent.utils import logger
from mindscience_agent.config import validator


class ModelConfig(BaseConfig):
    """Configuration class for model settings and API parameters.

    Args:
        model_name (str, optional): Name of the model to use. Defaults to None.
        provider (str, optional): Model provider name. Defaults to "openai".
        base_url (str, optional): Base URL for API. Defaults to None.
        api_key (str, optional): API key for authentication. Defaults to None.
        temperature (float, optional): Temperature for generation. Defaults to 0.2.
        max_tokens (int, optional): Maximum tokens per request. Defaults to 4096.
        timeout (int, optional): Request timeout in seconds. Defaults to 60.
        max_retries (int, optional): Maximum retry attempts. Defaults to 2.
        max_connections (int, optional): Maximum concurrent connections. Defaults to 8.
        **kwargs: Additional configuration parameters.
    """
    def __init__(
        self,
        model_name: str = None,
        provider: str = "openai",
        base_url: str = None,
        api_key: str = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: int = 60,
        max_retries: int = 2,
        max_connections: int = 8,
        **kwargs,
    ):
        super().__init__(config_name=f"{model_name}_model")

        self.model_name = model_name
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_connections = max_connections

        self.update_attrs(**kwargs)


@ModelConfig.validator("provider")
def validate_provider(config_instance: ModelConfig, provider):  # pylint: disable=W0613
    """Validate provider."""
    supported_providers = ModelFactory.get_available_models().keys()
    if provider not in supported_providers:
        logger.warning(f"{provider} provider is not supported, supported providers are {supported_providers}")
    return provider


@ModelConfig.validator("temperature")
def validate_temperature(config_instance: ModelConfig, temperature):  # pylint: disable=W0613
    """Validate temperature."""
    validator.check_type("temperature", value=temperature, expected_type=float)
    validator.check_non_negative("temperature", value=temperature)
    return temperature


@ModelConfig.validator("max_tokens")
def validate_max_tokens(config_instance: ModelConfig, max_tokens):  # pylint: disable=W0613
    """Validate max_tokens."""
    validator.check_type("max_tokens", value=max_tokens, expected_type=int)
    validator.check_non_negative("max_tokens", value=max_tokens)
    return max_tokens


@ModelConfig.validator("timeout")
def validate_timeout(config_instance: ModelConfig, timeout):    # pylint: disable=W0613
    """Validate timeout."""
    validator.check_type("timeout", value=timeout, expected_type=int)
    validator.check_number_range("timeout", value=timeout, min_value=30, max_value=300)
    return timeout


@ModelConfig.validator("max_retries")
def validate_max_retries(config_instance: ModelConfig, max_retries):    # pylint: disable=W0613
    """Validate max_retries."""
    validator.check_type("max_retries", value=max_retries, expected_type=int)
    validator.check_number_range("max_retries", value=max_retries, min_value=0, max_value=10)
    return max_retries


@ModelConfig.validator("max_connections")
def validate_max_connections(config_instance: ModelConfig, max_connections):    # pylint: disable=W0613
    """Validate max_connections."""
    validator.check_type("max_connections", value=max_connections, expected_type=int)
    validator.check_number_range("max_connections", value=max_connections, min_value=1, max_value=8)
    return max_connections
