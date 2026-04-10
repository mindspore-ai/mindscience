# Copyright 2026 Huawei Technologies Co., Ltd
# Copyright 2025 InternAgent
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
"""Model provider factory for creating and managing language model instances."""
import importlib
from typing import Dict

from vibescience_agent.utils import logger

MODEL_PROVIDER_MAP = {
    "openai": "vibescience_agent.model.openai_model.OpenAIModel",
}


class ModelFactory:
    """Factory for creating and managing language model instances.

    Args:
        config: Configuration dictionary for model creation.
    """
    registered_models = {}
    _model_cache = {}

    @staticmethod
    def create_model(config):
        """Create a model instance based on provided configuration."""
        provider = config.provider

        cache_key = ModelFactory._create_cache_key(provider, config)

        if cache_key in ModelFactory._model_cache:
            logger.debug( f"Reusing cached model for provider: {provider}")
            return ModelFactory._model_cache[cache_key]

        if provider not in MODEL_PROVIDER_MAP and provider not in ModelFactory.registered_models:
            raise ValueError(f"Unsupported model provider: {provider}")

        if provider in ModelFactory.registered_models:
            model_class = ModelFactory.registered_models[provider]
        else:
            module_path, class_name = MODEL_PROVIDER_MAP[provider].rsplit(".", 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

        model = model_class(config)
        ModelFactory._model_cache[cache_key] = model

        return model

    @classmethod
    def register_model(cls, provider_name: str, model_class) -> None:
        """Add a custom model implementation to the factory registry."""
        cls.registered_models[provider_name] = model_class
        logger.debug(f"New model provider registered: {provider_name}")

    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Get all registered model providers and their implementations."""
        available_models = MODEL_PROVIDER_MAP

        for provider, model_class in cls.registered_models.items():
            available_models[provider] = model_class.__name__

        return available_models

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached model instances and statistics."""
        cls._model_cache.clear()
        logger.debug("model_factory", "Model cache cleared")

    @staticmethod
    def _create_cache_key(provider: str, config) -> str:
        """Generate a unique identifier for model instance caching."""
        model_name = config.model_name
        base_url = config.base_url
        return f"{provider}:{model_name}:{base_url}"
