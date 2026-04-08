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
"""
Base Model Interface for VibeScienceAgent

Defines the core abstraction layer for language model interactions.
This module provides a standardized interface that all model implementations
must adhere to, enabling consistent access patterns across different backends.
"""

import abc
from typing import Dict, Any, Optional, List


class BaseModel(abc.ABC):
    """
    Abstract foundation for all language model implementations.

    This class defines the contract that all model providers must implement,
    providing a unified interface for text generation, structured output,
    and embedding creation regardless of the underlying model service.
    """

    def __init__(self, config):
        """
        Initialize model with performance tracking metrics.

        Args:
            **kwargs: Provider-specific configuration parameters
        """
        self.config = config

    @abc.abstractmethod
    async def generate(self,
                       prompt: str | list,
                       system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       stop_sequences: Optional[List[str]] = None,
                       **kwargs) -> str:
        """
        Generate text completion from the model.

        Args:
            prompt: The main input text to complete
            system_prompt: Context or instructions for the model
            temperature: Creativity control (higher = more random)
            max_tokens: Maximum generation length
            stop_sequences: Strings that will halt generation when produced
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text completion

        Raises:
            ModelError: On generation failure
        """

    @abc.abstractmethod
    async def generate_json(self,
                          prompt: str | list,
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate structured JSON output conforming to a schema.

        Args:
            prompt: The input text to process
            schema: JSON schema specification for the expected output format
            system_prompt: Context or instructions for the model
            temperature: Creativity control (higher = more random)
            default: Fallback response if generation fails
            **kwargs: Additional provider-specific parameters

        Returns:
            Structured data as a Python dictionary

        Raises:
            ModelError: On generation failure when no default is provided
        """
