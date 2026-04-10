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
"""Base model interface for VibeScienceAgent language model interactions."""
import abc
from typing import Dict, Any, Optional, List


class BaseModel(abc.ABC):
    """Abstract base class for all language model implementations.

    Args:
        config: Model configuration instance.
    """
    def __init__(self, config):
        """Initialize model with configuration."""
        self.config = config

    @abc.abstractmethod
    async def generate(self,
                        prompt: str | list,
                        system_prompt: Optional[str] = None,
                        temperature: Optional[float] = None,
                        max_tokens: Optional[int] = None,
                        stop_sequences: Optional[List[str]] = None,
                        **kwargs) -> str:
        """Generate text completion from model."""

    @abc.abstractmethod
    async def generate_json(self,
                          prompt: str | list,
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """Generate structured JSON output conforming to a schema."""
