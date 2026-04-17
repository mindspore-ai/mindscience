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
"""Model interface for MindScienceAgent."""
from typing import Dict, List, Optional, Any

import httpx
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from mindscience_agent.utils import logger


class Model:
    """Implementation of Model interface.

    Args:
        config: Model configuration instance.
    """
    def __init__(self, config):
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.model_name = config.model_name

        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.max_connections = config.max_connections

        connect_timeout = min(30.0, float(self.timeout))
        timeout_cfg = httpx.Timeout(self.timeout, connect=connect_timeout)
        limits = httpx.Limits(
            max_connections=self.max_connections,
            max_keepalive_connections=min(32, self.max_connections),
        )
        self._shared_async_client = httpx.AsyncClient(timeout=timeout_cfg, limits=limits)
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=self._shared_async_client,
        )
        logger.debug(
            f"OpenAI client initialized with model: {self.model_name} via {self.base_url}",
        )

    async def generate(self,
                       prompt: str | list,
                       system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       stop_sequences: Optional[List[str]] = None,
                       **kwargs) -> str:
        """Generate text based on provided prompt using OpenAI API."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if isinstance(prompt, list):
            messages += prompt
        else:
            messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
            **kwargs
        )
        content = response.choices[0].message.content
        return content

    def to_chat_openai(self):
        """Return LangChain ChatOpenAI with the same api_key, base_url, and model id as this adapter."""
        kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model_name,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "http_async_client": self._shared_async_client,
        }

        return ChatOpenAI(**kwargs)
