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
"""OpenAI model adapter implementing BaseModel interface for VibeScienceAgent."""
from typing import Dict, List, Optional, Any

import json
import httpx
from json_repair import repair_json
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from vibescience_agent.model.base_model import BaseModel
from vibescience_agent.utils import logger


class OpenAIModel(BaseModel):
    """OpenAI implementation of BaseModel interface.

    Args:
        config: Model configuration instance.
    """
    def __init__(self, config):
        super().__init__(config)
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

    async def generate_with_json_output(self,
                                       prompt: str | list,
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """Generate a response formatted as JSON according to the provided schema."""
        if system_prompt:
            enhanced_system_prompt = (
                f"{system_prompt}\n\n"
                f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"
            )
        else:
            enhanced_system_prompt = (
                f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"
            )

        try:
            req_messages = [
                {"role": "system", "content": enhanced_system_prompt},
            ]
            if isinstance(prompt, list):
                req_messages += prompt
            else:
                req_messages.append({"role": "user", "content": prompt})

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=req_messages,
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )

            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError as exc:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError as ex:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair") from ex
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON") from exc
            return result_dict

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}") from e
        except Exception as e:
            logger.error(f"Error generating JSON response from OpenAI: {e}")
            raise

    async def generate_json(self,
                          prompt: str | list,
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """Generate JSON output from the model."""
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

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
