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
"""
Pytest fixtures for VibeScienceAgent tests.
"""

from unittest.mock import Mock, AsyncMock
import pytest

from vibescience_agent.utils import init_logger
from vibescience_agent.config import LogConfig


@pytest.fixture(autouse=True)
def init_logger_for_tests():
    """Initialize logger for all tests."""
    init_logger(LogConfig(level="INFO"))


@pytest.fixture
def mock_base_model():
    """Create a mock base model for testing."""
    model = Mock()
    model.generate = AsyncMock(return_value="Test response")
    model.generate_json = AsyncMock(return_value={"result": "test"})

    # Add methods needed by agents
    model.to_chat_openai = Mock(return_value="openai:gpt-4o-mini")

    return model


@pytest.fixture
def mock_full_config():
    """Create a mock full configuration for testing."""
    return {
        "version": "1.0.0",
        "model_defaults": {
            "provider": "openai",
            "api_key": "test-api-key",
            "model_name": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        "agents": {
            "plan": {
                "model": {
                    "provider": "openai",
                    "api_key": "test-api-key",
                    "model_name": "gpt-4o-mini",
                },
                "agent": {
                    "use_tool_retriever": False,
                    "max_retries": 10,
                }
            },
            "execute": {
                "model": {
                    "provider": "openai",
                    "api_key": "test-api-key",
                    "model_name": "gpt-4o-mini",
                },
                "agent": {
                    "use_tool_retriever": False,
                    "max_retries": 10,
                }
            },
            "critic": {
                "model": {
                    "provider": "openai",
                    "api_key": "test-api-key",
                    "model_name": "gpt-4o-mini",
                },
                "agent": {
                    "max_retries": 10,
                }
            },
            "survey": {
                "model": {
                    "provider": "openai",
                    "api_key": "test-api-key",
                    "model_name": "gpt-4o-mini",
                },
                "agent": {
                    "max_papers": 10,
                }
            },
        },
        "tools": {
            "paper_survey": {
                "sources": ["pubmed", "arxiv", "semantic_scholar"],
                "max_results": 10,
            }
        },
        "logging": {
            "level": "INFO",
        }
    }


@pytest.fixture
def mock_tool_retriever():
    """Create a mock tool retriever for testing."""
    retriever = Mock()
    retriever.prompt_based_retrieval = Mock(return_value={
        "tools": [],
        "sciencedata": [],
        "libraries": [],
    })
    return retriever


@pytest.fixture
def mock_tool_registry():
    """Create a mock tool registry for testing."""
    registry = Mock()
    registry.get_all_tools = Mock(return_value=[])
    return registry
