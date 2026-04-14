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
Unit tests for ModelFactory.

Tests:: create_model functionality.
"""

import pytest
from mindscience_agent.model import ModelFactory, BaseModel
from mindscience_agent.config import ModelConfig


# =============================================================================
# Create Model Tests
# =============================================================================

class TestModelFactoryCreateModel:
    """Test suite for ModelFactory.create_model."""

    @pytest.mark.unit
    def test_create_model(self):
        """Test ModelFactory.create_model method."""
        config = ModelConfig(model_name="gpt-4o-mini", base_url="http://test-model-api.com", api_key="test-api-key")

        model = ModelFactory.create_model(config)

        assert model is not None
        assert isinstance(model, BaseModel)
