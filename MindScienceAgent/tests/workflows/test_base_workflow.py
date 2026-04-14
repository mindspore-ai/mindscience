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
Unit tests for BaseWorkflow.

Tests:: initialization and create.
"""

import pytest
from mindscience_agent.workflow import BaseWorkflow
from mindscience_agent.config import MindScienceConfig


# =============================================================================
# Initialization Tests
# =============================================================================

class TestBaseWorkflowInitialization:
    """Test suite for BaseWorkflow initialization."""

    @pytest.mark.unit
    def test_init_with_config(self, mock_full_config):
        """Test BaseWorkflow initialization with config."""
        config = MindScienceConfig._parse_config_data(mock_full_config)     # pylint: disable=W0212
        workflow = BaseWorkflow(config=config)

        assert workflow.config == config


# =============================================================================
# Create Tests
# =============================================================================

class TestBaseWorkflowCreate:
    """Test suite for BaseWorkflow create methods."""

    @pytest.mark.unit
    def test_create_agents(self, mock_full_config):
        """Test _create_agents method."""
        config = MindScienceConfig._parse_config_data(mock_full_config)    # pylint: disable=W0212
        workflow = BaseWorkflow(config=config)

        # BaseWorkflow AGENT_TYPES is empty tuple, so no agents will be created
        # Test that calling doesn't raise error
        workflow._create_agents()   # pylint: disable=W0212

        # Verify no agents created (because AGENT_TYPES is empty)
        assert not workflow.AGENT_TYPES
