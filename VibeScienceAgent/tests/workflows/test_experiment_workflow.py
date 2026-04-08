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
Integration tests for ExperimentWorkflow.

Tests:: initialization and enable_critic toggle.
"""

import pytest
from vibescience_agent.workflow import ExperimentWorkflow
from vibescience_agent.config import VibeScienceConfig


# =============================================================================
# Initialization Tests
# =============================================================================

class TestExperimentWorkflowInitialization:
    """Test suite for ExperimentWorkflow initialization."""

    @pytest.mark.integration
    def test_init(self, mock_full_config):
        """Test ExperimentWorkflow initialization."""
        config = VibeScienceConfig._parse_config_data(mock_full_config)   # pylint: disable=W0212
        workflow = ExperimentWorkflow(config=config)

        assert workflow.config == config
        assert hasattr(workflow, 'plan_agent')
        assert hasattr(workflow, 'execute_agent')


# =============================================================================
# Enable Critic Toggle Tests
# =============================================================================

class TestExperimentWorkflowEnableCritic:
    """Test suite for enable_critic toggle functionality."""

    @pytest.mark.integration
    def test_enable_critic_true(self, mock_full_config):
        """Test enable_critic=True creates workflow with critic node."""
        config = VibeScienceConfig._parse_config_data(mock_full_config)     # pylint: disable=W0212
        workflow = ExperimentWorkflow(config=config, enable_critic=True)

        # Verify enable_critic=True creates critic_agent and workflow contains critic node
        assert workflow.enable_critic is True
        assert hasattr(workflow, 'critic_agent')
        assert workflow.critic_agent is not None    # pylint: disable=E1101

        # Verify workflow nodes contain critic
        workflow_nodes = list(workflow.app.nodes.keys())
        assert "critic" in workflow_nodes

    @pytest.mark.integration
    def test_enable_critic_false(self, mock_full_config):
        """Test enable_critic=False creates workflow without critic node."""
        config = VibeScienceConfig._parse_config_data(mock_full_config)     # pylint: disable=W0212
        workflow = ExperimentWorkflow(config=config, enable_critic=False)

        # Verify enable_critic=False has critic_agent but workflow doesn't contain critic node
        assert workflow.enable_critic is False
        assert hasattr(workflow, 'critic_agent')
        assert workflow.critic_agent is not None    # pylint: disable=E1101

        # Verify workflow nodes don't contain critic
        workflow_nodes = list(workflow.app.nodes.keys())
        assert "critic" not in workflow_nodes
