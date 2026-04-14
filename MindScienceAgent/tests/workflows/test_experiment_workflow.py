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

Tests:: initialization, enable_critic toggle, and workflow.run execution.
"""
import re
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from mindscience_agent.workflow import ExperimentWorkflow
from mindscience_agent.config import MindScienceConfig

# =============================================================================
# Initialization Tests
# =============================================================================
class TestExperimentWorkflowInitialization:
    """Test suite for ExperimentWorkflow initialization."""

    @pytest.mark.integration
    def test_init(self, mock_full_config):
        """Test ExperimentWorkflow initialization."""
        config = MindScienceConfig._parse_config_data(mock_full_config)   # pylint: disable=W0212
        workflow = ExperimentWorkflow(config=config)

        assert workflow.config == config
        assert hasattr(workflow, 'plan_agent')
        assert hasattr(workflow, 'execute_agent')

# =============================================================================
# Enable Critic Toggle Tests
# =============================================================================
class TestExperimentWorkflowRun:
    """Test suite for ExperimentWorkflow.run method."""
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_without_critic(self, mock_full_config):
        """Test workflow.run without critic enabled - full workflow execution."""
        config = MindScienceConfig._parse_config_data(mock_full_config)
        workflow = ExperimentWorkflow(config=config, enable_critic=False)

        plan_call_count = [0]
        plan_responses = [
            {"role": "assistant", "content": "I'll execute a test command\n<execute>print('test')</execute>"},
            {"role": "assistant", "content": "Task completed successfully\n<solution>Test solution result</solution>"}
        ]

        async def mock_plan_execute(messages):
            response = plan_responses[plan_call_count[0]]
            plan_call_count[0] += 1
            return response

        workflow.plan_agent.execute = AsyncMock(side_effect=mock_plan_execute)
        workflow.execute_agent.execute = AsyncMock(
            return_value={"role": "assistant", "content": "Command executed successfully"}
        )

        result = await workflow.run("Test task")
        solution = re.search(r"<solution>(.*?)</solution>", result, re.DOTALL | re.IGNORECASE).group(1)

        assert solution == "Test solution result"
        assert workflow.plan_agent.execute.call_count == 2
        assert workflow.execute_agent.execute.call_count == 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_with_critic(self, mock_full_config):
        """Test workflow.run with critic enabled - full workflow execution."""
        config = MindScienceConfig._parse_config_data(mock_full_config)
        workflow = ExperimentWorkflow(config=config, enable_critic=True, test_time_scale_round=1)

        plan_call_count = [0]
        plan_responses = [
            {"role": "assistant", "content": " <think> Let me think about this</think> \n"},
            {"role": "assistant", "content": "Now I'll execute\n<execute>print('test')</execute>"},
            {"role": "assistant", "content": "Final solution\n<solution>Test solution with critic</solution>"}
        ]

        async def mock_plan_execute(messages):
            response = plan_responses[plan_call_count[0]]
            plan_call_count[0] += 1
            return response

        workflow.plan_agent.execute = AsyncMock(side_effect=mock_plan_execute)
        workflow.critic_agent.execute = AsyncMock(
            return_value={"role": "assistant", "content": "The plan looks good, proceed"}
        )
        workflow.execute_agent.execute = AsyncMock(
            return_value={"role": "assistant", "content": "Execution completed"}
        )

        result = await workflow.run("Test task")
        solution = re.search(r"<solution>(.*?)</solution>", result, re.DOTALL | re.IGNORECASE).group(1)

        assert solution == "Test solution with critic"
        assert workflow.plan_agent.execute.call_count == 3
        assert workflow.critic_agent.execute.call_count == 1
        assert workflow.execute_agent.execute.call_count == 1
