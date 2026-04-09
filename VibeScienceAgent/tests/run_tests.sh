#!/bin/bash
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

# VibeScienceAgent Test Runner Script

echo "======================================"
echo "VibeScienceAgent Test Runner"
echo "======================================"
echo ""

# Check Python version
echo "1. Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Current Python version: $PYTHON_VERSION"

# Check if Python 3.10+
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "❌ Error: Python 3.10+ required"
    echo "Current version: $PYTHON_VERSION"
    echo ""
    echo "Solutions:"
    echo "1. Upgrade Python to 3.10+"
    echo "2. Or modify type annotations in code (change 'str | None' to 'Optional[str]')"
    exit 1
else
    echo "✅ Python version meets requirements"
fi
echo ""

# Check pytest installation
echo "2. Checking pytest..."
if command -v pytest &> /dev/null; then
    echo "✅ pytest installed"
    pytest --version
else
    echo "❌ pytest not installed"
    echo "Please run: pip install pytest pytest-asyncio"
    exit 1
fi
echo ""

# Check test files
echo "3. Checking test files..."
TEST_FILES=(
    "conftest.py"
    "agents/test_plan_agent.py"
    "agents/test_execute_agent.py"
    "agents/test_survey_agent.py"
    "agents/test_critic_agent.py"
    "agents/test_idea_agent.py"
    "agents/test_idea_critic_agent.py"
    "agents/test_ranking_agent.py"
    "workflows/test_base_workflow.py"
    "workflows/test_experiment_workflow.py"
    "model/test_model_factory.py"
    "config/test_agent_config.py"
    "config/test_base_config.py"
    "config/test_log_config.py"
    "config/test_model_config.py"
    "config/test_tool_config.py"
    "config/test_vibescience_config.py"
)

ALL_EXIST=true
for file in "${TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file not found"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = false ]; then
    echo ""
    echo "❌ Some test files missing"
    exit 1
fi
echo ""

# Count test cases
echo "4. Counting test cases..."
echo "Test file distribution:"
echo "  - PlanAgent: $(grep -c 'def test_' agents/test_plan_agent.py) tests"
echo "  - ExecuteAgent: $(grep -c 'def test_' agents/test_execute_agent.py) tests"
echo "  - SurveyAgent: $(grep -c 'def test_' agents/test_survey_agent.py) tests"
echo "  - CriticAgent: $(grep -c 'def test_' agents/test_critic_agent.py) tests"
echo "  - IdeaAgent: $(grep -c 'def test_' agents/test_idea_agent.py) tests"
echo "  - IdeaCriticAgent: $(grep -c 'def test_' agents/test_idea_critic_agent.py) tests"
echo "  - RankingAgent: $(grep -c 'def test_' agents/test_ranking_agent.py) tests"
echo "  - BaseWorkflow: $(grep -c 'def test_' workflows/test_base_workflow.py) tests"
echo "  - ExperimentWorkflow: $(grep -c 'def test_' workflows/test_experiment_workflow.py) tests"
echo "  - ModelFactory: $(grep -c 'def test_' model/test_model_factory.py) tests"
echo "  - Config: $(grep 'def test_' config/*.py | wc -l) tests"
echo ""

# Run tests
echo "======================================"
echo "Running tests..."
echo "======================================"
echo ""

# Run tests and show results
pytest . -v --tb=short

# Check test results
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "✅ All tests passed!"
    echo "======================================"
else
    echo ""
    echo "======================================"
    echo "❌ Some tests failed"
    echo "======================================"
    echo ""
    echo "Common solutions:"
    echo "1. Python version issue: Ensure using Python 3.10+"
    echo "2. Dependency issue: Run pip install -r ../requirements.txt"
    echo "3. Import issue: Run pip install -e .."
    echo "4. Detailed error info: Run pytest . -v --tb=long"
    exit 1
fi
