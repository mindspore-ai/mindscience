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
"""Agent configuration classes for MindScienceAgent."""
from mindscience_agent.config import validator
from mindscience_agent.config.base_config import BaseConfig
from mindscience_agent.config.model_config import ModelConfig
from mindscience_agent.utils import logger

SUPPORTED_AGENT_LIST = ["survey", "plan", "critic", "execute", "ranking", "idea", "idea_critic"]
TOOL_RETRIEVER_SUPPORTED_AGENTS = ["plan", "execute", "idea"]
SKILL_SUPPORTED_AGENTS = ["plan", "execute", "idea"]


class AgentConfig(BaseConfig):
    """Configuration class for agent settings and model integration.

    Args:
        agent_type (str): Type of agent (survey, plan, critic, execute).
        model_config (ModelConfig): Model configuration for the agent.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 2.
        use_tool_retriever (bool, optional): Whether to use tool retriever. Defaults to False.
        skill_path (str, optional): Path to skill file. Defaults to "".
    """
    def __init__(
        self,
        agent_type: str,
        model_config: ModelConfig,
        max_retries: int = 2,
        use_tool_retriever: bool = False,
        skill_path: str = "",
        **kwargs,
    ):
        super().__init__(config_name=f"{agent_type}_agent")

        self.agent_type = agent_type
        self.model_config = model_config
        self.max_retries = max_retries
        self.use_tool_retriever = use_tool_retriever
        self.skill_path = skill_path

        self.update_attrs(**kwargs)


@AgentConfig.validator("agent_type")
def validate_agent_type(config_instance: AgentConfig, agent_type):  # pylint: disable=W0613
    """Validate agent_type."""
    if agent_type not in SUPPORTED_AGENT_LIST:
        raise ValueError(f"{agent_type} agent is not supported, supported agents are {SUPPORTED_AGENT_LIST}")
    return agent_type


@AgentConfig.validator("model_config")
def validate_model_config(config_instance: AgentConfig, model_config: ModelConfig):
    """Validate model_config."""
    validator.check_not_none(config_instance.get_config_name(), "model_name", model_config.model_name)
    validator.check_not_none(config_instance.get_config_name(), "base_url", model_config.base_url)
    validator.check_not_none(config_instance.get_config_name(), "api_key", model_config.api_key)
    return model_config


@AgentConfig.validator("max_retries")
def validate_max_retries(config_instance: AgentConfig, max_retries):  # pylint: disable=W0613
    """Validate max_retries."""
    validator.check_type("max_retries", value=max_retries, expected_type=int)
    validator.check_number_range("max_retries", value=max_retries, min_value=0, max_value=10)
    return max_retries


@AgentConfig.validator("use_tool_retriever")
def validate_use_tool_retriever(config_instance: AgentConfig, use_tool_retriever):
    """Validate use_tool_retriever."""
    validator.check_type("use_tool_retriever", value=use_tool_retriever, expected_type=bool)
    if config_instance.agent_type not in TOOL_RETRIEVER_SUPPORTED_AGENTS and use_tool_retriever:
        logger.warning(f"{config_instance.agent_type} agent does not support tool retriever.")
        return False
    return use_tool_retriever


@AgentConfig.validator("skill_path")
def validate_skill_path(config_instance: AgentConfig, skill_path):
    """Validate skill_path."""
    validator.check_type("skill_path", value=skill_path, expected_type=str)
    if config_instance.agent_type not in SKILL_SUPPORTED_AGENTS and skill_path:
        logger.warning(f"{config_instance.agent_type} agent does not support skills.")
        return ""
    return skill_path


class SurveyAgentConfig(AgentConfig):
    """Configuration class for survey agent with paper search settings.

    Args:
        max_papers (int, optional): Maximum number of papers to return. Defaults to 5.
    """
    def __init__(
        self,
        max_papers = 5,
        **kwargs,
    ):
        super().__init__(agent_type="survey", **kwargs)

        self.max_papers = max_papers


@SurveyAgentConfig.validator("max_papers")
def validate_max_papers(config_instance: SurveyAgentConfig, max_papers):  # pylint: disable=W0613
    """Validate max_papers."""
    validator.check_type("max_papers", value=max_papers, expected_type=int)
    validator.check_number_range("max_papers", value=max_papers, min_value=1)
    return max_papers


class IdeaAgentConfig(AgentConfig):
    """
    Configuration for IdeaAgent.

    Args:
        minimal_ideas (int, optional): Minimum number of ideas to generate. Default: ``5``.
    """

    def __init__(
        self,
        minimal_ideas = 5,
        **kwargs,
    ):
        super().__init__(agent_type="idea", **kwargs)

        self.minimal_ideas = minimal_ideas


@IdeaAgentConfig.validator("minimal_ideas")
def validate_minimal_ideas(config_instance: IdeaAgentConfig, minimal_ideas):  # pylint: disable=W0613
    """Validate minimal_ideas."""
    validator.check_type("minimal_ideas", value=minimal_ideas, expected_type=int)
    validator.check_number_range("minimal_ideas", value=minimal_ideas, min_value=1)
    return minimal_ideas
