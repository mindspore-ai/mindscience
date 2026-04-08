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
"""
Base Agent Interface for VibeScienceAgent Multi-Agent System

This module provides the foundational abstract base class (BaseAgent) that defines
the interface and common functionality for all specialized agents in the VibeScienceAgent
system. It establishes a standardized pattern for agent initialization, execution,
model interaction, and error handling that all derived agents must follow.

The module includes:
- BaseAgent: Abstract base class with template methods for agent operations
- AgentExecutionError: Custom exception for agent-specific failures
- Common utilities for model calls and retry logic
- Tool / library context and **per-agent tool retriever** (``run_tool_retrieval_once_if_enabled``,
  etc.); see section after :class:`BaseAgent`.
"""

from __future__ import annotations

import abc
import asyncio
from typing import Any, Dict, Optional, Union

from vibescience_agent.model.base_model import BaseModel
from vibescience_agent.tools.tool_registry import ToolRegistry
from vibescience_agent.tools.tool_retriever import ToolRetriever
from vibescience_agent.tools.env_desc import library_content_dict, sciencedata_dict
from vibescience_agent.utils.utils import (
    read_module2api,
    subset_module2api,
    build_tool_desc,
    library_names_for_prompt
)
from vibescience_agent.utils import logger
from vibescience_agent.config.agent_config import AgentConfig
from vibescience_agent.config.tool_config import ToolConfig


class AgentExecutionError(Exception):
    """
    Custom exception raised when an agent encounters an unrecoverable execution failure.

    This exception is raised after all retry attempts have been exhausted or when
    a critical error occurs that cannot be resolved through retries. It provides
    context about which agent failed and the nature of the failure.
    """


class BaseAgent(abc.ABC):
    """
    Abstract base class defining the interface and common functionality for all agents.

    BaseAgent establishes a standardized architecture for specialized agents within
    the VibeScienceAgent multi-agent system. Each agent encapsulates a specific cognitive
    task (e.g., hypothesis generation, critical evaluation, method development) and
    interacts with language models to perform that task.

    Key Responsibilities:
        - Define the contract that all concrete agents must implement
        - Provide model interaction utilities with automatic retry logic
        - Handle errors gracefully with configurable retry policies

    Attributes:
        model (BaseModel): Language model instance for text generation
        config (Dict[str, Any]): Configuration parameters for the agent
        name (str): Human-readable name for the agent
        description (str): Brief description of agent's purpose
        system_prompt (str): Default system-level instructions for the model
        max_retries (int): Maximum number of retry attempts on failures

    Abstract Methods:
        execute: Must be implemented by subclasses to define agent-specific logic

    Usage:
        Subclass BaseAgent and implement the execute() method to create a new
        specialized agent. Use _call_model() for all language model interactions
        to benefit from automatic retries and error handling.
    """

    def __init__(self, model: BaseModel, config: AgentConfig,
                 tool_config: Dict[str, ToolConfig] = None):
        """
        Initialize a new agent instance with model and configuration.

        Args:
            model (BaseModel): Language model instance for text generation.
            config (Dict[str, Any]): Configuration dictionary containing agent-specific
                settings. Common keys include:
                - name (str): Agent's display name
                - description (str): Purpose and capabilities description
                - system_prompt (str): Default system-level instructions
                - max_retries (int): Maximum retry attempts on failures (default: 10)
        """
        self.model = model
        self.config = config
        self.tool_config = tool_config
        self.agent_type = config.agent_type
        self.max_retries = config.max_retries
        self.skill_path = config.skill_path
        self.use_tool_retriever = config.use_tool_retriever

        self.module2api = read_module2api()
        if self.use_tool_retriever:
            self.tool_registry = ToolRegistry(self.module2api)
            self.retriever = ToolRetriever()
        self.ctx = {}
        self.system_prompt = ""

    @abc.abstractmethod
    async def execute(self, messages, **params) -> Dict[str, Any]:
        """
        Execute the agent's primary task (must be implemented by subclasses).

        Args:
            context (Dict[str, Any]): Contextual information needed for task execution.
            params (Dict[str, Any]): Task-specific parameters that control execution behavior.

        Returns:
            Dict[str, Any]: Execution results in a standardized dictionary format.

        Raises:
            AgentExecutionError: When execution fails after retries.
        """

    async def _call_model(self,
                        prompt: str | list,
                        system_prompt: Optional[str] = None,
                        schema: Optional[Dict[str, Any]] = None,
                        temperature: Optional[float] = None) -> Union[str, Dict[str, Any]]:
        """
        Protected method to call the language model with automatic retry logic.

        This is the primary interface for agents to interact with their language model.
        It provides robust error handling with exponential backoff retries, automatic
        selection between text and structured (JSON) generation based on the schema
        parameter, and comprehensive logging of failures.

        All concrete agent implementations should use this method rather than calling
        the model directly to benefit from standardized error handling.

        Args:
            prompt (str): The main user prompt describing the task for the model.
                Should be clear, specific, and include all necessary context.
            system_prompt (Optional[str]): System-level instructions that guide the
                model's behavior and response style. If ``None``, uses the agent's
                default ``system_prompt`` from configuration. Pass ``""`` to send no
                separate system message (provider may omit the system role). Defaults
                to None.
            schema (Optional[Dict[str, Any]]): JSON Schema definition for structured
                output. When provided, enforces the model to return JSON matching this
                schema. When None, returns freeform text. Defaults to None.
            temperature (Optional[float]): Sampling temperature for model generation.
                Higher values (e.g., 0.8-1.0) increase creativity, lower values
                (e.g., 0.1-0.3) increase determinism. If None, uses model default.

        Returns:
            Union[str, Dict[str, Any]]: Model's response in one of two formats:
                - str: Freeform text response when schema is None
                - Dict[str, Any]: Structured JSON response when schema is provided

        Raises:
            AgentExecutionError: When model calls fail consistently after exhausting
                all retry attempts (max_retries). Contains details of the final error.

        Note:
            The method sleeps for 1 second between retry attempts to avoid hammering
            the API and potentially triggering rate limits. Consider this latency when
            designing time-sensitive operations.
        """
        if system_prompt is None:
            system_prompt = self.system_prompt
        remaining_retries = self.max_retries

        while True:
            try:
                if schema:
                    return await self.model.generate_json(
                        prompt=prompt,
                        schema=schema,
                        system_prompt=system_prompt,
                        temperature=temperature
                    )
                return await self.model.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature
                )

            except Exception as e:
                # sleep for a short time before retrying, log time count
                await asyncio.sleep(1)

                remaining_retries -= 1
                logger.warning(
                    f"Agent {self.agent_type} model call failed: {str(e)}. Retries left: {remaining_retries}",
                )

                if remaining_retries <= 0:
                    raise AgentExecutionError(
                        f"Agent {self.agent_type} failed after max retries: {str(e)}"
                    ) from e

    def _build_agent_tool_context(
        self,
        class_tool_modules: frozenset[str] | None,
    ) -> dict[str, Any]:
        """Assemble the resource bundle PlanAgent / ExecuteAgent store on ``self``.

        Combines:
        - filtered built-in tools (from ``read_module2api``),
        - library name list for the prompt,
        - normalized ``custom_tools`` / ``custom_data`` / ``custom_software``,

        Returns:
            Dict with keys ``tool_desc``, ``library_content_list``, ``custom_tools``,
            ``custom_data``, ``custom_software``.
        """
        subset = subset_module2api(self.module2api, class_tool_modules)
        return {
            "tool_desc": build_tool_desc(subset),
            "library_content_list": library_names_for_prompt(),
            # Currently not supported
            "custom_tools": [],
            "custom_data": [],
            "custom_software": [],
        }

    def _update_selected_resources(self, selected_resources: Optional[Dict[str, Any]]) -> None:
        """Apply tool-retriever output (``tools`` / ``sciencedata`` / ``libraries`` keys)."""
        # Extract tool descriptions for the selected tools
        tool_desc = {}
        for tool in selected_resources["tools"]:
            # Get the module name from the tool
            if isinstance(tool, dict):
                module_name = tool.get("module", None)

                # If module is not specified, try to find it in the module2api
                if not module_name and hasattr(self, "module2api"):
                    for mod, apis in self.module2api.items():
                        for api in apis:
                            if api.get("name") == tool.get("name"):
                                module_name = mod
                                # Update the tool with the module information
                                tool["module"] = module_name
                                break
                        if module_name:
                            break
                    tool["module"] = module_name
            else:
                module_name = getattr(tool, "module_name", None)

                # If module is not specified, try to find it in the module2api
                if not module_name and hasattr(self, "module2api"):
                    tool_name = getattr(tool, "name", str(tool))
                    for mod, apis in self.module2api.items():
                        for api in apis:
                            if api.get("name") == tool_name:
                                module_name = mod
                                # Set the module_name attribute
                                tool.module_name = module_name
                                break
                        if module_name:
                            break

            if module_name not in tool_desc:
                tool_desc[module_name] = []

            # Add the tool to the appropriate module
            if isinstance(tool, dict):
                # Ensure the module is included in the tool description
                if "module" not in tool:
                    tool["module"] = module_name
                tool_desc[module_name].append(tool)
            else:
                # Convert tool object to dictionary
                tool_dict = {
                    "name": getattr(tool, "name", str(tool)),
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "parameters", {}),
                    "module": module_name,  # Explicitly include the module
                }
                tool_desc[module_name].append(tool_dict)

        # Prepare science data items with descriptions
        self.sciencedata_with_desc = []  # pylint: disable=W0201
        for item in selected_resources["sciencedata"]:
            description = sciencedata_dict.get(item, f"Science Data item: {item}")
            self.sciencedata_with_desc.append({"name": item, "description": description})

        self.ctx["tool_desc"] = tool_desc
        self.ctx["library_content_list"] = selected_resources["libraries"]

    def _prepare_resources_for_retrieval(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Prepare resources for retrieval and return selected resource names.

        Args:
            prompt: The user's query

        Returns:
            dict: Dictionary containing selected resource names for tools, science_data, and libraries
        """
        # Gather all available resources
        # 1. Tools from the registry
        all_tools = self.tool_registry.tools if hasattr(self, "tool_registry") else []

        # 2. Science Data items with descriptions
        # Add custom data items to retrieval if they exist
        if self.ctx.get("custom_data", None):
            for name, info in self.ctx["custom_data"].items():
                self.sciencedata_with_desc.append({"name": name, "description": info["description"]})

        # 3. Libraries with descriptions - use library_content_dict directly
        library_descriptions = []
        for lib_name, lib_desc in library_content_dict.items():
            library_descriptions.append({"name": lib_name, "description": lib_desc})

        # Add custom software items to retrieval if they exist
        if self.ctx.get("custom_software", None):
            for name, info in self.ctx["custom_software"].items():
                # Check if it's not already in the library descriptions to avoid duplicates
                if not any(lib["name"] == name for lib in library_descriptions):
                    library_descriptions.append({"name": name, "description": info["description"]})

        # Use retrieval to get relevant resources
        resources = {
            "tools": all_tools,
            "sciencedata": self.sciencedata_with_desc,
            "libraries": library_descriptions,
        }

        # Use prompt-based retrieval with the agent's LLM
        selected_resources = self.retriever.prompt_based_retrieval(prompt, resources, llm=self.model)
        logger.info("=" * 60)
        logger.info(f"🔍 RESOURCE RETRIEVAL FOR {self.agent_type.upper()} AGENT")
        logger.info("=" * 60)
        logger.info("Using prompt-based retrieval with the agent's LLM")

        # Extract the names from the selected resources for the system prompt
        selected_resources_names = {
            "tools": selected_resources["tools"],
            "sciencedata": [],
            "libraries": [lib["name"] if isinstance(lib, dict) else lib for lib in selected_resources["libraries"]],
        }

        # Process science data items to extract just the names
        for item in selected_resources["sciencedata"]:
            if isinstance(item, dict):
                selected_resources_names["sciencedata"].append(item["name"])
            elif isinstance(item, str) and ": " in item:
                # If the item already has a description, extract just the name
                name = item.split(": ")[0]
                selected_resources_names["sciencedata"].append(name)
            else:
                selected_resources_names["sciencedata"].append(item)

        # Print summary of what was retrieved
        logger.info("-" * 60)
        logger.info("📊 RETRIEVAL SUMMARY:")

        logger.info(f"  🔧 Tools: {len(selected_resources_names['tools'])} selected")
        logger.info(f"  📊 Science Data: {len(selected_resources_names['sciencedata'])} selected")
        logger.info(f"  ⚙️  Libraries: {len(selected_resources_names['libraries'])} selected")
        logger.info("=" * 60)

        return selected_resources_names

    def run_tool_retrieval_if_enabled(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Run tool retrieval if enabled and update the selected resources."""
        if self.use_tool_retriever:
            selected_resources = self._prepare_resources_for_retrieval(prompt)
            self._update_selected_resources(selected_resources)
