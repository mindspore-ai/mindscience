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
- Tool / library context and **per-agent tool retriever** (run_tool_retrieval_once_if_enabled,
  etc.); see section after :class:`BaseAgent`.
"""
from __future__ import annotations

import abc
import asyncio
import os
from typing import Any, Dict, Optional, Union

from langchain_core.tools import Tool
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
try:
    from langchain_experimental.utilities import PythonREPL
    python_repl_func = PythonREPL().run
except ImportError as e:
    from vibescience_agent.tools.support_tools import run_python_repl
    python_repl_func = run_python_repl

from vibescience_agent.model.base_model import BaseModel
from vibescience_agent.tools.tool_registry import ToolRegistry
from vibescience_agent.tools.tool_retriever import ToolRetriever
from vibescience_agent.tools.env_desc import library_content_dict
from vibescience_agent.utils.utils import (
    read_module2api,
    subset_module2api,
    build_tool_desc,
    library_names_for_prompt,
    extract_skill_description
)
from vibescience_agent.utils import logger
from vibescience_agent.config.agent_config import AgentConfig
from vibescience_agent.config.tool_config import ToolConfig
from vibescience_agent.config.vibescience_config import PROJECT_ROOT


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

    Args:
        model (BaseModel): Language model instance for text generation.
        config (AgentConfig): Configuration object containing agent-specific settings.
        tool_config (Dict[str, ToolConfig], optional): Tool configuration dictionary.

    Inputs:
        - messages (list): Conversation history (list of message dicts).
        - params (Dict[str, Any]): Task-specific parameters that control execution behavior.

    Outputs:
        - Dict[str, Any]: Execution results in a standardized dictionary format.
    """
    def __init__(self, model: BaseModel, config: AgentConfig,
                 tool_config: Dict[str, ToolConfig] = None):
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
        self._compiled_subgraph = None

    def _build_deep_agent(self):
        """Build a DeepAgent instance for tool execution."""
        chat_model = self.model.to_chat_openai()

        python_skill_tool = Tool(
            name="python_executor",
            func=python_repl_func,
            description=(
                "Execute Python code to process data, analyze results, or perform computations. "
                "Input should be a valid Python code snippet. Use this tool for tasks that require "
                "data manipulation, analysis, or any computation that can be done in Python."
            ),
        )

        backend = FilesystemBackend(root_dir=str(PROJECT_ROOT))
        execute_node = create_deep_agent(
            chat_model,
            backend=backend,
            tools=[python_skill_tool],
            skills=self.skill_path
        )
        return execute_node

    async def _invoke_subgraph(self, input_msg: list[tuple[str, str]]) -> dict:
        """Invoke the compiled subgraph with input messages and return results."""
        if not hasattr(self, "_compiled_subgraph"):
            raise AgentExecutionError(
                "Subgraph not compiled. Ensure _build_execute_subgraph "
                "is called during initialization."
            )
        if hasattr(self._compiled_subgraph, "ainvoke"):
            return await self._compiled_subgraph.ainvoke({"messages": input_msg})
        return await asyncio.to_thread(
            self._compiled_subgraph.invoke, {"messages": input_msg}
        )

    @abc.abstractmethod
    async def execute(self, messages, **params) -> Dict[str, Any]:
        """Execute the agent's primary task (must be implemented by subclasses)."""
    async def _call_model(self,
                        prompt: str | list,
                        system_prompt: Optional[str] = None,
                        schema: Optional[Dict[str, Any]] = None,
                        temperature: Optional[float] = None) -> Union[str, Dict[str, Any]]:
        """Protected method to call the language model with automatic retry logic."""
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
        """Assemble the resource bundle PlanAgent / ExecuteAgent store on self."""
        skills = []
        if self.skill_path and os.path.exists(self.skill_path):
            for root, _, files in os.walk(self.skill_path):
                if 'SKILL.md' in files:
                    markdown_path = os.path.join(root, 'SKILL.md')
                    extract_info = extract_skill_description(markdown_path)
                    if extract_info:
                        name, description = extract_info
                        dir_name = os.path.basename(root)
                        if name != dir_name:
                            continue
                        skills.append({"name": name, "description": description, "path": markdown_path})

        subset = subset_module2api(self.module2api, class_tool_modules)
        return {
            "skills": skills,
            "tool_desc": build_tool_desc(subset),
            "library_content_list": library_names_for_prompt(),
            # Currently not supported
            "custom_tools": [],
            "custom_data": [],
            "custom_software": [],
        }

    def _update_selected_resources(self, selected_resources: Optional[Dict[str, Any]]) -> None:
        """Apply tool-retriever output (tools / sciencedata / libraries keys)."""
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

        self.ctx["skills"] = selected_resources["skills"]
        self.ctx["tool_desc"] = tool_desc
        self.ctx["library_content_list"] = selected_resources["libraries"]

    def _prepare_resources_for_retrieval(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Prepare resources for retrieval and return selected resource names."""
        # Gather all available resources

        # 1. Tools from the registry
        all_tools = self.tool_registry.tools if hasattr(self, "tool_registry") else []

        # 2. Libraries with descriptions - use library_content_dict directly
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
            "skills": self.ctx.get("skills", []),
            "tools": all_tools,
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
            "skills": selected_resources["skills"],
            "tools": selected_resources["tools"],
            "libraries": [lib["name"] if isinstance(lib, dict) else lib for lib in selected_resources["libraries"]],
        }

        # Print summary of what was retrieved
        logger.info("-" * 60)
        logger.info("📊 RETRIEVAL SUMMARY:")

        if self.ctx.get("skills", []):
            logger.info(f"  🧠 Skills: {len(selected_resources_names['skills'])} selected")
        logger.info(f"  🔧 Tools: {len(selected_resources_names['tools'])} selected")
        logger.info(f"  ⚙️ Libraries: {len(selected_resources_names['libraries'])} selected")
        logger.info("=" * 60)

        return selected_resources_names

    def run_tool_retrieval_if_enabled(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Run tool retrieval if enabled and update the selected resources."""
        if self.use_tool_retriever:
            selected_resources = self._prepare_resources_for_retrieval(prompt)
            self._update_selected_resources(selected_resources)
