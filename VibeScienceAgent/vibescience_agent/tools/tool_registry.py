# Copyright 2026 Huawei Technologies Co., Ltd
# Copyright 2025 Biomni
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
"""Tool registry for managing and retrieving available tools."""
import pickle

import pandas as pd


class ToolRegistry:
    """Registry for managing tools and building document DataFrame for retrieval.

    Args:
        tools (dict): Dictionary of tools to register.
    """
    def __init__(self, tools):
        """Initialize tool registry and register all provided tools."""
        self.tools = []
        self.next_id = 0

        for j in tools.values():
            for tool in j:
                self.register_tool(tool)

        docs = []
        for tool_id in range(len(self.tools)):
            docs.append([int(tool_id), self.get_tool_by_id(int(tool_id))])
        self.document_df = pd.DataFrame(docs, columns=["docid", "document_content"])

    def register_tool(self, tool):
        """Register a new tool in the registry."""
        if self.validate_tool(tool):
            tool["id"] = self.next_id
            self.tools.append(tool)
            self.next_id += 1
        else:
            raise ValueError("Invalid tool format")

    def validate_tool(self, tool):
        """Validate that tool has required keys."""
        required_keys = ["name", "description", "required_parameters"]
        return all(key in tool for key in required_keys)

    def get_tool_by_name(self, name):
        """Get tool by name."""
        for tool in self.tools:
            if tool["name"] == name:
                return tool
        return None

    def get_tool_by_id(self, tool_id):
        """Get tool by ID."""
        for tool in self.tools:
            if tool["id"] == tool_id:
                return tool
        return None

    def get_id_by_name(self, name):
        """Get tool ID by name."""
        for tool in self.tools:
            if tool["name"] == name:
                return tool["id"]
        return None

    def get_name_by_id(self, tool_id):
        """Get tool name by ID."""
        for tool in self.tools:
            if tool["id"] == tool_id:
                return tool["name"]
        return None

    def list_tools(self):
        """List all tools in registry"""
        return [{"name": tool["name"], "id": tool["id"]} for tool in self.tools]

    def remove_tool_by_id(self, tool_id):
        """Remove the tool with the given id"""
        tool = self.get_tool_by_id(tool_id)
        if tool:
            self.tools = [t for t in self.tools if t["id"] != tool_id]
            return True
        return False

    def remove_tool_by_name(self, name):
        """Remove the tool with the given name"""
        tool = self.get_tool_by_name(name)
        if tool:
            self.tools = [t for t in self.tools if t["name"] != name]
            return True
        return False

    def save_registry(self, filename):
        """Save tool registry"""
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_registry(filename):
        """Load tool registry"""
        with open(filename, "rb") as file:
            return pickle.load(file)
