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
"""Tool retriever for selecting relevant resources based on queries."""
import contextlib
import re
import asyncio
import threading

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from vibescience_agent.utils import logger


class ToolRetriever:
    """Retrieve relevant tools and resources for queries using LLM-based selection."""
    def __init__(self):
        """Initialize tool retriever."""
        pass

    def prompt_based_retrieval(self, query: str, resources: dict, llm=None) -> dict:
        """Use prompt-based approach to retrieve most relevant resources for a query."""
        skills = resources.get("skills") or []
        tools_r = resources.get("tools") or []

        prompt_sections = []
        prompt_sections.append(f"""
You are an expert research assistant. Your task is to select the relevant resources to help answer a user's query.

USER QUERY: {query}

Below are the available resources. For each category, select items that are directly or indirectly relevant to answering the query.
Be generous in your selection - include resources that might be useful for the task, even if they're not explicitly mentioned in the query.
It's better to include slightly more resources than to miss potentially useful ones.

AVAILABLE SKILLS:
{self._format_resources_for_prompt(skills)}

AVAILABLE TOOLS:
{self._format_resources_for_prompt(tools_r)}""")

        response_format = """
For each category, respond with ONLY the indices of the relevant items in the following format:
SKILLS: [list of indices]
TOOLS: [list of indices]

For example:
SKILLS: [0, 2]
TOOLS: [0, 3, 5, 7, 9]

If a category has no relevant items, use an empty list, e.g., SKILLS: [] or TOOLS: []

IMPORTANT GUIDELINES:
1. Be generous but not excessive - aim to include all potentially relevant resources
2. ALWAYS prioritize skills over tools - if a skill provides functionality that overlaps with TOOLS, prefer the SKILLS
3. ALWAYS prioritize database tools for general queries - include as many database tools as possible
4. Include all literature search tools
5. For wet lab sequence type of queries, ALWAYS include molecular biology tools
7. Don't exclude resources just because they're not explicitly mentioned in the query
8. When in doubt about a database tool or molecular biology tool, include it rather than exclude it
"""
        prompt = "\n".join(prompt_sections) + response_format

        if llm is None:
            llm = ChatOpenAI(model="gpt-4o")

        logger.debug(f"tool retriever prompt: {prompt}")

        if hasattr(llm, "invoke"):
            response = llm.invoke([HumanMessage(content=prompt)])
            response_content = response.content
        elif hasattr(llm, "generate"):
            response_content = self._run_async(
                llm.generate(prompt=prompt)
            )
        else:
            response_content = str(llm(prompt))

        logger.debug(f"tool retriever response_content: {response_content}")

        selected_indices = self._parse_llm_response(response_content)

        logger.debug(f"tool retriever selected_indices: {selected_indices}")

        return {
            "skills": [skills[i] for i in selected_indices.get("skills", []) if i < len(skills)],
            "tools": [tools_r[i] for i in selected_indices.get("tools", []) if i < len(tools_r)]
        }

    def _run_async(self, coro):
        """Run async coroutine in sync code safely."""
        try:
            asyncio.get_running_loop()
            has_running_loop = True
        except RuntimeError:
            has_running_loop = False

        if not has_running_loop:
            return asyncio.run(coro)

        result_holder = {"result": None, "error": None}

        def _runner():
            try:
                result_holder["result"] = asyncio.run(coro)
            except Exception as e:  # pragma: no cover - defensive branch
                result_holder["error"] = e

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join()
        error = result_holder["error"]
        if error is not None:
            raise error
        return result_holder["result"]

    def _format_resources_for_prompt(self, resources: list) -> str:
        """Format resources for inclusion in the prompt."""
        formatted = []
        for i, resource in enumerate(resources):
            if isinstance(resource, dict):
                name = resource.get("name", f"Resource {i}")
                description = resource.get("description", "")
                formatted.append(f"{i}. {name}: {description}")
            elif isinstance(resource, str):
                formatted.append(f"{i}. {resource}")
            else:
                name = getattr(resource, "name", str(resource))
                desc = getattr(resource, "description", "")
                formatted.append(f"{i}. {name}: {desc}")

        return "\n".join(formatted) if formatted else "None available"

    def _parse_llm_response(self, response) -> dict:
        """Parse the LLM response to extract the selected indices.
        Accepts either a plain string or a Responses API-style list of content blocks.
        """
        if isinstance(response, list):
            parts = []
            for item in response:
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            response = "\n".join([p for p in parts if p])
        elif not isinstance(response, str):
            response = str(response)
        selected_indices = {"skills": [], "tools": []}

        skills_match = re.search(r"SKILLS:\s*\[(.*?)\]", response, re.IGNORECASE)
        if skills_match and skills_match.group(1).strip():
            with contextlib.suppress(ValueError):
                indices = skills_match.group(1).split(",")
                selected_indices["skills"] = [
                    int(idx.strip()) for idx in indices if idx.strip()
                ]

        tools_match = re.search(r"TOOLS:\s*\[(.*?)\]", response, re.IGNORECASE)
        if tools_match and tools_match.group(1).strip():
            with contextlib.suppress(ValueError):
                selected_indices["tools"] = [int(idx.strip()) for idx in tools_match.group(1).split(",") if idx.strip()]

        return selected_indices
