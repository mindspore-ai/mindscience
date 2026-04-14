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
# MODIFICATION NOTICE:
# This file contains code from Biomni, which is licensed under the Apache License, Version 2.0 (the "License").
# This file was modified by MindSpore Science Team on 2026.
# Changes include: 
# 1. remove library-related prompt.
# 2. add skill-related prompt.
# 3. add surveyresult-related prompt.
# ============================================================================
"""Prompt utilities for MindScienceAgent."""

def textify_api_dict(api_dict):
    """Convert a nested API dictionary to a nicely formatted string."""
    lines = []
    for category, methods in api_dict.items():
        lines.append(f"Import file: {category}")
        lines.append("=" * (len("Import file: ") + len(category)))
        for method in methods:
            lines.append(f"Method: {method.get('name', 'N/A')}")
            lines.append(f"  Description: {method.get('description', 'No description provided.')}")

            # Process required parameters
            req_params = method.get("required_parameters", [])
            if req_params:
                lines.append("  Required Parameters:")
                for param in req_params:
                    param_name = param.get("name", "N/A")
                    param_type = param.get("type", "N/A")
                    param_desc = param.get("description", "No description")
                    param_default = param.get("default", "None")
                    lines.append(f"    - {param_name} ({param_type}): {param_desc} [Default: {param_default}]")

            # Process optional parameters
            opt_params = method.get("optional_parameters", [])
            if opt_params:
                lines.append("  Optional Parameters:")
                for param in opt_params:
                    param_name = param.get("name", "N/A")
                    param_type = param.get("type", "N/A")
                    param_desc = param.get("description", "No description")
                    param_default = param.get("default", "None")
                    lines.append(f"    - {param_name} ({param_type}): {param_desc} [Default: {param_default}]")

            lines.append("")  # Empty line between methods
        lines.append("")  # Extra empty line after each category

    return "\n".join(lines)


def format_item_with_description(name, description):
    """Format an item with its description in a readable way."""
    # Handle None or empty descriptions
    if not description:
        description = f"Science Data item: {name}"

    # Check if the item is already formatted (contains a colon)
    if isinstance(name, str) and ": " in name:
        return name

    # Wrap long descriptions to make them more readable
    max_line_length = 80
    if len(description) > max_line_length:
        # Simple wrapping for long descriptions
        wrapped_desc = []
        words = description.split()
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= max_line_length:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                wrapped_desc.append(current_line)
                current_line = word

        if current_line:
            wrapped_desc.append(current_line)

        # Join with newlines and proper indentation
        formatted_desc = f"{name}:\n  " + "\n  ".join(wrapped_desc)
        return formatted_desc

    return f"{name}: {description}"


def generate_prompt(
    base_prompt,
    tool_desc=None,
    use_tool_retriever=False,
    survey_results=None,
    skill_path: str = "",
    skills=None
):
    """Generate a system prompt for plan and execute agent based on provided context."""
    prompt_modifier = base_prompt

    survey_results_formatted = []
    if survey_results:
        for paper in survey_results:
            if isinstance(paper, dict):
                title = paper.get("title", "Unknown")
                score = paper.get("score", "N/A")
                abstract = paper.get("abstract", "")
                doi = paper.get("doi", "")
                url = paper.get("url", "")
                source = paper.get("source", "Unknown")
                paper_info = f"Title: {title} (Relevance Score: {score})\n  Abstract: {abstract}"
                paper_info += f"\n  DOI: {doi}\n  URL: {url}\n  Source: {source}"

                if paper.get("is_deep_read"):
                    for field in ("background", "contributions", "methods", "challenges"):
                        val = paper.get(field)
                        if val:
                            paper_info += f"\n  {field.capitalize()}: {val}"
                survey_results_formatted.append(paper_info)

    # skill description
    skill_desc_formatted = []
    if skills:
        for skill in skills:
            skill_info = (
                f'- **{skill["name"]}**: {skill["description"]}\n'
                f'  -> Read `{skill["path"]}` for full instructions'
            )
            skill_desc_formatted.append(skill_info)

    if survey_results_formatted:
        prompt_modifier += """
📄 SURVEY RESULTS (RELEVANT LITERATURE):
{survey_results}

IMPORTANT: These papers have been collected through literature survey and are directly relevant to your task.

===============================
"""

    # Add environment resources
    if skill_desc_formatted or tool_desc:
        prompt_modifier += """

Environment Resources:
"""
    if skill_desc_formatted:
        prompt_modifier += """
- Skills System:
You have access to a skills library that provides specialized capabilities and domain knowledge.
---
**Scientific_skills Skills**: `{skill_path}` (higher priority)

**Available Skills:**
{skill_desc}

**How to Use Skills (Progressive Disclosure):**
Skills follow a **progressive disclosure** pattern - you see their name and description above, but only read full instructions when needed:
1. **Recognize when a skill applies**: Check if the user's task matches a skill's description
2. **Read the skill's full instructions**: Use the path shown in the skill list above
3. **Follow the skill's instructions**: SKILL.md contains step-by-step workflows, best practices, and examples
4. **Access supporting files**: Skills may include helper scripts, configs, or reference docs - use absolute paths

**When to Use Skills:**
- User's request matches a skill's domain (e.g., \"research X\" -> web-research skill)
- You need specialized knowledge or structured workflows
- A skill provides proven patterns for complex tasks

**Executing Skill Scripts:**
Skills may contain Python scripts or other executable files. Always use absolute paths from the skill list.

**Example Workflow:**

User: \"Can you research the latest developments in quantum computing?\"

1. Check available skills -> See \"web-research\" skill with its path
2. Read the skill using the path shown
3. Follow the skill's research workflow (search -> organize -> synthesize)
4. Use any helper scripts with absolute paths

Remember: Skills make you more capable and consistent. When in doubt, check if a skill exists for the task!
---
"""
    if tool_desc:
        prompt_modifier += """
- Function Dictionary:
{function_intro}
---
{tool_desc}
---
{import_instruction}
"""

    # Set appropriate text based on whether this is initial configuration or after retrieval
    if use_tool_retriever:
        function_intro = (
            "Based on your query, I've identified the following most relevant functions "
            "that you can use in your code:"
        )
        import_instruction = (
            "IMPORTANT: When using any function, you MUST first import it from its module. "
            "For example:\nfrom [module_name] import [function_name]"
        )
    else:
        function_intro = (
            "In your code, you will need to import the function location using the following "
            "dictionary of functions:"
        )
        import_instruction = ""

    # Format the prompt with appropriate values
    format_dict = {}

    if tool_desc:
        format_dict["tool_desc"] = textify_api_dict(tool_desc) if isinstance(tool_desc, dict) else tool_desc
        format_dict["function_intro"] = function_intro
        format_dict["import_instruction"] = import_instruction
    if survey_results_formatted:
        format_dict["survey_results"] = "\n".join(survey_results_formatted)
    if skill_desc_formatted:
        format_dict["skill_desc"] = "\n".join(skill_desc_formatted)
        format_dict["skill_path"] = skill_path

    formatted_prompt = prompt_modifier.format(**format_dict)

    return formatted_prompt
