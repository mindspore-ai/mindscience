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
import os

from vibescience_agent.tools.env_desc import library_content_dict


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
    library_content_list=None,
    use_tool_retriever=False,
    custom_tools=None,
    custom_data=None,
    custom_software=None,
    survey_results=None,
    skill_path: str = "",
    skills=None
):
    """
    Generate a system prompt for the plan and execute agent based on the provided context.

    Args:
        tool_desc: Description of available tools
        library_content_list: List of library contents available
        custom_tools: Optional list of custom tools available
        custom_data: Optional list of custom datasets available
        custom_software: Optional list of custom software available
        survey_results: Optional list of survey results to include in the prompt
        selected_resources: Optional dict from tool retriever (tools, libraries)
    """
    prompt_modifier = base_prompt
    # Separate custom and default resources
    default_library_content_list = []

    # Filter out custom items from default lists
    custom_software_names = set()

    if custom_software:
        custom_software_names = {item.get("name") if isinstance(item, dict) else item for item in custom_software}

    # Separate default library items
    if library_content_list:
        for lib in library_content_list:
            if isinstance(lib, dict):
                name = lib.get("name", "")
                if name not in custom_software_names:
                    default_library_content_list.append(lib)
            elif lib not in custom_software_names:
                default_library_content_list.append(lib)

        # Format default library content
        if isinstance(default_library_content_list, list) and all(
            isinstance(item, str) for item in default_library_content_list
        ):
            if (
                len(default_library_content_list) > 0
                and isinstance(default_library_content_list[0], str)
                and "," not in default_library_content_list[0]
            ):
                # Simple list of strings
                libraries_formatted = []
                for lib in default_library_content_list:
                    description = library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))
            else:
                # Already formatted string
                libraries_formatted = default_library_content_list
        else:
            # List with descriptions
            libraries_formatted = []
            for lib in default_library_content_list:
                if isinstance(lib, dict):
                    name = lib.get("name", "")
                    description = library_content_dict.get(name, f"Software library: {name}")
                    libraries_formatted.append(format_item_with_description(name, description))
                else:
                    description = library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))

    # Format custom resources with highlighting
    custom_tools_formatted = []
    if custom_tools:
        for tool in custom_tools:
            if isinstance(tool, dict):
                name = tool.get("name", "Unknown")
                desc = tool.get("description", "")
                module = tool.get("module", "custom_tools")
                custom_tools_formatted.append(f"🔧 {name} (from {module}): {desc}")
            else:
                custom_tools_formatted.append(f"🔧 {str(tool)}")

    custom_data_formatted = []
    if custom_data:
        for item in custom_data:
            if isinstance(item, dict):
                name = item.get("name", "Unknown")
                desc = item.get("description", "")
                custom_data_formatted.append(f"📊 {format_item_with_description(name, desc)}")
            else:
                desc = f"Custom data: {item}"
                custom_data_formatted.append(f"📊 {format_item_with_description(item, desc)}")

    custom_software_formatted = []
    if custom_software:
        for item in custom_software:
            if isinstance(item, dict):
                name = item.get("name", "Unknown")
                desc = item.get("description", "")
                custom_software_formatted.append(f"⚙️  {format_item_with_description(name, desc)}")
            else:
                desc = library_content_dict.get(item, f"Custom software: {item}")
                custom_software_formatted.append(f"⚙️ {format_item_with_description(item, desc)}")

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
            name = skill["name"]
            description = skill["description"]
            skill_info = f'- **{skill["name"]}**: {skill["description"]}\n  -> Read `{skill["path"]}` for full instructions'
            skill_desc_formatted.append(skill_info)

    # Add custom resources section first (highlighted)
    has_custom_resources = any(
        [custom_tools_formatted, custom_data_formatted, custom_software_formatted]
    )

    if has_custom_resources:
        prompt_modifier += """

PRIORITY CUSTOM RESOURCES
===============================
IMPORTANT: The following custom resources have been specifically added for planning use.
    PRIORITIZE using these resources as they are directly relevant to task planning.
    Always consider these FIRST and in the meantime using default resources.

"""

        if custom_tools_formatted:
            prompt_modifier += """

🔧 CUSTOM TOOLS (USE THESE FIRST):
{custom_tools}

"""

        if custom_data_formatted:
            prompt_modifier += """

📊 CUSTOM DATA (PRIORITIZE THESE DATASETS):
{custom_data}

"""

        if custom_software_formatted:
            prompt_modifier += """

⚙️  CUSTOM SOFTWARE (USE THESE LIBRARIES):
{custom_software}

"""

        prompt_modifier += """===============================
"""

    if survey_results_formatted:
        prompt_modifier += """

📄 SURVEY RESULTS (RELEVANT LITERATURE):
{survey_results}

IMPORTANT: These papers have been collected through literature survey and are directly relevant to your task.

===============================
"""

    # Add environment resources
    if skill_desc_formatted or tool_desc or library_content_list:
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
    if library_content_list:
        prompt_modifier += """
- Software Library:
{library_intro}
Each library is listed with its description to help you understand its functionality.
----
{library_content_formatted}
----

"""

    # Set appropriate text based on whether this is initial configuration or after retrieval
    if use_tool_retriever:
        function_intro = (
            "Based on your query, I've identified the following most relevant functions "
            "that you can use in your code:"
        )
        library_intro = (
            "Based on your query, I've identified the following most relevant libraries "
            "that you can use:"
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
        library_intro = (
            "The environment supports a list of libraries that can be directly used. "
            "Do not forget the import statement:"
        )
        import_instruction = ""

    # Format the prompt with appropriate values
    format_dict = {}

    if tool_desc:
        format_dict["tool_desc"] = textify_api_dict(tool_desc) if isinstance(tool_desc, dict) else tool_desc
        format_dict["function_intro"] = function_intro
        format_dict["import_instruction"] = import_instruction
    if library_content_list:
        # Format the content consistently
        library_content_formatted = "\n".join(libraries_formatted)
        format_dict["library_content_formatted"] = library_content_formatted
        format_dict["library_intro"] = library_intro
    if survey_results_formatted:
        format_dict["survey_results"] = "\n".join(survey_results_formatted)
    if skill_desc_formatted:
        format_dict["skill_desc"] = "\n".join(skill_desc_formatted)
        format_dict["skill_path"] = skill_path

    # Add custom resources to format dict
    if custom_tools_formatted:
        format_dict["custom_tools"] = "\n".join(custom_tools_formatted)
    if custom_data_formatted:
        format_dict["custom_data"] = "\n".join(custom_data_formatted)
    if custom_software_formatted:
        format_dict["custom_software"] = "\n".join(custom_software_formatted)

    formatted_prompt = prompt_modifier.format(**format_dict)

    return formatted_prompt
