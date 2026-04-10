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
"""Utility functions for VibeScienceAgent."""
import importlib
import re
import os
from pathlib import Path
from typing import Any
import sys
import certifi


TOOL_MODULE_PREFIX = "vibescience_agent.tools."
_EXCLUDED_FROM_PROMPTS = frozenset({"run_python_repl"})


def read_module2api():
    """Read module API descriptions from tool_description modules."""
    fields = [
        "literature",
        "support_tools",
    ]

    module2api = {}
    for field in fields:
        module_name = f"vibescience_agent.tools.tool_description.{field}"
        module = importlib.import_module(module_name)
        module2api[f"vibescience_agent.tools.{field}"] = module.description
    return module2api


def extract_between_regex(text, start_keyword, end_keyword):
    """Extract text between two keywords using regex."""
    keyword_1 = re.escape(start_keyword)
    keyword_2 = re.escape(end_keyword)
    pattern = f'({keyword_1}.*?{keyword_2})'
    match = re.findall(pattern, text, flags=re.DOTALL)
    return match[0] if match else None


def serialize_agent_messages(messages: list) -> str:
    """Serialize messages to string representation."""
    chunks: list[str] = []
    for msg in messages:
        role = "[User]" if msg.get("role") == "user" else "[Assistant]"
        content = msg.get("content", "")
        chunks.append(f"{role}: {content}")
    return "\n----------------\n".join(chunks)


def build_tool_desc(module2api: dict[str, Any]) -> dict[str, Any]:
    """Build structured tool specs for generate_prompt (drops REPL from text)."""
    return {
        mod: [t for t in tools if t.get("name") not in _EXCLUDED_FROM_PROMPTS]
        for mod, tools in module2api.items()
    }


def normalize_custom_tools(raw: Any) -> list[dict[str, Any]]:
    """Normalize custom tools input to consistent format."""
    if not raw:
        return []
    if isinstance(raw, list):
        return list(raw)
    if isinstance(raw, dict):
        out: list[dict[str, Any]] = []
        for name, info in raw.items():
            if isinstance(info, dict):
                out.append({
                    "name": name,
                    "description": info.get("description", ""),
                    "module": info.get("module", "custom_tools"),
                })
            else:
                out.append({"name": name, "description": str(info), "module": "custom_tools"})
        return out
    return []


def normalize_named_items(raw: Any) -> list[dict[str, str]]:
    """Normalize named items input to consistent format."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        return [
            {"name": name, "description": (info or {}).get("description", "") if isinstance(info, dict) else str(info)}
            for name, info in raw.items()
        ]
    return []


def load_env(env_path: str = ".env") -> None:
    """Read .env file and set environment variables."""
    env_file = Path(env_path)

    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse KEY=VALUE format
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                os.environ[key] = value


def extract_skill_description(markdown_path):
    """Extract skill description."""
    with open(markdown_path, 'r', encoding='utf-8') as file:
        content = file.read()

        # 匹配所有 YAML 块
        pattern = r'---\n(.*?)\n---'
        yaml_blocks = re.findall(pattern, content, re.DOTALL)

        for block in yaml_blocks:
            # 提取 name
            name_match = re.search(r'^name:\s*(.*?)$', block, re.MULTILINE)
            # 提取 description（可能跨越多行）
            desc_match = re.search(r'^description:\s*(.*?)(?=\n\w+:|$)', block, re.MULTILINE | re.DOTALL)

            if name_match and desc_match:
                name = name_match.group(1).strip()
                # 清理描述文本（移除多余空格和换行）
                description = desc_match.group(1).strip()
                description = re.sub(r'\n\s*', ' ', description)

                return name, description
    return None


def set_ssl_cert_file_path():
    """Set SSL certificate file path on Windows."""
    if sys.platform.startswith("win"):
        # Reset SSL-related environment variables on Windows
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

        cert_path = os.environ['SSL_CERT_FILE'].replace('/', '\\')
        os.environ['SSL_CERT_FILE'] = cert_path
        os.environ['REQUESTS_CA_BUNDLE'] = cert_path
