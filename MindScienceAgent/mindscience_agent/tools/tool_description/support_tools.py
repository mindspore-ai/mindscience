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
# Changes include: remove unused code snippets.
# ============================================================================
"""Descriptions for support tools used by agents."""
description = [
    {
        "description": "Executes the provided Python command in the notebook environment and returns the output.",
        "name": "run_python_repl",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Python command to execute in the notebook environment",
                "name": "command",
                "type": "str",
            }
        ],
    },
    {
        "description": "Read the source code of a function from any module path.",
        "name": "read_function_source_code",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Fully qualified function name "
                "(e.g., "
                "'bioagentos.tool.support_tools.write_python_code')",
                "name": "function_name",
                "type": "str",
            }
        ],
    },
    {
        "description": "Download data from Synapse using entity IDs. Requires SYNAPSE_AUTH_TOKEN environment variable for authentication. CRITICAL: Always specify entity_type parameter based on what you're downloading (file, dataset, folder, project). Check user hints like 'files' or search results to determine correct type. Multiple IDs only work with entity_type='file'. Recursive only works with entity_type='folder'.",
        "name": "download_synapse_data",
        "optional_parameters": [
            {
                "name": "download_location",
                "type": "str",
                "description": "Directory where files will be downloaded",
                "default": ".",
            },
            {
                "name": "follow_link",
                "type": "bool",
                "description": "Whether to follow links to download the linked entity",
                "default": False,
            },
            {
                "name": "recursive",
                "type": "bool",
                "description": "Whether to recursively download folders and their contents. ONLY valid with entity_type='folder'",
                "default": False,
            },
            {
                "name": "timeout",
                "type": "int",
                "description": "Timeout in seconds for each download operation",
                "default": 300,
            },
            {
                "name": "entity_type",
                "type": "str",
                "description": "Type of Synapse entity: 'file', 'dataset', 'folder', or 'project'. MUST match actual entity type! Check user hints (e.g., 'files' means entity_type='file') or search results ('node_type' field). Default 'dataset' should only be used for actual datasets.",
                "default": "dataset",
            },
        ],
        "required_parameters": [
            {
                "name": "entity_ids",
                "type": "str|list[str]",
                "description": "Synapse entity ID(s) to download. For files: single ID or list of IDs. For datasets/folders/projects: single ID only",
                "default": None,
            }
        ],
    },
]
