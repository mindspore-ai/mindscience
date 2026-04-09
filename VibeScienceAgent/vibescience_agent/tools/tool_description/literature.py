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
description = [
    {
        "description": "Fetches supplementary information for a paper given its DOI "
        "and saves it to a specified directory.",
        "name": "fetch_supplementary_info_from_doi",
        "optional_parameters": [
            {
                "default": "supplementary_info",
                "description": "Directory to save supplementary files",
                "name": "output_dir",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The paper DOI",
                "name": "doi",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query arXiv for papers based on provided search query.",
        "name": "query_arxiv",
        "optional_parameters": [
            {
                "default": 10,
                "description": "The maximum number of papers to retrieve.",
                "name": "max_papers",
                "type": "int",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The search query string.",
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query PubMed for papers based on the provided search query.",
        "name": "query_pubmed",
        "optional_parameters": [
            {
                "default": 10,
                "description": "The maximum number of papers to retrieve.",
                "name": "max_papers",
                "type": "int",
            },
            {
                "default": 3,
                "description": "Maximum number of retry attempts with modified queries.",
                "name": "max_retries",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The search query string.",
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Extract the text content of a webpage using requests and BeautifulSoup.",
        "name": "extract_url_content",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Webpage URL to extract content from",
                "name": "url",
                "type": "str",
            }
        ],
    },
    {
        "description": "Extract the text content of a PDF file given its URL.",
        "name": "extract_pdf_content",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "URL of the PDF file to extract text from",
                "name": "url",
                "type": "str",
            }
        ],
    },
    {
        "description": "Perform advanced web search using Qwen's built-in web search and extraction capabilities. This tool is ideal for finding current information, recent news, latest research, and up-to-date facts that may not be in the training data. Use this when you need information about recent events, current trends, latest scientific discoveries, or any time-sensitive information.",
        "name": "advanced_web_search_qwen",
        "optional_parameters": [
            {
                "default": 3,
                "description": "Maximum number of retry attempts with exponential backoff.",
                "name": "max_retries",
                "type": "int",
            },
            {
                "default": "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
                "description": "DashScope API base URL for Responses API.",
                "name": "base_url",
                "type": "str",
            },
            {
                "default": "qwen3.5-plus",
                "description": "Qwen model to use for search and synthesis.",
                "name": "model",
                "type": "str",
            },
            {
                "default": True,
                "description": "Enable thinking mode for better reasoning and analysis.",
                "name": "enable_thinking",
                "type": "bool",
            },
            {
                "default": 60,
                "description": "Request timeout in seconds.",
                "name": "timeout",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The search query or question you want to find information about. Be specific and detailed for better results.",
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query Semantic Scholar for academic papers and research articles. Semantic Scholar provides free, AI-powered search for scientific literature across all fields of study. Requires S2_API_KEY environment variable. Get your API key from https://www.semanticscholar.org/product/api. Use this when you need to find academic papers, research articles, or scholarly information.",
        "name": "query_semantic_scholar",
        "optional_parameters": [
            {
                "default": 10,
                "description": "Maximum number of papers to retrieve (default: 10).",
                "name": "max_papers",
                "type": "int",
            },
            {
                "default": "",
                "description": "Filter results by field of study (e.g., 'Medicine', 'Biology', 'Computer Science').",
                "name": "fields_of_study",
                "type": "str",
            },
            {
                "default": "",
                "description": "Filter results by year (e.g., '2024' for papers from 2024).",
                "name": "year",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The search query for academic papers or research topics.",
                "name": "query",
                "type": "str",
            }
        ],
    },
]
