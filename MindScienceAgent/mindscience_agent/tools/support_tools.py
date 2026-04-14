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
"""Support tools for MindScienceAgent providing various utility functions."""

import importlib
import inspect
import os
import random
import re
import time
from io import BytesIO

import arxiv
import PyPDF2
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from pymed import PubMed

from mindscience_agent.utils import logger


def read_function_source_code(function_name: str) -> str:
    """Read source code of a function from any module path."""
    # Split the function name into module path and function name
    parts = function_name.split(".")
    module_path = ".".join(parts[:-1])
    func_name = parts[-1]

    try:
        # Import the module
        module = importlib.import_module(module_path)

        # Get the function object from the module
        function = getattr(module, func_name)

        # Get the source code of the function
        source_code = inspect.getsource(function)

        return source_code
    except (ImportError, AttributeError) as e:
        return f"Error: Could not find function '{function_name}'. Details: {str(e)}"


def query_arxiv(query: str, max_papers: int = 10) -> str:
    """Query arXiv for papers based on the provided search query."""
    try:
        client = arxiv.Client()
        search_res = arxiv.Search(query=query, max_results=max_papers, sort_by=arxiv.SortCriterion.Relevance)
        results = "\n\n".join([
            f"Title: {paper.title}\nSummary: {paper.summary}" for paper in client.results(search_res)
        ])
        return results if results else "No papers found on arXiv."
    except Exception as e:
        return f"Error querying arXiv: {e}"


def query_pubmed(query: str, max_papers: int = 10, max_retries: int = 3) -> str:
    """Query PubMed for papers based on the provided search query."""
    try:
        pubmed = PubMed(tool="MyTool", email="your-email@example.com")  # Update with a valid email address

        # Initial attempt
        papers = list(pubmed.query(query, max_results=max_papers))

        # Retry with modified queries if no results
        retries = 0
        while not papers and retries < max_retries:
            retries += 1
            # Simplify query with each retry by removing the last word
            simplified_query = " ".join(query.split()[:-retries]) if len(query.split()) > retries else query
            time.sleep(1)  # Add delay between requests
            papers = list(pubmed.query(simplified_query, max_results=max_papers))

        if papers:
            results = "\n\n".join(
                [f"Title: {paper.title}\nAbstract: {paper.abstract}\nJournal: {paper.journal}" for paper in papers]
            )
            return results

        return "No papers found on PubMed after multiple query attempts."
    except Exception as e:
        return f"Error querying PubMed: {e}"


def advanced_web_search_qwen(
    query: str,
    max_retries: int = 3,
    base_url: str = "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
    model: str = "qwen3.5-plus",
    enable_thinking: bool = True,
    timeout: int = 60,
) -> str:
    """Advanced web search using Qwen API with built-in web_search tool."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY key must be set in environment variables.")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    delay = random.randint(1, 10)
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"Attempt {attempt}/{max_retries}: Processing query '{query[:50]}...'")

            response = client.responses.create(
                model=model,
                input=query,
                tools=[
                    {"type": "web_search"},
                    {"type": "web_extractor"},
                ],
                extra_body={"enable_thinking": enable_thinking}
            )

            formatted_response = ""
            citations = []

            for item in response.output:
                if item.type == "message":
                    for content in item.content:
                        if content.type == "output_text":
                            formatted_response += content.text

                elif item.type == "web_search_call":
                    if hasattr(item, 'action') and hasattr(item.action, 'sources'):
                        for source in item.action.sources:
                            if source.type == "url":
                                citations.append({"url": source.url})

                elif item.type == "web_extractor_call":
                    if hasattr(item, 'urls'):
                        for url in item.urls:
                            citations.append({"url": url})

            if citations:
                formatted_response += "\n\nSources:\n"
                unique_urls = list({cite["url"] for cite in citations})
                for i, url in enumerate(unique_urls, 1):
                    formatted_response += f"{i}. {url}\n"

            logger.debug(f"Successfully processed query. Found {len(citations)} citations.")
            return formatted_response

        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt}/{max_retries} failed: {str(e)}")

            if attempt < max_retries:
                sleep_time = delay * (2 ** (attempt - 1))
                logger.debug(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"All {max_retries} attempts failed for query: {query[:50]}...")
                return f"Error performing web search after {max_retries} attempts: {str(e)}"

    return f"Error performing web search after {max_retries} attempts: {str(last_error)}"


def extract_url_content(url: str) -> str:
    """Extract the text content of a webpage using requests and BeautifulSoup."""
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)

    # Check if the response is in text format
    if "text/plain" in response.headers.get("Content-Type", "") or "application/json" in response.headers.get(
        "Content-Type", ""
    ):
        return response.text.strip()  # Return plain text or JSON response directly

    # If it's HTML, use BeautifulSoup to parse
    soup = BeautifulSoup(response.text, "html.parser")

    # Try to find main content first, fallback to body
    content = soup.find("main") or soup.find("article") or soup.body

    # Remove unwanted elements
    for element in content(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
        element.decompose()

    # Extract text with better formatting
    paragraphs = content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    cleaned_text = []

    for p in paragraphs:
        text = p.get_text().strip()
        if text:  # Only add non-empty paragraphs
            cleaned_text.append(text)

    return "\n\n".join(cleaned_text)


def extract_pdf_content(url: str) -> str:
    """Extract text content of a PDF file given its URL."""
    try:
        # Check if the URL ends with .pdf
        if not url.lower().endswith(".pdf"):
            # If not, try to find a PDF link on the page
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Look for PDF links in the HTML content
                pdf_links = re.findall(r'href=[\'"]([^\'"]+\.pdf)[\'"]', response.text)
                if pdf_links:
                    # Use the first PDF link found
                    if not pdf_links[0].startswith("http"):
                        # Handle relative URLs
                        base_url = "/".join(url.split("/")[:3])
                        url = base_url + pdf_links[0] if pdf_links[0].startswith("/") else base_url + "/" + pdf_links[0]
                    else:
                        url = pdf_links[0]
                else:
                    return f"No PDF file found at {url}. Please provide a direct link to a PDF file."

        # Download the PDF
        response = requests.get(url, timeout=30)

        # Check if we actually got a PDF file (by checking content type or magic bytes)
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type and not response.content.startswith(b"%PDF"):
            return f"The URL did not return a valid PDF file. Content type: {content_type}"

        pdf_file = BytesIO(response.content)

        # Try with PyPDF2 first
        try:
            text = ""
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
        except Exception as e:
            logger.warning(f"Error extracting text from PDF: {str(e)}")

        # Clean up the text
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return "The PDF file did not contain any extractable text. It may be an image-based PDF requiring OCR."

        return text

    except requests.exceptions.RequestException as e:
        return f"Error downloading PDF: {str(e)}"
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"


def query_semantic_scholar(
    query: str,
    max_papers: int = 10,
    fields_of_study: str = "",
    year: str = ""
) -> str:
    """Query Semantic Scholar for academic papers and research articles."""
    try:
        semantic_scholar_key = os.environ.get("S2_API_KEY", None)
        if not semantic_scholar_key:
            raise ValueError(
                "Semantic Scholar API key not found. Please set the S2_API_KEY environment variable."
            )

        headers = {'x-api-key': semantic_scholar_key}

        search_url = "https://api.semanticscholar.org/graph/v1/paper/search"

        query_params = {
            'query': query,
            'limit': max_papers,
            'fields': 'title,year,citationCount,abstract,tldr,isOpenAccess,openAccessPdf,authors'
        }

        if fields_of_study:
            query_params['fields'] += ',fieldsOfStudy'

        if year:
            query_params['year'] = year

        response = requests.get(search_url, params=query_params, headers=headers, timeout=30)

        if response.status_code == 200:
            searched_data = response.json().get('data', [])

            if not searched_data:
                return "No papers found on Semantic Scholar."

            results = []
            for paper in searched_data:
                title = paper.get('title', 'N/A')
                year_pub = paper.get('year', 'N/A')
                citation_count = paper.get('citationCount', 'N/A')
                abstract = paper.get('abstract', paper.get('tldr', 'N/A'))
                is_open_access = paper.get('isOpenAccess', False)
                pdf_url = paper.get('openAccessPdf', 'N/A')
                authors = paper.get('authors', [])

                author_list = ', '.join([a.get('name', 'N/A') for a in authors]) if authors else 'N/A'

                result_text = f"Title: {title}\n"
                result_text += f"Authors: {author_list}\n"
                result_text += f"Year: {year_pub}\n"
                result_text += f"Citation Count: {citation_count}\n"
                result_text += f"Open Access: {'Yes' if is_open_access else 'No'}\n"

                if pdf_url != 'N/A':
                    result_text += f"PDF URL: {pdf_url}\n"

                result_text += f"Abstract: {abstract}\n"

                results.append(result_text)

            return "\n\n---\n\n".join(results)

        return f"Error querying Semantic Scholar: HTTP {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error querying Semantic Scholar: {str(e)}"
