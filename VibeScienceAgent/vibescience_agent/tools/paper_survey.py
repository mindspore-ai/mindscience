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
"""Literature search and paper survey utilities for VibeScienceAgent."""
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin
import httpx
import pdfplumber
import requests
from bs4 import BeautifulSoup

from vibescience_agent.utils import logger

GRAPH_URL = "https://api.semanticscholar.org/graph/v1/paper/"
REC_URL = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper/"

_ARXIV_SORT_RELEVANCE = "relevance"


@dataclass
class PaperMetadata:
    """Data class for paper metadata."""
    title: str
    authors: List[str]
    abstract: str
    year: Optional[int] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    url: Optional[str] = None
    citations: Optional[int] = None
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "year": self.year,
            "doi": self.doi,
            "journal": self.journal,
            "url": self.url,
            "citations": self.citations,
            "source": self.source,
        }


def fetch_semantic_papers(keyword, max_results=20, api_key: Optional[str] = None):
    """Fetch papers from Semantic Scholar based on keyword."""
    search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    query_params = {
        'query': keyword,
        'limit': max_results,
        'fields': 'title,year,citationCount,abstract,tldr,isOpenAccess,openAccessPdf'
    }

    headers = {'x-api-key': api_key}
    response = requests.get(search_url, params=query_params, headers=headers, verify=False, timeout=30)

    if response.status_code == 200:
        searched_data = response.json().get('data', [])
        papers = []
        for paper in searched_data:
            author_list = [author.get("name", "") for author in paper.get("authors", [])]

            paper = PaperMetadata(
                title=paper.get("title", ""),
                authors=author_list,
                abstract=paper.get("abstract", ""),
                year=paper.get("year"),
                doi=paper.get("doi"),
                journal=paper.get("journal", {}).get("name") if paper.get("journal") else None,
                url=paper.get("url"),
                citations=paper.get("citationCount"),
                source='semantic_scholar'
            )
            papers.append(paper.to_dict())

        return papers

    logger.debug(f"KeywordQuery: {response.status_code}")
    return []


def fetch_pubmed_papers(query: str, max_results: int = 20, sort: str = "relevance") -> list:
    """Fetch papers from PubMed based on the query."""
    logger.debug(f"Searching PubMed for: {query}")

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    search_url = f"{base_url}/esearch.fcgi"
    fetch_url = f"{base_url}/efetch.fcgi"

    sort_param = "relevance" if sort == "relevance" else "pub+date"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": sort_param
    }

    try:
        response = requests.get(search_url, params=search_params, verify=False, timeout=30)
        if response.status_code != 200:
            logger.error(f"PubMed search error: {response.status_code}")
            return []

        search_data = response.text
        soup = BeautifulSoup(search_data, "xml")
        pmids = [item.text for item in soup.find_all("Id")]

        if not pmids:
            logger.debug(f"No PubMed results found for query: {query}")
            return []

        # Request full PubMed records for the PMIDs returned by esearch.
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }

        fetch_response = requests.get(fetch_url, params=fetch_params, verify=False, timeout=30)
        if fetch_response.status_code != 200:
            logger.error(f"PubMed fetch error: {fetch_response.status_code}")
            return []

        xml_data = fetch_response.text
        papers = parse_pubmed_xml(xml_data)
        return papers

    except Exception as e:
        logger.error(f"Error searching PubMed: {str(e)}")
        return []


def fetch_arxiv_papers(query: str, max_results: int = 20, sort: str = "relevance", categories: list = None) -> list:
    """Fetch papers from arXiv based on the query."""
    logger.debug(f"Searching arXiv for: {query}")

    # arXiv API URL
    search_url = "http://export.arxiv.org/api/query"

    # Sort parameter
    sort_param = "relevance" if sort == "relevance" else "submittedDate"

    # Category filter
    cat_filter = ""
    if categories:
        cat_filter = " AND (" + " OR ".join([f"cat:{cat}" for cat in categories]) + ")"

    # Search parameters
    search_params = {
        "search_query": f"all:{query}{cat_filter}",
        "max_results": max_results,
        "sortBy": sort_param,
        "sortOrder": "descending"
    }

    try:
        response = requests.get(search_url, params=search_params, verify=False, timeout=30)
        if response.status_code != 200:
            logger.error(f"arXiv search error: {response.status_code}")
            return []

        xml_data = response.text
        papers = parse_arxiv_xml(xml_data)

        logger.debug(f"Get {len(papers)} papers from arXiv")

        return papers

    except Exception as e:
        logger.error(f"Error searching arXiv: {e}")
        return []


def select_papers(paper_bank, max_papers, rag_read_depth):
    """Select papers for deep reading based on scores and availability."""
    selected_for_deep_read = []
    count = 0
    for paper in sorted(paper_bank, key=lambda x: x['score'], reverse=True):
        if count >= rag_read_depth:
            break
        url = None
        if paper['source'] in ['arXiv', 'pubmed']:
            # For arXiv and pubmed, check if 'url' or 'doi' exists
            if 'url' in paper:
                url = paper['url']
            elif 'doi' in paper:
                url = paper['doi']
        elif paper['source'] == 'semantic_scholar':
            # For semantic_scholar, check if 'isOpenAccess' is True
            if paper.get('isOpenAccess', False):
                if 'openAccessPdf' in paper and 'url' in paper['openAccessPdf']:
                    url = paper['openAccessPdf']['url']

        if url:
            selected_for_deep_read.append(paper)
            count += 1

    selected_for_deep_read = selected_for_deep_read[:max_papers]
    return selected_for_deep_read


def parse_arxiv_xml(xml_data: str) -> list:
    """Parse arXiv XML response to extract paper metadata."""
    papers = []
    soup = BeautifulSoup(xml_data, "xml")

    for entry in soup.find_all("entry"):
        try:
            # Title
            title_elem = entry.find("title")
            title_text = title_elem.text.strip() if title_elem else ""

            # Abstract
            summary_elem = entry.find("summary")
            abstract_text = summary_elem.text.strip() if summary_elem else ""

            # Authors
            authors = []
            for author in entry.find_all("author"):
                name_elem = author.find("name")
                if name_elem:
                    authors.append(name_elem.text.strip())

            # Publication year
            published_elem = entry.find("published")
            year = None
            if published_elem:
                try:
                    pub_date = published_elem.text.strip()
                    match = re.search(r"(\d{4})", pub_date)
                    if match:
                        year = int(match.group(1))
                except ValueError:
                    pass

            # DOI and URL
            doi = None
            url = None
            for link in entry.find_all("link"):
                href = link.get("href", "")
                if link.get("title") == "doi":
                    doi = href.replace("http://dx.doi.org/", "")
                elif link.get("rel") == "alternate":
                    url = href.replace("abs", "pdf")

            paper = PaperMetadata(
                    title=title_text,
                    authors=authors,
                    abstract=abstract_text,
                    year=year,
                    doi=doi,
                    journal="arXiv",
                    url=url,
                    source='arXiv'
                )
            papers.append(paper.to_dict())

        except Exception as e:
            logger.error(f"Error parsing arXiv entry: {str(e)}")

    return papers


def parse_pubmed_xml(xml_data: str) -> list:
    """Parse PubMed XML response to extract paper metadata."""
    papers = []
    soup = BeautifulSoup(xml_data, "xml")

    for article in soup.find_all("PubmedArticle"):
        try:
            article_data = article.find("Article")
            if not article_data:
                continue

            # Title
            title = article_data.find("ArticleTitle")
            title_text = title.text if title else ""

            # Abstract
            abstract_elem = article_data.find("Abstract")
            abstract_text = ""
            if abstract_elem:
                abstract_parts = abstract_elem.find_all("AbstractText")
                if abstract_parts:
                    abstract_text = " ".join(part.text for part in abstract_parts)

            # Authors
            authors = []
            author_list = article_data.find("AuthorList")
            if author_list:
                for author in author_list.find_all("Author"):
                    last_name = author.find("LastName")
                    fore_name = author.find("ForeName")

                    if last_name and fore_name:
                        authors.append(f"{fore_name.text} {last_name.text}")
                    elif last_name:
                        authors.append(last_name.text)

            # Journal
            journal_elem = article_data.find("Journal")
            journal_name = ""
            if journal_elem:
                journal_title = journal_elem.find("Title")
                if journal_title:
                    journal_name = journal_title.text

            # Publication Date
            pub_date_elem = journal_elem.find("PubDate") if journal_elem else None
            year = None
            if pub_date_elem:
                year_elem = pub_date_elem.find("Year")
                if year_elem:
                    try:
                        year = int(year_elem.text)
                    except ValueError:
                        pass

            # DOI
            doi = None
            article_id_list = article.find("ArticleIdList")
            if article_id_list:
                for article_id in article_id_list.find_all("ArticleId"):
                    if article_id.get("IdType") == "doi":
                        doi = article_id.text
                        break

            # Create paper metadata
            paper = PaperMetadata(
                title=title_text,
                authors=authors,
                abstract=abstract_text,
                year=year,
                doi=doi,
                journal=journal_name + "@Pubmed",
                source='pubmed'
            )
            papers.append(paper.to_dict())

        except Exception as e:
            logger.error(f"Error parsing PubMed article: {str(e)}")

    return papers


def parse_io_description(output):
    """Parse input and output descriptions from string."""
    match_input = re.match(r'Input\("([^"]+)"\)', output)
    input_description = match_input.group(1) if match_input else None
    match_output = re.match(r'.*Output\("([^"]+)"\)', output)
    output_description = match_output.group(1) if match_output else None
    return input_description, output_description


def format_papers_for_printing_next_query(paper_lst: list) -> str:
    """Convert a list of papers to a string for use in LLM prompts."""
    parts: list[str] = []
    for paper in paper_lst:
        pid = paper.get("id", "")
        parts.append(f"paperId: {pid}")
        parts.append(f"title: {paper.get('title', '').strip()}")
        parts.append("")
    return "\n".join(parts)


def download_pdf(pdf_url, save_folder="pdfs"):
    """Download PDF file from URL and save to specified folder."""
    logger.debug(f"downloading pdf from {pdf_url}")

    if not pdf_url:
        return None

    os.makedirs(save_folder, exist_ok=True)

    file_name = pdf_url.split("/")[-1]
    if not file_name.endswith('.pdf'):
        file_name = file_name + '.pdf'
    save_path = os.path.join(save_folder, file_name)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
    }
    try:
        response = httpx.get(url=pdf_url,headers=headers, timeout=10, verify=False)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
            return save_path

        logger.error(f"Failed to download PDF from {pdf_url}: {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"Error downloading PDF from {pdf_url}: {e}")
        return None


def download_pdf_by_doi(doi: str, download_dir: str = "downloaded_papers") -> str | None:
    """Download PDF paper by DOI from publisher page."""
    doi = doi.strip()
    if doi.lower().startswith("doi:"):
        doi = doi[4:].strip()
    if doi.lower().startswith("https://doi.org/"):
        doi = doi[16:].strip()

    doi_url = f"https://doi.org/{doi}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(doi_url, headers=headers, allow_redirects=True, verify=False, timeout=30)
    publisher_url = response.url
    logger.debug(f"Redirected to the publisher page: {publisher_url}")

    soup = BeautifulSoup(response.text, "html.parser")
    pdf_links = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        link_text = link.get_text().lower()
        if (
            "pdf" in href.lower()
            or "pdf" in link_text
            or ("download" in link_text and ("full" in link_text or "article" in link_text))
            or "full text" in link_text
        ):
            pdf_links.append(urljoin(publisher_url, href))

    if pdf_links:
        logger.debug(f"Found {len(pdf_links)} candidate PDF link(s)")
        pdf_url = pdf_links[0]
        pdf_response = requests.get(pdf_url, headers=headers, stream=True, verify=False, timeout=30)
        if pdf_response.status_code == 200 and "application/pdf" in pdf_response.headers.get("Content-Type", ""):
            os.makedirs(download_dir, exist_ok=True)
            filename = f"{doi.replace('/', '_')}.pdf"
            filepath = os.path.join(download_dir, filename)
            with open(filepath, "wb") as f:
                for chunk in pdf_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.debug(f"PDF saved to {filepath}")
            return filepath
        logger.warning("Could not download a valid PDF from publisher links.")
    else:
        logger.warning("No PDF links found on publisher page.")

    return None


def extract_text_from_pdf(pdf_path: str) -> str | None:
    """Extract text content from PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        logger.warning(f"Error extracting text from PDF: {e}")
        return None


def paper_query(paper_id, api_key: Optional[str] = None):
    """Query paper recommendations by paper ID."""
    query_params = {
        'paperId': paper_id,
        'limit': 20,
        'fields': 'title,year,citationCount,abstract'
    }
    headers = {'x-api-key': api_key}
    response = requests.get(url=REC_URL + paper_id, params=query_params, headers=headers, verify=False, timeout=30)
    if response.status_code == 200:
        return response.json()

    return None


def paper_details(paper_id, fields=(
    'title,year,abstract,authors,citationCount,venue,citations,references,tldr'
), api_key: Optional[str] = None):
    """Get paper details based on paper ID."""
    paper_data_query_params = {'fields': fields}
    headers = {'x-api-key': api_key}
    response = requests.get(url=GRAPH_URL + paper_id, params=paper_data_query_params,
                            headers=headers, verify=False, timeout=30)
    if response.status_code == 200:
        return response.json()

    return None


def get_abstract(paper_id, api_key: Optional[str] = None):
    """Get the abstract of a paper based on paper ID."""
    details = paper_details(paper_id, api_key=api_key)

    if details is not None:
        return details["abstract"]

    return None


def get_citation_count(paper_id, api_key: Optional[str] = None):
    """Get the citation count of a paper based on paper ID."""
    details = paper_details(paper_id, api_key=api_key)

    if details is not None:
        return int(details["citationCount"])

    return None


def get_citations(paper_id, api_key: Optional[str] = None):
    """Get the citation list of a paper based on paper ID."""
    details = paper_details(paper_id, api_key=api_key)

    if details is not None:
        return details["citations"]

    return None


def get_references(paper_id, api_key: Optional[str] = None):
    """Get the reference list of a paper based on paper ID."""
    details = paper_details(paper_id, api_key=api_key)
    if details is None:
        return None
    references = (details.get("references") or [])[:100]

    ## get details of each reference, keep first 20 to save costs
    detailed_references = [
        paper_details(ref["paperId"], fields="title,year,abstract,citationCount", api_key=api_key)
        for ref in references
        if ref.get("paperId")
    ]
    detailed_references = [p for p in detailed_references if p and is_valid_paper(p)][:20]

    return detailed_references


def is_valid_paper(paper):
    """Check if paper is valid based on heuristics."""
    # Check for specific keywords indicating non-research papers
    title = paper.get("title", "").lower() if paper.get("title") else ""
    abstract = paper.get("abstract", "").lower() if paper.get("abstract") else ""
    if ("survey" in title or "survey" in abstract or
        "review" in title or "review" in abstract or
        "position paper" in title or "position paper" in abstract):
        return False

    # Check abstract length (new rule)
    if len(abstract.split()) <= 50:
        return False

    return True


def paper_filter(paper_lst):
    """Filter out papers based on basic heuristics."""
    filtered_paper_lst = {}

    for source, papers in paper_lst.items():
        if isinstance(papers, list):
            filtered_papers = [paper for paper in papers if is_valid_paper(paper)]
            filtered_paper_lst[source] = filtered_papers
        else:
            filtered_paper_lst[source] = papers

    return filtered_paper_lst


def multi_source_search(
    query: str,
    sources: Optional[List[str]] = None,
    max_results: int = 10,
    *,
    semantic_scholar_key: Optional[str] = None,
    **kwargs,
) -> dict[str, list[dict]]:
    """Search papers across multiple sources."""
    if not sources:
        sources = ["pubmed", "arxiv", "semantic_scholar"]

    combined_results = {}

    for source in sources:
        if source == "pubmed":
            combined_results[source] = fetch_pubmed_papers(query, max_results, **kwargs)
        elif source == "arxiv":
            combined_results[source] = fetch_arxiv_papers(
                query, max_results, sort=_ARXIV_SORT_RELEVANCE, **kwargs
            )
        elif source == "semantic_scholar":
            combined_results[source] = fetch_semantic_papers(query, max_results, api_key=semantic_scholar_key)
        else:
            logger.warning(f"Unknown source: {source}. Skipping.")

    return combined_results


class PaperSurvey:
    """
    Routes literature queries to search and Semantic Scholar helper APIs.

    Args:
        config (PaperSurveyConfig): config for paper survey tool
    """
    def __init__(
        self,
        config,
    ):
        self.max_results = config.max_results
        self.sources = config.sources

    def query_route(self, query):
        """Route query to appropriate parsing and execution function."""
        return parse_and_execute(
            query,
            self.max_results,
            self.sources,
        )


def parse_and_execute(
    output,
    max_results,
    sources: Optional[List[str]] = None,
):
    """Parse output string and execute corresponding API function."""
    semantic_scholar_key = os.environ.get("S2_API_KEY", None)
    if not semantic_scholar_key:
        raise ValueError(
            "Semantic Scholar API key not found. Please set the S2_API_KEY environment variable."
        )

    if output.startswith("KeywordQuery"):
        match = re.match(r'KeywordQuery\("([^"]+)"\)', output)
        keyword = match.group(1) if match else None
        if keyword:
            response = multi_source_search(
                keyword,
                sources=sources,
                max_results=max_results,
                semantic_scholar_key=semantic_scholar_key,
            )
            if response is not None:
                return paper_filter(response)
        return None
    if output.startswith("PaperQuery"):
        match = re.match(r'PaperQuery\("([^"]+)"\)', output)
        paper_id = match.group(1) if match else None
        if paper_id:
            response = paper_query(paper_id, api_key=semantic_scholar_key)
            if response is not None and response.get("recommendedPapers"):
                recs = response["recommendedPapers"]
                if isinstance(recs, list):
                    return paper_filter({"semantic_scholar": recs})
                if isinstance(recs, dict):
                    return paper_filter(recs)
        return None
    if output.startswith("GetAbstract"):
        match = re.match(r'GetAbstract\("([^"]+)"\)', output)
        paper_id = match.group(1) if match else None
        return get_abstract(paper_id, api_key=semantic_scholar_key) if paper_id else None
    if output.startswith("GetCitationCount"):
        match = re.match(r'GetCitationCount\("([^"]+)"\)', output)
        paper_id = match.group(1) if match else None
        return get_citation_count(paper_id, api_key=semantic_scholar_key) if paper_id else None
    if output.startswith("GetCitations"):
        match = re.match(r'GetCitations\("([^"]+)"\)', output)
        paper_id = match.group(1) if match else None
        return get_citations(paper_id, api_key=semantic_scholar_key) if paper_id else None
    if output.startswith("GetReferences"):
        match = re.match(r'GetReferences\("([^"]+)"\)', output)
        paper_id = match.group(1) if match else None
        return get_references(paper_id, api_key=semantic_scholar_key) if paper_id else None
    return None
