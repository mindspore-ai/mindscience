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
Survey Agent for VibeScienceAgent

This module implements the Survey Agent, which performs comprehensive literature
surveys on research topics. The agent generates intelligent search queries, retrieves
relevant academic papers from multiple sources, scores papers based on relevance,
and performs deep reading analysis to extract methodological details from top papers.
This agent supports automated, iterative literature review with query refinement.
"""
import os
from typing import Any, Dict

from vibescience_agent.agents.base_agent import BaseAgent
from vibescience_agent.tools.paper_survey import (
    PaperSurvey, parse_io_description, format_papers_for_printing_next_query,
    download_pdf, extract_text_from_pdf, download_pdf_by_doi, select_papers
)
from vibescience_agent.utils import logger
from vibescience_agent.config.agent_config import SurveyAgentConfig
from vibescience_agent.config.tool_config import ToolConfig

_PREPARE_BASE_PROMPT = """
You are an expert problem analyzer. Your task is to process a user's question and generate two critical outputs:
1. **Key Information Needed**: Background knowledge required to solve the user's question.
2. **Domain Classification**: The primary academic/technical domain the question belongs to.

**Instructions**
- **Input Handling**: Accept a single user question (text) as input.
- **Key Information Extraction**: Identify necessary components to address the question comprehensively. Avoid redundancy and summary to ONE SINGLE sentence LESS THAN 25 words.
- **Domain Classification**: Assign 1 most relevant domains using standard terminology (e.g., "Machine Learning" instead of "AI").

ONLY output the JSON dict with NO additional text. DO NOT output newline characters. DO NOT output any markdown modifier so that we can call json.loads() on the output later.
"""


class SurveyAgent(BaseAgent):
    """
    Survey Agent conducts comprehensive literature surveys for research topics.

    Args:
        model: Language model to use
        config: Configuration dictionary
        tool_config: Tool configuration dictionary

    Inputs:
        - messages (list): Conversation history; the user task is taken from messages[0].
        - params (Dict[str, Any]): Unused; reserved for extensions.

    Outputs:
        - Dict[str, Any]: List of papers with metadata, scores, and deep reading analysis.
    """
    def __init__(self, model, config: SurveyAgentConfig, tool_config: Dict[str, ToolConfig] = None):
        super().__init__(model, config, tool_config)

        # Load agent-specific configuration
        self.max_papers = config.max_papers

        # Initialize tools
        self.paper_survey = PaperSurvey(tool_config.get("paper_survey", {}))

        self.prepare_system_prompt = _PREPARE_BASE_PROMPT
        logger.debug("SurveyAgent system prompt for preparing process:\n" + self.prepare_system_prompt)

    async def execute(self, messages, **params) -> Dict[str, Any]:
        """Prepare description/domain, then conduct advanced paper survey."""
        description, domain = await self.prepare_description_domain(messages)
        papers = await self.advanced_query_paper(description, domain)

        return papers

    async def prepare_description_domain(self, messages):
        """Extract problem description and domain from user query using structured output."""
        problem = messages[0]["content"]

        logger.debug("SurveyAgent preparing call model inputs:\n" + problem)

        output_schema = {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Core problem context"
                },
                "domain": {
                    "type": "string",
                    "description": "Problem domain"
                }
            },
            "required": ["description", "domain"]
        }

        response = await self._call_model(
            prompt=problem,
            system_prompt=self.prepare_system_prompt,
            schema=output_schema
        )

        logger.debug("SurveyAgent preparing call model output: ", response)

        description = response.get("description", "")
        domain = response.get("domain", "")

        return description, domain

    async def advanced_query_paper(self, goal_description, domain) -> Dict[str, Any]:
        """Conduct iterative paper survey: query generation, retrieval, scoring, and deep reading."""
        search_queries = []

        output_schema_paper_score={
            "type": "object",
            "Properties": {
                "^[a-zA-Z0-9_]+$": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "description": "A dictionary where each key is a paperID and each value is a score between 1 and 10."
        }
        output_schema_paper_details={
            "type": "object",
            "properties": {
                "background": {
                    "type": "string",
                    "description": "Core problem context and motivation"
                },
                "contributions": {
                    "type": "string",
                    "description": "Novel contributions to the field"
                },
                "methods": {
                    "type": "string",
                    "description": "Key technical approaches/methods used"
                },
                "challenges": {
                    "type": "string",
                    "description": "Limitations or challenges mentioned"
                }
            },
            "required": ["background", "contributions", "methods", "challenges"]
        }

        ###
        define_task_attribute_prompt = (
            f"You are a researcher doing research on the topic of {domain}. "
            f"You should define the task attribute such as the model input and output "
            f"of the topic for better searching relevant papers. "
            f"Formulate the input and output as: Attribute(\"attribute\"). "
            f"For example, Input(\"input\"), Output(\"output\"). "
            f"The attribute: (just return the task attribute itself with no additional text):"
        )
        logger.debug(
            "SurveyAgent define task attribute call model inputs: " + define_task_attribute_prompt
        )
        response = await self._call_model(
            prompt=define_task_attribute_prompt
        )
        logger.debug("SurveyAgent define task attribute call model output: " + response)
        io_description = parse_io_description(response)

        ###
        init_keyword_query_prompt = (
            f"You are a researcher doing literature review on the topic of {goal_description}.\n"
            f"You should propose some keywords for using the Semantic Scholar API to find the "
            f"most relevant papers to this topic.Formulate your query as: KeywordQuery(\"keyword\"). \n"
            f"Just give me one query, with the most important keyword, the keyword can be a "
            f"concatenation of multiple keywords (just put a space between every word) but please "
            f"be concise and try to cover all the main aspects.\n"
            f"Your query (just return the query itself with no additional text):"
        )
        logger.debug(
            "SurveyAgent init keyword query call model inputs:\n" + init_keyword_query_prompt
        )
        response = await self._call_model(
            prompt=init_keyword_query_prompt
        )
        logger.debug("SurveyAgent init keyword query call model outputs: " + response)
        init_query = response

        init_paper_lst = self.paper_survey.query_route(init_query)
        search_queries.append(init_query)

        # make paper bank
        if init_paper_lst:
            flattened_papers = []
            for _, papers in init_paper_lst.items():
                if isinstance(papers, list):
                    flattened_papers.extend(papers)
                elif isinstance(papers, dict) and "data" in papers:
                    flattened_papers.extend(papers["data"])

            paper_bank = {str(i): paper for i, paper in enumerate(flattened_papers)}
            logger.debug(f"init paper bank size: {len(paper_bank)}")
        else:
            logger.debug("No papers found for the initial query")
            paper_bank = {}

        # make advanced query
        grounding_k = 10
        iteration = 0
        while len(paper_bank) < self.max_papers and iteration < 10:
            ## select the top k papers with highest scores for grounding
            data_list = [{'id': id, **info} for id, info in paper_bank.items()]
            grounding_papers = data_list[: grounding_k]
            grounding_papers_str = format_papers_for_printing_next_query(grounding_papers)
            if io_description is not None:
                new_query_prompt = (
                    f"You are a researcher doing literature review on the topic of {domain}.\n"
                    f"You should propose some queries for using the Semantic Scholar API to find the "
                    f"most relevant papers to this topic.\n"
                    f"The input and output of the queries should be same with: "
                    f"input: {io_description[0]}, output: {io_description[1]}\n"
                    f"(1) KeywordQuery(\"keyword\"): find most relevant papers to the given keyword "
                    f"(the keyword shouldn't be too long and specific, otherwise the search engine "
                    f"will fail; it is ok to combine a few short keywords with spaces, such as "
                    f"\"lanaguage model reasoning\").\n"
                    f"(2) PaperQuery(\"paperId\"): find the most similar papers to the given paper "
                    f"(as specified by the paperId).\n"
                    f"(3) GetReferences(\"paperId\"): get the list of papers referenced in the given "
                    f"paper (as specified by the paperId).\n"
                    f"Right now you have already collected the following relevant papers: \n"
                    f"{grounding_papers_str}\n"
                    f"You can formulate new search queries based on these papers. And you have "
                    f"already asked the following queries:\n"
                    f"{search_queries}\n"
                    f"Please formulate a new query to expand our paper collection with more diverse "
                    f"and relevant papers (you can do so by diversifying the types of queries to "
                    f"generate and minimize the overlap with previous queries). Directly give me "
                    f"your new query without any explanation or additional text, just the query itself:"
                )
            else:
                new_query_prompt = (
                    f"You are a researcher doing literature review on the topic of {domain}.\n"
                    f"You should propose some queries for using the Semantic Scholar API to find the "
                    f"most relevant papers to this topic.\n"
                    f"(1) KeywordQuery(\"keyword\"): find most relevant papers to the given keyword "
                    f"(the keyword shouldn't be too long and specific, otherwise the search engine "
                    f"will fail; it is ok to combine a few short keywords with spaces, such as "
                    f"\"lanaguage model reasoning\").\n"
                    f"(2) PaperQuery(\"paperId\"): find the most similar papers to the given paper "
                    f"(as specified by the paperId).\n"
                    f"(3) GetReferences(\"paperId\"): get the list of papers referenced in the given "
                    f"paper (as specified by the paperId).\n"
                    f"Right now you have already collected the following relevant papers: \n"
                    f"{grounding_papers_str}\n"
                    f"You can formulate new search queries based on these papers. And you have "
                    f"already asked the following queries:\n"
                    f"{search_queries}\n"
                    f"Please formulate a new query to expand our paper collection with more diverse "
                    f"and relevant papers (you can do so by diversifying the types of queries to "
                    f"generate and minimize the overlap with previous queries). Directly give me "
                    f"your new query without any explanation or additional text, just the query itself:"
                )
            logger.debug(f"SurveyAgent generate query round {iteration} call model inputs:\n" + new_query_prompt)
            response = await self._call_model(
                prompt=new_query_prompt
            )
            logger.debug(f"SurveyAgent generate query round {iteration} call model output:\n" + response)
            new_query = response

            search_queries.append(new_query)

            new_paper_lst = None
            try:
                logger.debug(f"Searching new query {new_query}")
                new_paper_lst = self.paper_survey.query_route(new_query)
            except Exception as e:
                # Network/API failures should not crash the whole survey loop.
                logger.warning(f"survey error: {e}")

            if new_paper_lst:
                flattened_papers = []
                for _, papers in new_paper_lst.items():
                    if isinstance(papers, list):
                        flattened_papers.extend(papers)
                    elif isinstance(papers, dict) and "data" in papers:
                        flattened_papers.extend(papers["data"])
                existing_titles = {paper['title'] for paper in paper_bank.values()}
                new_papers = [paper for paper in flattened_papers if paper['title'] not in existing_titles]
                logger.debug(f"Size of new_papers after filtering: {len(new_papers)}")
                if new_papers:
                    # Assign new unique indices to new papers
                    start_index = len(paper_bank)
                    new_paper_bank = {str(start_index + i): paper for i, paper in enumerate(new_papers)}

                    # Update paper_bank with new papers
                    paper_bank.update(new_paper_bank)
                else:
                    logger.debug("No NEW papers found for the query")
            else:
                logger.debug("No papers found for the query")

            iteration += 1

        data_list = [{'id': paper_id, **info} for paper_id, info in paper_bank.items()]
        paper_bank = data_list[:]
        batch_size = 10

        for batch_index in range(0, len(paper_bank), batch_size):
            batch = paper_bank[batch_index:batch_index + batch_size]
            abs_batch = [
                {'id': paper['id'], 'title': paper['title'], 'abstract': paper['abstract']}
                for paper in batch
            ]
            if io_description is not None:
                paper_score_prompt = (
                    f"You are a helpful literature review assistant whose job is to read the below "
                    f"set of papers and score each paper.The criteria for scoring is: \n"
                    f"The paper is directly relevant to the topic of: {domain}. Note that it should "
                    f"be specific to solve the problem of focus, rather than just generic methods. \n"
                    f"The papers are: \n {abs_batch} \n Please score each paper from 1 to 10. \n"
                    f"Write the response in JSON format with \"paperID: score\" as the key and value "
                    f"for each paper. \n\n ONLY output the JSON dict with NO additional text. "
                    f"DO NOT output newline characters. DO NOT output any markdown modifier so that "
                    f"we can call json.loads() on the output later."
                )
            else:
                paper_score_prompt = (
                    f"You are a helpful literature review assistant whose job is to read the below "
                    f"set of papers and score each paper.The criteria for scoring is: \n"
                    f"The paper is directly relevant to the topic of: {domain}. Note that it should "
                    f"be specific to solve the problem of focus, rather than just generic methods. \n"
                    f"The papers are: \n {abs_batch} \n Please score each paper from 1 to 10. \n"
                    f"MUST Write the response in JSON format with \"paperID: score\" as the key and "
                    f"value for each paper. \n\n ONLY output the JSON dict with NO additional text. "
                    f"DO NOT output newline characters. DO NOT output any markdown modifier so that "
                    f"we can call json.loads() on the output later."
                )
            logger.debug("SurveyAgent scoring call model inputs:\n" + paper_score_prompt)
            response = await self._call_model(
                prompt=paper_score_prompt,
                schema=output_schema_paper_score
            )
            logger.debug(f"SurveyAgent scoring call model output:\n{response}")

            for key, score in response.items():
                # actual_paper_id = batch_index + int(key)
                actual_paper_id = int(key)
                if 0 <= actual_paper_id < len(paper_bank):
                    paper_bank[actual_paper_id]['score'] = score
                else:
                    logger.warning(f"Index '{actual_paper_id}' out of range in paper_bank.")

        logger.debug(f"Final paper_bank: {paper_bank}")

        rag_read_depth = 3
        selected_for_deep_read = select_papers(paper_bank, self.max_papers, rag_read_depth)

        for paper in selected_for_deep_read:
            paper_id = paper["id"]
            url = None
            if paper['source'] in ['arXiv', 'pubmed']:
                url = paper.get('url') or paper.get('doi')
            elif paper['source'] == 'semantic_scholar':
                if paper.get('isOpenAccess', False):
                    url = paper['openAccessPdf']['url']

            logger.debug(f"deep read paper_id: {paper_id}, url: {url}")
            base_dir = 'tmp'
            if url:
                pdf_dir = os.path.join(base_dir, "pdf")
                if not os.path.exists(pdf_dir):
                    os.makedirs(pdf_dir)

                pdf_path = None
                if paper['source'] in ["semantic_scholar", "arXiv"]:
                    pdf_path = download_pdf(url, save_folder=pdf_dir)
                elif paper['source'] == "pubmed":
                    pdf_path = download_pdf_by_doi(doi=url, download_dir=pdf_dir)

                if pdf_path:
                    text = extract_text_from_pdf(pdf_path)
                    if text:
                        get_detail_prompt = (
                            f"Analyze the following paper text and extract structured information:"
                            f"{text}\nExtract:\n"
                            f"- Background: Core problem context and motivation\n"
                            f"- Contributions: Novel contributions to the field\n"
                            f"- Methods: Key technical approaches/methods used\n"
                            f"- Challenges: Limitations or challenges mentioned\n\n"
                            f"Return JSON format with keys: methods, contributions, background, challenges. "
                            f"Use concise technical language.\n\n"
                            f"Using JSON for response format: \"background: ...\", \"contributions: ...\","
                            f"\"methods: ...\", \"challenges: ...\" ONLY output the JSON dict with NO "
                            f"additional text. DO NOT output newline characters. DO NOT output any "
                            f"markdown modifier so that we can call json.loads() on the output later."
                        )
                        response = await self._call_model(
                            prompt=get_detail_prompt,
                            schema=output_schema_paper_details
                        )
                        details = response

                        if details:
                            paper["background"] = details.get("background", "")
                            paper["contributions"] = details.get("contributions", "")
                            paper["methods"] = details.get("methods", "")
                            paper["challenges"] = details.get("challenges", "")
                        else:
                            paper["background"] = None
                            paper["contributions"] = None
                            paper["methods"] = None
                            paper["challenges"] = None

        for paper in paper_bank:
            paper['is_deep_read'] = paper['id'] in [p['id'] for p in selected_for_deep_read]

        if len(paper_bank) > self.max_papers:
            logger.debug(f"Number of papers before filter: {len(paper_bank)}")
            # sort papers by is_deep_read and score, prioritize deep read papers and higher scored papers
            paper_bank.sort(key=lambda x: (x.get('is_deep_read', False), x.get('score', 0)), reverse=True)
            # select top papers based on max_papers limit
            paper_bank = paper_bank[:self.max_papers]
            logger.debug(f"Number of papers after filter: {len(paper_bank)}")

        return paper_bank
