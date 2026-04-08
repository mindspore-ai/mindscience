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
from typing import List

from vibescience_agent.config.base_config import BaseConfig
from vibescience_agent.config import validator

SUPPORTED_PAPER_SOURCES = ["pubmed", "arxiv", "semantic_scholar"]


class ToolConfig(BaseConfig):
    def __init__(
        self,
        tool_name,
        **kwargs,
    ):
        super().__init__(config_name=f"{tool_name}_tool")

        self._tool_name = tool_name

        self.update_attrs(**kwargs)

    def get_tool_name(self):
        return self._tool_name


class PaperSurveyConfig(ToolConfig):
    def __init__(   # pylint: disable=W0102
        self,
        max_results: int = 10,
        sources: List[str] = SUPPORTED_PAPER_SOURCES,
        **kwargs,
    ):
        super().__init__(tool_name="paper_survey", **kwargs)

        self.max_results = max_results
        self.sources = sources


@PaperSurveyConfig.validator("max_results")
def validate_max_results(config_instance: PaperSurveyConfig, max_results):  # pylint: disable=W0613
    """Validate max_results."""
    validator.check_type("max_results", value=max_results, expected_type=int)
    validator.check_number_range("max_results", value=max_results, min_value=0, max_value=20)
    return max_results


@PaperSurveyConfig.validator("sources")
def validate_sources(config_instance: PaperSurveyConfig, sources):  # pylint: disable=W0613
    """Validate sources."""
    validator.check_type("sources", value=sources, expected_type=list)
    validator.check_list_subset("sources", sources, SUPPORTED_PAPER_SOURCES)
    return sources


# Tool config class registry for extensibility
TOOL_CONFIG_CLASSES = {
    "paper_survey": PaperSurveyConfig,
}
