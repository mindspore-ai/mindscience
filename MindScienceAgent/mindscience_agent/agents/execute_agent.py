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
"""Execute Agent — Execute subagent (adapted from end_to_end generate node).
Uses deepagents + Tool/PythonREPL + StateGraph like the original stack,
while inheriting BaseAgent and bridging OpenAIModel → ChatOpenAI.
"""
from __future__ import annotations

from typing import TypedDict, Dict

from langchain_core.messages import ToolMessage
from langchain_core.globals import set_debug
from langgraph.graph import START, StateGraph

from mindscience_agent.config.agent_config import AgentConfig
from mindscience_agent.config.tool_config import ToolConfig
from mindscience_agent.agents.base_agent import AgentExecutionError, BaseAgent
from mindscience_agent.utils.prompts import generate_prompt
from mindscience_agent.utils.utils import extract_between_regex
from mindscience_agent.utils.message import create_assistant_msg
from mindscience_agent.utils import logger


class _ExecuteSubgraphState(TypedDict):
    messages: list


_EXECUTE_BASE_PROMPT = """
You are a precise Code Extraction, Troubleshooting, and Execution Specialist.
Your ONLY job is to extract Python code provided by the previous agent, execute it,
and debug it if necessary.

# Workflow & Execution Guidelines
You MUST strictly follow this step-by-step process:

1. **Extract**: Locate the code wrapped between `<execute>` and `</execute>` tags in the user prompt. Extract ONLY the pure Python code inside.
   - *CRITICAL*: DO NOT include the `<execute>` and `</execute>` tags themselves in your extracted code.
   - *CRITICAL*: Strip any markdown code blocks (like ```python) if they are present inside the tags.

2. **Execute**: You MUST use the `python_executor` tool to run the extracted code.
   - *CRITICAL*: Ensure the input parameter for the tool (e.g., the code string) is strictly populated with the extracted Python script. IT MUST NOT BE EMPTY.
   - STRICTLY DO NOT use default tools like `execute`, `grep`, or `glob`.

3. **Evaluate & Debug**:
   - If the code runs successfully, return the final output.
   - If you encounter errors (e.g., SyntaxError, ModuleNotFoundError), you must debug, rewrite the code, and call the `python_executor` tool again until it runs successfully.

4. **Code Requirements**:
   - Always include `print()` statements to expose intermediate results and variable states.
   - Ensure all required modules (e.g., numpy, pandas) are imported at the very beginning of the code block.

# Output Format
When calling the `python_executor` tool, your internal thought process should look like this:
"I have located the `<execute>` tags. I will now extract the pure Python code and pass it exactly into the `python_executor` tool."
"""


class ExecuteAgent(BaseAgent):
    """
    Execute Agent runs <execute> code via python_executor
    inside a deep agent subgraph and returns the result wrapped in <observation>.

    Args:
        model (BaseModel): LLM model.
        config (Dict[str, Any]): Agent config.
        tool_config (Dict[str, ToolConfig]): Tool configuration dict.

    Inputs:
        - messages (list): Full history; last message contentmust include <execute>...</execute>.
        - params (Dict[str, Any]): Unused; reserved for extensions.

    Outputs:
        - Dict[str, Any]: Assistant message wrapped in <observation>.
    """
    def __init__(self, model, config: AgentConfig, tool_config: Dict[str, ToolConfig] = None):
        super().__init__(model, config, tool_config)
        self.debug = logger.LOG_LEVEL == logger.LOG_LEVEL_MAP["DEBUG"]

        self._compiled_subgraph = self._build_execute_subgraph()
        self.ctx = self._build_agent_tool_context()
        self.ctx["skills"] = []     # skills already passed to deep agent

    def _build_execute_subgraph(self):
        """Build a single-node LangGraph: START → deep agent with python_executor and skills."""
        deep_agent = self._build_deep_agent()
        workflow = StateGraph(_ExecuteSubgraphState)
        workflow.add_node("execute_agent", deep_agent)
        workflow.add_edge(START, "execute_agent")
        return workflow.compile()

    async def execute(self, messages, **params):
        """Run the subgraph on code extracted from the last message; return an observation assistant turn."""
        user_query = messages[0]["content"]
        self._build_system_prompt(user_query)

        processed_input = self._process_input(messages)
        logger.debug(f"ExecuteAgent call model inputs:\n{processed_input}")
        input_msg = [("system", self.system_prompt)] + processed_input
        remaining_retries = self.config.max_retries

        if self.debug:
            set_debug(self.debug)

        while True:
            try:
                final_state = await self._invoke_subgraph(input_msg)
                break
            except Exception as e:
                remaining_retries -= 1
                logger.warning(f"ExecuteAgent execution error: {e}. Retries left: {remaining_retries}")

                if remaining_retries <= 0:
                    logger.warning(f"ExecuteAgent failed after max retries. Error: {e}")

                    if self.debug:
                        set_debug(False)

                    return {"messages": []}

        if self.debug:
            set_debug(False)

        return self._process_output(final_state)

    def _collect_executor_outputs(self, final_state: dict) -> list[str]:
        """Collect string contents of python_executor tool messages from final state."""
        outputs: list[str] = []
        for msg in final_state.get("messages", []):
            if isinstance(msg, ToolMessage):
                tname = getattr(msg, "name", None)
                if tname == "python_executor":
                    outputs.append(str(msg.content).strip())
            else:
                typ = getattr(msg, "type", None)
                name = getattr(msg, "name", None)
                if typ == "tool" and name == "python_executor":
                    outputs.append(str(msg.content).strip())
        return outputs

    def _process_output(self, final_state: dict) -> dict:
        """Wrap the last python_executor output (or a fallback string) as an assistant message."""
        outputs = self._collect_executor_outputs(final_state)
        if len(outputs) > 0:
            return create_assistant_msg("\n<observation>" + outputs[-1].strip() + "</observation>")
        return create_assistant_msg("\n<observation>ExecuteAgent returned with no results</observation>")

    def _process_input(self, query: list) -> list[tuple[str, str]]:
        """Extract inner text between <execute> and </execute> from the last message."""
        input_msg: list[tuple[str, str]] = []
        msg_content = query[-1]["content"]
        if "<execute>" in msg_content and "</execute>" in msg_content:
            msg_content = "\n" + extract_between_regex(msg_content, '<execute>', '</execute>')
            input_msg.append(("user", msg_content))
            return input_msg

        raise AgentExecutionError(
            "ExecuteAgent input message must contain code wrapped in <execute> tags. "
            "Please ensure the previous agent formats the message correctly."
        )

    def _build_system_prompt(self, user_query: str):
        """Build system prompt for execute agent."""
        if self.system_prompt:
            return
        self.run_tool_retrieval_if_enabled(user_query)

        self.system_prompt = generate_prompt(
            base_prompt=_EXECUTE_BASE_PROMPT,
            tool_desc=self.ctx["tool_desc"],
            use_tool_retriever=self.use_tool_retriever,
        )

        logger.debug("ExecuteAgent system prompt:\n" + self.system_prompt)
