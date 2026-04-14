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
"""Run MindScienceAgent from the command line.
This module lives next to ``mindscience_agent.yaml`` at the repository root. Run from the
``MindScienceAgent`` root directory or set PYTHONPATH:
    python main.py
    python main.py --config-path mindscience_agent.yaml
    python main.py --enable-critic  # Enable critic agent for idea evaluation
    python main.py --prompt "Your custom prompt here"  # Run with custom prompt
    python main.py --enable-critic --test-time-scale-round 3 # Enable critic and set 3 test rounds
"""
from __future__ import annotations

import os
import argparse
import asyncio
from mindscience_agent.utils import load_env, set_ssl_cert_file_path
from mindscience_agent.workflow import ExperimentWorkflow
from mindscience_agent.config import MindScienceConfig


async def main() -> None:
    # read args
    parser = argparse.ArgumentParser(description="MindScienceAgent command-line entry.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="./mindscience_agent.yaml",
        help="Path to YAML config. If omitted, loads mindscience_agent.yaml in the repo root when present.",
    )
    parser.add_argument("--enable-critic", action="store_true")
    parser.add_argument(
        "--test-time-scale-round",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt to pass to the agent. If provided, overrides the default prompt.",
    )
    args = parser.parse_args()

    # create and validate config
    config = MindScienceConfig.init_config_from_yaml(args.config_path)

    # set env variables
    if os.path.exists(".env"):
        load_env(".env")
	
    set_ssl_cert_file_path()


    # init workflow
    agent = ExperimentWorkflow(
        config=config,
        enable_critic=args.enable_critic,
        test_time_scale_round=args.test_time_scale_round
    )

    # set a prompt
    default_prompt = 'Chlorine perchlorate (Cl2O4) is an interesting oxide of chlorine. The chlorine atoms have different oxidation states. What is the product of their oxidation states (e.g. If the oxidation states are +3 and +5, provide "15" as your answer)?'

    prompt = args.prompt if args.prompt else default_prompt

    # run workflow
    await agent.run(prompt)


if __name__ == "__main__":
    asyncio.run(main())
