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
    python run_workflow.py --prompt "Your custom prompt here"
    python run_workflow.py --prompt "Your custom prompt here" --config-path mindscience_agent.yaml
    python run_workflow.py --prompt "Your custom prompt here" --enable-critic --test-time-scale-round 3
"""
from __future__ import annotations

import sys

import asyncio
import argparse
import os
from mindscience_agent.utils import load_env, set_ssl_cert_file_path
from mindscience_agent.workflow import ExperimentWorkflow
from mindscience_agent.config import MindScienceConfig


async def main() -> None:
    """Main function to run MindScienceAgent from command line."""
    # read args
    parser = argparse.ArgumentParser(description="MindScienceAgent command-line entry.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt to pass to the agent.",
    )
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
    args = parser.parse_args()

    if not args.prompt.strip():
        print("="*55, flush=True)
        print("Please provide a science-related question as --prompt", flush=True)
        print("="*55, flush=True)
        print("Example:", flush=True)
        print(
            'python run_workflow.py --prompt "Please help me analyse the molecular weight of '
            'the following drug molecule: Aspirin (acetylsalicylic acid) '
            'SMILES: CC(=O)OC1=CC=CC=C1C(=O)O"',
            flush=True
        )
        sys.exit(0)

    # set env variables
    if os.path.exists(".env"):
        load_env(".env")

    # create and validate config
    config = MindScienceConfig.init_config_from_yaml(args.config_path)

    set_ssl_cert_file_path()

    # init workflow
    agent = ExperimentWorkflow(
        config=config,
        enable_critic=args.enable_critic,
        test_time_scale_round=args.test_time_scale_round
    )

    # run workflow
    await agent.run(args.prompt)


if __name__ == "__main__":
    asyncio.run(main())
