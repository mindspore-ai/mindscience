"""Run VibeScienceAgent from the command line.

This module lives next to ``vibescience_agent.yaml`` at the repository root. Run from the
``MindScienceAgent`` root directory or set PYTHONPATH:
    python main.py
    python main.py --config-path vibescience_agent.yaml
"""

from __future__ import annotations

import os
import argparse
import asyncio
from vibescience_agent.utils import load_env
from vibescience_agent.workflow import ExperimentWorkflow
from vibescience_agent.config import VibeScienceConfig


async def main() -> None:
    # read args
    parser = argparse.ArgumentParser(description="VibeScienceAgent command-line entry.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="./vibescience_agent.yaml",
        help="Path to YAML config. If omitted, loads vibescience_agent.yaml in the repo root when present.",
    )
    parser.add_argument(
        "--sciencedata-path",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument("--enable-critic", action="store_true")
    parser.add_argument(
        "--test-time-scale-round",
        type=int,
        default=1,
        help="",
    )
    args = parser.parse_args()

    # create and validate config
    config = VibeScienceConfig.init_config_from_yaml(args.config_path)

    # set env variables
    if os.path.exists(".env"):
        load_env(".env")

    # init workflow
    agent = ExperimentWorkflow(
        config=config,
        sciencedata_path=args.sciencedata_path,
        enable_critic=args.enable_critic,
        test_time_scale_round=args.test_time_scale_round
    )

    # set a prompt
    prompt = (
        "Question: A researcher is interested in studying insects in the family Micromalthidae. "
        "They develop a lab colony where larvae are supplied with decaying wood to feed on.\n"
        "\n"
        "Suppose an adult male is found in the colony. Upon its death, what will be the only "
        "thing that this individual will have fed on?\n"
        "\n"
        "Answer Choices:\n"
        "A. Its mother\n"
        "B. Fungus\n"
        "C. Decaying wood\n"
        "D. Cellulose\n"
        "E. Nothing"
    )
    # the right answer is A. Its mother

    # run workflow
    await agent.run(prompt)


if __name__ == "__main__":
    asyncio.run(main())
