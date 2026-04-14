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
"""
Eval script for OpenAI Frontierscience olympiad chemistry dataset (olympiad benchmark).
This script runs evaluation on olympiad chemistry problems with concurrent process control.

Usage:
  python eval_frontierscience.py --data_path <path_to_data> [--log_dir <log_directory>]
    [--concurrent_processes <num>] [--n <runs>]

Data path:
  Download from https://huggingface.co/datasets/openai/frontierscience/raw/main/olympiad/test.jsonl
  Or use: wget https://huggingface.co/datasets/openai/frontierscience/resolve/main/olympiad/test.jsonl

Example:
  python eval_frontierscience.py --data_path test.jsonl --log_dir results --concurrent_processes 4 --n 4
"""
import argparse
import os
import subprocess
import time
from typing import List, Tuple, Union

import pandas as pd
import psutil

from mindscience_agent.config.mindscience_config import PROJECT_ROOT


def parse_args():
    """Parse command line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Run frontierscience evaluation")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset JSON file")
    parser.add_argument("--log_dir", type=str, default="frontierscience_results", help="Directory to save logs")
    parser.add_argument("--concurrent_processes", type=int, default=8, help="Maximum number of concurrent processes")
    parser.add_argument("--n", type=int, default=8, help="Number of runs per question")
    return parser.parse_args()


def find_processes_by_keyword(
    keywords: Union[str, List[str]],
    match_attrs: Tuple[str, ...] = ("name", "cmdline"),
    case_sensitive: bool = False,
    exact_match: bool = False
) -> List[dict]:
    """find processes by keyword in specified attributes (name, cmdline, etc.)"""
    if isinstance(keywords, str):
        keywords = [keywords]

    if not case_sensitive:
        keywords = [kw.lower() for kw in keywords]

    matched_processes = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "username", "exe"]):
        try:
            proc_info = proc.as_dict(attrs=match_attrs + ("pid", "username", "exe"))
            proc_info["cmdline"] = proc_info.get("cmdline") or []
            is_matched = False
            for attr in match_attrs:
                attr_value = proc_info.get(attr)

                if attr_value is None:
                    continue

                if isinstance(attr_value, list):
                    attr_str = " ".join(map(str, attr_value))
                else:
                    attr_str = str(attr_value)

                if not case_sensitive:
                    attr_str = attr_str.lower()

                for kw in keywords:
                    if exact_match:
                        if kw in attr_str.split() if exact_match else kw == attr_str:
                            is_matched = True
                            break
                    else:
                        if kw in attr_str:
                            is_matched = True
                            break

                if is_matched:
                    break

            if is_matched:
                matched_processes.append({
                    "pid": proc_info["pid"],
                    "name": proc_info.get("name", ""),
                    "cmdline": " ".join(proc_info["cmdline"]),  # convert cmdline list to string
                    "username": proc_info.get("username", ""),
                    "exe": proc_info.get("exe", "")  # executable file path
                })

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return matched_processes


def main():
    """Main function to run the frontierscience evaluation."""
    args = parse_args()
    data_path = args.data_path
    log_dir = args.log_dir
    concurrent_processes = args.concurrent_processes
    n = args.n

    dataset = pd.read_json(data_path, lines=True)
    condition = dataset['subject'] == 'chemistry'
    dataset = dataset[condition].reset_index(drop=True)

    os.makedirs(log_dir, exist_ok=True)
    process_count = 0
    pid_to_log = {}

    start_time = time.time()

    for i, row in dataset.iterrows():
        q = row["problem"].split("Think step by step and solve the problem below.")[0].strip()
        q_log_dir = f"{log_dir}/q{i}"
        os.makedirs(q_log_dir, exist_ok=True)

        for j in range(n):
            log_filename = f"{q_log_dir}/n{j}.log"
            with open(log_filename, "w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    ["python", f"{PROJECT_ROOT}/run_workflow.py", "--prompt", q, "--config-path",
                        f"{PROJECT_ROOT}/mindscience_agent.yaml"],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )

            print(f"[{i}/40] Saved to {log_filename}, PID: {process.pid}", flush=True)
            process_count += 1
            pid_to_log[process.pid] = log_filename

            while process_count >= concurrent_processes:
                time.sleep(60)
                processes = find_processes_by_keyword(
                    keywords=f"{PROJECT_ROOT}/run_workflow.py",
                    match_attrs=("cmdline",),
                    case_sensitive=False
                )
                if processes:
                    print(f"{len(processes)} processes running:", flush=True)
                    for p in processes:
                        log_file = pid_to_log.get(p['pid'], "unknown")
                        print(f"PID: {p['pid']}, Log: {log_file}", flush=True)
                    process_count = len(processes)
                else:
                    process_count = 0
                    break

    while process_count > 0:
        time.sleep(60)
        processes = find_processes_by_keyword(
            keywords=f"{PROJECT_ROOT}/run_workflow.py",
            match_attrs=("cmdline",),
            case_sensitive=False
        )
        if processes:
            print(f"{len(processes)} processes running:", flush=True)
            for p in processes:
                log_file = pid_to_log.get(p['pid'], "unknown")
                print(f"PID: {p['pid']}, Log: {log_file}", flush=True)
            process_count = len(processes)
        else:
            process_count = 0
            break

    print(f"Total evaluation time: {time.time() - start_time:.2f} seconds.", flush=True)


if __name__ == "__main__":
    main()
