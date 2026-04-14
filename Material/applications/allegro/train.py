# Copyright 2024 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
train
"""
import argparse
import random
import warnings

import mindspore as ms
import numpy as np

from mindchemistry.utils.load_config import load_yaml_config_from_path
from src import trainer


def main():
    code_graph = "GRAPH"
    parser = argparse.ArgumentParser(description='Nequip problem')
    parser.add_argument(
        "--mode",
        type=str,
        default=code_graph,
        choices=[code_graph, "PYNATIVE"],
        help="Context mode, support 'GRAPH', 'PYNATIVE'"
    )
    parser.add_argument(
        "--save_graphs",
        type=bool,
        default=False,
        choices=[True, False],
        help="Whether to save intermediate compilation graphs"
    )
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument(
        "--device_target",
        type=str,
        default="Ascend",
        help="The target device to run, support 'Ascend'"
    )
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./rmd.yaml")
    parser.add_argument(
        "--dtype", type=str, default='float32', help="type of float to use, e.g. float16, float32 and float64"
    )
    parser.add_argument(
        "--parallel_mode",
        type=str,
        default="NONE",
        choices=["DATA_PARALLEL", "NONE"],
        help="Parallel mode, support 'DATA_PARALLEL', 'NONE'"
    )

    args = parser.parse_args()

    configs = load_yaml_config_from_path(args.config_file_path)

    parallel_mode = args.parallel_mode
    if parallel_mode == "DATA_PARALLEL":
        ms.set_context(
            mode=ms.GRAPH_MODE if args.mode.upper().startswith(code_graph) else ms.PYNATIVE_MODE,
            save_graphs=args.save_graphs,
            save_graphs_path=args.save_graphs_path,
            device_target=args.device_target
        )
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        ms.communication.init()
        ms.set_seed(123)
    else:
        ms.set_context(
            mode=ms.GRAPH_MODE if args.mode.upper().startswith(code_graph) else ms.PYNATIVE_MODE,
            save_graphs=args.save_graphs,
            save_graphs_path=args.save_graphs_path,
            device_target=args.device_target,
            device_id=args.device_id
        )

    dtype_map = {"float16": ms.float16, "float32": ms.float32, "float64": ms.float64}
    dtype = dtype_map.get(args.dtype, None)

    ms.set_seed(123)
    ms.dataset.config.set_seed(1)
    np.random.seed(1)
    random.seed(1)

    warnings.filterwarnings("ignore")

    trainer.train(dtype=dtype, configs=configs, parallel_mode=parallel_mode)


if __name__ == '__main__':
    main()
