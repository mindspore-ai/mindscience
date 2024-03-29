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
import warnings
import argparse
import mindspore as ms
from mindchemistry.utils.load_config import load_yaml_config_from_path

from src import trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Steerable-GNN problem')
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--save_graphs", type=int, default=0, choices=[1, 0],
                        help="Whether to save intermediate compilation graphs, 0(False), 1(True)")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./qm9.yaml")
    parser.add_argument("--dtype", type=str, default='float32',
                        help="type of float to use, e.g. float16, float32 and float64")
    args = parser.parse_args()
    configs = load_yaml_config_from_path(args.config_file_path)

    ms.set_context(mode=ms.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else ms.PYNATIVE_MODE,
                   save_graphs=args.save_graphs, save_graphs_path=args.save_graphs_path,
                   device_target=args.device_target, device_id=args.device_id)

    dtype_map = {"float16": ms.float16, "float32": ms.float32, "float64": ms.float64}
    dtype = dtype_map.get(args.dtype, None)
    if dtype is None:
        print("Error: invalid value for dtype")

    ncon_dtype_map = {"float16": ms.float16, "float32": ms.float32}
    ncon_dtype = ncon_dtype_map.get(configs['model']['ncon_dtype'], None)
    if ncon_dtype is None:
        raise ValueError("Invalid value for ncon_dtype")

    warnings.filterwarnings("ignore")

    trainer.train(dtype=dtype, configs=configs, ncon_dtype=ncon_dtype)
