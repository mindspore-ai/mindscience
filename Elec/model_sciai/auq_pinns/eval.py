# Copyright 2023 Huawei Technologies Co., Ltd
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

"""auq pinns eval"""
import argparse
import mindspore as ms
from mindspore import context, amp, set_context
from sciai.context import init_project
from sciai.utils.python_utils import print_time

from src.network import get_all_networks
from src.process import ValDataset, TrainDataset, post_process, prepare

def parse_args():
    """Parse input args"""
    parser = argparse.ArgumentParser(description="auq_pinns")
    parser.add_argument(
        "--mode",
        type=str,
        default="GRAPH",
        choices=["GRAPH", "PYNATIVE"],
        help="Running in GRAPH_MODE OR PYNATIVE_MODE",
    )
    parser.add_argument(
        "--device_target",
        type=str,
        default="Ascend",
        choices=["GPU", "Ascend", "CPU"],
        help="The target device to run, support 'Ascend', 'GPU'",
    )
    parser.add_argument(
        "--device_id", type=int, default=7, help="ID of the target device"
    )
    input_args = parser.parse_args()
    return input_args

@print_time("eval")
def main(args):
    print("args= ",args)
    print("type(args)= ",type(args))
    args.load_ckpt = True
    _, decoder, _, _, _ = get_all_networks(args)
    decoder = amp.auto_mixed_precision(decoder, args.amp_level)
    train_dataset = TrainDataset(args)
    val_dataset = ValDataset(args)
    post_process(args, decoder, train_dataset, val_dataset)


if __name__ == "__main__":
    args_ = prepare()
    set_context(
        mode=context.GRAPH_MODE
        if args_[0].mode.upper().startswith("GRAPH")
        else context.PYNATIVE_MODE,
        device_target=args_[0].device_target,
    )
    main(*args_)
    
