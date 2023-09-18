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
"""validation"""

import mindspore
from sciai.context import init_project
from sciai.utils import print_time, print_log
from src.network import prepare_network
from src.process import prepare_dataset, post_process, prepare


@print_time("eval")
def main(args, data_config):
    # prepare dataset
    dataset_val = prepare_dataset(args, data_config, mode='val')

    # prepare neuaral networks:
    network, _ = prepare_network(args)

    mindspore.load_checkpoint(args.load_ckpt_path, network)
    print_log('Loaded model checkpoint at {}'.format(args.load_ckpt_path))

    post_process(args, data_config, network, dataset_val)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
