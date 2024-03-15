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
# ============================================================================

# !/usr/bin/env python
# ========================= #
# Training of network model #
# ========================= #
"""
Training of network model
"""

import os
import argparse
import mindspore as ms
from deephe3.kernel import DeepHE3Kernel


parser = argparse.ArgumentParser(description='Train DeepH-E3 network')
parser.add_argument('config', type=str, metavar='CONFIG', help='Config file for training')
parser.add_argument('-n', type=int, default=None, help='Maximum number of threads')
args = parser.parse_args()

if args.n is not None:
    os.environ["OMP_NUM_THREADS"] = f"{args.n}"
    os.environ["MKL_NUM_THREADS"] = f"{args.n}"
    os.environ["NUMEXPR_NUM_THREADS"] = f"{args.n}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{args.n}"
    os.environ["VECLIB_MAXIMUM_THREADS"] = f"{args.n}"


ms.set_context(device_target="Ascend", device_id=1, mode=ms.GRAPH_MODE, max_call_depth=3000)

kernel = DeepHE3Kernel()
kernel.train(args.config)
