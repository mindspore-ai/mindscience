# Copyright 2025 Huawei Technologies Co., Ltd
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
"""run ColumnParallelLinear testing"""

import argparse
import numpy as np

import mindspore as ms
from mindspore import mint, Tensor
from mindspore.communication import init

from mindscience.distributed.manager import initialize_parallel, get_tensor_parallel_rank
from mindscience.distributed.modules import ColumnParallelLinear

def test_column_parallel_linear(args):
    """
    Feature: Column-parallel linear layer correctness.

    Description: Initializes tensor parallelism and compares a standard Linear layer
    with the ColumnParallelLinear implementation for equivalent outputs.

    Expectation: When gather_output is True, outputs match the non-parallel Linear; when
    gather_output is False, each parallel shard matches the corresponding slice of the
    full Linear output within tolerance.
    """
    initialize_parallel(tensor_parallel_size=args.num_workers)
    ms.set_seed(2048)
    linear = mint.nn.Linear(args.in_features, args.out_features, bias=args.bias)

    ms.set_seed(2048)
    column_parallel_linear = ColumnParallelLinear(args.in_features, args.out_features, bias=args.bias,
                                                  gather_output=args.gather_output,
                                                  use_sequence_parallel=args.use_sequence_parallel,
                                                  compute_dtype=ms.float32)

    bs = 8
    x = Tensor(np.random.randn(bs, 4, args.in_features), ms.float32)

    linear_out = linear(x)

    if args.use_sequence_parallel:
        tp_rank = get_tensor_parallel_rank()
        stride = bs // args.num_workers
        parallel_linear_out = column_parallel_linear(x[tp_rank*stride:(tp_rank+1)*stride])
    else:
        parallel_linear_out = column_parallel_linear(x)

    if args.gather_output:
        assert mint.allclose(linear_out, parallel_linear_out, rtol=1e-4, atol=1e-7)
    else:
        tp_rank = get_tensor_parallel_rank()
        stride = args.out_features // args.num_workers
        assert mint.allclose(linear_out[:, :, tp_rank*stride:(tp_rank+1)*stride], parallel_linear_out,
                             rtol=1e-4, atol=1e-7)


if __name__ == "__main__":
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device("Ascend")
    init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker")
    parser.add_argument("--in_features", type=int, default=32, help="Number of input features dim")
    parser.add_argument("--out_features", type=int, default=32, help="Number of output features dim")
    parser.add_argument("--bias", action="store_true", help="Whether add bias in linear")
    parser.add_argument("--gather_output", action="store_true", help="Whether gather output")
    parser.add_argument("--use_sequence_parallel", action="store_true", help="Whether use sequence parallel")
    args_ = parser.parse_args()

    test_column_parallel_linear(args_)
