# ============================================================================
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
"""Test Validate Checkpoint"""
from pathlib import Path

import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from .utils import compare_output

def validate_checkpoint(model_1, model_2, in_args, rtol: float = 1e-5,
                        atol: float = 1e-5,):
    r"""
    Check network's checkpoint safely saves and loads the state of the model
    Args:
        model_1 (mindspore.nn.cell): mindspore model to save checkpoint from
        model_2 (mindspore.nn.cell): mindspore model to load checkpoint to
        in_args (Tuple[Tensor]): input arguments
        rtol (float): relative tolerance of error allowed, by default 1e-5
        atol (float): absolute tolerance of error allowed, by default 1e-5

    Returns:
        bool: Whether the test passes (i.e., output matches the target data within the tolerance)
    """
    try:
        ms.save_checkpoint(model_1, "checkpoint.ckpt")
    except IOError:
        pass

    try:
        params = load_checkpoint("checkpoint.ckpt")
        load_param_into_net(model_1, params)
    except IOError:
        pass
    # Now test forward passes
    output_1 = model_1(*in_args).asnumpy()
    output_2 = model_2(*in_args).asnumpy()

    # Model outputs should initially be different
    assert not compare_output(
        output_1, output_2, rtol, atol
    ), "Model outputs should initially be different"

    # Save checkpoint from model 1 and load it into model 2
    ms.save_checkpoint(model_1, "checkpoint.ckpt")
    params = load_checkpoint("checkpoint.ckpt")
    ms.load_param_into_net(model_2, params)

    # Forward with loaded checkpoint
    output_1 = model_1(*in_args).asnumpy()
    output_2 = model_2(*in_args).asnumpy()
    loaded_checkpoint = compare_output(output_1, output_2, rtol, atol)

    # Delete checkpoint file (it should exist!)
    Path("checkpoint.ckpt").unlink() # missing_ok=False
    return loaded_checkpoint
