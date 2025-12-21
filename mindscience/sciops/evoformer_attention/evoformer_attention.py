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
"""evoformer_attention python api"""

import os
from mindspore.ops import CustomOpBuilder

_LOADED_OPS = None

def evo_attention(query, key, value, head_num, bias, attn_mask, scale_value, input_layout):
    """Performs evoformer attention computation using a custom NPU operator.

    This function implements the attention mechanism used in Evoformer, which is a key component
    in protein structure prediction models like AlphaFold2. It computes attention scores and
    applies them to the value tensor to produce the output.

    Args:
        query (Tensor): Query tensor for attention computation.
        key (Tensor): Key tensor for attention computation.
        value (Tensor): Value tensor for attention computation.
        head_num (int): Number of attention heads.
        bias (Tensor): Bias tensor to be added to attention scores.
        attn_mask (Tensor): Attention mask to mask out certain positions.
        scale_value (float): Scaling factor applied to attention scores.
        input_layout (str): Layout of the input tensors (e.g., 'BHMK' or 'BMKH').

    Returns:
        Tensor. Output tensor after applying attention mechanism.

    Raises:
        RuntimeError: If the custom operator fails to load or execute.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindscience.sciops import evo_attention
        >>>
        >>> # Example with BSND layout
        >>> b, n, s, d = 2048, 1, 2048, 8
        >>> query = Tensor(np.random.uniform(-0.1, 0.1, (b, s, n, d)), ms.bfloat16)
        >>> key = Tensor(np.random.uniform(-0.1, 0.1, (b, s, n, d)), ms.bfloat16)
        >>> value = Tensor(np.random.uniform(-0.1, 0.1, (b, s, n, d)), ms.bfloat16)
        >>> bias = Tensor(np.random.uniform(-0.1, 0.1, (1, n, s, s)), ms.bfloat16)
        >>> mask = np.concatenate((np.ones((b, 1, 1, s - 5)).astype(np.float32),
        ...                        np.zeros((b, 1, 1, 5)).astype(np.float32)), axis=-1)
        >>> evo_mask = Tensor(1 - mask.astype(np.uint8))
        >>> output = evo_attention(query, key, value, n, bias, evo_mask, scale_value=1.0, input_layout="BSND")
        >>> print(output.shape)
        (2048, 2048, 1, 8)
    """
    global _LOADED_OPS
    if _LOADED_OPS is None:
        ops_dir = os.path.dirname(__file__)
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{ops_dir}/binary:" + os.environ["ASCEND_CUSTOM_OPP_PATH"]
        build_dir = f"{ops_dir}/build"
        ccsrc_file = f"{ops_dir}/evoformer_attention.cpp"
        op_builder = CustomOpBuilder("evoformer_attention", ccsrc_file, "Ascend", build_dir=build_dir)
        _LOADED_OPS = op_builder.load()
    return _LOADED_OPS.npu_evoformer_attention(query, key, value, bias, None, None,
                                               attn_mask, None, scale_value, None, None,
                                               None, head_num, input_layout, None, None)
