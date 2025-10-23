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
