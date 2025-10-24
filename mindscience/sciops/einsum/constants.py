# Copyright 2023-2025 Huawei Technologies Co., Ltd
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
constants.py
This module contains various constants used throughout the application.

Constants:
`T_INPUT, T_MUL, T_MATMUL, T_OUT` are the flags of tensor type.
T_MATMUL represents the matrix multiplication generated the tensor.

`INPUT, OUT, MID_MUL, MID_MATMUL, MUL, MATMUL` are the flags of
constraint types.

`MUST_MK, MUST_KM, MUST_ALL` are the flags for the order of the K-axis
in batch matrix multiplication. ALL means that both KM and MK are allowed.

`MIN_WEIGHT_PROD, SEARCH_K_THRE` are the constants used in
label order optimization.
"""

# Tensor type
T_INPUT = 0
T_MUL = 1
T_MATMUL = 2
T_OUT = 3


# Constraint Types
INPUT = "INPUT"
OUT = "OUT"
MID_MUL = "MID_MUL"
MID_MATMUL = "MID_MATMUL"
MUL = "MUL"
MATMUL = "MATMUL"


# bmm
MUST_MK = 1
MUST_KM = 2
MUST_ALL = 3


# optimization
MIN_WEIGHT_PROD = 1048576
SEARCH_K_THRE = 20
