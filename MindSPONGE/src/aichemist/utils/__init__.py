# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
utility module initialization
"""

from .scatter import scatter, scatter_, scatter_add, scatter_log_softmax, scatter_max, \
    scatter_mean, scatter_min, scatter_mul, scatter_softmax, scatter_update
from .operator import flatten, to_tensor, \
    to_array, is_numeric, to_device
from .file import download, compute_md5, extract, get_line_count, fdopen, read_yaml, write_yaml
from .pretty import SEP, LINE, time, long_array
from .functional import masked_mean, mean_with_nan, shifted_softplus, multi_slice, \
    multi_slice_mask, as_mask, size_to_index, extend, variadic_sum, variadic_arange, variadic_cross_entropy, \
    variadic_log_softmax, variadic_max, variadic_mean, variadic_meshgrid, variadic_randperm, variadic_sample, \
    variadic_softmax, variadic_sort, variadic_to_padded, variadic_topk, padded_to_variadic, \
    clipped_policy_gradient_objective, policy_gradient_objective, margin_ranking_loss
