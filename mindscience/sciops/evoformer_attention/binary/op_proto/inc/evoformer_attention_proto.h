/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EVOFORMER_ATTENTION_PROTO_H_
#define EVOFORMER_ATTENTION_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(EvoformerAttention)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, ge::TensorType::ALL())
    .INPUT(value, ge::TensorType::ALL())
    .OPTIONAL_INPUT(real_shift, ge::TensorType::ALL())
    .OPTIONAL_INPUT(drop_mask, ge::TensorType::ALL())
    .OPTIONAL_INPUT(padding_mask, ge::TensorType::ALL())
    .OPTIONAL_INPUT(atten_mask, ge::TensorType::ALL())
    .OPTIONAL_INPUT(prefix, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_qlen, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_kvlen, ge::TensorType::ALL())
    .OPTIONAL_INPUT(q_start_idx, ge::TensorType::ALL())
    .OPTIONAL_INPUT(kv_start_idx, ge::TensorType::ALL())
    .OUTPUT(softmax_max, ge::TensorType::ALL())
    .OUTPUT(softmax_sum, ge::TensorType::ALL())
    .OUTPUT(softmax_out, ge::TensorType::ALL())
    .OUTPUT(attention_out, ge::TensorType::ALL())
    .ATTR(scale_value, Float, 1)
    .ATTR(keep_prob, Float, 1)
    .ATTR(pre_tockens, Int, 2147483647)
    .ATTR(next_tockens, Int, 2147483647)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .ATTR(inner_precise, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(pse_type, Int, 1)
    .OP_END_FACTORY_REG(EvoformerAttention);

}  // namespace ge

#endif
