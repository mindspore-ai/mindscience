/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_FLASH_ATTENTION_SCORE_H_
#define OP_API_INC_LEVEL2_ACLNN_FLASH_ATTENTION_SCORE_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnEvoformerAttention,
 * calculates the size of the workspace based on the specific calculation process.
 * @domain aclnn_ops_infer
 */
aclnnStatus aclnnEvoformerAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, double scaleValueOptional, double keepProbOptional, int64_t preTokensOptional,
    int64_t nextTokensOptional, int64_t headNum, char *inputLayout, int64_t innerPreciseOptional,
    int64_t sparseModeOptional, const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut, const aclTensor *attentionOutOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnEvoformerAttention, used to perform calculation.
 */
aclnnStatus aclnnEvoformerAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream);

/**
 * @brief The first interface of aclnnFlashAttentionVarLenScore,
 * calculates the size of the workspace based on the specific calculation process.
 * @domain aclnn_ops_infer
 */
aclnnStatus aclnnFlashAttentionVarLenScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, double scaleValueOptional, double keepProbOptional,
    int64_t preTokensOptional, int64_t nextTokensOptional, int64_t headNum, char *inputLayout,
    int64_t innerPreciseOptional, int64_t sparseModeOptional, const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut, const aclTensor *attentionOutOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnFlashAttentionVarLenScore, used to perform calculation.
 */
aclnnStatus aclnnFlashAttentionVarLenScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           const aclrtStream stream);


/**
 * @brief The first interface of aclnnEvoformerAttentionV2,
 * calculates the size of the workspace based on the specific calculation process.
 * @domain aclnn_ops_infer
*/
aclnnStatus aclnnEvoformerAttentionV2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional,
    const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional,
    double scaleValueOptional,
    double keepProbOptional,
    int64_t preTokensOptional,
    int64_t nextTokensOptional,
    int64_t headNum,
    char *inputLayout,
    int64_t innerPreciseOptional,
    int64_t sparseModeOptional,
    int64_t pseTypeOptional,
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnEvoformerAttentionV2, used to perform calculation.
*/
aclnnStatus aclnnEvoformerAttentionV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

/**
 * @brief The first interface of aclnnFlashAttentionVarLenScoreV2,
 * calculates the size of the workspace based on the specific calculation process.
 * @domain aclnn_ops_infer
*/
aclnnStatus aclnnFlashAttentionVarLenScoreV2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional,
    double scaleValueOptional,
    double keepProbOptional,
    int64_t preTokensOptional,
    int64_t nextTokensOptional,
    int64_t headNum,
    char *inputLayout,
    int64_t innerPreciseOptional,
    int64_t sparseModeOptional,
    int64_t pseTypeOptional,
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnFlashAttentionVarLenScoreV2, used to perform calculation.
*/
aclnnStatus aclnnFlashAttentionVarLenScoreV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_FLASH_ATTENTION_SCORE_H_
