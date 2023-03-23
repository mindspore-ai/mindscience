# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Test API examples."""
import numpy as np
import pytest
import mindspore
import mindspore.context as context
from mindspore import Tensor
from mindspore import set_seed
from mindspore import dtype as mstype
from mindspore.ops import operations as P

from mindsponge.cell import (Attention, GlobalAttention,
                             InvariantPointAttention, MSAColumnAttention,
                             MSAColumnGlobalAttention,
                             MSARowAttentionWithPairBias, OuterProductMean,
                             Transition, TriangleAttention,
                             TriangleMultiplication)
from mindsponge.common import get_aligned_seq

np.random.seed(123)
set_seed(123)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_attention():
    """
    Feature: test attention
    Description: None
    Expectation: assert np.allclose(attn_out, ans)
    """
    model = Attention(num_head=4, hidden_size=64, gating=True, q_data_dim=64,
                      m_data_dim=64, output_dim=64)
    q_data = Tensor(np.ones((32, 128, 64)), mstype.float32)
    m_data = Tensor(np.ones((32, 256, 64)), mstype.float32)
    attention_mask = Tensor(np.ones((32, 4, 128, 256)), mstype.float32)
    attn_out = model(q_data, m_data, attention_mask).asnumpy()
    ans = np.load('/home/workspace/mindspore_dataset/mindsponge_data/examples/Attention.npy')
    assert np.allclose(attn_out, ans)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_globalattention():
    """
    Feature: test globalattention
    Description: None
    Expectation: assert np.allclose(attn_out, ans)
    """
    model = GlobalAttention(num_head=4, input_dim=64,
                            gating=True, output_dim=256)
    q_data = Tensor(np.ones((32, 128, 64)), mstype.float32)
    m_data = Tensor(np.ones((32, 128, 64)), mstype.float32)
    q_mask = Tensor(np.ones((32, 128, 1)), mstype.float32)
    attn_out = model(q_data, m_data, q_mask).asnumpy()
    ans = np.load('/home/workspace/mindspore_dataset/mindsponge_data/examples/GlobalAttention.npy')
    assert np.allclose(attn_out, ans)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_invariantpointattention():
    """
    Feature: test invariantpointattention
    Description: None
    Expectation: assert np.allclose(attn_out, ans)
    """
    context.set_context(mode=mindspore.GRAPH_MODE)
    model = InvariantPointAttention(num_head=12, num_scalar_qk=16, num_scalar_v=16,
                                    num_point_v=8, num_point_qk=4,
                                    num_channel=384, pair_dim=128)
    inputs_1d = Tensor(np.ones((256, 384)), mstype.float32)
    inputs_2d = Tensor(np.ones((256, 256, 128)), mstype.float32)
    mask = Tensor(np.ones((256, 1)), mstype.float32)
    rotation = tuple([Tensor(np.ones(256), mstype.float16) for _ in range(9)])
    translation = tuple([Tensor(np.ones(256), mstype.float16)
                         for _ in range(3)])
    attn_out = model(inputs_1d, inputs_2d, mask, rotation, translation).asnumpy()
    ans = np.load('/home/workspace/mindspore_dataset/mindsponge_data/examples/InvariantPointAttention.npy')
    assert np.allclose(attn_out, ans)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_msarowattentionwithpairbias():
    """
    Feature: test msarowattentionwithpairbias
    Description: None
    Expectation: assert np.allclose(msa_out, ans)
    """
    model = MSARowAttentionWithPairBias(num_head=4, key_dim=4, gating=True,
                                        msa_act_dim=64, pair_act_dim=128,
                                        batch_size=None)
    msa_act = Tensor(np.ones((4, 256, 64)), mstype.float32)
    msa_mask = Tensor(np.ones((4, 256)), mstype.float32)
    pair_act = Tensor(np.ones((256, 256, 128)), mstype.float32)
    index = None
    msa_out = model(msa_act, msa_mask, pair_act, index).asnumpy()
    ans = np.load('/home/workspace/mindspore_dataset/mindsponge_data/examples/MSARowAttentionWithPairBias.npy')
    assert np.allclose(msa_out, ans)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_msacolumnattention():
    """
    Feature: test msacolumnattention
    Description: None
    Expectation: assert np.allclose(attn_out, ans)
    """
    model = MSAColumnAttention(num_head=8, key_dim=256, gating=True,
                               msa_act_dim=256, batch_size=1, slice_num=0)
    msa_act = Tensor(np.ones((512, 256, 256)), mstype.float32)
    msa_mask = Tensor(np.ones((512, 256)), mstype.float32)
    index = Tensor(0, mstype.int32)
    attn_out = model(msa_act, msa_mask, index).asnumpy()
    ans = np.load('/home/workspace/mindspore_dataset/mindsponge_data/examples/MSAColumnAttention.npy')
    assert np.allclose(attn_out, ans)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_msacolumnglobalattention():
    """
    Feature: test msacolumnglobalattention
    Description: None
    Expectation: assert np.allclose(msa_out, ans)
    """
    model = MSAColumnGlobalAttention(
        num_head=4, gating=True, msa_act_dim=64, batch_size=None)
    msa_act = Tensor(np.ones((4, 256, 64)), mstype.float32)
    msa_mask = Tensor(np.ones((4, 256)), mstype.float32)
    index = None
    msa_out = model(msa_act, msa_mask, index).asnumpy()
    ans = np.load('/home/workspace/mindspore_dataset/mindsponge_data/examples/MSAColumnGlobalAttention.npy')
    assert np.allclose(msa_out, ans)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transition():
    """
    Feature: test transition
    Description: None
    Expectation: assert np.allclose(output, ans)
    """
    model = Transition(num_intermediate_factor=4, input_dim=128)
    input_0 = Tensor(np.ones((32, 128, 128)), mstype.float32)
    output = model(input_0).asnumpy()
    ans = np.load('/home/workspace/mindspore_dataset/mindsponge_data/examples/Transition.npy')
    assert np.allclose(output, ans)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_triangleattention():
    """
    Feature: test triangleattention
    Description: None
    Expectation: assert np.allclose(out, ans)
    """
    model = TriangleAttention(
        orientation="per_row", num_head=4, key_dim=64, gating=True, layer_norm_dim=64)
    input_0 = Tensor(np.ones((256, 256, 64)), mstype.float32)
    input_1 = Tensor(np.ones((256, 256)), mstype.float32)
    out = model(input_0, input_1, index=0).asnumpy()
    ans = np.load('/home/workspace/mindspore_dataset/mindsponge_data/examples/TriangleAttention.npy')
    assert np.allclose(out, ans)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_trianglemultiplication():
    """
    Feature: test trianglemultiplication
    Description: None
    Expectation: assert np.allclose(out, ans)
    """
    model = TriangleMultiplication(num_intermediate_channel=64,
                                   equation="ikc,jkc->ijc", layer_norm_dim=64, batch_size=0)
    input_0 = Tensor(np.ones((256, 256, 64)), mstype.float32)
    input_1 = Tensor(np.ones((256, 256)), mstype.float32)
    out = model(input_0, input_1, index=0).asnumpy()
    ans = np.load('/home/workspace/mindspore_dataset/mindsponge_data/examples/TriangleMultiplication.npy')
    assert np.allclose(out, ans)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_outerproductmean():
    """
    Feature: test outerproductmean
    Description: None
    Expectation: assert np.allclose(output, ans)
    """
    model = OuterProductMean(num_outer_channel=32,
                             act_dim=128, num_output_channel=256)
    act = Tensor(np.ones((32, 64, 128)), mstype.float32)
    mask = Tensor(np.ones((32, 64)), mstype.float32)
    mask_norm = P.ExpandDims()(P.MatMul(transpose_a=True)(mask, mask), -1)
    output = model(act, mask, mask_norm).asnumpy()
    ans = np.load('/home/workspace/mindspore_dataset/mindsponge_data/examples/OuterProductMean.npy')
    assert np.allclose(output, ans)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_get_aligned_seq():
    """
    Feature: test get_aligned_seq
    Description: None
    Expectation: assert aligned_gt_seq == "ABAAABAA"
                 aligned_info == "|-||.|.|"
                 aligned_pr_seq == "A-AABBBA"
    """
    gt_seq = "ABAAABAA"
    pr_seq = "AAABBBA"
    aligned_gt_seq, aligned_info, aligned_pr_seq = get_aligned_seq(
        gt_seq, pr_seq)
    assert aligned_gt_seq == "ABAAABAA"
    assert aligned_info == "|-||.|.|"
    assert aligned_pr_seq == "A-AABBBA"
