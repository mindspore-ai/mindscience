# Copyright 2023 Huawei Technologies Co., Ltd
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
"""attention testcase"""
import os
import sys
import pytest

import numpy as np
from mindspore import Tensor, ops, load_checkpoint, load_param_into_net, jit_class
from mindspore import dtype as mstype

from mindflow.cell import Attention, MultiHeadAttention, AttentionBlock, DropPath, ViT
from mindflow.core import RelativeRMSELoss

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)

from common.cell import compare_output, validate_checkpoint, validate_model_infer, validate_output_dtype
from common.cell import FP32_RTOL, FP32_ATOL, FP16_RTOL, FP16_ATOL

BATCH_SIZE, NUM_HEADS, SEQ_LEN, IN_CHANNELS = 2, 4, 15, 64


def load_inputs():
    x = Tensor(np.load('input.npy').astype(np.float32))
    mask = Tensor(np.load('mask.npy').astype(np.int32))
    return x, mask


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_attention_softmax_dtype():
    """
    Feature: attention softmax
    Description: test forward result dtype
    Expectation: success
    """
    net = Attention(IN_CHANNELS, NUM_HEADS, compute_dtype=mstype.float32)
    x, _ = load_inputs()
    net_scores_32 = net.softmax(x, mstype.float32)
    net_scores_16 = net.softmax(x, mstype.float16)
    assert net_scores_16.dtype == mstype.float16
    compare_output(net_scores_32.numpy(),
                   net_scores_16.numpy(), FP32_RTOL, FP32_ATOL)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_attention_dtype():
    """
    Feature: attention
    Description: test forward result dtype
    Expectation: success
    """
    net = Attention(IN_CHANNELS, NUM_HEADS, compute_dtype=mstype.float16)
    x, _ = load_inputs()
    q, k, v = net.get_qkv(x)
    assert q.dtype == mstype.float16
    assert k.dtype == mstype.float16
    assert v.dtype == mstype.float16


# pylint: disable=W0212
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('compute_dtype', [mstype.float16, mstype.float32])
def test_attention_mask1(compute_dtype):
    """
    Feature: attention
    Description: test attention mask function
    Expectation: success
    """
    net = Attention(IN_CHANNELS, NUM_HEADS, compute_dtype=mstype.float16)
    scores = ops.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN], dtype=compute_dtype)
    attn_mask = ops.randint(0, 2, (BATCH_SIZE, SEQ_LEN, SEQ_LEN))
    key_padding_mask = ops.randint(0, 2, (BATCH_SIZE, SEQ_LEN))
    y = net._mask_scores(scores, attn_mask, key_padding_mask)
    assert y.shape == (BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN)
    assert y.dtype == compute_dtype


# pylint: disable=W0212
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('compute_dtype', [mstype.float16, mstype.float32])
def test_attention_mask2(compute_dtype):
    """
    Feature: attention
    Description: test attention mask function
    Expectation: success
    """
    net = Attention(IN_CHANNELS, NUM_HEADS, compute_dtype=mstype.float16)
    scores = ops.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN], dtype=compute_dtype)
    attn_mask = ops.randint(0, 2, (SEQ_LEN, SEQ_LEN))
    key_padding_mask = ops.randint(0, 2, (BATCH_SIZE, SEQ_LEN, SEQ_LEN))
    y = net._mask_scores(scores, attn_mask, key_padding_mask)
    assert y.shape == (BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN)
    assert y.dtype == compute_dtype


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_multihead_attention_multi_dtype():
    """
    Feature: MultiHeadAttention
    Description: test result dtype
    Expectation: success
    """
    net_32 = MultiHeadAttention(
        IN_CHANNELS, NUM_HEADS, compute_dtype=mstype.float32)
    net_16 = MultiHeadAttention(
        IN_CHANNELS, NUM_HEADS, compute_dtype=mstype.float16)
    x, mask = load_inputs()
    validate_checkpoint(net_32, net_16, (x, mask), FP16_RTOL, FP16_ATOL)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_multihead_attention():
    """
    Feature: MultiHeadAttention
    Description: test forward result shape
    Expectation: success
    """
    net = MultiHeadAttention(in_channels=IN_CHANNELS, num_heads=NUM_HEADS)
    x, mask = load_inputs()
    validate_model_infer(net, (x, mask), './multihead.ckpt',
                         './multihead_output.npy', FP32_RTOL, FP32_ATOL)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_multihead_attention_dtype():
    """
    Feature: MultiHeadAttention
    Description: test forward result dtype
    Expectation: success
    """
    net_16 = MultiHeadAttention(
        in_channels=IN_CHANNELS, num_heads=NUM_HEADS, compute_dtype=mstype.float16)
    x, mask = load_inputs()
    validate_output_dtype(net_16, (x, mask), mstype.float16)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_attn_block():
    """
    Feature: AttentionBlock
    Description: test forward result
    Expectation: success
    """
    net = AttentionBlock(in_channels=IN_CHANNELS, num_heads=NUM_HEADS)
    x, mask = load_inputs()
    validate_model_infer(net, (x, mask), './attention_block.ckpt',
                         './attention_block_output.npy', FP32_RTOL, FP32_ATOL)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_vit_forward():
    """
    Feature: ViT
    Description: test forward result dtype
    Expectation: success
    """
    x = ops.rand(32, 3, 192, 384)
    model = ViT(in_channels=3,
                out_channels=3,
                encoder_depths=6,
                encoder_embed_dim=768,
                encoder_num_heads=12,
                decoder_depths=6,
                decoder_embed_dim=512,
                decoder_num_heads=16,
                compute_dtype=mstype.float32
                )
    output = model(x)
    assert output.dtype == mstype.float32
    assert output.shape == (32, 288, 768)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_droppath():
    """
    Feature: DropPath train eval mode
    Description: test forward result shape
    Expectation: success
    """
    net = DropPath()
    x = np.random.rand(BATCH_SIZE, SEQ_LEN, IN_CHANNELS)
    net.set_train(True)
    out = net(Tensor(x)).numpy()
    assert out.shape == x.shape
    net.set_train(False)
    out = net(Tensor(x)).numpy()
    assert np.array_equal(out, x)


@jit_class
class Trainer:
    """Trainer"""

    def __init__(self, net, loss_fn):
        self.net = net
        self.loss_fn = loss_fn

    def get_loss(self, data, label):
        "get loss"
        pred = self.net(data)
        return self.loss_fn(label, pred)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_multihead_attention_grad():
    """
    Feature: MultiHeadAttention
    Description: test backward result
    Expectation: success
    """
    ckpt_path = './multihead.ckpt'
    model = MultiHeadAttention(
        IN_CHANNELS, NUM_HEADS, compute_dtype=mstype.float32)
    params = load_checkpoint(ckpt_path)
    load_param_into_net(model, params)

    input_data = Tensor(np.load('./input.npy'))
    input_label = Tensor(np.load('./label.npy'))

    trainer = Trainer(model, RelativeRMSELoss())

    def forward_fn(data, label):
        loss = trainer.get_loss(data, label)
        return loss

    grad_fn = ops.value_and_grad(
        forward_fn, None, model.trainable_params(), has_aux=False)

    _, grads = grad_fn(input_data, input_label)

    convert_grads = tuple(grad.asnumpy() for grad in grads)
    with np.load('./grads.npz') as data:
        output_target = tuple(data[key] for key in data.files)

    validate_ans = compare_output(
        convert_grads, output_target, rtol=1e-6, atol=1e-6)
    assert validate_ans, "The verification of scaleddot grad case failed."
