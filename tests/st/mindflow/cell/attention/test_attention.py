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
from mindspore import Tensor, ops, load_checkpoint, load_param_into_net, jit_class, context
from mindspore import dtype as mstype

from mindflow.cell import Attention, MultiHeadAttention, TransformerBlock, DropPath, ViT
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
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('compute_dtype', [mstype.float16, mstype.float32])
def test_attention_qkv(mode, compute_dtype):
    """
    Feature: attention
    Description: test qkv dtype and shape
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Attention(IN_CHANNELS, NUM_HEADS, compute_dtype=compute_dtype)
    x = ops.randn((BATCH_SIZE, SEQ_LEN, IN_CHANNELS))
    qkv = net.get_qkv(x)
    for tensor in qkv:
        assert tensor.dtype == compute_dtype
        assert tensor.shape == (BATCH_SIZE, NUM_HEADS, SEQ_LEN, IN_CHANNELS//NUM_HEADS)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('fa_dtype', [mstype.float16, mstype.bfloat16])
def test_flash_attn(mode, fa_dtype):
    """
    Feature: FlashAttn
    Description: test forward result
    Expectation: success
    """
    context.set_context(mode=mode)
    in_shape = (BATCH_SIZE, NUM_HEADS, SEQ_LEN, IN_CHANNELS//NUM_HEADS)
    query, key, value = ops.randn(in_shape), ops.randn(in_shape), ops.randn(in_shape)
    mask = ops.randint(0, 2, (SEQ_LEN, SEQ_LEN))
    net = MultiHeadAttention(IN_CHANNELS, NUM_HEADS, enable_flash_attn=True, fa_dtype=fa_dtype)
    output = net.attn(query, key, value, mask)
    assert output.dtype == fa_dtype
    assert output.shape == in_shape


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('fa_dtype', [mstype.float16, mstype.bfloat16])
def test_multihead_fa(mode, fa_dtype):
    """
    Feature: FlashAttention
    Description: test forward result
    Expectation: success
    """
    context.set_context(mode=mode)
    net = MultiHeadAttention(IN_CHANNELS, NUM_HEADS, enable_flash_attn=True, fa_dtype=fa_dtype)
    in_shape = (BATCH_SIZE, SEQ_LEN, IN_CHANNELS)
    x = ops.randn(in_shape)
    mask = ops.randint(0, 2, (BATCH_SIZE, 1, SEQ_LEN, SEQ_LEN))
    output = net(x, mask)
    assert output.dtype == mstype.float32
    assert output.shape == in_shape


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('fa_dtype', [mstype.float16, mstype.bfloat16])
def test_fa_forward(mode, fa_dtype):
    """
    Feature: FlashAttention
    Description: test FlashAttention forward result
    Expectation: success
    """
    context.set_context(mode=mode)
    net = MultiHeadAttention(IN_CHANNELS, NUM_HEADS, enable_flash_attn=False)
    fa_net = MultiHeadAttention(IN_CHANNELS, NUM_HEADS, enable_flash_attn=True, fa_dtype=fa_dtype)
    batch_size, seq_len = 256, 512
    in_shape = (batch_size, seq_len, IN_CHANNELS)
    x = ops.randn(in_shape)
    mask = ops.randint(0, 2, (batch_size, 1, seq_len, seq_len))
    validate_checkpoint(net, fa_net, (x, mask), FP32_RTOL, FP32_ATOL)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_attention_mask1(mode):
    """
    Feature: attention
    Description: test attention mask function
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Attention(IN_CHANNELS, NUM_HEADS, compute_dtype=mstype.float16)
    attn_mask = ops.randint(0, 2, (SEQ_LEN, SEQ_LEN))
    key_padding_mask = ops.randint(0, 2, (BATCH_SIZE, SEQ_LEN))
    mask = net.merge_mask(attn_mask, key_padding_mask)
    assert mask.shape == (BATCH_SIZE, 1, SEQ_LEN, SEQ_LEN)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_attention_mask2(mode):
    """
    Feature: attention
    Description: test attention mask function
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Attention(IN_CHANNELS, NUM_HEADS)
    attn_mask = ops.randint(0, 2, (BATCH_SIZE, 1, SEQ_LEN, SEQ_LEN))
    key_padding_mask = ops.randint(0, 2, (BATCH_SIZE, SEQ_LEN))
    mask = net.merge_mask(attn_mask, key_padding_mask)
    assert mask.shape == (BATCH_SIZE, 1, SEQ_LEN, SEQ_LEN)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_multihead_attention_forward(mode):
    """
    Feature: MultiHeadAttention
    Description: test result dtype
    Expectation: success
    """
    context.set_context(mode=mode)
    net_32 = MultiHeadAttention(
        IN_CHANNELS, NUM_HEADS, compute_dtype=mstype.float32)
    net_16 = MultiHeadAttention(
        IN_CHANNELS, NUM_HEADS, compute_dtype=mstype.float16)
    x, mask = load_inputs()
    validate_checkpoint(net_32, net_16, (x, mask), FP16_RTOL, FP16_ATOL)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_multihead_attention(mode):
    """
    Feature: MultiHeadAttention
    Description: test forward result shape
    Expectation: success
    """
    context.set_context(mode=mode)
    net = MultiHeadAttention(in_channels=IN_CHANNELS, num_heads=NUM_HEADS)
    x, mask = load_inputs()
    validate_model_infer(net, (x, mask), './multihead.ckpt',
                         './multihead_output.npy', FP32_RTOL, FP32_ATOL)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('compute_dtype', [mstype.float16, mstype.bfloat16])
def test_multihead_attention_dtype(mode, compute_dtype):
    """
    Feature: MultiHeadAttention
    Description: test forward result dtype
    Expectation: success
    """
    context.set_context(mode=mode)
    net = MultiHeadAttention(
        in_channels=IN_CHANNELS, num_heads=NUM_HEADS, compute_dtype=compute_dtype)
    x, mask = load_inputs()
    validate_output_dtype(net, (x, mask), compute_dtype)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_attn_block(mode):
    """
    Feature: TransformerBlock
    Description: test forward result
    Expectation: success
    """
    context.set_context(mode=mode)
    net = TransformerBlock(in_channels=IN_CHANNELS, num_heads=NUM_HEADS)
    x, mask = load_inputs()
    validate_model_infer(net, (x, mask), './attention_block.ckpt',
                         './attention_block_output.npy', FP32_RTOL, FP32_ATOL)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_vit_forward(mode):
    """
    Feature: ViT
    Description: test forward result dtype
    Expectation: success
    """
    context.set_context(mode=mode)
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
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_droppath(mode):
    """
    Feature: DropPath train eval mode
    Description: test forward result shape
    Expectation: success
    """
    context.set_context(mode=mode)
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
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_multihead_attention_grad(mode):
    """
    Feature: MultiHeadAttention
    Description: test backward result
    Expectation: success
    """
    context.set_context(mode=mode)
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
