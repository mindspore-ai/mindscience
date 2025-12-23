# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 ByteDance and/or its affiliates.
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

"""transformer"""

from dataclasses import dataclass
from typing import Optional

import mindspore as ms
from mindspore import mint, nn

from protenix.model import base_config
from protenix.model.modules import base_modules as bm
from protenix.model.modules.module_utils.gated_linear_unit import gated_linear_unit
from protenix.model.modules.primitives import AdaptiveLayernorm, AdaptiveZeroInit
from protenix.model.modules.utils import (
    broadcast_token_to_atom,
    rearrange_to_dense_trunk,
)
from mindscience.e3nn.utils import Ncon


class TransitionBlock(nn.Cell):
    """
    A neural network layer that combines adaptive layer normalization, a gated linear unit (GLU),
    and adaptive zero initialization to process input data with optional conditional inputs.

    Args:
        in_channels (int): Number of input channels.
        num_intermediate_factor (int): Factor to determine the number of intermediate channels.
        single_channels (int, optional): Number of single conditional channels. Default: ``None``.
        ndim (int, optional): Number of dimensions for input tensor. Default: ``3``.
        with_single_cond (bool, optional): Whether to use single conditional processing. Default: ``True``.
        use_glu_kernel (bool, optional): Whether to use GLU. Default: ``True``.
        dtype (dtype, optional): Data type. Default: ``ms.float32``.
        mode (str, optional): Computation mode. Default: ``'ncon'``.

    Inputs:
        - **x** (Tensor) - Input tensor to the layer.
        - **single_cond** (Tensor, optional) - Single conditional tensor. Required if `with_single_cond` is True.

    Outputs:
        - **output** (Tensor) - Output tensor after processing through the TransitionBlock.
    """

    def __init__(
        self,
        in_channels,
        num_intermediate_factor,
        single_channels=None,
        ndim=3,
        with_single_cond=True,
        use_glu_kernel=False,
        dtype=ms.float32,
    ):
        super().__init__()
        self.num_intermediate = num_intermediate_factor * in_channels
        self.with_single_cond = with_single_cond
        self.ndim = ndim
        if single_channels is None:
            single_channels = in_channels
        self.adaptive_layernorm = AdaptiveLayernorm(
            in_channels,
            single_channels,
            with_single_cond=self.with_single_cond,
            dtype=dtype,
        )
        self.use_glu_kernel = use_glu_kernel
        if self.use_glu_kernel:
            self.weights = bm.custom_initializer(
                "relu", [in_channels, self.num_intermediate * 2], dtype=ms.float32
            )
            self.weights = ms.Parameter(
                ms.Tensor(self.weights).reshape(in_channels, 2, self.num_intermediate)
            )
        else:
            self.linear1 = bm.LinearAMP(
                in_channels,
                self.num_intermediate,
                weight_init="ones",
                has_bias=False,
                dtype=dtype,
            )
            self.linear2 = bm.LinearAMP(
                in_channels,
                self.num_intermediate,
                weight_init="ones",
                has_bias=False,
                dtype=dtype,
            )
        self.adaptive_zero_init = AdaptiveZeroInit(
            self.num_intermediate,
            in_channels,
            single_channels,
            with_single_cond=self.with_single_cond,
            dtype=dtype,
        )

    def construct(self, x, single_cond=None):
        """construct"""
        x = x.squeeze()
        if single_cond is not None:
            single_cond = single_cond.squeeze()
        x = self.adaptive_layernorm(x, single_cond)
        if self.use_glu_kernel:
            c = gated_linear_unit(
                x=x,
                weight=self.weights.astype(x.dtype),
                activation=mint.nn.functional.silu,
            ).astype(x.dtype)
        else:
            c = ms.ops.silu((self.linear1(x))) * self.linear2(x)

        output = self.adaptive_zero_init(c, single_cond)
        return output


@dataclass
class SelfAttentionConfig(base_config.BaseConfig):
    num_head: int = 16
    key_dim: Optional[int] = None
    value_dim: Optional[int] = None


class SelfAttention(nn.Cell):
    """
    A self-attention mechanism implementation with adaptive layer normalization and adaptive zero initialization.

    This class implements the self-attention mechanism commonly used in transformer models.
    It includes adaptive layer normalization for input processing and adaptive zero initialization
    for the final output. The mechanism computes attention scores using query, key, and value
    transformations, applies masking, and optionally incorporates pair-wise logits.

    Args:
        config: Configuration object containing parameters such as key dimension, value dimension,
            and number of attention heads.
        global_config: Global configuration object for additional settings.
        num_channels (int): Number of channels in the input tensor.
        pair_shape (tuple, optional): Shape of the pair logits.
        ndim (int, optional): Number of dimensions for the dense layers. Default: ``3``.
        with_single_cond (bool, optional): Whether to include single condition adaptation. Default: ``True``.
        dtype (dtype, optional): Data type. Default: ``ms.float32``.
        mode (str, optional): Computation mode. Default: ``'ncon'``.

    Inputs:
        - **x** (Tensor) - Input tensor to the self-attention layer.
        - **single_cond** (Tensor, optional) - Single condition tensor for adaptation.
        - **pair_logits** (Tensor, optional) - Additional logits to incorporate into attention scores.

    Outputs:
        - **output** (Tensor) - The output tensor after self-attention and adaptive zero initialization.
    """

    def __init__(
        self,
        config,
        global_config,
        num_channels,
        pair_shape=None,
        ndim=None,
        with_single_cond=True,
        dtype=ms.float32,
        mode="ncon",
    ):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.num_channels = num_channels
        self.with_single_cond = with_single_cond
        self.adaptive_layernorm = AdaptiveLayernorm(
            num_channels,
            int(num_channels // 2),
            with_single_cond=self.with_single_cond,
            dtype=dtype,
        )
        key_dim = (
            self.config.key_dim if self.config.key_dim is not None else num_channels
        )
        value_dim = (
            self.config.value_dim if self.config.value_dim is not None else num_channels
        )
        num_head = self.config.num_head
        if key_dim % num_head != 0:
            raise ValueError("key_dim must be divisible by num_head")
        if value_dim % num_head != 0:
            raise ValueError("value_dim must be divisible by num_head")
        key_dim = key_dim // num_head
        self.key_dim = key_dim
        value_dim = value_dim // num_head
        qk_shape = (num_head, key_dim)
        v_shape = (num_head, value_dim)
        self.q_linear = bm.CustomDense(
            num_channels, qk_shape, use_bias=True, ndim=ndim, dtype=dtype, mode=mode
        )
        self.k_linear = bm.CustomDense(
            num_channels, qk_shape, use_bias=False, ndim=ndim, dtype=dtype, mode=mode
        )
        self.v_linear = bm.CustomDense(
            num_channels, v_shape, use_bias=False, ndim=ndim, dtype=dtype, mode=mode
        )
        self.linear = bm.LinearAMP(
            num_channels,
            num_head * value_dim,
            weight_init="zeros",
            dtype=dtype,
            has_bias=False,
        )
        self.adaptive_zero_init = AdaptiveZeroInit(
            num_channels,
            num_channels,
            int(num_channels // 2),
            with_single_cond=self.with_single_cond,
            dtype=dtype,
        )
        self.ncon1 = Ncon([[-2, -1, 1], [-3, -1, 1]])
        self.ncon2 = Ncon([[-2, -1, 2], [2, -2, -3]])

        # add for protenix
        self.layernorm_pair = bm.LayerNorm(pair_shape, create_beta=False)
        self.linear_pair = bm.CustomDense(
            pair_shape[-1], self.config.num_head, ndim=3, dtype=dtype, mode=mode
        )

    def construct(self, x, single_cond, pair_logits):
        """construct"""
        x = self.adaptive_layernorm(x, single_cond)
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        if pair_logits is not None:
            pair_logits = self.linear_pair(self.layernorm_pair(pair_logits)).transpose(
                (2, 0, 1)
            )  # pay attention to the trasnpose order

        logits = mint.einsum(
            "...qhc,...khc->...hqk", q * self.key_dim ** (-0.5), k
        )  # + bias
        if pair_logits is not None:
            logits = logits + pair_logits  # (num_heads, seq_len, seq_len)
        weights = mint.softmax(logits, dim=-1)
        weights = weights.astype(q.dtype)
        weighted_avg = mint.einsum("...hqk,...khc->...qhc", weights, v)

        weighted_avg = weighted_avg.reshape(q.shape[:-2] + (-1,))
        gate_logits = self.linear(x)
        weighted_avg = weighted_avg * mint.sigmoid(
            gate_logits.astype(ms.float32)
        ).astype(gate_logits.dtype)
        output = self.adaptive_zero_init(weighted_avg, single_cond)
        return output


class Transformer(nn.Cell):
    """
    Transformer module for processing sequential data.

    Args:
        config: Configuration object.
        global_config: Global configuration object.
        in_shape (tuple): Input shape.
        pair_shape (tuple): Shape of the pair logits.
        ndim (int, optional): Number of dimensions. Default: 3.
        using_pair_act (bool, optional): Whether to use pair activation. Default: False.
        dtype (dtype, optional): Data type. Default: ms.float32.
        mode (str, optional): Computation mode. Default: 'ncon'.
    """

    @dataclass
    class Config(base_config.BaseConfig):
        attention: SelfAttentionConfig = base_config.autocreate()
        num_blocks: int = 24
        block_remat: bool = False
        super_block_size: int = 4
        num_intermediate_factor: int = 2

    def __init__(
        self,
        config,
        global_config,
        in_shape,
        pair_shape,
        ndim=3,
        using_pair_act=False,
        dtype=ms.float32,
    ):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.using_pair_act = using_pair_act
        self.act = []
        if self.config.num_blocks % self.config.super_block_size != 0:
            raise ValueError("num_blocks must be divisible by super_block_size")
        self.num_super_blocks = self.config.num_blocks // self.config.super_block_size
        self.super_blocks = ms.nn.CellList(
            [
                SuperBlock(
                    config,
                    global_config,
                    self.config.num_blocks,
                    using_pair_act,
                    in_shape,
                    pair_shape,
                    ndim=ndim,
                    dtype=dtype,
                )
                for _ in range(self.num_super_blocks)
            ]
        )

    @ms.jit
    def construct(self, act, single_cond, pair_cond=None):
        """construct"""
        for i in range(self.num_super_blocks):
            act = self.super_blocks[i](act, single_cond, pair_cond)
        return act


class Block(nn.Cell):
    """
    Basic building block of the transformer.

    Args:
        config: Configuration object.
        global_config: Global configuration object.
        in_shape (tuple): Input shape.
        pair_shape (tuple, optional): Shape of the pair logits.
        ndim (int, optional): Number of dimensions. Default: 2.
        dtype (dtype, optional): Data type. Default: ms.float32.
        mode (str, optional): Computation mode. Default: 'einsum'.
    """

    def __init__(
        self,
        config,
        global_config,
        in_shape,
        pair_shape=None,
        ndim=2,
        dtype=ms.float32,
    ):
        super().__init__()
        self.self_attention = SelfAttention(
            config.attention,
            global_config,
            in_shape[-1],
            pair_shape,
            ndim=3,
            dtype=dtype,
        )
        self.transition_block = TransitionBlock(
            in_shape[-1],
            config.num_intermediate_factor,
            int(in_shape[-1] // 2),
            ndim=ndim,
            dtype=dtype,
        )

    def construct(self, act, single_cond, pair_cond):
        """construct"""
        act = act + self.self_attention(act, single_cond, pair_cond)
        act = act + self.transition_block(act, single_cond)
        return act


class SuperBlock(nn.Cell):
    """
    A super block consisting of multiple basic blocks.

    Args:
        config: Configuration object.
        global_config: Global configuration object.
        num_blocks (int): Number of blocks.
        using_pair_act (bool): Whether to use pair activation.
        in_shape (tuple): Input shape.
        pair_shape (tuple, optional): Shape of the pair logits.
        ndim (int, optional): Number of dimensions. Default: 2.
        dtype (dtype, optional): Data type. Default: ms.float32.
        mode (str, optional): Computation mode. Default: 'einsum'.
    """

    def __init__(
        self,
        config,
        global_config,
        num_blocks,
        using_pair_act,
        in_shape,
        pair_shape=None,
        ndim=2,
        dtype=ms.float32,
    ):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.num_blocks = num_blocks
        self.using_pair_act = using_pair_act
        self.blocks = ms.nn.CellList(
            [
                Block(
                    config,
                    global_config,
                    in_shape,
                    pair_shape,
                    ndim=ndim,
                    dtype=dtype,
                )
                for _ in range(self.config.super_block_size)
            ]
        )

    def construct(self, act, single_cond, pair_cond):  # pylint: disable=unused-argument
        """construct"""
        for j in range(self.config.super_block_size):
            act = self.blocks[j](act, single_cond, pair_cond)
        return act


@dataclass
class CrossAttentionConfig(base_config.BaseConfig):
    num_head: int = 4
    key_dim: int = 128
    value_dim: int = 128


class CrossAttention(nn.Cell):
    """
    A CrossAttention class implementing multi-head cross-attention mechanism for processing sequential data.

    Args:
        config (Config): Configuration object containing attention settings.
        global_config (GlobalConfig): Global configuration object.
        in_channel (int): Input dimension for the attention mechanism.
        logits_in_shape (tuple): Shape of the input logits.
        dtype (dtype, optional): Data type. Default: ms.float32.

    Inputs:
        - **x_q** (Tensor) - Query tensor.
        - **x_k** (Tensor) - Key tensor.
        - **pair_logits** (Tensor, optional) - Optional pair logits tensor. Default: ``None``.
        - **single_cond_q** (Tensor) - Single condition tensor for queries.

    Outputs:
        - **output** (Tensor) - Output tensor after cross-attention processing.
    """

    def __init__(
        self, config, global_config, in_channel, logits_in_shape, dtype=ms.float32
    ):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.adaptive_layernorm_q = AdaptiveLayernorm(
            in_channel, in_channel, dtype=dtype
        )
        self.adaptive_layernorm_k = AdaptiveLayernorm(
            in_channel, in_channel, dtype=dtype
        )
        if config.key_dim % config.num_head != 0:
            raise ValueError("key_dim must be divisible by num_head")
        if config.value_dim % config.num_head != 0:
            raise ValueError("value_dim must be divisible by num_head")
        self.key_dim = config.key_dim // config.num_head
        self.value_dim = config.value_dim // config.num_head
        self.linear_q = bm.LinearAMP(
            in_channel, self.config.num_head * self.key_dim, has_bias=True, dtype=dtype
        )
        self.linear_k = bm.LinearAMP(
            in_channel, self.config.num_head * self.key_dim, has_bias=False, dtype=dtype
        )
        self.linear_v = bm.LinearAMP(
            in_channel,
            self.config.num_head * self.value_dim,
            has_bias=False,
            dtype=dtype,
        )
        self.ncon1 = Ncon([[-1, -3, -2, 1], [-1, -4, -2, 1]])
        self.ncon2 = Ncon([[-1, -3, -2, 1], [-1, 1, -3, -4]])
        self.gating_query = bm.LinearAMP(
            in_channel,
            self.config.num_head * self.value_dim,
            has_bias=False,
            dtype=dtype,
        )
        self.adaptive_zero_init = AdaptiveZeroInit(
            in_channel, in_channel, in_channel, dtype=dtype
        )
        self.layernorm_bias = bm.LayerNorm(logits_in_shape, create_beta=False)
        self.linear_bias = bm.CustomDense(
            logits_in_shape[-1], self.config.num_head, ndim=4
        )

    def construct(self, x_q, x_k, pair_logits, single_cond_q):
        """Multihead self-attention."""
        pair_logits = self.linear_bias(self.layernorm_bias(pair_logits.squeeze())).transpose(
            (3, 0, 1, 2)
        )

        x_q = x_q.squeeze()
        single_cond_q = single_cond_q.squeeze()
        x_q = self.adaptive_layernorm_q(x_q, single_cond_q)
        x_k = self.adaptive_layernorm_k(x_q, single_cond_q)

        q = self.linear_q(x_q)
        k = self.linear_k(x_k)
        v = self.linear_v(x_k)

        q = q.view(q.shape[:-1] + (self.config.num_head, -1))
        k = k.view(k.shape[:-1] + (self.config.num_head, -1))
        v = v.view(v.shape[:-1] + (self.config.num_head, -1))

        q = q.swapaxes(-2, -3)
        k = k.swapaxes(-2, -3)
        v = v.swapaxes(-2, -3)
        q = q * self.key_dim ** (-0.5)

        # add for protenix
        q_trunked, k_trunked, v_trunked, attn_bias_trunked, q_pad_length = (
            rearrange_to_dense_trunk(
                q=q,
                k=k,
                v=v,
                n_queries=pair_logits.shape[-2],
                n_keys=pair_logits.shape[-1],
                attn_bias=None,
                inf=1e10,
            )
        )
        if pair_logits is not None:
            bias = attn_bias_trunked + pair_logits
        else:
            bias = 0
        logits = mint.matmul(q_trunked, k_trunked.transpose(-1, -2)) + bias

        weights = ms.ops.softmax(logits, axis=-1)
        weighted_avg = mint.matmul(weights, v_trunked)
        weighted_avg = ms.ops.reshape(
            weighted_avg, weighted_avg.shape[:-3] + (-1,) + (weighted_avg.shape[-1],)
        )
        if q_pad_length > 0:
            weighted_avg = weighted_avg[..., :-q_pad_length, :]

        weighted_avg = weighted_avg.transpose(-2, -3)
        gate_logits = self.gating_query(x_q)
        weighted_avg = weighted_avg * ms.ops.sigmoid(
            gate_logits.astype(ms.float32)
        ).astype(gate_logits.dtype).view(
            gate_logits.shape[:-1] + (self.config.num_head, -1)
        )
        weighted_avg = weighted_avg.reshape(shape=weighted_avg.shape[:-2] + (-1,))

        output = self.adaptive_zero_init(
            weighted_avg,
            single_cond_q,
        )
        return output


class CrossAttTransformer(nn.Cell):
    """
    A CrossAttTransformer class implementing a transformer that applies cross attention between two sets of subsets.

    Args:
        config (Config): Configuration object containing settings for the transformer.
        global_config (GlobalConfig): Global configuration object.
        in_shape (tuple): Input shape for the transformer.
        ndim (int, optional): Number of dimensions. Default: 3.
        dtype (dtype, optional): Data type. Default: ms.float32.

    Inputs:
        - **queries_act** (Tensor) - Query activations tensor.
        - **queries_single_cond** (Tensor) - Single condition tensor for queries.
        - **pair_cond** (Tensor) - Pair condition tensor.

    Outputs:
        - **queries_act** (Tensor) - Processed query activations tensor after cross attention.
    """

    @dataclass
    class Config(base_config.BaseConfig):
        num_intermediate_factor: int
        num_blocks: int
        attention: CrossAttentionConfig = base_config.autocreate()

    def __init__(self, config, global_config, in_shape, ndim=3, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.in_shape = in_shape
        self.block = ms.nn.CellList(
            [
                CrossAttTransformerBlock(
                    config,
                    global_config,
                    self.in_shape[-2],
                    in_shape,
                    ndim=ndim,
                    dtype=dtype,
                )
                for _ in range(self.config.num_blocks)
            ]
        )

    def construct(self, queries_act, queries_single_cond, pair_cond):
        for i in range(self.config.num_blocks):
            queries_act = self.block[i](queries_act, pair_cond, queries_single_cond)
        return queries_act


class CrossAttTransformerBlock(nn.Cell):
    """
    CrossAttentionTransformerBlock class implementing a transformer block that applies
    cross attention between two sets of subsets.

    Args:
        config (Config): Configuration object containing settings for the transformer block.
        global_config (GlobalConfig): Global configuration object.
        in_channel (int): Input dimension for the transformer block.
        in_shape (tuple): Input shape for the transformer block.
        ndim (int): Number of dimensions for the transformer block. Defaults to 3.
        dtype (ms.dtype): Data type for the transformer block. Defaults to ms.float32.

    Inputs:
        - **queries_act** (Tensor) - Query activations tensor.
        - **queries_mask** (Tensor) - Mask tensor for queries.
        - **keys_mask** (Tensor) - Mask tensor for keys.
        - **pair_logits** (Tensor) - Pair logits tensor.
        - **queries_single_cond** (Tensor) - Single condition tensor for queries.

    Outputs:
        - **queries_act** (Tensor) - Processed query activations tensor after cross attention.

    Example:
        >>> config = CrossAttTransformerBlock.Config(num_intermediate_factor=2, num_blocks=3)
        >>> global_config = GlobalConfig()
        >>> in_channel = 128
        >>> in_shape = (128, 128)
        >>> ndim = 3
        >>> dtype = ms.float32
        >>> block = CrossAttTransformerBlock(config, global_config, in_channel, in_shape, ndim=ndim, dtype=dtype)
        >>> queries_act = ms.Tensor(np.random.randn(10, 128, 128), dtype=dtype)
        >>> queries_mask = ms.Tensor(np.random.randn(10, 128), dtype=dtype)
        >>> queries_to_keys = ms.Tensor(np.random.randn(10, 128), dtype=dtype)
        >>> keys_mask = ms.Tensor(np.random.randn(10, 128), dtype=dtype)
        >>> pair_logits = ms.Tensor(np.random.randn(10, 128, 128), dtype=dtype)
        >>> queries_single_cond = ms.Tensor(np.random.randn(10, 128), dtype=dtype)
        >>> keys_single_cond = ms.Tensor(np.random.randn(10, 128), dtype=dtype)
        >>> queries_act = block(queries_act, queries_mask, queries_to_keys, keys_mask, pair_logits,
            queries_single_cond, keys_single_cond)
        >>> print(queries_act.shape)
        (10, 128, 128)
    """

    def __init__(
        self, config, global_config, in_channel, in_shape, ndim=3, dtype=ms.float32
    ):
        super().__init__()
        self.cross_attention = CrossAttention(
            config.attention, global_config, in_channel, in_shape, dtype=dtype
        )
        self.transition = TransitionBlock(
            in_channel, config.num_intermediate_factor, ndim=ndim, dtype=dtype
        )

    def construct(
        self, queries_act, pair_logits, queries_single_cond
    ):  # pylint: disable=unused-argument
        """Construct cross attention transformer block."""
        keys_act = None
        queries_act = queries_act + self.cross_attention(
            queries_act,
            keys_act,
            pair_logits,
            queries_single_cond,
        )
        queries_act = queries_act + self.transition(queries_act, queries_single_cond)
        return queries_act


@dataclass
class AtomCrossAttDecoderConfig(base_config.BaseConfig):
    per_token_channels: int = 768
    per_atom_channels: int = 128
    per_atom_pair_channels: int = 16
    atom_transformer: CrossAttTransformer.Config = base_config.autocreate(
        num_intermediate_factor=2, num_blocks=3
    )


class AtomCrossAttDecoder(nn.Cell):
    """Mapping to per-atom features and self-attention on subsets.

    Args:
        config: Configuration object containing model parameters.
        global_config: Global configuration object with additional parameters.

    Inputs:
        - **token_act** (Tensor) - Tensor representing token activations.
        - **enc** (AtomCrossAttEncoderOutput) - Output from the encoder containing necessary features and masks.
        - **batch** (feat_batch.Batch) - Batch containing atom cross attention features.

    Outputs:
        - **position_update** (Tensor) - Tensor representing the updated positions after processing.
    """

    def __init__(self, config, global_config, dtype=ms.float32):
        super().__init__()
        self.c = config
        self._project_token_features_for_broadcast = bm.LinearAMP(
            self.c.per_token_channels,
            self.c.per_atom_channels,
            has_bias=False,
            dtype=dtype,
        )
        self._atom_features_layer_norm = bm.LayerNorm(
            (self.c.per_atom_channels,),
            create_beta=False,
            gamma_init="ones",
            dtype=dtype,
        )
        self._atom_features_to_position_update = bm.LinearAMP(
            self.c.per_atom_channels,
            3,
            weight_init=global_config.final_init,
            has_bias=False,
            dtype=dtype,
        )
        self._atom_transformer_decoder = CrossAttTransformer(
            self.c.atom_transformer,
            global_config,
            in_shape=[self.c.per_atom_channels, self.c.per_atom_pair_channels],
            ndim=2,
            dtype=dtype,
        )

    def construct(
        self,
        token_act,  # (num_tokens, ch)
        enc,
        batch,
    ):
        """atom cross attention decoder"""
        token_act = self._project_token_features_for_broadcast(token_act)
        queries_act = broadcast_token_to_atom(
            x_token=token_act,  # [..., N_token, c_atom]
            atom_to_token_idx=batch.atom_cross_att.token_atoms_to_queries.astype(
                ms.int32
            ),
        )  # [..., N_atom, c_atom]
        queries_act = queries_act + enc.skip_connection
        # Run the atom cross attention transformer.
        queries_act = self._atom_transformer_decoder(
            queries_act=queries_act,
            queries_single_cond=enc.queries_single_cond,
            pair_cond=enc.pair_cond,
        )

        queries_position_update = self._atom_features_to_position_update(
            self._atom_features_layer_norm(queries_act)
        )
        return queries_position_update
