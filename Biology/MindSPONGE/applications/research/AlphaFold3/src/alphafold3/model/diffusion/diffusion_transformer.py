# Copyright 2025 Huawei Technologies Co., Ltd
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Diffusion transformer model."""

from dataclasses import dataclass
from alphafold3.model import base_config
from alphafold3.utils.gated_linear_unit import gated_linear_unit
from alphafold3.model.atom_layout import atom_layout
from alphafold3.model.components import base_modules as bm

from mindspore import mint
import mindspore as ms
from mindspore import nn, ops
from mindchemistry.e3.utils import Ncon


class AdaptiveLayernorm(nn.Cell):
    """
    If single condition is None, this layer is the same as layernorm.
    If single condition is given, the layer is modified from Scalable Diffusion Models with Transformers
    https://arxiv.org/abs/2212.09748

    Args:
        num_channels (int): Number of channels in the input tensor.
        single_channel (int, optional): Number of channels in the single condition tensor. Required if `with_single_cond` is True. Default: ``None``.
        ndim (int, optional): Number of dimensions for the dense layers. Default: ``3``.
        with_single_cond (bool, optional): Whether to include the single condition adaptation. Default: ``True``.

    Inputs:
        - **x** (Tensor) - Input tensor to be normalized.
        - **single_cond** (Tensor, optional) - Optional single condition tensor used to adapt the normalization parameters. Required if `with_single_cond` is True.

    Outputs:
        - **output** (Tensor) - The normalized output tensor.
    """

    def __init__(self, num_channels, single_channel=None, ndim=3, with_single_cond=True, dtype=ms.float32):
        super().__init__()
        self.with_single_cond = with_single_cond
        if self.with_single_cond:
            self.layernorm = bm.LayerNorm([num_channels], name='layer_norm',
                                          create_gamma=False, create_beta=False,
                                          gamma_init='ones', beta_init='zeros', dtype=ms.float32)
            self.single_cond_layer_norm = bm.LayerNorm([single_channel], name='single_cond_layer_norm',
                                                       create_beta=False, gamma_init='ones', beta_init='zeros',
                                                       dtype=ms.float32)
            self.single_cond_scale = bm.CustomDense(single_channel, num_channels, weight_init='zeros',
                                                    use_bias=True, bias_init='ones', ndim=ndim, dtype=dtype)
            self.single_cond_bias = bm.CustomDense(
                single_channel, num_channels, weight_init='zeros', ndim=ndim, dtype=dtype)
        else:
            self.layernorm = bm.LayerNorm([num_channels], dtype=ms.float32)

    def construct(self, x, single_cond=None):
        if not self.with_single_cond:
            x = self.layernorm(x)
        else:
            x = self.layernorm(x)
            single_cond = self.single_cond_layer_norm(single_cond)
            single_scale = self.single_cond_scale(single_cond)
            single_bias = self.single_cond_bias(single_cond)
            x = mint.add(mint.mul(mint.sigmoid(single_scale), x), single_bias)
        return x


class AdaptiveZeroInit(nn.Cell):
    """
    An adaptive initialization layer that combines two conditional linear transformations.

    Args:
        global_config: Configuration object containing initialization settings.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        single_channels (int, optional): Number of single conditional channels. Default: ``None``.
        ndim (int, optional): Number of dimensions for the dense layer input. Default: ``3``.
        with_single_cond (bool, optional): Whether to use single conditional transformation. Default: ``True``.

    Inputs:
        - **x** (Tensor) - Input tensor to the layer.
        - **single_cond** (Tensor, optional) - Single conditional tensor. Required if `with_single_cond` is True.

    Outputs:
        - **output** (Tensor) - Output tensor after applying the adaptive initialization.
    """

    def __init__(self, global_config, in_channels, out_channels, single_channels=None, ndim=3, with_single_cond=True, dtype=ms.float32):
        super().__init__()
        self.with_single_cond = with_single_cond
        self.cond_linear1 = bm.CustomDense(
            in_channels, out_channels, weight_init='zeros', ndim=ndim, dtype=dtype)
        if self.with_single_cond:
            if single_channels is None:
                single_channels = in_channels
            self.cond_linear2 = bm.CustomDense(single_channels, out_channels, weight_init='zeros',
                                               use_bias=True, bias_init='zeros', ndim=ndim, dtype=dtype)
            self.cond_linear2.bias = ms.Parameter(self.cond_linear2.bias * (-2))

    def construct(self, x, single_cond=None):
        if not self.with_single_cond:
            output = self.cond_linear1(x)
        else:
            output = self.cond_linear1(x)
            cond = self.cond_linear2(single_cond)
            output = mint.mul(mint.sigmoid(cond), output)
        return output


class TransitionBlock(nn.Cell):
    """
    A neural network layer that combines adaptive layer normalization, a gated linear unit (GLU), and adaptive zero initialization to process input data with optional conditional inputs.

    Args:
        global_config: Configuration object containing initialization settings.
        in_channels (int): Number of input channels.
        num_intermediate_factor (int): Factor to determine the number of intermediate channels.
        single_channels (int, optional): Number of single conditional channels. Default: ``None``.
        ndim (int, optional): Number of dimensions for input tensor. Default: ``3``.
        with_single_cond (bool, optional): Whether to use single conditional processing. Default: ``True``.
        use_glu_kernel (bool, optional): Whether to use GLU. Default: ``True``.
        name (str, optional): Name of the layer. Default: ``''``.

    Inputs:
        - **x** (Tensor) - Input tensor to the layer.
        - **single_cond** (Tensor, optional) - Single conditional tensor. Required if `with_single_cond` is True.

    Outputs:
        - **output** (Tensor) - Output tensor after processing through the TransitionBlock.
    """

    def __init__(self, global_config, in_channels, num_intermediate_factor, single_channels=None, ndim=3, with_single_cond=True, use_glu_kernel=True, name='', dtype=ms.float32):
        super().__init__()
        self.num_intermediate = num_intermediate_factor * in_channels
        if single_channels is None:
            single_channels = in_channels
        self.adaptive_layernorm = AdaptiveLayernorm(
            in_channels, single_channels, ndim=ndim, with_single_cond=with_single_cond, dtype=dtype)
        self.use_glu_kernel = use_glu_kernel
        if self.use_glu_kernel:
            self.weights = bm.custom_initializer(
                'relu', [in_channels, self.num_intermediate * 2], dtype=dtype)
            self.weights = ms.Parameter(ms.Tensor(self.weights).reshape(
                in_channels, 2, self.num_intermediate))
        else:
            self.linear = bm.CustomDense(
                in_channels, self.num_intermediate * 2, weight_init='zeros', ndim=3, dtype=dtype)
        self.adaptive_zero_init = AdaptiveZeroInit(
            global_config, self.num_intermediate, in_channels, single_channels, ndim=ndim, with_single_cond=with_single_cond, dtype=dtype)

    def construct(self, x, single_cond=None):
        x = self.adaptive_layernorm(x, single_cond)
        if self.use_glu_kernel:
            c = gated_linear_unit.gated_linear_unit(
                x=x, weight=self.weights.astype(x.dtype),
                implementation=None, activation=mint.nn.functional.silu, precision=None
            ).astype(x.dtype)
        else:
            x = self.linear(x)
            x0, x1 = ops.split(x, int(x.shape[-1]/2), axis=-1)
            c = ops.silu(x0) * x1
        output = self.adaptive_zero_init(c, single_cond)
        return output

@dataclass
class SelfAttentionConfig(base_config.BaseConfig):
    num_head: int = 16
    key_dim: int | None = None
    value_dim: int | None = None


class SelfAttention(nn.Cell):
    """
    A self-attention mechanism implementation with adaptive layer normalization and adaptive zero initialization.

    This class implements the self-attention mechanism commonly used in transformer models. It includes adaptive layer normalization for input processing and adaptive zero initialization for the final output. The mechanism computes attention scores using query, key, and value transformations, applies masking, and optionally incorporates pair-wise logits.

    Args:
        config: Configuration object containing parameters such as key dimension, value dimension, and number of attention heads.
        global_config: Global configuration object for additional settings.
        num_channels (int): Number of channels in the input tensor.
        in_shape (tuple): Shape of the input tensor.
        ndim (int, optional): Number of dimensions for the dense layers. Default: ``3``.
        with_single_cond (bool, optional): Whether to include single condition adaptation. Default: ``True``.

    Inputs:
        - **x** (Tensor) - Input tensor to the self-attention layer.
        - **mask** (Tensor) - Attention mask to apply.
        - **single_cond** (Tensor, optional) - Single condition tensor for adaptation.
        - **pair_logits** (Tensor, optional) - Additional logits to incorporate into attention scores.

    Outputs:
        - **output** (Tensor) - The output tensor after self-attention and adaptive zero initialization.

    Notes:
        - The class uses adaptive layer normalization and adaptive zero initialization for processing inputs and outputs.
        - The attention mechanism supports optional single condition adaptation and pair-wise logits.
    """

    def __init__(self, config, global_config, num_channels, in_shape, ndim=3, with_single_cond=True, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.adaptive_layernorm = AdaptiveLayernorm(num_channels, int(
            num_channels//2), ndim=ndim, with_single_cond=with_single_cond, dtype=dtype)
        key_dim = self.config.key_dim if self.config.key_dim is not None else num_channels
        value_dim = self.config.value_dim if self.config.value_dim is not None else num_channels
        num_head = self.config.num_head
        assert key_dim % num_head == 0, f'{key_dim=} % {num_head=} != 0'
        assert value_dim % num_head == 0, f'{value_dim=} % {num_head=} != 0'
        key_dim = key_dim // num_head
        self.key_dim = key_dim
        value_dim = value_dim // num_head
        qk_shape = (num_head, key_dim)
        v_shape = (num_head, value_dim)
        self.q_linear = bm.CustomDense(num_channels, qk_shape, use_bias=True, dtype=dtype)
        self.k_linear = bm.CustomDense(num_channels, qk_shape, use_bias=False, dtype=dtype)
        self.v_linear = bm.CustomDense(num_channels, v_shape, use_bias=False, dtype=dtype)
        self.linear = bm.CustomDense(
            num_channels, num_head * value_dim, weight_init='zeros', dtype=dtype)
        self.adaptive_zero_init = AdaptiveZeroInit(global_config, num_channels, num_channels, int(
            num_channels//2), 2, with_single_cond=with_single_cond, dtype=dtype)
        self.ncon1 = Ncon([[-2, -1, 1], [-3, -1, 1]])
        self.ncon2 = Ncon([[-2, -1, 2], [2, -2, -3]])

    def construct(self, x, mask, single_cond, pair_logits):
        bias = (1e9 * (mask - 1.0))[..., None, None, :].astype(x.dtype)
        x = self.adaptive_layernorm(x, single_cond)
        q = self.q_linear(x)
        k = self.k_linear(x)
        logits = mint.einsum('...qhc,...khc->...hqk', q * self.key_dim ** (-0.5), k) + bias
        if pair_logits is not None:
            logits += pair_logits
        weights = mint.softmax(logits, dim=-1)
        weights = weights.astype(q.dtype)
        v = self.v_linear(x)
        weighted_avg = mint.einsum('...hqk,...khc->...qhc', weights, v)
        weighted_avg = weighted_avg.reshape(weighted_avg.shape[:-2] + (-1,))
        gate_logits = self.linear(x)
        weighted_avg *= mint.sigmoid(gate_logits)
        output = self.adaptive_zero_init(weighted_avg, single_cond)
        return output


class Transformer(nn.Cell):
    @dataclass
    class Config(base_config.BaseConfig):
        attention: SelfAttentionConfig = base_config.autocreate()
        num_blocks: int = 24
        block_remat: bool = False
        super_block_size: int = 4
        num_intermediate_factor: int = 2

    def __init__(self, config, global_config, in_shape, pair_shape, using_pair_act=False, name="transformer", dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.using_pair_act = using_pair_act
        self.act = []
        if using_pair_act:
            self.pair_layernorm = bm.LayerNorm(pair_shape, create_beta=False, dtype=ms.float32)
        else:
            self.pair_layernorm = None
        assert self.config.num_blocks % self.config.super_block_size == 0
        self.num_super_blocks = self.config.num_blocks // self.config.super_block_size
        self.super_blocks = ms.nn.CellList(
            [
                SuperBlock(
                    config, global_config, self.config.num_blocks,
                    using_pair_act, in_shape, pair_shape, name, dtype=dtype
                )
                for _ in range(self.num_super_blocks)
            ]
        )

    @ms.jit
    def construct(self, act, single_cond, mask, pair_cond=None):
        if pair_cond is None:
            pair_act = None
        else:
            pair_act = self.pair_layernorm(pair_cond)
        for i in range(self.num_super_blocks):
            act = self.super_blocks[i](act, mask, single_cond, pair_act)
        return act


class Block(nn.Cell):
    def __init__(self, config, global_config, in_shape, dtype=ms.float32):
        super().__init__()
        self.self_attention = SelfAttention(
            config.attention, global_config, in_shape[-1], in_shape, ndim=2, dtype=dtype)
        self.transition_block = TransitionBlock(global_config, in_shape[-1],
                                                config.num_intermediate_factor, int(in_shape[-1]//2), ndim=2, dtype=dtype)

    def construct(self, act, mask, single_cond, pair_logits):
        act += self.self_attention(act, mask, single_cond, pair_logits)
        act += self.transition_block(act, single_cond)
        return act


class SuperBlock(nn.Cell):
    def __init__(self, config, global_config, num_blocks, using_pair_act, in_shape, pair_shape=None, name='', dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.num_blocks = num_blocks
        self.using_pair_act = using_pair_act
        self.blocks = ms.nn.CellList(
            [
                Block(
                    config, global_config, in_shape, dtype=dtype
                )
                for _ in range(self.config.super_block_size)
            ]
        )
        if self.using_pair_act:
            self.pair_linear = bm.CustomDense(
                pair_shape[-1], (self.config.super_block_size, self.config.attention.num_head), ndim=3, dtype=dtype)
        else:
            self.pair_linear = None

    def construct(self, act, mask, single_cond, pair_act):
        if pair_act is None:
            pair_logits = None
        else:
            pair_logits = self.pair_linear(pair_act).transpose([2, 3, 0, 1])
        for j in range(self.config.super_block_size):
            act = self.blocks[j](act, mask, single_cond, pair_logits[j])
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

    Inputs:
        - **x_q** (Tensor) - Query tensor.
        - **x_k** (Tensor) - Key tensor.
        - **mask_q** (Tensor) - Query mask tensor.
        - **mask_k** (Tensor) - Key mask tensor.
        - **pair_logits** (Tensor, optional) - Optional pair logits tensor. Default: ``None``.
        - **single_cond_q** (Tensor) - Single condition tensor for queries.
        - **single_cond_k** (Tensor) - Single condition tensor for keys.

    Outputs:
        - **output** (Tensor) - Output tensor after cross-attention processing.
    """

    def __init__(self, config, global_config, in_channel, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.adaptive_layernorm_q = AdaptiveLayernorm(in_channel, in_channel, dtype=dtype)
        self.adaptive_layernorm_k = AdaptiveLayernorm(in_channel, in_channel, dtype=dtype)
        assert config.key_dim % config.num_head == 0
        assert config.value_dim % config.num_head == 0
        self.key_dim = config.key_dim // config.num_head
        self.value_dim = config.value_dim // config.num_head
        self.linear_q = bm.CustomDense(
            in_channel, (self.config.num_head, self.key_dim), use_bias=True, ndim=3, dtype=dtype)
        self.linear_k = bm.CustomDense(
            in_channel, (self.config.num_head, self.key_dim), use_bias=False, ndim=3, dtype=dtype)
        self.linear_v = bm.CustomDense(
            in_channel, (self.config.num_head, self.value_dim), use_bias=False, ndim=3, dtype=dtype)
        self.ncon1 = Ncon([[-1, -3, -2, 1], [-1, -4, -2, 1]])
        self.ncon2 = Ncon([[-1, -3, -2, 1], [-1, 1, -3, -4]])
        self.gating_query = bm.CustomDense(
            in_channel, self.config.num_head * self.value_dim, use_bias=False,
            weight_init='zeros', bias_init='ones', ndim=3, dtype=dtype)
        self.adaptive_zero_init = AdaptiveZeroInit(
            global_config, in_channel, in_channel, in_channel, dtype=dtype)

    def construct(self, x_q, x_k, mask_q, mask_k, pair_logits, single_cond_q, single_cond_k):
        """Multihead self-attention."""
        bias = (
            1e9
            * (mask_q - 1.0)[..., None, :, None]
            * (mask_k - 1.0)[..., None, None, :]
        )
        x_q = self.adaptive_layernorm_q(x_q, single_cond_q)
        x_k = self.adaptive_layernorm_k(x_k, single_cond_k)
        q = self.linear_q(x_q)
        k = self.linear_k(x_k)
        logits = mint.einsum('...qhc,...khc->...hqk', q * self.key_dim ** (-0.5), k) + bias
        if pair_logits is not None:
            logits += pair_logits
        weights = ops.softmax(logits, axis=-1)
        v = self.linear_v(x_k)
        weighted_avg = mint.einsum('...hqk,...khc->...qhc', weights, v)
        weighted_avg = ops.reshape(
            weighted_avg, weighted_avg.shape[:-2] + (-1,))

        gate_logits = self.gating_query(x_q)
        weighted_avg *= ops.sigmoid(gate_logits)

        output = self.adaptive_zero_init(weighted_avg, single_cond_q,)
        return output


class CrossAttTransformer(nn.Cell):
    """
    A CrossAttTransformer class implementing a transformer that applies cross attention between two sets of subsets.

    Args:
        config (Config): Configuration object containing settings for the transformer.
        global_config (GlobalConfig): Global configuration object.
        in_shape (tuple): Input shape for the transformer.

    Inputs:
        - **queries_act** (Tensor) - Query activations tensor.
        - **queries_mask** (Tensor) - Mask tensor for queries.
        - **queries_to_keys** (Tensor) - Tensor mapping queries to keys.
        - **keys_mask** (Tensor) - Mask tensor for keys.
        - **queries_single_cond** (Tensor) - Single condition tensor for queries.
        - **keys_single_cond** (Tensor) - Single condition tensor for keys.
        - **pair_cond** (Tensor) - Pair condition tensor.

    Outputs:
        - **queries_act** (Tensor) - Processed query activations tensor after cross attention.
    """
    @dataclass
    class Config(base_config.BaseConfig):
        num_intermediate_factor: int
        num_blocks: int
        attention: CrossAttentionConfig = base_config.autocreate()

    def __init__(self, config, global_config, in_shape, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.pair_input_layer_norm = bm.LayerNorm(in_shape, create_beta=False, dtype=ms.float32)
        self.pair_logits_projection = bm.CustomDense(
            in_shape[-1], (self.config.num_blocks, self.config.attention.num_head), ndim=4, dtype=dtype)
        self.block = ms.nn.CellList(
            [
                CrossAttTransformerBlock(
                    config, global_config, in_shape[-2], dtype=dtype
                )
                for _ in range(self.config.num_blocks)
            ]
        )

    def construct(self, queries_act, queries_mask, queries_to_keys,
                  keys_mask, queries_single_cond, keys_single_cond,
                  pair_cond):
        pair_act = self.pair_input_layer_norm(pair_cond)
        pair_logits = self.pair_logits_projection(pair_act)
        pair_logits = ops.transpose(pair_logits, (3, 0, 4, 1, 2))
        for i in range(self.config.num_blocks):
            queries_act = self.block[i](queries_act, queries_mask, queries_to_keys, keys_mask, pair_logits[i],
                                        queries_single_cond, keys_single_cond)
        return queries_act


class CrossAttTransformerBlock(nn.Cell):
    def __init__(self, config, global_config, in_channel, dtype=ms.float32):
        super().__init__()
        self.cross_attention = CrossAttention(
            config.attention, global_config, in_channel, dtype=dtype)
        self.transition = TransitionBlock(
            global_config, in_channel, config.num_intermediate_factor, dtype=dtype)

    def construct(self, queries_act, queries_mask, queries_to_keys, keys_mask, pair_logits,
                  queries_single_cond, keys_single_cond):
        keys_act = atom_layout.convert_ms(
            queries_to_keys, queries_act, layout_axes=(-3, -2)
        )
        queries_act += self.cross_attention(queries_act, keys_act, queries_mask, keys_mask,
                                            pair_logits, queries_single_cond, keys_single_cond)
        queries_act += self.transition(queries_act, queries_single_cond)
        return queries_act
