# Copyright 2024 DeepMind Technologies Limited
# Copyright (C) 2025 Huawei Technologies Co., Ltd
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
#
# Modifications by Huawei Technologies Co., Ltd: Adapt to run by MindSpore on Ascend

"""modules for the Diffuser model."""

from dataclasses import dataclass
from typing import Literal

import mindspore as ms
from mindspore import nn, ops, Tensor, mint
from mindchemistry.e3.utils import Ncon
from alphafold3.model import base_config
from alphafold3.utils.attention import attention
from alphafold3.utils.gated_linear_unit.gated_linear_unit import gated_linear_unit
from alphafold3.model.components import base_modules as bm
from alphafold3.model.components import mapping
from alphafold3.model.diffusion import diffusion_transformer
from alphafold3.model.diffusion.triangle import TriangleMultiplication as Triangle
from alphafold3.model.diffusion.triangle import OuterProductMean as ProductMean


def get_shard_size(num_residues, shard_spec):
    shard_size = shard_spec[0][-1]
    for num_residues_upper_bound, num_residues_shard_size in shard_spec:
        shard_size = num_residues_shard_size
        if (
                num_residues_upper_bound is None
                or num_residues <= num_residues_upper_bound
        ):
            break
    return shard_size


class TransitionBlock(nn.Cell):
    """
    A transition block for transformer networks, implementing either a GLU-based or linear-based transformation.

    Args:
        config (Config): Configuration object containing parameters for the transition block.
        global_config (GlobalConfig): Global configuration object.
        normalized_shape (tuple): Shape of the input tensor for normalization.
        ndim (int): Number of dimensions of the input tensor. Default: ``3``.

    Inputs:
        - **act** (Tensor) - Input activation tensor to be processed.

    Outputs:
        - **output** (Tensor) - Output tensor after processing through the transition block.
    """
    @dataclass
    class Config(base_config.BaseConfig):
        num_intermediate_factor: int = 4
        use_glu_kernel: bool = True

    def __init__(
            self, config, global_config, normalized_shape, ndim=3, dtype=ms.float32
    ):
        super().__init__()
        self.config = config
        self.global_config = global_config
        num_channels = normalized_shape[-1]
        self.num_intermediate = int(
            num_channels * self.config.num_intermediate_factor)
        self.layernorm = bm.LayerNorm(
            normalized_shape, name='input_layer_norm', dtype=ms.float32)
        if self.config.use_glu_kernel:
            self.glu_weight = bm.custom_initializer(
                'relu', (num_channels, 2 * self.num_intermediate), dtype=dtype)
            self.glu_weight = ms.Parameter(Tensor(self.glu_weight).reshape(
                num_channels, 2, self.num_intermediate))
        else:
            self.linear = bm.CustomDense(num_channels, self.num_intermediate * 2,
                                         weight_init='zeros', ndim=ndim, dtype=dtype)
            self.linear.weight = bm.custom_initializer(
                'zeros', self.linear.weight.shape, dtype=dtype)
        self.out_linear = bm.CustomDense(self.num_intermediate, num_channels,
                                         weight_init=self.global_config.final_init, ndim=ndim, dtype=dtype)

    def construct(self, act, broadcast_dim=0):
        act = self.layernorm(act)
        if self.config.use_glu_kernel:
            c = gated_linear_unit(
                x=act,
                weight=self.glu_weight,
                implementation=None,
                activation=mint.nn.functional.silu,
                precision=None
            )
        else:
            act = self.linear(act)
            a, b = mint.split(act, act.shape[-1]//2, axis=-1)
            c = mint.nn.functional.silu(a) * b
        return self.out_linear(c)


class MSAAttention(nn.Cell):
    """
    Multi-Head Self-Attention (MSA) attention mechanism for processing sequence and pair data.

    Args:
        config (Config): Configuration object containing parameters for the attention mechanism.
        global_config (GlobalConfig): Global configuration object.
        act_shape (tuple): Shape of the activation tensor.
        pair_shape (tuple): Shape of the pair tensor.

    Inputs:
        - **act** (Tensor) - Input activation tensor.
        - **mask** (Tensor) - Mask tensor to prevent attention weights from focusing on invalid positions.
        - **pair_act** (Tensor) - Pair activation tensor.

    Outputs:
        - **output** (Tensor) - Output tensor after processing through the attention mechanism.
    """
    @dataclass
    class Config(base_config.BaseConfig):
        num_head: int = 8

    def __init__(self, config, global_config, act_shape, pair_shape, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.actnorm = bm.LayerNorm(act_shape, dtype=ms.float32)
        self.pairnorm = bm.LayerNorm(pair_shape, dtype=ms.float32)
        num_channel = act_shape[-1]
        value_dim = num_channel // self.config.num_head
        self.pair_logits = bm.CustomDense(pair_shape[-1], self.config.num_head, use_bias=False,
                                          weight_init='zeros', ndim=3, dtype=dtype)
        self.v_projection = bm.CustomDense(num_channel, (self.config.num_head, value_dim),
                                           use_bias=False, ndim=len(act_shape), dtype=dtype)
        ncon_list1 = [-3, -2, 1]
        ncon_list2 = [-1, 1, -3, -4]
        self.ncon = Ncon([ncon_list1, ncon_list2])
        self.gating_query = bm.CustomDense(
            num_channel, self.config.num_head * value_dim, weight_init='zeros', use_bias=False, ndim=3, dtype=dtype)
        self.output_projection = bm.CustomDense(self.config.num_head * value_dim, num_channel,
                                                weight_init=self.global_config.final_init,
                                                use_bias=False, ndim=3, dtype=dtype)

    def construct(self, act, mask, pair_act):
        act = self.actnorm(act)
        pair_act = self.pairnorm(pair_act)
        logits = self.pair_logits(pair_act).transpose([2, 0, 1])
        logits += 1e9 * (mint.max(mask, dim=0)[0] - 1.0)
        weights = mint.softmax(logits, dim=-1)
        v = self.v_projection(act)
        v_avg = self.ncon([weights, v])
        v_avg = v_avg.reshape(v_avg.shape[:-2]+(-1,))
        gate_value = self.gating_query(act)
        v_avg *= mint.sigmoid(gate_value)
        out = self.output_projection(v_avg)
        return out


class GridSelfAttention(nn.Cell):
    """
    Self-attention mechanism that operates either per-sequence or per-residue.

    Args:
        config (Config): Configuration object containing parameters for the attention mechanism.
        global_config (GlobalConfig): Global configuration object.
        transpose (bool): Whether to transpose the activation tensor during processing.
        normalized_shape (tuple): Shape of the input tensor for normalization.

    Inputs:
        - **act** (Tensor) - Input activation tensor.
        - **pair_mask** (Tensor) - Mask tensor indicating valid regions in the input.

    Outputs:
        - **output** (Tensor) - Output tensor after processing through the self-attention mechanism.
    """
    @dataclass
    class Config(base_config.BaseConfig):
        num_head: int = 4

    def __init__(
            self, config, global_config, transpose, normalized_shape, dtype=ms.float32
    ):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.transpose = transpose
        num_channels = normalized_shape[-1]
        in_shape = normalized_shape[-1]
        assert num_channels % self.config.num_head == 0
        qkv_dim = max(num_channels // self.config.num_head, 16)
        qkv_shape = (self.config.num_head, qkv_dim)
        self.q_projection = bm.CustomDense(
            in_shape, qkv_shape, use_bias=False, ndim=3, dtype=dtype)
        self.k_projection = bm.CustomDense(
            in_shape, qkv_shape, use_bias=False, ndim=3, dtype=dtype)
        self.v_projection = bm.CustomDense(
            in_shape, qkv_shape, use_bias=False, ndim=3, dtype=dtype)
        self.gating_query = bm.CustomDense(
            num_channels, self.config.num_head * qkv_dim, weight_init='zeros', use_bias=False, ndim=3, dtype=dtype)
        self.output_projection = bm.CustomDense(self.config.num_head * qkv_dim, num_channels,
                                                weight_init=self.global_config.final_init, ndim=3, dtype=dtype)
        self.act_norm = bm.LayerNorm(normalized_shape, dtype=ms.float32)
        self.pair_bias_projection = bm.CustomDense(
            num_channels, self.config.num_head, use_bias=False, weight_init='linear', ndim=3, dtype=dtype)
        num_residues = normalized_shape[0]
        self.chunk_size = get_shard_size(
            num_residues, self.global_config.pair_attention_chunk_size
        )

    def _attention(self, act, mask, bias):
        q = self.q_projection(act)
        k = self.k_projection(act)
        v = self.v_projection(act)
        bias = ops.expand_dims(bias, 0)
        weighted_avg = attention.dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            bias=bias,
            logits_dtype=ms.float32,
            precision=None,
            implementation=self.global_config.flash_attention_implementation,
        )
        weighted_avg = weighted_avg.reshape(weighted_avg.shape[:-2] + (-1,))
        gate_value = self.gating_query(act)
        weighted_avg *= mint.sigmoid(gate_value)
        return self.output_projection(weighted_avg)

    def construct(self, act, pair_mask):
        """Builds a module.

        Arguments:
            act: [num_seq, num_res, channels] activations tensor
            pair_mask: [num_seq, num_res] mask of non-padded regions in the tensor.
                Only used in inducing points attention currently.

        Returns:
            Result of the self-attention operation.
        """
        pair_mask = mint.swapaxes(pair_mask, -1, -2)
        act = self.act_norm(act)

        non_batched_bias = self.pair_bias_projection(act)
        non_batched_bias = non_batched_bias.transpose(2, 0, 1)
        if self.transpose:
            act = mint.swapaxes(act, -2, -3)
        pair_mask = pair_mask[:, None, None, :].astype(ms.bool_)
        act = self._attention(act, pair_mask, non_batched_bias)
        if self.transpose:
            act = mint.swapaxes(act, -2, -3)
        return act


class TriangleMultiplication(nn.Cell):
    """
    Implements triangle multiplication for tensor operations.

    Args:
        config (Config): Configuration object specifying the equation and whether to use a GLU kernel.
        global_config (GlobalConfig): Global configuration object.
        in_channel (int): Number of input channels.
        normalized_shape (tuple): Shape of the input tensor for normalization.
        batch_size (int, optional): Batch size for processing. Default: ``None``.

    Inputs:
        - **act** (Tensor) - Input activation tensor.
        - **mask** (Tensor) - Mask tensor indicating valid regions in the input.

    Outputs:
        - **out** (Tensor) - Output tensor after triangle multiplication.
    """
    @dataclass
    class Config(base_config.BaseConfig):
        equation: Literal['ikc,jkc->ijc', 'kjc,kic->ijc']
        use_glu_kernel: bool = True

    def __init__(self, config, global_config, in_channel, normalized_shape, batch_size=None, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.triangle_multi = Triangle(
            self.config,
            self.global_config,
            num_intermediate_channel=in_channel,
            equation=self.config.equation,
            normalized_shape=normalized_shape,
            batch_size=batch_size,
            dtype=dtype)

    def construct(self, act, mask):
        out = self.triangle_multi(act, mask)
        return out


class OuterProductMean(nn.Cell):
    """
    Implements the OuterProductMean operation for tensor computations.

    Args:
        config (Config): Configuration object containing parameters for the operation.
        global_config (GlobalConfig): Global configuration object.
        num_output_channel (int): Number of output channels.
        in_channel (int): Number of input channels.

    Inputs:
        - **act** (Tensor) - Input activation tensor.
        - **mask** (Tensor) - Mask tensor indicating valid regions in the input.

    Outputs:
        - **out** (Tensor) - Output tensor after applying the outer product mean operation.
    """
    @dataclass
    class Config(base_config.BaseConfig):
        chunk_size: int = 128
        num_outer_channel: int = 32

    def __init__(self, config, global_config, num_output_channel, in_channel, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.num_output_channel = num_output_channel
        self.outer_product_mean = ProductMean(self.config.num_outer_channel,
                                              in_channel,
                                              self.num_output_channel,
                                              dtype=dtype)

    def construct(self, act, mask):
        mask_norm = ops.expand_dims(mint.matmul(mask.T, mask), -1)
        out = self.outer_product_mean(act, mask, mask_norm)
        return out


class PairFormerIteration(nn.Cell):
    """
    Single Iteration of PairFormer, which processes pairwise and single activations in a single iteration.

    Args:
        config (PairFormerIteration.Config): Configuration for the PairFormerIteration module.
        global_config: Global configuration for the model.
        normalized_shape (tuple): Shape of the input tensor for normalization.
        single_shape (tuple | None): Shape of the single activation tensor. Default: ``None``.
        with_single (bool): Whether to include single activation processing. Default: ``False``.

    Inputs:
        - **act** (Tensor) - Pairwise activations tensor.
        - **pair_mask** (Tensor) - Padding mask for pairwise activations.
        - **single_act** (Tensor | None) - Single activations tensor, optional.
        - **seq_mask** (Tensor | None) - Sequence mask, optional.

    Outputs:
        - **act** (Tensor) - Processed pairwise activations tensor.
        - **single_act** (Tensor) - Processed single activations tensor (if `with_single` is True).
    """
    @dataclass
    class Config(base_config.BaseConfig):
        """Config for PairFormerIteration."""
        num_layer: int = 1
        pair_attention: GridSelfAttention.Config = base_config.autocreate()
        pair_transition: TransitionBlock.Config = base_config.autocreate()
        single_attention: diffusion_transformer.SelfAttentionConfig | None = base_config.autocreate()
        single_transition: TransitionBlock.Config | None = base_config.autocreate()
        triangle_multiplication_incoming: TriangleMultiplication.Config = (
            base_config.autocreate(equation='kjc,kic->ijc')
        )
        triangle_multiplication_outgoing: TriangleMultiplication.Config = (
            base_config.autocreate(equation='ikc,jkc->ijc')
        )
        shard_transition_blocks: bool = True

    def __init__(self, config, global_config, normalized_shape, single_shape=None, with_single=False, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.with_single = with_single
        num_channel = normalized_shape[-1]
        self.triangle_multiplication1 = TriangleMultiplication(
            self.config.triangle_multiplication_outgoing,
            self.global_config,
            num_channel,
            normalized_shape,
            dtype=dtype
        )
        self.triangle_multiplication2 = TriangleMultiplication(
            self.config.triangle_multiplication_incoming,
            self.global_config,
            num_channel,
            normalized_shape,
            dtype=dtype
        )
        self.grid_self_attention1 = GridSelfAttention(
            self.config.pair_attention,
            self.global_config,
            False,
            normalized_shape,
            dtype=dtype
        )
        self.grid_self_attention2 = GridSelfAttention(
            self.config.pair_attention,
            self.global_config,
            True,
            normalized_shape,
            dtype=dtype
        )
        self.transition_block = TransitionBlock(
            self.config.pair_transition, self.global_config, normalized_shape, dtype=dtype
        )
        num_residues = normalized_shape[0]
        if self.config.shard_transition_blocks:
            self.transition_block = mapping.sharded_apply(
                self.transition_block,
                get_shard_size(
                    num_residues, self.global_config.pair_transition_shard_spec
                )
            )
        if self.with_single:
            assert self.config.single_attention is not None
            self.single_pair_logits_projection = bm.CustomDense(
                num_channel, self.config.single_attention.num_head, ndim=3, dtype=dtype
            )
            self.single_pair_logits_norm = bm.LayerNorm(normalized_shape, dtype=ms.float32)
            self.single_attention = diffusion_transformer.SelfAttention(
                self.config.single_attention, self.global_config,
                single_shape[-1], normalized_shape, with_single_cond=False, dtype=dtype)
            self.single_transition = TransitionBlock(
                self.config.single_transition,
                self.global_config,
                single_shape,
                2,
                dtype=dtype
            )

    def construct(self, act, pair_mask, single_act=None, seq_mask=None):
        act += self.triangle_multiplication1(act, pair_mask)
        act += self.triangle_multiplication2(act, pair_mask)
        act += self.grid_self_attention1(act, pair_mask)
        act += self.grid_self_attention2(act, pair_mask)
        act += self.transition_block(act)
        if self.with_single:
            norm_act = self.single_pair_logits_norm(act)
            pair_logits = self.single_pair_logits_projection(norm_act)
            pair_logits = pair_logits.transpose((2, 0, 1))
            single_act += self.single_attention(
                single_act, seq_mask, None, pair_logits
            )
            single_act += self.single_transition(single_act,
                                                 broadcast_dim=None)
            return act, single_act
        return act


class EvoformerIteration(nn.Cell):
    """
    EvoformerIteration is a single iteration of the Evoformer main stack, which processes
    activations and masks through a series of attention and transformation layers to
    update the MSA (Multiple Sequence Alignment) and pair representations.

    Args:
        config (EvoformerIteration.Config): Configuration for the EvoformerIteration.
        global_config (base_config.BaseConfig): Global configuration for the model.
        act_shape (tuple): Shape of the activation tensor.
        pair_shape (tuple): Shape of the pair tensor.

    Inputs:
        - **activations** (dict): A dictionary containing the MSA and pair activations.
        - **masks** (dict): A dictionary containing the MSA and pair masks.

    Outputs:
        - **activations** (dict): A dictionary containing the updated MSA and pair activations.
    """
    @dataclass
    class Config(base_config.BaseConfig):
        """Configuration for EvoformerIteration."""

        num_layer: int = 4
        msa_attention: MSAAttention.Config = base_config.autocreate()
        outer_product_mean: OuterProductMean.Config = base_config.autocreate()
        msa_transition: TransitionBlock.Config = base_config.autocreate()
        pair_attention: GridSelfAttention.Config = base_config.autocreate()
        pair_transition: TransitionBlock.Config = base_config.autocreate()
        triangle_multiplication_incoming: TriangleMultiplication.Config = (
            base_config.autocreate(equation='kjc,kic->ijc')
        )
        triangle_multiplication_outgoing: TriangleMultiplication.Config = (
            base_config.autocreate(equation='ikc,jkc->ijc')
        )
        shard_transition_blocks: bool = False

    def __init__(self, config, global_config, act_shape, pair_shape, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        num_channel = pair_shape[-1]
        self.outer_product_mean = OuterProductMean(
            config=self.config.outer_product_mean,
            global_config=self.global_config,
            num_output_channel=num_channel,
            in_channel=act_shape[-1],
            dtype=dtype
        )
        self.msa_attention = MSAAttention(self.config.msa_attention,
                                          self.global_config, act_shape, pair_shape, dtype=dtype)
        self.msa_transition = TransitionBlock(
            self.config.msa_transition, self.global_config, act_shape, dtype=dtype
        )
        self.triangle_multiplication1 = TriangleMultiplication(
            self.config.triangle_multiplication_outgoing,
            self.global_config,
            num_channel,
            pair_shape,
            dtype=dtype
        )
        self.triangle_multiplication2 = TriangleMultiplication(
            self.config.triangle_multiplication_incoming,
            self.global_config,
            num_channel,
            pair_shape,
            dtype=dtype
        )
        self.pair_attention1 = GridSelfAttention(
            self.config.pair_attention,
            self.global_config,
            False,
            pair_shape,
            dtype=dtype
        )
        self.pair_attention2 = GridSelfAttention(
            self.config.pair_attention,
            self.global_config,
            True,
            pair_shape,
            dtype=dtype
        )
        self.transition_block = TransitionBlock(
            self.config.msa_transition, self.global_config, pair_shape, dtype=dtype
        )
        num_residues = act_shape[0]
        if self.config.shard_transition_blocks:
            self.transition_block = mapping.sharded_apply(
                self.transition_block,
                get_shard_size(
                    num_residues, self.global_config.pair_transition_shard_spec
                )
            )

    def construct(self, activations, masks):
        msa_act, pair_act = activations["msa"], activations["pair"]
        msa_mask, pair_mask = masks['msa'], masks['pair']
        pair_act += self.outer_product_mean(msa_act, msa_mask)
        msa_act += self.msa_attention(msa_act, msa_mask, pair_act)
        msa_act += self.msa_transition(msa_act)
        pair_act += self.triangle_multiplication1(pair_act, pair_mask)
        pair_act += self.triangle_multiplication2(pair_act, pair_mask)
        pair_act += self.pair_attention1(pair_act, pair_mask)
        pair_act += self.pair_attention2(pair_act, pair_mask)
        pair_act += self.transition_block(pair_act)
        return {"msa": msa_act, "pair": pair_act}
