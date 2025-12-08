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

"""Triangle"""
import numpy as np
import mindspore as ms
from mindspore import nn, ops
import mindspore.common.dtype as mstype
from mindspore import Parameter, mint
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from alphafold3.utils.gated_linear_unit import gated_linear_unit
from alphafold3.model.components.base_modules import LayerNorm, CustomDense
from mindscience.common.memory_reduce import _memory_reduce
from mindscience.common.initializer import lecun_init
from mindscience.models.layers.mask import MaskedLayerNorm
from mindscience.e3nn.utils import Ncon


class TriangleMultiplication(nn.Cell):
    r"""
    Triangle multiplication layer. for the detailed implementation process, refer to
    `TriangleMultiplication <https://www.nature.com/articles/s41586-021-03819-2>`_.

    The information between the amino acid pair is integrated through the information of three edges ij, ik, jk, and
    the result of the dot product between ik and jk is added to the edge of ij.

    Args:
        num_intermediate_channel (float):   The number of intermediate channel.
        equation (str):                     The equation used in triangle multiplication layer. edge update forms
                                            corresponding to 'incoming' and 'outgoing',
                                            :math:`(ikc,jkc->ijc, kjc,kic->ijc)`.
        layer_norm_dim (int):               The last dimension length of the layer norm.
        batch_size (int):                   The batch size of parameters in triangle multiplication. Default: ``None``.

    Inputs:
        - **pair_act** (Tensor) - Tensor of pair_act. shape :math:`(N{res}, N{res}, layer\_norm\_dim)`.
        - **pair_mask** (Tensor) - The mask for TriangleAttention matrix with shape. shape :math:`(N{res}, N{res})`.
        - **index** (Tensor) - The index of while loop, only used in case of while control
          flow.

    Outputs:
        Tensor, the float tensor of the pair_act of the layer with shape :math:`(N{res}, N{res}, layer\_norm\_dim)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, config, global_config, num_intermediate_channel, equation, normalized_shape,
                 batch_size=None, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.num_intermediate_channel = num_intermediate_channel
        self.left_norm_input = LayerNorm(normalized_shape, dtype=ms.float32)
        self.center_norm = LayerNorm(normalized_shape, dtype=ms.float32)
        self.projection = nn.Dense(
            normalized_shape[-1], num_intermediate_channel * 2, has_bias=False, dtype=dtype)
        self.gate = nn.Dense(normalized_shape[-1], num_intermediate_channel * 2,
                             weight_init=self.global_config.final_init, has_bias=False, dtype=dtype)
        self.output_projection = CustomDense(
            normalized_shape[-1], num_intermediate_channel, weight_init=self.global_config.final_init,
            ndim=3, dtype=dtype)
        self.gating_linear = CustomDense(
            num_intermediate_channel, num_intermediate_channel, weight_init=self.global_config.final_init,
            ndim=3, dtype=dtype)
        self.weight_glu = mint.stack(
            [self.gate.weight.T, self.projection.weight.T], dim=1)
        if self.config.equation == "ikc,jkc->ijc":
            ncon_list = [[-1, -2, 1], [-1, -3, 1]]
        elif self.config.equation == "kjc,kic->ijc":
            ncon_list = [[-1, 1, -3], [-1, 1, -2]]
        else:
            raise ValueError("Not support this equation.")
        self.ncon = Ncon(ncon_list)

    def construct(self, act, mask, use_glu=True):
        r"""
        Builds triangle multiplication module.

        Args:
            act(Tensor):     Pair activations. Data type is float.
            mask(Tensor):    Pair mask. Data type is float.

        Returns:
            act(Tensor), the shape is same as act_shape[:-1].
        """
        self.weight_glu = mint.stack(
            [self.gate.weight.T, self.projection.weight.T], dim=1)

        mask = mask[None, ...]
        act = self.left_norm_input(act)
        input_act = act

        if use_glu is True:
            projection = gated_linear_unit(
                x=act,
                weight=self.weight_glu,
                activation=ms.mint.sigmoid,
            )
            projection = ops.transpose(projection, (2, 0, 1))
            projection *= mask
        else:
            projection = self.projection(act)
            projection = ops.transpose(projection, (2, 0, 1))
            projection *= mask
            gate = self.gate(act)
            gate = ops.transpose(gate, (2, 0, 1))
            projection *= ms.mint.sigmoid(gate)
        projection = projection.reshape(
            self.num_intermediate_channel, 2, *projection.shape[1:])
        a, b = projection[:, 0], projection[:, 1]
        act = self.ncon([a, b])
        act = self.center_norm(act.transpose((1, 2, 0)))
        act = self.output_projection(act)
        gate_out = self.gating_linear(input_act)
        act *= mint.sigmoid(gate_out)
        return act


class OuterProductMean(nn.Cell):
    r"""
    Computing the correlation of the input tensor along its second dimension, the computed correlation
    could be used to update the correlation features(e.g. the Pair representation).

    .. math::
        OuterProductMean(\mathbf{act}) = Linear(flatten(mean(\mathbf{act}\otimes\mathbf{act})))

    Args:
        num_outer_channel (float):  The last dimension size of intermediate layer in OuterProductMean.
        act_dim (int):              The last dimension size of the input act.
        num_output_channel (int):   The last dimension size of output.
        batch_size(int):            The batch size of parameters in OuterProductMean,
                                    used in while control flow. Default: "None".
        slice_num (int):            The slice num used in OuterProductMean layer
                                    when the memory is overflow. Default: 0.

    Inputs:
        - **act** (Tensor) - The input tensor with shape :math:`(dim_1, dim_2, act\_dim)`.
        - **mask** (Tensor) - The mask for OuterProductMean with shape :math:`(dim_1, dim_2)`.
        - **mask_norm** (Tensor) - Squared L2-norm along the first dimension of **mask**,
          pre-computed to avoid re-computing, its shape is :math:`(dim_2, dim_2, 1)`.
        - **index** (Tensor) - The index of while loop, only used in case of while control
          flow. Default: "None".

    Outputs:
        Tensor, the float tensor of the output of OuterProductMean layer with
        shape :math:`(dim_2, dim_2, num\_output\_channel)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, num_outer_channel, act_dim, num_output_channel, batch_size=None, slice_num=0, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.num_output_channel = num_output_channel
        self.num_outer_channel = num_outer_channel
        self.layer_norm_input = MaskedLayerNorm()
        self.matmul_trans_b = P.MatMul(transpose_b=True)
        self.matmul = P.MatMul()
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.idx = Tensor(0, mstype.int32)
        self._init_parameter()

    def construct(self, act, mask, mask_norm, index=None):
        """Compute outer product mean."""
        mask = P.ExpandDims()(mask, -1)
        act = self.layer_norm_input(
            act, self.layer_norm_input_gamma, self.layer_norm_input_beta)
        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        out_shape = act_shape[:-1] + (-1,)
        left_act = mask * P.Reshape()(
            P.BiasAdd()(self.matmul_trans_b(act, self.left_projection_weight), self.left_projection_bias), out_shape)
        right_act = mask * P.Reshape()(
            P.BiasAdd()(self.matmul_trans_b(act, self.right_projection_weight), self.right_projection_bias), out_shape)
        _, d, e = right_act.shape
        batched_inputs = (left_act,)
        nonbatched_inputs = (right_act, self.linear_output_weight,
                             self.o_biases, d, e)
        act = _memory_reduce(self._compute, batched_inputs,
                             nonbatched_inputs, self.slice_num, 1)
        epsilon = 1e-3
        act = P.RealDiv()(act, epsilon + mask_norm)
        return act

    def _init_parameter(self):
        '''init parameter'''
        self.layer_norm_input_gamma = Parameter(
            Tensor(np.ones((self.act_dim)), self.dtype))
        self.layer_norm_input_beta = Parameter(
            Tensor(np.zeros((self.act_dim)), self.dtype))
        self.left_projection_weight = Parameter(
            initializer(lecun_init(self.act_dim), [self.num_outer_channel, self.act_dim], self.dtype))
        self.left_projection_bias = Tensor(
            np.zeros((self.num_outer_channel)), self.dtype)
        self.right_projection_weight = Parameter(
            initializer(lecun_init(self.act_dim), [self.num_outer_channel, self.act_dim], self.dtype))
        self.right_projection_bias = Tensor(
            np.zeros((self.num_outer_channel)), self.dtype)
        self.linear_output_weight = Parameter(
            Tensor(np.zeros((self.num_outer_channel, self.num_outer_channel, self.num_output_channel)),
                   self.dtype))
        self.o_biases = Parameter(
            Tensor(np.zeros((self.num_output_channel)), self.dtype))

    def _compute(self, left_act, right_act, linear_output_weight, linear_output_bias, d, e):
        '''compute outer product mean'''

        left_act = left_act.transpose((0, 2, 1))
        act = Ncon([[1, -2, -4], [1, -1, -3]])([left_act, right_act])
        act = Ncon([[-1, 1, 2, -2], [1, 2, -3]]
                   )([act, linear_output_weight]) + linear_output_bias
        act = P.Transpose()(act, (1, 0, 2))
        return act
