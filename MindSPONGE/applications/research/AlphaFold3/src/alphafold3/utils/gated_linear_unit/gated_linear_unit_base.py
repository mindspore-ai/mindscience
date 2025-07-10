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

"""Common types for gated linear unit kernels."""
import abc
import mindspore as ms
from mindspore import mint


class GatedLinearUnit(abc.ABC):
    """Gated linear unit."""

    def __call__(self, x, weight, *, activation, precision, **kwargs):
        """Applies a gated linear unit (https://arxiv.org/abs/1612.08083).

        Computes `activation(x @ weight[:, 0]) * x @ weight[:, 1]`.

        This is SwiGLU when `activation=swish`, GEGLU when
        `activation=gelu`, REGLU when `activation=relu`, and GLU when
        `activation=sigmoid` (https://arxiv.org/abs/2002.05202).

        Args:
            x: the input array.
            weight: the combined weight array.
            activation: optional activation function.
            precision: specifies the matrix multiplication precision. Either `None`
            (default), which means the default precision for the backend, or an
            enum of "DEFAULT/HIGH/...".

        Returns:
            The output array.
        """

        return self._fwd(
            x, weight, activation=activation, precision=precision, **kwargs
        )

    @abc.abstractmethod
    def _fwd(self, x, weight, *, activation, precision):
        """Gated linear unit."""
        ...


def gated_linear_unit_ms(x, weight, *, activation, precision=None):
    """Applies a gated linear unit (https://arxiv.org/abs/1612.08083).

    Computes `activation(x @ weight[:, 0]) * x @ weight[:, 1]`.

    This is SwiGLU when `activation=swish`, GEGLU when
    `activation=gelu`, REGLU when `activation=relu`, and GLU when
    `activation=sigmoid` (https://arxiv.org/abs/2002.05202).

    Args:
        x: the input array.
        weight: the combined weight array.
        activation: optional activation function.
        precision: specifies the matrix multiplication precision. Either `None`
        (default), which means the default precision for the backend, or an
        enum of "DEFAULT/HIGH/...".

    Returns:
        The output array.
    """

    weight_reshaped = mint.reshape(
        weight, (-1, weight.shape[-2] * weight.shape[-1]))
    # y = ops.dot(x.astype('float32'), weight_reshaped.astype('float32'))
    y1 = mint.matmul(x, weight_reshaped)
    y = y1.astype(ms.float32)
    a, b = y.split(y.shape[-1] // 2, axis=-1)
    out = mint.mul(a, b) if activation is None else mint.mul(activation(a), b)
    out = out.astype(x.dtype)

    return out
