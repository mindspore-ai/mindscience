# Copyright 2025 Huawei Technologies Co., Ltd
#
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Public API for gated linear unit functions."""

import typing
from typing import Literal, TypeAlias
from alphafold3.utils.gated_linear_unit import gated_linear_unit_base

Implementation: TypeAlias = Literal['ms']


def gated_linear_unit(x, weight, *, activation, precision, implementation=None):
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
      implementation: if `None` (default), an implementation is automatically
        chosen. 'ms' will use standard MS and work on any platform.

    Raises:
      ValueError: if the arguments are invalid.

    Returns:
      The output array.
    """

    if x.dtype != weight.dtype:
        raise ValueError(
            f'Input and weight must have the same dtype. {x.dtype} !='
            f' {weight.dtype}'
        )

    if implementation is not None:
        named_args = typing.get_args(Implementation)
        if implementation not in named_args:
            raise ValueError(
                f'Unsupported named implementation. Must be one of {named_args}.'
            )

    return gated_linear_unit_base.gated_linear_unit_ms(
        x=x,
        weight=weight,
        activation=activation,
        precision=precision,
    )
