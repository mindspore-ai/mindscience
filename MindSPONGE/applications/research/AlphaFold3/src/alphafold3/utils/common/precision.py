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

"""Precision classes and utilities."""

import enum
import mindspore as ms


@enum.unique
class DotPrecision(enum.Enum):
    """Precision for `dot` operation.

    Naming scheme: {OPERAND_DTYPE}_{ACCUMULATOR_DTYPE}[_{NUM_PASSES}x]
    """

    BF16_F32 = "bf16_f32"

    # NPU only precisions.
    F32_F32 = "f32_f32"  # Full f32 precision (doesn't use TensorCores).
    F16_F16 = "f16_f16"
    F16_F32 = "f16_f32"

    @property
    def operand_dtype(self) -> ms.dtype:
        match self:
            case DotPrecision.BF16_F32:
                return ms.bfloat16
            case DotPrecision.F16_F16 | DotPrecision.F16_F32:
                return ms.float16
            case _:
                return ms.float32

    @property
    def accumulator_dtype(self) -> ms.dtype:
        return ms.float16 if (self == DotPrecision.F16_F16) else ms.float32


_MS_NPU_PRECISION_MAP = {
    (ms.float16, "DEFAULT"): DotPrecision.F16_F32,
    (ms.bfloat16, "DEFAULT"): DotPrecision.BF16_F32,
    (ms.float32, "DEFAULT"): DotPrecision.F32_F32,
    (ms.float32, "HIGH"): DotPrecision.F32_F32,
    (ms.float32, "HIGHEST"): DotPrecision.F32_F32,
}

_MS_CPU_PRECISION_MAP = {
    (ms.float16, "DEFAULT"): DotPrecision.F16_F32,
    (ms.bfloat16, "DEFAULT"): DotPrecision.F32_F32,
    (ms.float32, "DEFAULT"): DotPrecision.F32_F32,
    (ms.float32, "HIGH"): DotPrecision.F32_F32,
    (ms.float32, "HIGHEST"): DotPrecision.F32_F32,
}


def _create_ms_precision_map():
    precision_map = {}
    for (dtype, ms_precision), dot_precision in _MS_NPU_PRECISION_MAP.items():
        precision_map[("ascend", dtype, ms_precision)] = dot_precision
    for (dtype, ms_precision), dot_precision in _MS_CPU_PRECISION_MAP.items():
        precision_map[("cpu", dtype, ms_precision)] = dot_precision
    return precision_map


_MS_PRECISION_MAP = _create_ms_precision_map()


def get_equivalent_dot_precision(
    a_dtype: ms.dtype, b_dtype: ms.dtype, ms_precision: str
) -> DotPrecision:
    """Returns `DotPrecision` replicating default behaviour."""
    if a_dtype != b_dtype:
        raise ValueError("Cannot infer precision if operand types differ.")

    backend = ms.context.get_context("device_target").lower()
    if (ms_precision != "DEFAULT") and (a_dtype != ms.float32):
        raise ValueError(
            "`Precision` values other than `DEFAULT` only have an effect if"
            " the operand type is `float32`."
        )
    return _MS_PRECISION_MAP[(backend, a_dtype, ms_precision)]
