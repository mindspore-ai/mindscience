# Copyright 2022 Huawei Technologies Co., Ltd
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
from .tensor_product import TensorProduct


class FullyConnectedTensorProduct(TensorProduct):
    r"""
    Fully-connected weighted tensor product. All the possible path allowed by :math:`|l_1 - l_2| \leq l_{out} \leq l_1 + l_2` are made.

    Equivalent to `TensorProduct` with `instructions='connect'`. For details, see `mindchemistry.e3.TensorProduct`.

    Args:
        irreps_in1 (Union[str, Irrep, Irreps]): Irreps for the first input.
        irreps_in2 (Union[str, Irrep, Irreps]): Irreps for the second input.
        irreps_out (Union[str, Irrep, Irreps]): Irreps for the output.
        irrep_norm (str): {'component', 'norm'}, the assumed normalization of the input and output representations. Default: 'component'. Default: 'component'.
        path_norm (str): {'element', 'path'}, the normalization method of path weights. Default: 'element'.
        weight_init (str): {'zeros', 'ones', 'truncatedNormal', 'normal', 'uniform', 'he_uniform', 'he_normal', 'xavier_uniform'}, the initial method of weights. Default: 'normal'.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
        >>> FullyConnectedTensorProduct('2x1o', '1x1o+3x0e', '5x2e+4x1o')
        TensorProduct [connect] (2x1o x 1x1o+3x0e -> 5x2e+4x1o)

    """

    def __init__(self, irreps_in1, irreps_in2, irreps_out, **kwargs):
        super().__init__(irreps_in1, irreps_in2, irreps_out, instructions='connect', **kwargs)


class FullTensorProduct(TensorProduct):
    r"""
    Full tensor product between two irreps. 

    Equivalent to `TensorProduct` with `instructions='full'`. For details, see `mindchemistry.e3.TensorProduct`.

    Args:
        irreps_in1 (Union[str, Irrep, Irreps]): Irreps for the first input.
        irreps_in2 (Union[str, Irrep, Irreps]): Irreps for the second input.
        filter_ir_out (Union[str, Irrep, Irreps, None]): Filter to select only specific `Irrep` of the output. Default: None.
        irrep_norm (str): {'component', 'norm'}, the assumed normalization of the input and output representations. Default: 'component'. Default: 'component'.
        path_norm (str): {'element', 'path'}, the normalization method of path weights. Default: 'element'.
        weight_init (str): {'zeros', 'ones', 'truncatedNormal', 'normal', 'uniform', 'he_uniform', 'he_normal', 'xavier_uniform'}, the initial method of weights. Default: 'normal'.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
        >>> FullTensorProduct('2x1o+4x0o', '1x1o+3x0e')
        TensorProduct [full] (2x1o+4x0o x 1x1o+3x0e -> 2x0e+12x0o+6x1o+2x1e+4x1e+2x2e)

    """

    def __init__(self, irreps_in1, irreps_in2, filter_ir_out=None, **kwargs):
        super().__init__(irreps_in1, irreps_in2, filter_ir_out, instructions='full', **kwargs)


class ElementwiseTensorProduct(TensorProduct):
    r"""
    Elementwise connected tensor product.

    Equivalent to `TensorProduct` with `instructions='element'`. For details, see `mindchemistry.e3.TensorProduct`.

    Args:
        irreps_in1 (Union[str, Irrep, Irreps]): Irreps for the first input.
        irreps_in2 (Union[str, Irrep, Irreps]): Irreps for the second input.
        filter_ir_out (Union[str, Irrep, Irreps, None]): Filter to select only specific `Irrep` of the output. Default: None.
        irrep_norm (str): {'component', 'norm'}, the assumed normalization of the input and output representations. Default: 'component'. Default: 'component'.
        path_norm (str): {'element', 'path'}, the normalization method of path weights. Default: 'element'.
        weight_init (str): {'zeros', 'ones', 'truncatedNormal', 'normal', 'uniform', 'he_uniform', 'he_normal', 'xavier_uniform'}, the initial method of weights. Default: 'normal'.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
        >>> ElementwiseTensorProduct('2x2e+4x1o', '3x1e+3x0o')
        TensorProduct [element] (2x2e+1x1o+3x1o x 2x1e+1x1e+3x0o -> 2x1e+2x2e+2x3e+1x0o+1x1o+1x2o+3x1e)

    """

    def __init__(self, irreps_in1, irreps_in2, filter_ir_out=None, **kwargs):
        super().__init__(irreps_in1, irreps_in2, filter_ir_out, instructions='element', **kwargs)


class Linear(TensorProduct):
    r"""
    Linear operation equivariant.

    Equivalent to `TensorProduct` with `instructions='linear'`. For details, see `mindchemistry.e3.TensorProduct`.

    Args:
        irreps_in (Union[str, Irrep, Irreps]): Irreps for the input.
        irreps_out (Union[str, Irrep, Irreps]): Irreps for the output.
        irrep_norm (str): {'component', 'norm'}, the assumed normalization of the input and output representations. Default: 'component'. Default: 'component'.
        path_norm (str): {'element', 'path'}, the normalization method of path weights. Default: 'element'.
        weight_init (str): {'zeros', 'ones', 'truncatedNormal', 'normal', 'uniform', 'he_uniform', 'he_normal', 'xavier_uniform'}, the initial method of weights. Default: 'normal'.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
        >>> Linear('2x2e+3x1o+3x0e', '3x2e+5x1o+2x0e')
        TensorProduct [linear] (2x2e+3x1o+3x0e x 1x0e -> 3x2e+5x1o+2x0e)

    """

    def __init__(self, irreps_in, irreps_out, **kwargs):
        super().__init__(irreps_in, None, irreps_out, instructions='linear', **kwargs)


class TensorSquare(TensorProduct):
    r"""
    Compute the square tensor product of a tensor.

    Equivalent to `TensorProduct` with `irreps_in2=None and instructions='full' or 'connect'`. For details, see `mindchemistry.e3.TensorProduct`.

    If `irreps_out` is given, this operation is fully connected.
    If `irreps_out` is not given, the operation has no parameter and is like full tensor product.

    Args:
        irreps_in (Union[str, Irrep, Irreps]): Irreps for the input.
        irreps_out (Union[str, Irrep, Irreps, None]): Irreps for the output. Default: None.
        filter_ir_out (Union[str, Irrep, Irreps, None]): Filter to select only specific `Irrep` of the output. Default: None.
        irrep_norm (str): {'component', 'norm'}, the assumed normalization of the input and output representations. Default: 'component'. Default: 'component'.
        path_norm (str): {'element', 'path'}, the normalization method of path weights. Default: 'element'.
        weight_init (str): {'zeros', 'ones', 'truncatedNormal', 'normal', 'uniform', 'he_uniform', 'he_normal', 'xavier_uniform'}, the initial method of weights. Default: 'normal'.

    Raises:
        ValueError: If both `irreps_out` and `filter_ir_out` are not None.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
        >>> TensorSquare('2x1o', irreps_out='5x2e+4x1e+7x1o')
        TensorProduct [connect] (2x1o x 2x1o -> 5x2e+4x1e)
        >>> TensorSquare('2x1o+3x0e', filter_ir_out='5x2o+4x1e+2x0e')
        TensorProduct [full] (2x1o+3x0e x 2x1o+3x0e -> 4x0e+9x0e+4x1e)

    """

    def __init__(self, irreps_in, irreps_out=None, filter_ir_out=None, **kwargs):
        if irreps_out is None:
            super().__init__(irreps_in, None, filter_ir_out, instructions='full', **kwargs)
        else:
            if filter_ir_out is None:
                super().__init__(irreps_in, None, irreps_out, instructions='connect', **kwargs)
            else:
                raise ValueError("Both `irreps_out` and `filter_ir_out` are not None, this is ambiguous.")
