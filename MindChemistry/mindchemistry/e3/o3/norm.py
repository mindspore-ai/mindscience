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
"""norm"""
from mindspore import nn, ops, float32

from .irreps import Irreps
from .tensor_product import TensorProduct


class Norm(nn.Cell):
    r"""
    Norm of each irrep in a direct sum of irreps.

    Args:
        irreps_in (Union[str, Irrep, Irreps]): Irreps for the input.
        squared (bool): whether to return the squared norm. Default: False.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
        >>> n = Norm('3x1o')
        >>> v = ms.Tensor(np.linspace(1., 2., n.irreps_in.dim), dtype=ms.float32)
        >>> n(v).shape
        (1, 3)

    """

    def __init__(self, irreps_in, squared=False, dtype=float32, ncon_dtype=float32):
        super().__init__()

        self.squared = squared
        irreps_in = Irreps(irreps_in).simplify()
        irreps_out = Irreps([(mul, "0e") for mul, _ in irreps_in])

        instr = [(i, i, i, "uuu", False, ir.dim) for i, (mul, ir) in enumerate(irreps_in)]

        self.tp = TensorProduct(irreps_in,
                                irreps_in,
                                irreps_out,
                                instr,
                                irrep_norm="component",
                                dtype=dtype,
                                ncon_dtype=ncon_dtype)

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out.simplify()

    def construct(self, v):
        """Implement the norm-activation function for the input tensor."""
        out = self.tp(v, v)
        if self.squared:
            return out
        return ops.sqrt(ops.relu(out))

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in})"
