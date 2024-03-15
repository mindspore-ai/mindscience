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
"""normact"""
from mindspore import nn, Parameter, float32, ops
from mindspore.common.initializer import initializer

from ..o3.irreps import Irreps
from ..o3.tensor_product import TensorProduct
from ..o3.norm import Norm


class NormActivation(nn.Cell):
    r"""Activation function for the norm of irreps.
    Applies a scalar activation to the norm of each irrep and outputs a (normalized) version of that irrep multiplied
      by the scalar output of the scalar activation.

    Args:
        irreps_in (Union[str, Irrep, Irreps]): the input irreps.
        act (Func): an activation function for each part of the norm of `irreps_in`.
        normalize (bool): whether to normalize the input features before multiplying them by the scalars from the
          nonlinearity. Default: True.
        epsilon (float): when ``normalize``ing, norms smaller than ``epsilon`` will be clamped up to ``epsilon``
          to avoid division by zero. Not allowed when `normalize` is False. Default: None.
        bias (bool): whether to apply a learnable additive bias to the inputs of the `act`. Default: False.

    Raises:
        ValueError: If `epsilon` is not None and `normalize` is False.
        ValueError: If `epsilon` is not positive.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
        >>> NormActivation("2x1e", ops.sigmoid, bias=True)
        NormActivation [sigmoid] (2x1e -> 2x1e)
    """

    def __init__(self,
                 irreps_in,
                 act,
                 normalize=True,
                 epsilon=None,
                 bias=False,
                 init_method='zeros',
                 dtype=float32,
                 ncon_dtype=float32):
        super().__init__()

        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_in)

        if epsilon is None and normalize:
            epsilon = 1e-8
        elif epsilon is not None and not normalize:
            raise ValueError("`epsilon` and `normalize = False` don't make sense together.")
        elif not epsilon > 0:
            raise ValueError(f"epsilon {epsilon} is invalid, must be strictly positive.")
        self.epsilon = epsilon
        if self.epsilon is not None:
            self._eps_squared = epsilon * epsilon
        else:
            self._eps_squared = 0.0

        self.norm = Norm(irreps_in, squared=(epsilon is not None), dtype=dtype)
        self.act = act
        self.normalize = normalize
        if bias:
            self.bias = Parameter(initializer(init_method, (self.irreps_in.num_irreps,), dtype),
                                  name=self.__class__.__name__)
        else:
            self.bias = None

        self.scalar_multiplier = TensorProduct(irreps_in1=self.norm.irreps_out,
                                               irreps_in2=irreps_in,
                                               instructions='element',
                                               dtype=dtype,
                                               ncon_dtype=ncon_dtype)

    def construct(self, v):
        """Implement the norm-activation function for the input tensor."""
        norms = self.norm(v)
        if self._eps_squared > 0:
            norms[norms < self._eps_squared] = self._eps_squared
            norms = ops.sqrt(norms)

        nonlin_arg = norms
        if self.bias is not None:
            nonlin_arg = nonlin_arg + self.bias

        scalings = self.act(nonlin_arg)
        if self.normalize:
            scalings = scalings / norms

        return self.scalar_multiplier(scalings, v)

    def __repr__(self):
        return f"{self.__class__.__name__} [{self.act.__name__}] ({self.irreps_in} -> {self.irreps_in})"
