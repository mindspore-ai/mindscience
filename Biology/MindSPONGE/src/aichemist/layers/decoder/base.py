# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
Decoder networks for readout function
"""

from typing import Union

from mindspore import Tensor
from mindspore.nn import Cell
from ...configs import Registry as R


class Decoder(Cell):
    r"""Decoder network to reduce the dimension of representation

    Args:
        dim_in (int): Input dimension.

        dim_out (int): Output dimension. Default: 1

        activation (Union[Cell, str]): Activation function. Default: ``None``.

        n_layers (int): Number of hidden layers. Default: 1

    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int = 1,
                 activation: Union[Cell, str] = None,
                 n_layers: int = 1,
                 **kwargs
                 ):

        super().__init__()
        self._kwargs = kwargs

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_layers = n_layers

        self.output: Cell = None
        self.activation = R.build('activation', activation)

    def __str__(self):
        return 'Decoder<>'

    def construct(self, x: Tensor):
        # pylint: disable=not-callable
        return self.output(x)
