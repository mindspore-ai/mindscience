# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of Cybertron package.
#
# The Cybertron is open-source software based on the AI-framework:
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

from mindspore import nn
from mindspore import Tensor
from mindspore.nn import Cell

from .block import MLP, Dense
from .block import PreActResidual
from .block import SeqPreActResidual
from .block import PreActDense

__all__ = [
    "Decoder",
    "get_decoder",
    "HalveDecoder",
    "ResidualOutputBlock",
]

_DECODER_BY_KEY = dict()


def _decoder_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _DECODER_BY_KEY:
            _DECODER_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _DECODER_BY_KEY:
                _DECODER_BY_KEY[alias] = cls
        return cls
    return alias_reg


class Decoder(Cell):
    r"""Decoder network to reduce the dimension of representation

    Args:

        n_in (int):         Input dimension.

        n_out (int):        Output dimension. Default: 1

        activation (Cell):  Activation function. Default: None

        n_layers (int):     Number of hidden layers. Default: 1

    """
    def __init__(self,
                 n_in: int,
                 n_out: int = 1,
                 activation: Cell = None,
                 n_layers: int = 1,
                 ):

        super().__init__()

        self.reg_key = 'none'
        self.name = 'decoder'

        self.n_in = n_in
        self.n_out = n_out
        self.n_layers = n_layers

        self.output = None
        self.activation = activation

    def construct(self, x: Tensor):
        #pylint: disable=not-callable
        return self.output(x)


@_decoder_register('halve')
class HalveDecoder(Decoder):
    r"""A MLP decoder with halve number of layers.

    Args:

        n_in (int):         Input dimension.

        n_out (int):        Output dimension. Default: 1

        activation (Cell):  Activation function. Default: None

        n_layers (int):     Number of hidden layers. Default: 1

    """
    def __init__(self,
                 n_in: int,
                 n_out: int = 1,
                 activation: Cell = None,
                 n_layers: int = 1,
                 ):

        super().__init__(
            n_in=n_in,
            n_out=n_out,
            activation=activation,
            n_layers=n_layers,
        )

        self.reg_key = 'halve'
        self.name = 'halve'

        if self.n_layers > 0:
            n_hiddens = []
            dim = self.n_in
            for _ in range(self.n_layers):
                dim = dim // 2
                if dim < n_out:
                    raise ValueError(
                        "The dimension of hidden layer is smaller than output dimension")
                n_hiddens.append(dim)
            self.output = MLP(self.n_in, self.n_out, n_hiddens, activation=self.activation)
        else:
            self.output = Dense(self.n_in, self.n_out, activation=self.activation)

    def __str__(self):
        return 'halve'


@_decoder_register('residual')
class ResidualOutputBlock(Decoder):
    r"""Residual block type decoder

    Args:

        n_in (int):         Input dimension.

        n_out (int):        Output dimension. Default: 1

        activation (Cell):  Activation function. Default: None

        n_layers (int):     Number of hidden layers. Default: 1

    """
    def __init__(self,
                 n_in: int,
                 n_out: int = 1,
                 activation: Cell = None,
                 n_layers: int = 1,
                 ):

        super().__init__(
            n_in=n_in,
            n_out=n_out,
            activation=activation,
            n_layers=n_layers,
        )

        self.reg_key = 'residual'
        self.name = 'residual'

        if self.n_layers == 1:
            output_residual = PreActResidual(self.n_in, activation=self.activation)
        else:
            output_residual = SeqPreActResidual(
                self.n_in, activation=self.activation, n_res=self.n_layers)

        self.output = nn.SequentialCell([
            output_residual,
            PreActDense(self.n_in, self.n_out, activation=self.activation),
        ])

    def __str__(self):
        return 'residual'


_DECODER_BY_NAME = {
    decoder.__name__: decoder for decoder in _DECODER_BY_KEY.values()}


def get_decoder(decoder: str,
                n_in: int,
                n_out: int,
                activation: Cell = None,
                n_layers: int = 1,
                ) -> Decoder:
    """get decoder by name"""
    if decoder is None or isinstance(decoder, Decoder):
        return decoder

    if isinstance(decoder, str):
        if decoder.lower() == 'none':
            return None
        if decoder.lower() in _DECODER_BY_KEY.keys():
            return _DECODER_BY_KEY[decoder.lower()](
                n_in=n_in,
                n_out=n_out,
                activation=activation,
                n_layers=n_layers,
            )
        if decoder in _DECODER_BY_NAME.keys():
            return _DECODER_BY_NAME[decoder](
                n_in=n_in,
                n_out=n_out,
                activation=activation,
                n_layers=n_layers,
            )

        raise ValueError(
            "The Decoder corresponding to '{}' was not found.".format(decoder))

    raise TypeError("Unsupported init type '{}'.".format(type(decoder)))
