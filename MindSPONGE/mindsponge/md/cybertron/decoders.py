# ============================================================================
# Copyright 2021 The AIMM team at Shenzhen Bay Laboratory & Peking University
#
# People: Yi Isaac Yang, Jun Zhang, Diqing Chen, Yaqiang Zhou, Huiyang Zhang,
#         Yupeng Huang, Yijie Xia, Yao-Kun Lei, Lijiang Yang, Yi Qin Gao
#
# This code is a part of Cybertron-Code package.
#
# The Cybertron-Code is open-source software based on the AI-framework:
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
"""decoders"""

from mindspore import nn

from .blocks import MLP, Dense
from .blocks import PreActResidual
from .blocks import SeqPreActResidual
from .blocks import PreActDense

__all__ = [
    "Decoder",
    "get_decoder",
    "SimpleDecoder",
    "ResidualOutputBlock",
]

_DECODER_ALIAS = dict()


def _decoder_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _DECODER_ALIAS:
            _DECODER_ALIAS[name] = cls

        for alias in aliases:
            if alias not in _DECODER_ALIAS:
                _DECODER_ALIAS[alias] = cls
        return cls
    return alias_reg


class Decoder(nn.Cell):
    """Decoder"""
    def __init__(self, n_in, n_out=1, activation=None, n_layers=1, output=None):
        super().__init__()

        self.name = 'decoder'
        self.n_in = n_in
        self.n_out = n_out
        self.n_layers = n_layers
        self.output = output
        self.activation = activation

    def construct(self, x):
        return self.output(x)


@_decoder_register('halve')
class SimpleDecoder(Decoder):
    """SimpleDecoder"""
    def __init__(self, n_in, n_out, activation, n_layers=1,):
        super().__init__(
            n_in=n_in,
            n_out=n_out,
            activation=activation,
            n_layers=n_layers,
        )

        self.name = 'halve'

        if n_layers > 0:
            n_hiddens = []
            dim = n_in
            for _ in range(n_layers):
                dim = dim // 2
                if dim < n_out:
                    raise ValueError(
                        "The dimension of hidden layer is smaller than output dimension")
                n_hiddens.append(dim)
            self.output = MLP(n_in, n_out, n_hiddens, activation=activation)
        else:
            self.output = Dense(n_in, n_out, activation=activation)

    def __str__(self):
        return 'halve'


@_decoder_register('residual')
class ResidualOutputBlock(Decoder):
    """ResidualOutputBlock"""
    def __init__(self, n_in, n_out, activation, n_layers=1,):
        super().__init__(
            n_in=n_in,
            n_out=n_out,
            activation=activation,
            n_layers=n_layers,
        )

        self.name = 'residual'

        if n_layers == 1:
            output_residual = PreActResidual(n_in, activation=activation)
        else:
            output_residual = SeqPreActResidual(
                n_in, activation=activation, n_res=n_layers)

        self.output = nn.SequentialCell([
            output_residual,
            PreActDense(n_in, n_out, activation=activation),
        ])

    def __str__(self):
        return 'residual'


def get_decoder(obj, n_in, n_out, activation=None, n_layers=1,):
    """get_decoder"""
    if obj is None or isinstance(obj, Decoder):
        return obj
    if isinstance(obj, str):
        if obj.lower() not in _DECODER_ALIAS.keys():
            raise ValueError(
                "The class corresponding to '{}' was not found.".format(obj))
        return _DECODER_ALIAS[obj.lower()](
            n_in=n_in,
            n_out=n_out,
            activation=activation,
            n_layers=n_layers,
        )
    raise TypeError("Unsupported init type '{}'.".format(type(obj)))
