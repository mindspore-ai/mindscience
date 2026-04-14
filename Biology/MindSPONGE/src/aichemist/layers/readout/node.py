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
Readout functions
"""

from typing import Union

import mindspore as ms
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.numpy import count_nonzero
from mindspore.ops import functional as F

from ...configs import Config

from .readout import Readout
from ..aggregator import NodeAggregator
from ..decoder import Decoder
from ...transforms import ScaleShift
from ...configs import Registry as R


class NodeReadout(Readout):
    r"""
    Readout function

    Args:
        dim_output (int):           Output dimension.

        activation (Cell):          Activation function

        decoder (str):              Decoder network for atom representation. Default: 'halve'

        aggregator (str):           Aggregator network for atom representation. Default: 'sum'

        scale (float):              Scale value for output. Default: 1

        shift (float):              Shift value for output. Default: 0

        type_ref (Tensor):          Tensor of shape `(T, Y)`. Data type is float.
                                    Reference value for atom types. Default: ``None``.

        atomwise_scaleshift (bool): To use atomwise scaleshift (True) or graph scaleshift (False).
                                    Default: ``False``.

        axis (int):                 Axis to readout. Default: -2

        n_decoder_layers (list):    number of neurons in each hidden layer of the decoder network.
                                    Default: 1

        energy_unit (str):          Energy unit of output. Default: ``None``.

        hyper_param (dict):         Hyperparameter. Default: ``None``.

    Note:

        B:  Batch size.

        A:  Number of atoms.

        T:  Number of atom types.

        Y:  Output dimension.

    """

    def __init__(self,
                 dim_output: int = 1,
                 dim_node_rep: int = None,
                 dim_edge_rep: int = None,
                 activation: Union[Cell, str] = None,
                 decoder: Union[Decoder, dict, str] = 'halve',
                 aggregator: Union[NodeAggregator, dict, str] = None,
                 scaleshift: ScaleShift = (1, 0, None, True),
                 axis: int = -2,
                 **kwargs,
                 ):
        super().__init__(
            dim_node_rep=dim_node_rep,
            dim_edge_rep=(dim_node_rep if dim_edge_rep is None else dim_edge_rep),
            activation=activation,
            scaleshift=scaleshift,
            axis=axis,
            ndim=1,
            shape=(dim_output,),
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

        self.dim_output = int(dim_output)

        self.decoder = decoder
        if isinstance(decoder, (Decoder, dict)) or self.dim_edge_rep is not None:
            self.decoder = R.build('decoder', self.decoder,
                                   dim_in=self.dim_node_rep, dim_out=self.dim_output,
                                   activation=self.activation)
            if self.decoder is None:
                self.dim_node_rep = None
            else:
                self.dim_node_rep = self.decoder.dim_in

        self.aggregator = R.build('aggregator.node', aggregator, dim_out=self.dim_output, axis=self.axis)

    def set_dimension(self, dim_node_rep: int, dim_edge_rep: int):
        super().set_dimension(dim_node_rep, dim_edge_rep)
        if self.dim_node_rep is not None and isinstance(self.decoder, str):
            self.decoder = R.build('decoder', self.decoder,
                                   dim_in=self.dim_node_rep, dim_out=self.dim_output,
                                   activation=self.activation)

    def print_info(self, num_retraction: int = 0, num_gap: int = 3, char: str = '-'):
        """print the information of readout"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+" Activation function: "+str(self.activation))
        if self.decoder is not None:
            print(ret+gap+" Decoder: "+str(self.decoder.cls_name))
        if self.aggregator is not None:
            print(ret+gap+" Aggregator: "+str(self.aggregator.cls_name))
        print(ret+gap+" Representation dimension: "+str(self.dim_node_rep))
        print(ret+gap+" Readout dimension: "+str(self.dim_output))
        print(ret+gap+f" Reduce axis: {self.axis}")
        print('-'*80)
        return self

    def construct(self,
                  node_rep: Tensor,
                  edge_rep: Tensor,
                  node_emb: Tensor = None,
                  edge_emb: Tensor = None,
                  edge_cutoff: Tensor = None,
                  atom_type: Tensor = None,
                  atom_mask: Tensor = None,
                  distance: Tensor = None,
                  dis_mask: Tensor = None,
                  dis_vec: Tensor = None,
                  bond: Tensor = None,
                  bond_mask: Tensor = None,
                  **kwargs,
                  ) -> Tensor:

        raise NotImplementedError


@R.register('readout.atomwise')
class AtomwiseReadout(NodeReadout):
    r"""
    Readout function

    Args:
        dim_output (int):           Output dimension.

        activation (Cell):          Activation function

        decoder (str):              Decoder network for atom representation. Default: 'halve'

        aggregator (str):           Aggregator network for atom representation. Default: 'sum'

        scale (float):              Scale value for output. Default: 1

        shift (float):              Shift value for output. Default: 0

        type_ref (Tensor):          Tensor of shape `(T, Y)`. Data type is float.
                                    Reference value for atom types. Default: ``None``.

        atomwise_scaleshift (bool): To use atomwise scaleshift (True) or graph scaleshift (False).
                                    Default: ``False``.

        axis (int):                 Axis to readout. Default: -2

        n_decoder_layers (list):    number of neurons in each hidden layer of the decoder network.
                                    Default: 1

        energy_unit (str):          Energy unit of output. Default: ``None``.

        hyper_param (dict):         Hyperparameter. Default: ``None``.

    Note:

        B:  Batch size.

        A:  Number of atoms.

        T:  Number of atom types.

        Y:  Output dimension.

    """

    def __init__(self,
                 dim_output: int = 1,
                 dim_node_rep: int = None,
                 activation: Union[Cell, str] = None,
                 decoder: Union[Decoder, dict, str] = 'halve',
                 aggregator: Union[NodeAggregator, dict, str] = 'sum',
                 scaleshift: ScaleShift = (1, 0, None, True),
                 axis: int = -2,
                 **kwargs,
                 ):
        super().__init__(
            dim_output=dim_output,
            dim_node_rep=dim_node_rep,
            dim_edge_rep=dim_node_rep,
            activation=activation,
            decoder=decoder,
            aggregator=aggregator,
            scaleshift=scaleshift,
            axis=axis,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

    def construct(self,
                  node_rep: Tensor,
                  edge_rep: Tensor,
                  node_emb: Tensor = None,
                  edge_emb: Tensor = None,
                  edge_cutoff: Tensor = None,
                  atom_type: Tensor = None,
                  atom_mask: Tensor = None,
                  distance: Tensor = None,
                  dis_mask: Tensor = None,
                  dis_vec: Tensor = None,
                  bond: Tensor = None,
                  bond_mask: Tensor = None,
                  **kwargs,
                  ) -> Tensor:

        if atom_mask is None:
            num_atoms = node_rep.shape[-2]
        else:
            num_atoms = count_nonzero(atom_mask.astype(ms.int16), axis=-1, keepdims=True)

        y = node_rep
        if self.decoder is not None:
            # The shape changes from (B, A, Y) to (B, A, F)
            y = self.decoder(node_rep)

        if self.aggregator is not None:
            # The shape changes from (B, Y) to (B, A, Y)
            y = self.aggregator(y, atom_mask, num_atoms)

        return y


@R.register('readout.graph')
class GraphReadout(NodeReadout):
    r"""
    Readout function

    Args:
        dim_output (int):           Output dimension.

        activation (Cell):          Activation function

        decoder (str):              Decoder network for atom representation. Default: 'halve'

        aggregator (str):           Aggregator network for atom representation. Default: 'sum'

        scale (float):              Scale value for output. Default: 1

        shift (float):              Shift value for output. Default: 0

        type_ref (Tensor):          Tensor of shape `(T, Y)`. Data type is float.
                                    Reference value for atom types. Default: ``None``.

        atomwise_scaleshift (bool): To use atomwise scaleshift (True) or graph scaleshift (False).
                                    Default: ``False``.

        axis (int):                 Axis to readout. Default: -2

        n_decoder_layers (list):    number of neurons in each hidden layer of the decoder network.
                                    Default: 1

        energy_unit (str):          Energy unit of output. Default: ``None``.

        hyper_param (dict):         Hyperparameter. Default: ``None``.

    Note:

        B:  Batch size.

        A:  Number of atoms.

        T:  Number of atom types.

        Y:  Output dimension.

    """

    def __init__(self,
                 dim_output: int = 1,
                 dim_node_rep: int = None,
                 activation: Union[Cell, str] = None,
                 decoder: Union[Decoder, dict, str] = 'halve',
                 aggregator: Union[NodeAggregator, dict, str] = 'mean',
                 scaleshift: ScaleShift = (1, 0, None, False),
                 axis: int = -2,
                 **kwargs,
                 ):
        super().__init__(
            dim_output=dim_output,
            dim_node_rep=dim_node_rep,
            dim_edge_rep=dim_node_rep,
            activation=activation,
            decoder=decoder,
            aggregator=aggregator,
            scaleshift=scaleshift,
            axis=axis,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

        if aggregator is None:
            raise ValueError('The aggreator cannot be None under Graph mode!')

        self.shift_by_atoms = False

    def construct(self,
                  node_rep: Tensor,
                  edge_rep: Tensor,
                  node_emb: Tensor = None,
                  edge_emb: Tensor = None,
                  edge_cutoff: Tensor = None,
                  atom_type: Tensor = None,
                  atom_mask: Tensor = None,
                  distance: Tensor = None,
                  dis_mask: Tensor = None,
                  dis_vec: Tensor = None,
                  bond: Tensor = None,
                  bond_mask: Tensor = None,
                  **kwargs,
                  ) -> Tensor:

        if atom_mask is None:
            num_atoms = node_rep.shape[-2]
        else:
            num_atoms = count_nonzero(F.cast(atom_mask, ms.int16), axis=-1, keepdims=True)

        y = self.aggregator(node_rep, atom_mask, num_atoms)

        if self.decoder is not None:
            y = self.decoder(y)

        return y
