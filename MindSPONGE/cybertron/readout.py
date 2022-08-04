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
Readout functions
"""

import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from mindsponge.data import get_class_parameters, get_hyper_string, get_hyper_parameter
from mindsponge.data import set_class_into_hyper_param, set_hyper_parameter
from mindsponge.function import Units
from mindsponge.function import get_integer

from .activation import get_activation
from .aggregator import Aggregator, get_aggregator
from .decoder import Decoder, get_decoder
from .model import MolecularModel

__all__ = [
    "AtomwiseReadout",
    "GraphReadout",
]


class Readout(nn.Cell):
    r"""
    Readout function

    Args:
        model (MolecularModel):     Molecular model.

        dim_output (int):           Output dimension.

        activation (Cell):          Activation function

        decoder (str):              Decoder network for atom representation. Default: 'halve'

        aggregator (str):           Aggregator network for atom representation. Default: 'sum'

        scale (float):              Scale value for output. Default: 1

        shift (float):              Shift value for output. Default: 0

        type_ref (Tensor):          Tensor of shape (B, T, Y). Data type is float.
                                    Reference value for atom types. Default: None

        atomwise_scaleshift (bool): To use atomwise scaleshift (True) or graph scaleshift (False).
                                    Default: False

        axis (int):                 Axis to readout. Default: -2

        n_decoder_layers (list):    number of neurons in each hidden layer of the decoder network.
                                    Default: 1

        energy_unit (str):          Energy unit of output. Default: None

        hyper_param (dict):         Hyperparameter. Default: None

    Symbols:

        B:  Batch size.

        A:  Number of atoms.

        T:  Number of atom types.

        Y:  Output dimension.

    """

    def __init__(self,
                 model: MolecularModel = None,
                 dim_output: int = 1,
                 activation: Cell = None,
                 decoder: Decoder = None,
                 aggregator: Aggregator = None,
                 scale: float = 1,
                 shift: float = 0,
                 type_ref: Tensor = None,
                 atomwise_scaleshift: bool = False,
                 axis: int = -2,
                 n_decoder_layers: int = 1,
                 energy_unit: str = None,
                 hyper_param: dict = None,
                 ):

        super().__init__()

        dim_represent = None
        if hyper_param is not None:
            dim_represent = get_hyper_parameter(hyper_param, 'dim_represent')
            dim_output = get_hyper_parameter(hyper_param, 'dim_output')
            activation = get_class_parameters(hyper_param, 'activation')
            decoder = get_class_parameters(hyper_param, 'decoder')
            aggregator = get_class_parameters(hyper_param, 'aggregator')
            scale = get_hyper_parameter(hyper_param, 'scale')
            shift = get_hyper_parameter(hyper_param, 'shift')
            type_ref = get_hyper_parameter(hyper_param, 'type_ref')
            atomwise_scaleshift = get_hyper_parameter(
                hyper_param, 'atomwise_scaleshift')
            axis = get_hyper_parameter(hyper_param, 'axis')
            n_decoder_layers = get_hyper_parameter(
                hyper_param, 'n_decoder_layers')
            energy_unit = get_hyper_string(hyper_param, 'energy_unit')

        self.units = Units(energy_unit=energy_unit)
        self.energy_unit = energy_unit
        self.dim_represent = dim_represent
        self.dim_output = get_integer(dim_output)

        if model is not None:
            self.dim_represent = model.dim_feature
            self.activation = model.activation
        if activation is not None:
            self.activation = get_activation(activation)
        self.activation_name = 'none' if self.activation is None else self.activation.cls_name

        self.n_decoder_layers = get_integer(n_decoder_layers)
        self.decoder = get_decoder(
            decoder, self.dim_represent, self.dim_output, self.activation, self.n_decoder_layers)
        self.decoder_name = 'none' if self.decoder is None else self.decoder.cls_name

        self.aggregator = get_aggregator(aggregator, self.dim_output, axis)
        self.aggregator_name = 'none' if self.aggregator is None else self.aggregator.cls_name

        self.scale = Tensor(scale, ms.float32)
        self.shift = Tensor(shift, ms.float32)
        self.axis = get_integer(axis)

        self.atomwise_scaleshift = Tensor(atomwise_scaleshift, ms.bool_)
        self.type_ref = None if type_ref is None else Tensor(
            type_ref, ms.float32)

        if self.decoder is None and self.dim_represent != self.dim_output:
            raise ValueError("When decoder is None, dim_output ("+str(dim_output) +
                             ") must be equal to dim_represent ("+str(self.dim_represent)+")")
        self.reduce_sum = P.ReduceSum()

        self.hyper_param = dict()
        self.hyper_types = {
            'dim_represent': 'int',
            'dim_output': 'int',
            'activation': 'Cell',
            'decoder': 'Cell',
            'aggregator': 'Cell',
            'scale': 'float',
            'shift': 'float',
            'type_ref': 'Tensor',
            'atomwise_scaleshift': 'bool',
            'axis': 'int',
            'n_decoder_layers': 'int',
            'energy_unit': 'str',
        }

    def set_hyper_param(self):
        """set hyperparameters"""
        set_hyper_parameter(self.hyper_param, 'name', self.cls_name)
        set_class_into_hyper_param(self.hyper_param, self.hyper_types, self)
        return self

    def set_scaleshift(self,
                       scale: float = 1,
                       shift: float = 0,
                       type_ref: Tensor = None,
                       atomwise_scaleshift: bool = None,
                       unit: str = None
                       ):

        """set scale and shift"""
        if unit is not None:
            self.units.set_energy_unit(unit)
            set_hyper_parameter(self.hyper_param, 'energy_unit',
                                self.units.energy_unit)
        self.scale = Tensor(scale, ms.float32).reshape(-1)
        if self.scale.shape[-1] != self.dim_output and self.scale.shape[-1] != 1:
            raise ValueError('The dimension of "scale" ('+str(self.scale.shape[-1]) +
                             ') does not match the output dimension ('+str(self.dim_output)+').')
        self.shift = Tensor(shift, ms.float32).reshape(-1)
        if self.shift.shape[-1] != self.dim_output and self.shift.shape[-1] != 1:
            raise ValueError('The dimension of "shift" ('+str(self.shift.shape[-1]) +
                             ') does not match the output dimension ('+str(self.dim_output)+').')
        if type_ref is not None:
            self.type_ref = Tensor(type_ref, ms.float32)
            if self.type_ref.shape[-1] != self.dim_output and self.type_ref.shape[-1] != 1:
                raise ValueError('The dimension of "type_ref" ('+str(self.type_ref.shape[-1]) +
                                 ') does not match the output dimension ('+str(self.dim_output)+').')
        set_hyper_parameter(self.hyper_param, 'scale', self.scale)
        set_hyper_parameter(self.hyper_param, 'shift', self.shift)
        set_hyper_parameter(self.hyper_param, 'type_ref', self.type_ref)
        if atomwise_scaleshift is not None:
            self.atomwise_scaleshift = Tensor(atomwise_scaleshift, ms.bool_)
            if self.atomwise_scaleshift.size != 1:
                raise ValueError(
                    'The size of "atomwise_scaleshift" must be 1!')
            set_hyper_parameter(
                self.hyper_param, 'atomwise_scaleshift', self.atomwise_scaleshift)
        return self

    def change_unit(self, units):
        """change units"""
        scale = self.units.convert_energy_to(units)
        self.scale *= scale
        self.shift *= scale
        if self.type_ref is not None:
            self.type_ref *= scale
        set_hyper_parameter(self.hyper_param, 'scale', self.scale)
        set_hyper_parameter(self.hyper_param, 'shift', self.shift)
        set_hyper_parameter(self.hyper_param, 'type_ref', self.type_ref)
        set_hyper_parameter(self.hyper_param, 'energy_unit',
                            self.units.energy_unit)
        return self

    def print_info(self, num_retraction: int = 0, num_gap: int = 3, char: str = '-'):
        """print the information of readout"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+" Activation function: "+str(self.activation.cls_name))
        if self.decoder is not None:
            print(ret+gap+" Decoder: "+str(self.decoder.cls_name))
        if self.aggregator is not None:
            print(ret+gap+" Aggregator: "+str(self.aggregator.cls_name))
        print(ret+gap+" Representation dimension: "+str(self.dim_represent))
        print(ret+gap+" Readout dimension: "+str(self.dim_output))
        print(ret+gap+" Scale: "+str(self.scale.asnumpy()))
        print(ret+gap+" Shift: "+str(self.shift.asnumpy()))
        print(ret+gap+" Scaleshift mode: " +
              ("Atomwise" if self.atomwise_scaleshift else "Graph"))
        if self.type_ref is None:
            print(ret+gap+" Reference value for atom types: None")
        else:
            print(ret+gap+" Reference value for atom types:")
            for i, ref in enumerate(self.type_ref):
                print(ret+gap+gap+' No.{: <5}'.format(str(i)+': ')+str(ref))
        print(ret+gap+" Output unit: "+str(self.units.energy_unit_name))
        print(ret+gap+" Reduce axis: "+str(self.axis))
        print('-'*80)

    def construct(self,
                  x: Tensor,
                  xlist: list = None,
                  atoms_types: Tensor = None,
                  atom_mask: Tensor = None,
                  num_atoms: Tensor = None
                  ):
        r"""Compute readout network.

        Args:
            x (Tensor):             Tensor of shape (B, A, F). Data type is float.
                                    Representation of atoms.
            xlist (list):           List of atom representations.
                                    Default: None
            atoms_types (Tensor):   Tensor of shape (B, A). Data type is int.
                                    Index of atom types. Default: None
            atom_mask (Tensor):     Tensor of shape (B, A). Data type is bool
                                    Mask for atom types
            num_atoms (Tensor):     Tensor of shape (B, A). Data type is int
                                    Number of atoms for each batch.

        Returns:
            y: (Tensor):    Tensor of shape (B, A, Y). Data type is float

        Symbols:

            B:  Batch size.
            A:  Number of atoms in system.
            F:  Feature dimension of representation.
            Y:  Output dimension.

        """
        raise NotImplementedError


class AtomwiseReadout(Readout):
    r"""
    Atomwise readout function

    Args:
        model (MolecularModel):     Molecular model.

        dim_output (int):           Output dimension.

        activation (Cell):          Activation function

        decoder (str):              Decoder network for atom representation. Default: 'halve'

        aggregator (str):           Aggregator network for atom representation. Default: 'sum'

        scale (float):              Scale value for output. Default: 1

        shift (float):              Shift value for output. Default: 0

        type_ref (Tensor):          Tensor of shape (B, T, Y). Data type is float.
                                    Reference value for atom types. Default: None

        atomwise_scaleshift (bool): To use atomwise scaleshift (True) or graph scaleshift (False).
                                    Default: False

        axis (int):                 Axis to readout. Default: -2

        n_decoder_layers (list):    number of neurons in each hidden layer of the decoder network.
                                    Default: 1

        energy_unit (str):          Energy unit of output. Default: None

        hyper_param (dict):         Hyperparameter. Default: None

    Symbols:

        B:  Batch size.

        A:  Number of atoms.

        T:  Number of atom types.

        Y:  Output dimension.

    """

    def __init__(self,
                 model: MolecularModel = None,
                 dim_output: int = 1,
                 activation: Cell = None,
                 decoder: Decoder = 'halve',
                 aggregator: Aggregator = 'sum',
                 scale: float = 1,
                 shift: float = 0,
                 type_ref: Tensor = None,
                 atomwise_scaleshift: bool = True,
                 axis: int = -2,
                 n_decoder_layers: int = 1,
                 energy_unit: str = None,
                 hyper_param: dict = None,
                 ):

        super().__init__(
            model=model,
            dim_output=dim_output,
            activation=activation,
            decoder=decoder,
            aggregator=aggregator,
            scale=scale,
            shift=shift,
            type_ref=type_ref,
            atomwise_scaleshift=atomwise_scaleshift,
            axis=axis,
            n_decoder_layers=n_decoder_layers,
            energy_unit=energy_unit,
            hyper_param=hyper_param,
        )

        self.set_hyper_param()

    def construct(self,
                  x: Tensor,
                  xlist: list = None,
                  atoms_types: Tensor = None,
                  atom_mask: Tensor = None,
                  num_atoms: Tensor = None
                  ):
        r"""Compute atomwise readout network.

        Args:
            x (Tensor):             Tensor of shape (B, A, F). Data type is float.
                                    Representation of atoms.
            xlist (list):           List of atom representations.
                                    Default: None
            atoms_types (Tensor):   Tensor of shape (B, A). Data type is int.
                                    Index of atom types. Default: None
            atom_mask (Tensor):     Tensor of shape (B, A). Data type is bool
                                    Mask for atom types
            num_atoms (Tensor):     Tensor of shape (B, A). Data type is int
                                    Number of atoms for each batch.

        Returns:
            y: (Tensor):    Tensor of shape (B, A, Y). Data type is float

        Symbols:

            B:  Batch size.
            A:  Number of atoms in system.
            F:  Feature dimension of representation.
            Y:  Output dimension.

        """
        y = x
        if self.decoder is not None:
            y = self.decoder(y)

        if self.aggregator is not None:
            if self.atomwise_scaleshift:
                y = y * self.scale + self.shift
                if self.type_ref is not None:
                    y += F.gather(self.type_ref, atoms_types, 0)
                y = self.aggregator(y, atom_mask, num_atoms)
            else:
                y = self.aggregator(y, atom_mask, num_atoms) / num_atoms
                y = y * self.scale + self.shift
                if self.type_ref is not None:
                    ref = F.gather(self.type_ref, atoms_types, 0)
                    y += self.reduce_sum(ref, self.axis)

        return y


class GraphReadout(Readout):
    r"""
    Graph readout function

    Args:
        model (MolecularModel):     Molecular model.

        dim_output (int):           Output dimension.

        activation (Cell):          Activation function

        decoder (str):              Decoder network for atom representation. Default: 'halve'

        aggregator (str):           Aggregator network for atom representation. Default: 'mean'

        scale (float):              Scale value for output. Default: 1

        shift (float):              Shift value for output. Default: 0

        type_ref (Tensor):          Tensor of shape (B, T, Y). Data type is float.
                                    Reference value for atom types. Default: None

        atomwise_scaleshift (bool): To use atomwise scaleshift (True) or graph scaleshift (False).
                                    Default: False

        axis (int):                 Axis to readout. Default: -2

        n_decoder_layers (list):    number of neurons in each hidden layer of the decoder network.
                                    Default: 1

        energy_unit (str):          Energy unit of output. Default: None

        hyper_param (dict):         Hyperparameter. Default: None

    Symbols:

        B:  Batch size.

        A:  Number of atoms.

        T:  Number of atom types.

        Y:  Output dimension.

    """

    def __init__(self,
                 model: MolecularModel = None,
                 dim_output: int = 1,
                 activation: Cell = None,
                 decoder: Decoder = 'halve',
                 aggregator: Aggregator = 'mean',
                 scale: float = 1,
                 shift: float = 0,
                 type_ref: Tensor = None,
                 atomwise_scaleshift: bool = False,
                 axis: int = -2,
                 n_decoder_layers: int = 1,
                 energy_unit: str = None,
                 hyper_param: dict = None,
                 ):

        super().__init__(
            model=model,
            dim_output=dim_output,
            activation=activation,
            decoder=decoder,
            aggregator=aggregator,
            scale=scale,
            shift=shift,
            type_ref=type_ref,
            atomwise_scaleshift=atomwise_scaleshift,
            axis=axis,
            n_decoder_layers=n_decoder_layers,
            energy_unit=energy_unit,
            hyper_param=hyper_param,
        )

        if self.aggregator is None:
            raise ValueError("aggregator cannot be None at GraphReadout")

        self.set_hyper_param()

    def construct(self,
                  x: Tensor,
                  xlist: list = None,
                  atoms_types: Tensor = None,
                  atom_mask: Tensor = None,
                  num_atoms: Tensor = None
                  ):
        r"""Compute graph readout network.

        Args:
            x (Tensor):             Tensor of shape (B, A, F). Data type is float.
                                    Representation of atoms.
            xlist (list):           List of atom representations.
                                    Default: None
            atoms_types (Tensor):   Tensor of shape (B, A). Data type is int.
                                    Index of atom types. Default: None
            atom_mask (Tensor):     Tensor of shape (B, A). Data type is bool
                                    Mask for atom types
            num_atoms (Tensor):     Tensor of shape (B, A). Data type is int
                                    Number of atoms for each batch.

        Returns:
            y: (Tensor):    Tensor of shape (B, A, Y). Data type is float

        Symbols:

            B:  Batch size.
            A:  Number of atoms in system.
            F:  Feature dimension of representation.
            Y:  Output dimension.

        """

        y = self.aggregator(x, atom_mask, num_atoms)

        if self.decoder is not None:
            y = self.decoder(y)

        y = y * self.scale + self.shift
        if self.atomwise_scaleshift:
            y *= num_atoms

        if self.type_ref is not None:
            ref = F.gather(self.type_ref, atoms_types, 0)
            y += self.reduce_sum(ref, self.axis)

        return y


_READOUT_BY_KEY = {
    'atomwise': AtomwiseReadout,
    'graph': GraphReadout,
}

_READOUT_BY_NAME = {out.__name__: out for out in _READOUT_BY_KEY.values()}


def get_readout(readout: str = None,
                model: MolecularModel = None,
                dim_output: int = 1,
                energy_unit: str = None,
                ) -> Readout:
    """get readout function

    Args:
        readout (str):          Name of readout function. Default: None
        model (MolecularModel): Molecular model. Default: None
        dim_output (int):       Output dimension. Default: 1
        energy_unit (str):      Energy Unit. Default: None

    """
    if isinstance(readout, Readout):
        return readout
    if readout is None:
        return None

    hyper_param = None
    if isinstance(readout, dict):
        if 'name' not in readout.keys():
            raise KeyError('Cannot find the key "name"! in readout dict')
        hyper_param = readout
        readout = get_hyper_string(hyper_param, 'name')

    if isinstance(readout, str):
        if readout.lower() == 'none':
            return None
        if readout.lower() in _READOUT_BY_KEY.keys():
            return _READOUT_BY_KEY[readout.lower()](
                model=model,
                dim_output=dim_output,
                energy_unit=energy_unit,
                hyper_param=hyper_param,
            )
        if readout in _READOUT_BY_NAME.keys():
            return _READOUT_BY_NAME[readout](
                model=model,
                dim_output=dim_output,
                energy_unit=energy_unit,
                hyper_param=hyper_param,
            )
        raise ValueError(
            "The Readout corresponding to '{}' was not found.".format(readout))
    raise TypeError("Unsupported Readout type '{}'.".format(type(readout)))
