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
Main program of Cybertron
"""

import os
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore.nn import Cell, CellList
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.train._utils import _make_directory

from mindspore.train import save_checkpoint

from mindsponge.function import Units, global_units
from mindsponge.function import get_integer
from mindsponge.data import get_class_parameters
from mindsponge.data import get_hyper_parameter, get_hyper_string
from mindsponge.data import set_class_into_hyper_param
from mindsponge.colvar import IndexDistances
from mindsponge.partition import FullConnectNeighbours
from mindsponge.potential import PotentialCell

from .readout import Readout, get_readout
from .model import MolecularModel, get_molecular_model

_cur_dir = os.getcwd()

class Cybertron(Cell):
    """Cybertron: An architecture to perform deep molecular model for molecular modeling.

    Args:

        model (Cell):           Deep molecular model.

        readout (Cell):         Readout function.

        dim_output (int):       Output dimension. Default: 1.

        num_atoms (int):        Maximum number of atoms in system. Default: None.

        atom_types (Tensor):    Tensor of shape (B, A). Data type is int.
                                Index of atom types.
                                Default: None,

        bond_types (Tensor):    Tensor of shape (B, A, N). Data type is int.
                                Index of bond types. Default: None.

        pbc_box (Tensor):       Tensor of shape (B, D).
                                Box size of periodic boundary condition. Default: None

        use_pbc (bool):         Whether to use periodic boundary condition. Default: None

        length_unit (str):      Unit of position coordinate. Default: None

        energy_unit (str):      Unit of output energy. Default: None.

        hyper_param (dict):     Hyperparameters of Cybertron. Default: None.

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        N:  Number of neighbour atoms.

        D:  Dimension of position coordinates, usually is 3.

        O:  Output dimension of the predicted properties.

    """

    def __init__(self,
                 model: MolecularModel = None,
                 readout: Readout = 'atomwise',
                 dim_output: int = 1,
                 num_atoms: int = None,
                 atom_types: Tensor = None,
                 bond_types: Tensor = None,
                 pbc_box: Tensor = None,
                 use_pbc: bool = None,
                 length_unit: str = None,
                 energy_unit: str = None,
                 hyper_param: dict = None,
                 ):

        super().__init__()

        if hyper_param is not None:
            model = get_class_parameters(hyper_param, 'hyperparam.model')
            num_readout = get_hyper_parameter(
                hyper_param, 'hyperparam.num_readout')
            readout = get_class_parameters(
                hyper_param, 'hyperparam.readout', num_readout)
            dim_output = get_hyper_parameter(
                hyper_param, 'hyperparam.dim_output')
            num_atoms = get_hyper_parameter(
                hyper_param, 'hyperparam.num_atoms')
            atom_types = get_hyper_parameter(
                hyper_param, 'hyperparam.atom_types')
            bond_types = get_hyper_parameter(
                hyper_param, 'hyperparam.bond_types')
            pbc_box = get_hyper_parameter(hyper_param, 'hyperparam.pbc_box')
            use_pbc = get_hyper_parameter(hyper_param, 'hyperparam.use_pbc')
            length_unit = get_hyper_string(
                hyper_param, 'hyperparam.length_unit')
            energy_unit = get_hyper_string(
                hyper_param, 'hyperparam.energy_unit')

        if length_unit is None and energy_unit is None:
            self.units = global_units
        else:
            self.units = Units(length_unit, energy_unit)

        self.length_unit = self.units.length_unit
        self.energy_unit = self.units.energy_unit
        self.model = get_molecular_model(model, length_unit=self.length_unit)

        self.model_name = self.model.network_name
        self.model_hyper_param = self.model.hyper_param
        self.dim_feature = self.model.dim_feature
        self.activation = self.model.activation
        self.input_unit_scale = self.units.convert_energy_to(self.model.units)
        self.calc_distance = self.model.calc_distance

        self.dim_output = Tensor(dim_output).reshape(-1)
        self.num_output = self.dim_output.size
        if self.num_output == 1:
            self.dim_output = get_integer(self.dim_output)

        if readout is None:
            self.num_readout = 0
            self.dim_output = self.dim_feature
            self.num_output = 0
        elif isinstance(readout, (list, tuple)):
            self.num_readout = len(readout)
            if self.num_readout == 1:
                readout = readout[0]
        else:
            self.num_readout = 1

        if self.num_output != self.num_readout:
            if self.num_output == 1:
                self.dim_output = Tensor(
                    [self.dim_output,] * self.num_readout)
                self.num_output = self.num_readout
            elif self.num_readout == 1:
                if isinstance(readout, Readout):
                    raise TypeError(
                        'The class "Readout" cannot be broadcast, please use "str" or "dict" instead.')
                readout = [readout,] * self.num_output
                self.num_readout = self.num_output
            else:
                raise ValueError('"The number of "readout" ('+str(self.num_readout) +
                                 ') mismatch does not match the number of "dim_output" (' +
                                 str(self.num_output)+').')

        self.multi_readouts = False
        if self.num_readout > 1:
            self.multi_readouts = True
            self.readout = CellList(
                [
                    get_readout(
                        readout[i],
                        model=self.model,
                        dim_output=self.dim_output[i],
                        energy_unit=self.units.energy_unit,
                    )
                    for i in range(self.num_readout)
                ]
            )
            self.tot_out_dim = 0
            for i in range(self.num_readout):
                self.dim_output[i] = self.readout[i].dim_output
                self.tot_out_dim += self.readout[i].dim_output
        elif self.num_readout == 1:
            self.readout = get_readout(
                readout,
                model=self.model,
                dim_output=self.dim_output,
                energy_unit=self.units.energy_unit,
            )
            self.dim_output = self.readout.dim_output
            self.tot_out_dim = self.dim_output
        else:
            self.readout = None

        self.atomwise_scaleshift = self.get_atomwise_scaleshift()

        self.output_unit_scale = self.get_output_unit_scale()

        if atom_types is None:
            self.atom_types = None
            self.atom_mask = None
            self.fixed_atoms = False
            if num_atoms is None:
                raise ValueError(
                    '"num_atoms" must be assigned when "atom_types" is None')
            natoms = get_integer(num_atoms)
            self.num_atoms = natoms
        else:
            self.fixed_atoms = True
            # (1,A)
            self.atom_types = Tensor(atom_types, ms.int32).reshape(1, -1)
            self.atom_mask = atom_types > 0
            natoms = self.atom_types.shape[-1]
            if self.atom_mask.all():
                self.num_atoms = natoms
            else:
                self.num_atoms = F.cast(atom_types > 0, ms.int32)
                self.num_atoms = msnp.sum(num_atoms, -1, keepdims=True)

        self.bond_types = None
        self.bond_mask = None
        if bond_types is not None:
            self.bond_types = Tensor(
                bond_types, ms.int32).reshape(1, natoms, -1)
            self.bond_mask = bond_types > 0

        self.fc_neighbours = FullConnectNeighbours(natoms)
        self.neighbours = None
        self.neighbour_mask = None
        if self.atom_types is not None:
            self.neighbours, self.neighbour_mask = self.fc_neighbours(
                self.atom_types > 0)

        self.pbc_box = None
        self.use_pbc = use_pbc
        if pbc_box is not None:
            # (1,D)
            self.pbc_box = Tensor(pbc_box, ms.float32).reshape(1, -1)
            self.use_pbc = True

        cutoff = self.model.cutoff
        self.distances = IndexDistances(
            self.use_pbc, length_unit, cutoff*10, keep_dims=False)

        self.hyper_param = dict()
        self.hyper_types = {
            'model': 'Cell',
            'num_readout': 'int',
            'readout': 'Cell',
            'dim_output': 'int',
            'num_atoms': 'int',
            'atom_types': 'int',
            'bond_types': 'int',
            'pbc_box': 'bool',
            'use_pbc': 'bool',
            'length_unit': 'str',
            'energy_unit': 'str',
        }

        self.set_hyper_param()

        self.concat = P.Concat(-1)

    def set_units(self, length_unit: str = None, energy_unit: str = None, units: Units = None):
        """set units"""
        if units is None:
            if length_unit is not None:
                self.set_length_unit(length_unit)
            if energy_unit is not None:
                self.set_energy_unit(energy_unit)
        else:
            self.units = units
            self.length_unit = self.units.length_unit
            self.energy_unit = self.units.energy_unit
            self.input_unit_scale = self.units.convert_energy_to(
                self.model.units)
            self.output_unit_scale = self.get_output_unit_scale()
        return self

    def set_length_unit(self, length_unit: str):
        """set length unit"""
        self.units = self.units.set_length_unit(length_unit)
        self.length_unit = self.units.length_unit
        self.input_unit_scale = self.units.convert_energy_to(self.model.units)
        return self

    def set_energy_unit(self, energy_units: str):
        """set energy unit"""
        self.units.set_energy_unit(energy_units)
        self.energy_unit = self.units.energy_unit
        self.output_unit_scale = self.get_output_unit_scale()
        return self

    def get_output_unit_scale(self) -> Tensor:
        """get the scale factor of output unit"""
        if self.num_readout == 1:
            output_unit_scale = self.units.convert_energy_from(
                self.readout.energy_unit)
        elif self.num_readout > 1:
            output_unit_scale = ()
            for readout in self.readout:
                unit = readout.energy_unit
                scale = self.units.convert_energy_from(unit)
                output_unit_scale = output_unit_scale + \
                    (scale,) * readout.dim_output
        return Tensor(output_unit_scale, ms.float32)

    def set_hyper_param(self):
        """set hyperparameters"""
        set_class_into_hyper_param(
            self.hyper_param, self.hyper_types, self, 'hyperparam')
        return self

    def get_atomwise_scaleshift(self):
        """get the atomwaise scale and shift"""
        if self.readout is None:
            return None
        if self.num_readout == 1:
            return self.readout.atomwise_scaleshift
        atomwise_scaleshift = []
        for i in range(self.num_readout):
            for _ in range(self.readout[i].dim_output):
                atomwise_scaleshift.append(
                    self.readout[i].atomwise_scaleshift)
        return F.stack(atomwise_scaleshift)

    def print_info(self, num_retraction: int = 3, num_gap: int = 3, char: str = ' '):
        """print the information of Cybertron"""
        ret = char * num_retraction
        gap = char * num_gap
        print("================================================================================")
        print("Cybertron Engine, Ride-on!")
        print('-'*80)
        print(ret+' Length unit: ' + self.units.length_unit_name)
        print(ret+' Input unit scale: ' + str(self.input_unit_scale))
        if self.atom_types is not None:
            print(ret+' Using fixed atom type index:')
            for i, atom in enumerate(self.atom_types[0]):
                print(
                    ret+gap+' Atom {: <7}'.format(str(i)+': ')+str(atom.asnumpy()))
        if self.bond_types is not None:
            print(ret+' Using fixed bond connection:')
            for b in self.bond_types[0]:
                print(ret+gap+' '+str(b.asnumpy()))
            print(ret+' Fixed bond mask:')
            for m in self.bond_mask[0]:
                print(ret+gap+' '+str(m.asnumpy()))
        print('-'*80)
        self.model.print_info(num_retraction=num_retraction,
                              num_gap=num_gap, char=char)

        if self.multi_readouts:
            print(ret+" With "+str(self.num_readout)+" readout networks: ")
            print('-'*80)
            for i in range(self.num_readout):
                print(ret+" "+str(i)+(". "+self.readout[i].cls_name))
                self.readout[i].print_info(
                    num_retraction=num_retraction, num_gap=num_gap, char=char)
            print(ret+" Output dimension: "+str(self.dim_output))
            print(ret+" Total output dimension: "+str(self.tot_out_dim))
        else:
            print(ret+" Readout network: "+self.readout.cls_name)
            print('-'*80)
            self.readout.print_info(
                num_retraction=num_retraction, num_gap=num_gap, char=char)
            print(ret+" Output dimension: "+str(self.dim_output))
        print(ret+" Output unit for Cybertron: "+self.units.energy_unit_name)
        print(ret+" Output unit scale: "+str(self.output_unit_scale))
        print("================================================================================")

    def get_multi_scaleshift(self,
                             scale: float = 1,
                             shift: float = 0,
                             type_ref: Tensor = None,
                             atomwise_scaleshift: bool = None,
                             readout_id: int = None
                             ):
        """get the scale and shift of multiple readouts"""

        if readout_id is None:
            readout_id = list(range(self.num_readout))
            num_scale = self.num_readout
            dim_scale = self.dim_output.asnumpy().tolist()
        elif isinstance(readout_id, (int)):
            num_scale = 1
            dim_scale = [self.readout[readout_id].dim_output,]
            readout_id = [readout_id,]
        elif isinstance(readout_id, (list, tuple)):
            num_scale = len(readout_id)
            if num_scale >= self.num_readout:
                raise ValueError('The number of "readout_id" ({:d}) is out of range of readout function ({:d})'
                                 .format(num_scale, self.num_readout))
            dim_scale = []
            for r in readout_id:
                dim_scale.append(int(self.dim_output[r].asnumpy()))
        else:
            raise TypeError(
                'The type of "readout_id" must be int, list or tuple but got type "'+str(type(readout_id))+').')

        split_index = []
        dim_output = 0
        for i in range(num_scale):
            dim_output += dim_scale[i]
            split_index.append(dim_output)
        split_index.pop()

        def _get_multi_argument(argument, name, dtype, shape=None):
            if argument is None:
                return [None,] * num_scale
            if isinstance(argument, (list, tuple)):
                if len(argument) != num_scale:
                    if len(argument) == 1:
                        argument *= num_scale
                    else:
                        raise ValueError('The number of "'+name+'" ('+str(len(argument)) +
                                         ') does not match the number of readout_id ('+str(num_scale)+').')
            else:
                argument = Tensor(argument, dtype)
                if shape is not None:
                    argument = F.reshape(argument, shape)
                if argument.shape[-1] == 1:
                    argument = [argument,] * num_scale
                elif argument.shape[-1] == dim_output:
                    argument = msnp.split(argument, split_index, axis=-1)
                else:
                    raise ValueError('The size of "'+name+
                                     '" ({:d}) does not match the output number ({:d}).'.
                                     format(argument.shape[-1], dim_output))
            return argument

        scale = _get_multi_argument(scale, 'scale', ms.float32, (-1,))
        shift = _get_multi_argument(shift, 'shift', ms.float32, (-1,))
        type_ref = _get_multi_argument(type_ref, 'type_ref', ms.float32)
        atomwise_scaleshift = _get_multi_argument(
            atomwise_scaleshift, 'atomwise_scaleshift', ms.bool_, (-1,))

        return scale, shift, type_ref, atomwise_scaleshift, readout_id

    def set_scaleshift(self,
                       scale: float = 1,
                       shift: float = 0,
                       type_ref: Tensor = None,
                       atomwise_scaleshift: bool = None,
                       unit: str = None,
                       readout_id: int = None
                       ):
        """set the scale and shift"""

        if self.num_readout == 1:
            self.readout.set_scaleshift(
                scale=scale, shift=shift, type_ref=type_ref, atomwise_scaleshift=atomwise_scaleshift, unit=unit)
            self.atomwise_scaleshift = self.readout.atomwise_scaleshift
        else:
            scale, shift, type_ref, atomwise_scaleshift, readout_id = \
                self.get_multi_scaleshift(
                    scale, shift, type_ref, atomwise_scaleshift, readout_id)
            for i, r in enumerate(readout_id):
                if r >= self.num_readout:
                    raise ValueError('readout_id[{:d}]={:d} is out of the range of readout function ({:d})'
                                     .format(i, r, self.num_readout))
                self.readout[r].set_scaleshift(
                    scale=scale[i],
                    shift=shift[i],
                    type_ref=type_ref[i],
                    atomwise_scaleshift=atomwise_scaleshift[i],
                    unit=unit
                )
        self.atomwise_scaleshift = self.get_atomwise_scaleshift()
        if unit is not None:
            self.set_energy_unit(unit)
        self.set_hyper_param()
        return self

    def save_checkpoint(self, ckpt_file_name: str, directory: str = None):
        """save checkpoint file"""
        if directory is not None:
            directory = _make_directory(directory)
        else:
            directory = _cur_dir
        ckpt_file = os.path.join(directory, ckpt_file_name)
        if os.path.exists(ckpt_file):
            os.remove(ckpt_file)
        save_checkpoint(self, ckpt_file, append_dict=self.hyper_param)
        return self

    def construct(self,
                  positions: Tensor = None,
                  atom_types: Tensor = None,
                  pbc_box: Tensor = None,
                  distances: Tensor = None,
                  neighbours: Tensor = None,
                  neighbour_mask: Tensor = None,
                  bonds: Tensor = None,
                  bond_mask: Tensor = None,
                  ):
        """Compute the properties of the molecules.

        Args:
            positions (Tensor):         Tensor of shape (B, A, D). Data type is float.
                                        Cartesian coordinates for each atom.
            atom_types (Tensor):        Tensor of shape (B, A). Data type is int.
                                        Type index (atomic number) of atom types.
                                        Default: self.atom_types
            pbc_box (Tensor):           Tensor of shape (B, D). Data type is float.
                                        Box size of periodic boundary condition
            distances (Tensor):         Tensor of shape (B, A, N). Data type is float.
                                        Distances between atoms
            neighbours (Tensor):        Tensor of shape (B, A, N). Data type is int.
                                        Indices of other near neighbour atoms around a atom
            neighbour_mask (Tensor):    Tensor of shape (B, A, N). Data type is bool.
                                        Mask for neighbours
            bond_types (Tensor):        Tensor of shape (B, A, N). Data type is int.
                                        Types index of bond connected with two atoms
            bond_mask (Tensor):         Tensor of shape (B, A, N). Data type is bool.
                                        Mask for bonds

        Returns:
            properties (Tensor):    Tensor of shape (B, A, O). Data type is float.

        """

        if self.fixed_atoms:
            # (1,A)
            atom_types = self.atom_types
            num_atoms = self.num_atoms
            atom_mask = self.atom_mask
        else:
            # (1,A)
            atom_mask = atom_types > 0
            num_atoms = F.cast(atom_mask, ms.int32)
            num_atoms = msnp.sum(num_atoms, -1, keepdims=True)

        if self.calc_distance:
            if distances is None:
                if neighbours is None:
                    if self.fixed_atoms:
                        neighbours = self.neighbours
                        neighbour_mask = self.neighbour_mask
                    else:
                        neighbours, neighbour_mask = self.fc_neighbours(
                            atom_mask)
                if self.pbc_box is not None:
                    pbc_box = self.pbc_box
                distances = self.distances(
                    positions, neighbours, neighbour_mask, pbc_box) * self.input_unit_scale
        else:
            distances = 1
            neighbour_mask = bond_mask

        x, xlist = self.model(distances, atom_types, atom_mask,
                              neighbours, neighbour_mask, bonds, bond_mask)

        if self.readout is None:
            return x

        if self.multi_readouts:
            ytuple = ()
            for i in range(self.num_readout):
                yi = self.readout[i](x, xlist, atom_types,
                                     atom_mask, num_atoms)
                ytuple = ytuple + (yi,)
            return self.concat(ytuple) * self.output_unit_scale

        return self.readout(x, xlist, atom_types, atom_mask, num_atoms) * self.output_unit_scale

class CybertronFF(PotentialCell):
    """Cybertron as potential for Mindmindsponge.

    Args:

        model (Cell):           Deep molecular model. Default: None

        readout (Cell):         Readout function. Default: 'atomwise'

        num_atoms (int):        Maximum number of atoms in system. Default: None.

        atom_types (Tensor):    Tensor of shape (B, A). Data type is int.
                                Index of atom types.
                                Default: None,

        bond_types (Tensor):    Tensor of shape (B, A, N). Data type is int.
                                Index of bond types. Default: None.

        use_pbc (bool):         Whether to use periodic boundary condition. Default: None

        length_unit (str):      Unit of position coordinate. Default: None

        energy_unit (str):      Unit of output energy. Default: None.

        hyper_param (dict):     Hyperparameters of Cybertron. Default: None.

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        N:  Number of neighbour atoms.

        D:  Dimension of position coordinates, usually is 3.

        O:  Output dimension of the predicted properties.

    """
    def __init__(self,
                 model: MolecularModel = None,
                 readout: Readout = 'atomwise',
                 num_atoms: int = None,
                 atom_types: Tensor = None,
                 bond_types: Tensor = None,
                 use_pbc: bool = None,
                 length_unit: str = None,
                 energy_unit: str = None,
                 hyper_param: dict = None,
                 ):

        super().__init__(
            exclude_index=None,
            length_unit=length_unit,
            energy_unit=energy_unit,
            use_pbc=use_pbc,
        )

        dim_output = 1
        pbc_box = None
        if hyper_param is not None:
            model = get_class_parameters(hyper_param, 'hyperparam.model')
            num_readout = get_hyper_parameter(
                hyper_param, 'hyperparam.num_readout')
            readout = get_class_parameters(
                hyper_param, 'hyperparam.readout', num_readout)
            dim_output = get_hyper_parameter(
                hyper_param, 'hyperparam.dim_output')
            num_atoms = get_hyper_parameter(
                hyper_param, 'hyperparam.num_atoms')
            atom_types = get_hyper_parameter(
                hyper_param, 'hyperparam.atom_types')
            bond_types = get_hyper_parameter(
                hyper_param, 'hyperparam.bond_types')
            pbc_box = get_hyper_parameter(hyper_param, 'hyperparam.pbc_box')
            use_pbc = get_hyper_parameter(hyper_param, 'hyperparam.use_pbc')
            length_unit = get_hyper_string(
                hyper_param, 'hyperparam.length_unit')
            energy_unit = get_hyper_string(
                hyper_param, 'hyperparam.energy_unit')

        if dim_output != 1:
            raise ValueError('The output dimension of CybertronFF must be 1 but got: '+str(dim_output))
        if readout is None:
            raise ValueError('The readout function in CybertronFF cannot be None!')

        self.model = get_molecular_model(model, length_unit=self.length_unit)

        self.model_name = self.model.network_name
        self.model_hyper_param = self.model.hyper_param
        self.dim_feature = self.model.dim_feature
        self.activation = self.model.activation
        self.input_unit_scale = self.units.convert_energy_to(self.model.units)
        self.calc_distance = self.model.calc_distance

        if isinstance(readout, (list, tuple)):
            raise ValueError('CybertronFF cannot accept multiple readouts!')

        self.readout = get_readout(
            readout,
            model=self.model,
            dim_output=1,
            energy_unit=self.units.energy_unit,
        )
        if self.readout.dim_output != 1:
            raise ValueError('The output dimension of readout in CybertronFF must be 1 but got: '+
                             str(self.readout.dim_output))
        self.dim_output = self.readout.dim_output

        self.atomwise_scaleshift = self.readout.atomwise_scaleshift
        self.output_unit_scale = self.get_output_unit_scale()

        if atom_types is None:
            raise ValueError('For CybertronFF, atom_types cannot be None')

        # (1,A)
        self.atom_types = Tensor(atom_types, ms.int32).reshape(1, -1)
        self.atom_mask = atom_types > 0
        natoms = self.atom_types.shape[-1]
        if self.atom_mask.all():
            self.num_atoms = natoms
        else:
            self.num_atoms = F.cast(atom_types > 0, ms.int32)
            self.num_atoms = msnp.sum(num_atoms, -1, keepdims=True)

        self.bond_types = None
        self.bond_mask = None
        if bond_types is not None:
            self.bond_types = Tensor(
                bond_types, ms.int32).reshape(1, natoms, -1)
            self.bond_mask = bond_types > 0

        self.pbc_box = None
        if pbc_box is not None:
            # (1,D)
            self.pbc_box = Tensor(pbc_box, ms.float32).reshape(1, -1)

        self.cutoff = self.model.cutoff

        self.hyper_param = dict()
        self.hyper_types = {
            'model': 'Cell',
            'num_readout': 'int',
            'readout': 'Cell',
            'dim_output': 'int',
            'num_atoms': 'int',
            'atom_types': 'int',
            'bond_types': 'int',
            'pbc_box': 'bool',
            'use_pbc': 'bool',
            'length_unit': 'str',
            'energy_unit': 'str',
        }

        self.set_hyper_param()

        self.concat = P.Concat(-1)

    def set_units(self, length_unit: str = None, energy_unit: str = None, units: Units = None):
        """set units"""
        if units is None:
            if length_unit is not None:
                self.units.set_length_unit(length_unit)
            if energy_unit is not None:
                self.units.set_energy_unit(energy_unit)
        else:
            self.units = units
            self.length_unit = self.units.length_unit
            self.energy_unit = self.units.energy_unit
            self.input_unit_scale = self.units.convert_energy_to(
                self.model.units)
            self.output_unit_scale = self.get_output_unit_scale()
        return self

    def get_output_unit_scale(self) -> Tensor:
        """get the scale factor of output unit"""
        output_unit_scale = self.units.convert_energy_from(
            self.readout.energy_unit)
        return Tensor(output_unit_scale, ms.float32)

    def set_hyper_param(self):
        """set hyperparameters"""
        set_class_into_hyper_param(self.hyper_param, self.hyper_types, self, 'hyperparam')
        return self

    def print_info(self, num_retraction: int = 3, num_gap: int = 3, char: str = ' '):
        """print the information of CybertronFF"""
        ret = char * num_retraction
        gap = char * num_gap
        print("================================================================================")
        print("Cybertron Force Field:")
        print('-'*80)
        print(ret+' Length unit: ' + self.units.length_unit_name)
        print(ret+' Input unit scale: ' + str(self.input_unit_scale))
        for i, atom in enumerate(self.atom_types[0]):
            print(
                ret+gap+' Atom {: <7}'.format(str(i)+': ')+str(atom.asnumpy()))
        if self.bond_types is not None:
            print(ret+' Using fixed bond connection:')
            for b in self.bond_types[0]:
                print(ret+gap+' '+str(b.asnumpy()))
            print(ret+' Fixed bond mask:')
            for m in self.bond_mask[0]:
                print(ret+gap+' '+str(m.asnumpy()))
        print('-'*80)
        self.model.print_info(num_retraction=num_retraction,
                              num_gap=num_gap, char=char)

        print(ret+" Readout network: "+self.readout.cls_name)
        print('-'*80)
        self.readout.print_info(
            num_retraction=num_retraction, num_gap=num_gap, char=char)
        print(ret+" Output unit for Cybertron: "+self.units.energy_unit_name)
        print(ret+" Output unit scale: "+str(self.output_unit_scale))
        print("================================================================================")

    def set_scaleshift(self,
                       scale: float = 1,
                       shift: float = 0,
                       type_ref: Tensor = None,
                       atomwise_scaleshift: bool = None,
                       unit: str = None,
                       ):
        """set the scale and shift"""

        self.readout.set_scaleshift(
            scale=scale, shift=shift, type_ref=type_ref, atomwise_scaleshift=atomwise_scaleshift, unit=unit)
        self.atomwise_scaleshift = self.readout.atomwise_scaleshift

        if unit is not None:
            self.units.set_energy_unit(unit)
            self.output_unit_scale = self.get_output_unit_scale()

        self.set_hyper_param()
        return self

    def save_checkpoint(self, ckpt_file_name: str, directory: str = None):
        """save checkpoint file"""
        if directory is not None:
            directory = _make_directory(directory)
        else:
            directory = _cur_dir
        ckpt_file = os.path.join(directory, ckpt_file_name)
        if os.path.exists(ckpt_file):
            os.remove(ckpt_file)
        save_checkpoint(self, ckpt_file, append_dict=self.hyper_param)
        return self

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate potential energy.

        Args:
            coordinate (Tensor):           Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system.
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms. Default: None
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour atoms. Default: None
            neighbour_coord (Tensor):       Tensor of shape (B, A, N, D). Data type is bool.
                                            Position coorindates of neighbour atoms.
            neighbour_distance (Tensor):   Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms. Default: None
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            potential (Tensor): Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            N:  Maximum number of neighbour atoms.
            D:  Dimension of the simulation system. Usually is 3.

        """

        x, xlist = self.model(neighbour_distance, self.atom_types, self.atom_mask,
                              neighbour_index, neighbour_mask)

        energy = self.readout(x, xlist, self.atom_types, self.atom_mask, self.num_atoms)

        return energy * self.output_unit_scale
