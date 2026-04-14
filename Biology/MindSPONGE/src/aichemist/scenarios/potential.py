# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
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
from typing import Union, List, Tuple
from numpy import ndarray

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter
from mindspore.nn import Cell, CellList
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.train import save_checkpoint
from mindspore.train._utils import _make_directory

from ..utils.units import Units, GLOBAL_UNITS
from ..utils.fullconnect import FullConnectNeighbours

from ..utils import write_yaml
from ..configs import Config
from ..layers.embedding import GraphEmbedding
from ..layers.readout import Readout
from ..nets.geonet import MolecularGNN
from ..transforms import ScaleShift
from ..configs import Registry as R

_cur_dir = os.getcwd()


@R.register('scenario.cybertron')
# pylint: disable=E1102
class Cybertron(Cell):
    """Cybertron: An architecture to perform deep molecular model for molecular modeling.

    Args:
        model (Cell):           Deep molecular model.

        readout (Cell):         Readout function.

        dim_output (int):       Output dimension. Default: 1.

        num_atoms (int):        Maximum number of atoms in system. Default: ``None``.

        atom_type (Tensor):    Tensor of shape (B, A). Data type is int.
                                Index of atom types.
                                Default: ``None``.,

        bond_types (Tensor):    Tensor of shape (B, A, N). Data type is int.
                                Index of bond types. Default: ``None``.

        num_atom_types (int):   Maximum number of atomic types. Default: 64

        pbc_box (Tensor):       Tensor of shape (B, D).
                                Box size of periodic boundary condition. Default: ``None``.

        use_pbc (bool):         Whether to use periodic boundary condition. Default: ``None``.

        length_unit (str):      Unit of position coordinate. Default: ``None``.

        energy_unit (str):      Unit of output energy. Default: ``None``.

        hyper_param (dict):     Hyperparameters of Cybertron. Default: ``None``.

    Note:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        N:  Number of neighbour atoms.

        D:  Dimension of position coordinates, usually is 3.

        O:  Output dimension of the predicted properties.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 model: Union[MolecularGNN, dict, str],
                 embedding: Union[GraphEmbedding, dict, str] = None,
                 readout: Union[Readout, dict, str, List[Readout]] = 'atomwise',
                 num_atoms: int = None,
                 atom_type: Union[Tensor, ndarray, List[int]] = None,
                 bond_types: Union[Tensor, ndarray, List[int]] = None,
                 pbc_box: Union[Tensor, ndarray, List[float]] = None,
                 use_pbc: bool = None,
                 scale: Union[float, Tensor, List[Union[float, Tensor]]] = 1,
                 shift: Union[float, Tensor, List[Union[float, Tensor]]] = 0,
                 type_ref: Union[Tensor, ndarray, List[Union[Tensor, ndarray]]] = None,
                 length_unit: Union[str, Units] = None,
                 energy_unit: Union[str, Units] = None,
                 **kwargs
                 ):

        super().__init__()
        self._kwargs = Config.get_arguments(locals(), kwargs)

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        self._units = Units(length_unit, energy_unit)

        if atom_type is None:
            self.atom_type = None
            self.atom_mask = None
            if num_atoms is None:
                raise ValueError('"num_atoms" must be assigned when "atom_type" is None')
            natoms = int(num_atoms)
            self.num_atoms = natoms
        else:
            # The shape looks like (1,A)
            self.atom_type = ms.Tensor(atom_type, ms.int32).reshape(1, -1)
            self.atom_mask = self.atom_type > 0
            natoms = self.atom_type.shape[-1]
            if self.atom_mask.all():
                self.num_atoms = natoms
            else:
                self.num_atoms = F.cast(atom_type > 0, ms.int16)
                self.num_atoms = msnp.sum(num_atoms, -1, keepdims=True)

        self.bonds = None
        self.bond_mask = None
        if bond_types is not None:
            self.bonds = ms.Tensor(bond_types, ms.int16).reshape(1, natoms, -1)
            self.bond_mask = bond_types > 0

        model = R.build('net', model)

        if embedding is None or isinstance(embedding, str):
            embedding = model.default_embedding

        self.embedding = R.build('embedding', embedding,
                                 dim_node=model.dim_node_emb,
                                 dim_edge=model.dim_edge_emb,
                                 activation=model.activation,
                                 length_unit=length_unit)

        self.dim_node_emb = self.embedding.dim_node
        self.dim_edge_emb = self.embedding.dim_edge

        model.set_dimension(self.dim_node_emb, self.dim_edge_emb)
        self.model = model

        self.dim_node_rep = self.model.dim_node_rep
        self.dim_edge_rep = self.model.dim_edge_rep

        self.calc_distance = self.embedding.emb_dis

        self.neighbours = None
        self.neighbour_mask = None
        self.get_neigh_list = None
        self.pbc_box = None
        self.use_pbc = use_pbc
        self.num_neighbours = None
        self.cutoff = None
        self.large_dis = 5e4

        if self.calc_distance:
            self.cutoff = self.embedding.cutoff

            self.get_neigh_list = FullConnectNeighbours(natoms)
            self.num_neighbours = self.num_atoms - 1

            if self.atom_type is not None:
                self.neighbours, self.neighbour_mask = self.get_neigh_list(
                    self.atom_type > 0)

            if pbc_box is not None:
                # The shape looks like (1,D)
                self.pbc_box = ms.Tensor(pbc_box, ms.float32).reshape(1, -1)
                self.use_pbc = True

            self.large_dis = self.cutoff * 10

        self.norm_last_dim = None

        self.activation = self.model.activation

        self.num_readouts = 0
        self.num_outputs = 2
        self.output_ndim = (2, 3)
        # The shape looks like [(A, F), (A, N, F)]
        self.output_shape = ((self.num_atoms, self.dim_node_rep),
                             (self.num_atoms, self.num_neighbours, self.dim_edge_rep))

        self.readout: List[Readout] = None
        self.scaleshift: List[ScaleShift] = None
        if readout is not None:
            if isinstance(readout, (Readout, str, dict)):
                self.num_readouts = 1
                self.num_outputs = 1
                readout = [readout]
            if isinstance(readout, (list, tuple)):
                self.num_readouts = len(readout)
                self.num_outputs = len(readout)
                readout = [R.build('readout', r,
                                   dim_node_rep=self.dim_node_rep,
                                   dim_edge_rep=self.dim_edge_rep,
                                   activation=self.activation,
                                   ) for r in readout]
            else:
                readout = None
                raise TypeError(f'Unsupported `readout` type: {type(readout)}')

            self.output_ndim = []
            self.output_shape = []
            for i in range(self.num_outputs):
                readout[i].set_dimension(self.dim_node_rep, self.dim_edge_rep)
                self.output_ndim.append(readout[i].ndim)
                self.output_shape.append(readout[i].shape)

            self.readout = CellList(readout)

            self.set_scaleshift(scale, shift, type_ref)

        self.input_unit_scale = self.embedding.convert_length_from(self._units)
        self.use_scaleshift = True

    @property
    def units(self) -> Units:
        return self._units

    @units.setter
    def units(self, units_: Units):
        self._units = units_
        self.input_unit_scale = self.embedding.convert_length_from(self._units)
        if self.readout is not None:
            self.output_unit_scale = \
                (Tensor(self.scaleshift[i].convert_energy_to(self._units), ms.float32)
                 for i in range(self.num_readouts))

    @property
    def length_unit(self) -> str:
        return self._units.length_unit

    @length_unit.setter
    def length_unit(self, length_unit_: Union[str, Units]):
        self.set_length_unit(length_unit_)

    @property
    def energy_unit(self) -> str:
        return self._units.energy_unit

    @energy_unit.setter
    def energy_unit(self, energy_unit_: Union[str, Units]):
        self.set_energy_unit(energy_unit_)

    @property
    def model_name(self) -> str:
        return self.model.cls_name

    def set_train(self, mode: bool = True):
        super().set_train(mode)
        self.use_scaleshift = not mode
        return self

    def set_inference(self, mode: bool = True):
        """Sets the cell to inference mode."""
        self.set_train(not mode)
        return self

    def set_scaleshift(self,
                       scale: Union[float, Tensor, List[Union[float, Tensor]]] = 1,
                       shift: Union[float, Tensor, List[Union[float, Tensor]]] = 0,
                       type_ref: Union[Tensor, ndarray, List[Union[Tensor, ndarray]]] = None,
                       ):
        """set scale, shift and type_ref"""
        if self.readout is None:
            return self

        def _check_data(value, name: str):
            if not isinstance(value, (list, tuple)):
                value = [value]
            if len(value) == self.num_readouts:
                return value
            if len(value) == 1:
                return value * self.num_readouts
            raise ValueError(f'The number of {name} {len(value)} must be equal to '
                             f'the number of readout functions {self.num_readouts}')

        def _check_scaleshift(value, shape: Tuple[int], name: str) -> Tensor:
            if value is None:
                return None
            if not isinstance(value, (float, int, Tensor, Parameter, ndarray)):
                raise TypeError(f'The type of {name} must be float, Tensor or ndarray, '
                                f'but got: {type(value)}')
            value = ms.Tensor(value).float()
            if value.ndim == 0:
                value = F.reshape(value, (-1,))
            if value.shape == shape:
                return value

            if value.size == 1:
                # The shape looks like (1, ..., 1) <- (1)
                return F.reshape(value, (1,) * len(shape) + (-1,))

            raise ValueError(f'The shape of {name} ({value.shape}) does not match '
                             f'the shape of readout function: {shape}')

        scale = _check_data(scale, 'scale')
        shift = _check_data(shift, 'shift')

        scale = [_check_scaleshift(scale[i], self.readout[i].shape, 'scale')
                 for i in range(self.num_readouts)]
        shift = [_check_scaleshift(shift[i], self.readout[i].shape, 'shift')
                 for i in range(self.num_readouts)]

        def _check_type_ref(ref, shape: Tuple[int]) -> Tensor:
            if ref is None:
                return None
            if not isinstance(ref, (Tensor, Parameter, ndarray)):
                raise TypeError(f'The type of type_ref must be Tensor, Parameter or ndarray, '
                                f'but got: {type(ref)}')
            ref = ms.Tensor(ref).float()
            if ref.ndim < 2:
                raise ValueError(f'The rank (ndim) of type_ref should be at least 2, '
                                 f'but got : {ref.ndim}')
            if ref.shape[1:] != shape:
                raise ValueError(f'The shape of type_ref {ref.shape} does not match '
                                 f'the shape of readout function: {shape}')
            return ref

        if not isinstance(type_ref, (list, tuple)):
            type_ref = [type_ref]
        if len(type_ref) != self.num_readouts:
            if len(type_ref) == 1:
                type_ref *= self.num_readouts
            else:
                raise ValueError(f'The number of type_ref {len(type_ref)} must be equal to '
                                 f'the number of readout functions {self.num_readouts}')

        type_ref = [_check_type_ref(type_ref[i], self.readout[i].shape)
                    for i in range(self.num_readouts)]

        if self.scaleshift is None:
            self.scaleshift = CellList([
                ScaleShift(scale=scale[i],
                           shift=shift[i],
                           type_ref=type_ref[i],
                           shift_by_atoms=self.readout[i].shift_by_atoms)
                for i in range(self.num_readouts)
            ])
        else:
            for i in range(self.num_readouts):
                self.scaleshift[i].set_scaleshift(scale=scale[i], shift=shift[i], type_ref=type_ref[i])
        return self

    def readout_ndim(self, readout_idx: int) -> int:
        """returns the rank (ndim) of a specific readout function"""
        if self.readout is None:
            return None
        self._check_readout_index(readout_idx)
        return self.readout[readout_idx].ndim

    def readout_shape(self, readout_idx: int) -> Tuple[int]:
        """returns the shape of a specific readout function"""
        if self.readout is None:
            return None
        self._check_readout_index(readout_idx)
        return self.readout[readout_idx].shape

    def scale(self, readout_idx: int = None) -> Union[Tensor, List[Tensor]]:
        """returns the scale"""
        if self.readout is None:
            return [self.scaleshift[i].scale for i in range(self.num_readouts)]
        self._check_readout_index(readout_idx)
        return self.scaleshift[readout_idx].scale

    def shift(self, readout_idx: int = None) -> Union[Tensor, List[Tensor]]:
        """returns the shift"""
        if self.readout is None:
            return [self.scaleshift[i].shift for i in range(self.num_readouts)]
        self._check_readout_index(readout_idx)
        return self.scaleshift[readout_idx].shift

    def type_ref(self, readout_idx: int = None) -> Union[Tensor, List[Tensor]]:
        """returns the type_ref"""
        if self.readout is None:
            return [self.scaleshift[i].type_ref for i in range(self.num_readouts)]
        self._check_readout_index(readout_idx)
        return self.scaleshift[readout_idx].type_ref

    def set_units(self, length_unit: str = None, energy_unit: str = None):
        """set units"""
        if length_unit is not None:
            self.set_length_unit(length_unit)
        if energy_unit is not None:
            self.set_energy_unit(energy_unit)
        return self

    def set_length_unit(self, length_unit: str):
        """set length unit"""
        self._units = self._units.set_length_unit(length_unit)
        self.input_unit_scale = self.embedding.convert_length_from(self._units)
        return self

    def set_energy_unit(self, energy_units: str):
        """set energy unit"""
        self._units.set_energy_unit(energy_units)
        if self.readout is not None:
            self.output_unit_scale = (
                Tensor(self.scaleshift[i].convert_energy_to(self._units), ms.float32)
                for i in range(self.num_readouts))
        return self

    def save_configure(self, filename: str, directory: str = None):
        """save configure to file"""
        write_yaml(self._kwargs, filename, directory)
        return self

    def save_checkpoint(self, ckpt_file_name: str, directory: str = None, append_dict: str = None):
        """save checkpoint file"""
        if directory is not None:
            directory = _make_directory(directory)
        else:
            directory = _cur_dir
        ckpt_file = os.path.join(directory, ckpt_file_name)
        if os.path.exists(ckpt_file):
            os.remove(ckpt_file)
        save_checkpoint(self, ckpt_file, append_dict=append_dict)
        return self

    def print_info(self, num_retraction: int = 3, num_gap: int = 3, char: str = ' '):
        """print the information of Cybertron"""
        ret = char * num_retraction
        gap = char * num_gap
        print("================================================================================")
        print("Cybertron Engine, Ride-on!")
        print('-'*80)
        if self.atom_type is None:
            print(f'{ret} Using variable atom types with maximum number of atoms: {self.num_atoms}')
        else:
            print(f'{ret} Using fixed atom type index:')
            for i, atom in enumerate(self.atom_type[0]):
                print(ret+gap+' Atom {: <7}'.format(str(i))+f': {atom.asnumpy()}')
        if self.bonds is not None:
            print(ret+' Using fixed bond connection:')
            for b in self.bonds[0]:
                print(ret+gap+' '+str(b.asnumpy()))
            print(ret+' Fixed bond mask:')
            for m in self.bond_mask[0]:
                print(ret+gap+' '+str(m.asnumpy()))
        print('-'*80)
        self.embedding.print_info(num_retraction=num_retraction,
                                  num_gap=num_gap, char=char)
        self.model.print_info(num_retraction=num_retraction,
                              num_gap=num_gap, char=char)

        print(ret+" With "+str(self.num_readouts)+" readout networks: ")
        print('-'*80)
        for i in range(self.num_readouts):
            print(ret+" "+str(i)+(". "+self.readout[i].cls_name))
            self.readout[i].print_info(
                num_retraction=num_retraction, num_gap=num_gap, char=char)
            self.scaleshift[i].print_info(
                num_retraction=num_retraction, num_gap=num_gap, char=char)

        print(f'{ret} Input unit: {self._units.length_unit_name}')
        print(f'{ret} Output unit: {self._units.energy_unit_name}')
        print(f'{ret} Input unit scale: {self.input_unit_scale}')
        print("================================================================================")

    def construct(self,
                  coordinate: Tensor = None,
                  atom_type: Tensor = None,
                  pbc_box: Tensor = None,
                  bonds: Tensor = None,
                  bond_mask: Tensor = None,
                  ) -> Union[Tensor, Tuple[Tensor]]:
        """Compute the properties of the molecules.

        Args:
            coordinate (Tensor): Tensor of shape (B, A, D). Data type is float.
                Cartesian coordinates for each atom.
            atom_type (Tensor): Tensor of shape (B, A). Data type is int.
                Type index (atomic number) of atom types.
            pbc_box (Tensor): Tensor of shape (B, D). Data type is float.
                Box size of periodic boundary condition
            bonds (Tensor): Tensor of shape (B, A, A). Data type is int.
                Types index of bond connected with two atoms
            bond_mask (Tensor): Tensor of shape (B, A, A). Data type is bool.
                Mask for bonds

        Returns:
            outputs (Tensor):    Tensor of shape (B, A, O). Data type is float.
        """

        if self.atom_type is None:
            # The shape looks like (B, A)
            atom_mask = atom_type > 0
        else:
            # The shape looks like (1, A)
            atom_type = self.atom_type
            atom_mask = self.atom_mask

        num_atoms = atom_mask.shape[-1]

        distance = None
        vectors = None
        dis_mask = None
        if coordinate is not None:
            # The shape looks like (B, A, D)
            coordinate *= self.input_unit_scale

            if self.pbc_box is not None:
                pbc_box = self.pbc_box

            # The shape looks like (B, A, A, D) = (B, 1, A, D) - (B, A, 1, D)
            vectors = coordinate.expand_dims(-2) - coordinate.expand_dims(-3)
            if self.use_pbc and self.pbc_box is not None:
                if vectors.ndim > 2:
                    shape = pbc_box.shape[:1] + (1,) * (vectors.ndim - 2) + pbc_box.shape[-1:]
                    pbc_box = pbc_box.reshape(shape)
                box_nograd = F.stop_gradient(pbc_box)
                inv_box = msnp.reciprocal(box_nograd)
                vectors -= box_nograd * F.floor(vectors * inv_box + 0.5)
                vectors = vectors * inv_box * pbc_box
            # The shape looks like (B, A, A) = (B, A, 1) & (B, 1, A)
            dis_mask = F.logical_and(F.expand_dims(atom_mask, -1), F.expand_dims(atom_mask, -2))
            # The shape looks like (A, A)
            diagonal = F.logical_not(F.eye(num_atoms, num_atoms, ms.bool_))
            # The shape looks like (B, A, A) & (A, A)
            dis_mask = F.logical_and(dis_mask, diagonal)

            # Add a non-zero value to the neighbour_vector whose mask value is False
            # to prevent them from becoming zero values after Norm operation,
            # which could lead to auto-differentiation errors
            # The shape looks like (B, A, A)
            large_dis = F.broadcast_to(self.large_dis, dis_mask.shape)
            large_dis = F.select(dis_mask, F.zeros_like(large_dis), large_dis)
            # The shape looks like (B, A, A, D) = (B, A, A, D) + (B, A, A, 1)
            vectors += F.expand_dims(large_dis, -1)

            if self.norm_last_dim is None:
                distance = ops.norm(vectors, 2, -1)
            else:
                distance = self.norm_last_dim(vectors)

        if self.bonds is not None:
            bonds = self.bonds
            bond_mask = self.bond_mask

        node_emb, node_mask, edge_emb, edge_mask, edge_cutoff = self.embedding(atom_type=atom_type,
                                                                               atom_mask=atom_mask,
                                                                               distance=distance,
                                                                               dis_mask=dis_mask,
                                                                               bond=bonds,
                                                                               bond_mask=bond_mask,
                                                                               )

        node_rep, edge_rep = self.model(node_emb=node_emb,
                                        node_mask=node_mask,
                                        edge_emb=edge_emb,
                                        edge_mask=edge_mask,
                                        edge_cutoff=edge_cutoff,
                                        )

        if self.readout is None:
            return node_rep, node_rep

        if atom_mask is not None:
            num_atoms = msnp.count_nonzero(F.cast(atom_mask, ms.int16), axis=-1, keepdims=True)

        outputs = ()
        for i in range(self.num_readouts):
            output = self.readout[i](node_rep=node_rep,
                                     edge_rep=edge_rep,
                                     node_emb=node_emb,
                                     edge_emb=edge_emb,
                                     edge_cutoff=edge_cutoff,
                                     atom_type=atom_type,
                                     atom_mask=atom_mask,
                                     distance=distance,
                                     dis_mask=dis_mask,
                                     dis_vec=vectors,
                                     bond=bonds,
                                     bond_mask=bond_mask,
                                     )

            if self.use_scaleshift and self.scaleshift is not None:
                output = self.scaleshift[i](output, atom_type, num_atoms)

            outputs += (output,)

        if self.num_readouts == 1:
            return outputs[0]

        return outputs

    def _check_readout_index(self, readout_idx: int):
        if readout_idx >= self.num_readouts:
            raise ValueError(f'The index ({readout_idx}) is exceed '
                             f'the number of readout ({self.num_readouts})')
        return self
