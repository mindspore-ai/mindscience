# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
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
Residue
"""

from operator import itemgetter
from typing import Union, List
import numpy as np
from numpy import ndarray
import mindspore as ms
from mindspore import numpy as msnp
try:
    # MindSpore 2.X
    from mindspore import jit_class
except ImportError:
    # MindSpore 1.X
    from mindspore import ms_class as jit_class
from mindspore.ops import functional as F
from mindspore.common import Tensor

from ...function.functions import get_integer, get_ms_array, get_arguments
from ...data.element import elements, element_set, element_dict, atomic_mass
from ...data.template import get_template, get_template_index


@jit_class
class Residue:
    r"""
    Base class for residue.
    The `Residue` Cell is the component of the `Molecule` (System) Cell.
    A `Residue` can represent not only an amino acid residue, but also a small molecule in a molecular system,
    such as a water molecule, an inorganic salt ion, etc. This means that the `Residue` Cell has
    a similar concept to the "residue" in a PDB file.

    Note:
        `Residue` Cell is only used to represent the atomic properties and bond connections,
        but does NOT contain atomic coordinates.

    Args:
        atom_name (Union[List[str], ndarray]): Array of atom name with data type `str`. Default: ``None``.
        atom_type (Union[List[str], ndarray]): Array of atom type with data type `str`. Default: ``None``.
        atom_mass (Union[Tensor, ndarray, List[float]]): Array of atom mass of shape :math:`(B, A)`
            with data type `float`. Default: ``None``.
        atom_charge (Union[Tensor, ndarray, List[float]]): Array of atom charge of shape :math:`(B, A)`
            with data type `float`. Default: ``None``.
        atomic_number (Union[Tensor, ndarray, List[float]]): Array of atomic number of shape :math:`(B, A)`
            with data type `int`. Default: ``None``.
        bonds (Union[Tensor, ndarray, List[int]]): Array of bond connection of shape :math:`(B, b, 2)`
            with data type `int`. Default: ``None``.
        settle_index (Union[Tensor, ndarray, List[int]]): Array of atom indices for SETTLE constraint algorithm,
            The shape of the array is :math:`(B, 3)` with data type `int`. The order of index is vertex atoms and two
            base atoms. Default: ``None``.
        settle_length (Union[Tensor, ndarray, List[float]]): Array of length for SETTLE constraint algorithm,
            The shape of the array is :math:`(B, 2)` with data type `int`. The order of length is leg and base.
            Default: ``None``.
        settle_unit (str): Unit value for SETTLE constraint algorithm. Default: ``None``.
        head_atom (int): Index of the head atom to connect with the previous residue. Default: ``None``.
        tail_atom (int): Index of the tail atom to connect with the next residue. Default: ``None``.
        start_index (int): The start index of the first atom in this residue. Default: ``0``.
        name (str): Name of the residue. Default: ``'MOL'``.
        template (Union[dict, str]): Template for residue. It can be a `dict` in MindSPONGE template
            format or a `str` for the filename of a MindSPONGE template file. If a `str` is given,
            it will first look for a file with the same name in the current directory. If file does
            not exist, it will search in the built-in template directory of MindSPONGE
            (`mindsponge.data.template`). Default: ``None``.
        length_unit (str): Unit of length. Default: ``None``.
        kwargs (dict): Other parameters dictionary.

    Note:
        B:  Batchsize, i.e. number of walkers in simulation
        A:  Number of atoms.
        b:  Number of bonds.
    """

    def __init__(self,
                 atom_name: Union[List[str], ndarray] = None,
                 atom_type: Union[List[str], ndarray] = None,
                 atom_mass: Union[Tensor, ndarray, List[float]] = None,
                 atom_charge: Union[Tensor, ndarray, List[float]] = None,
                 atomic_number: Union[Tensor, ndarray, List[float]] = None,
                 bonds: Union[Tensor, ndarray, List[int]] = None,
                 settle_index: Union[Tensor, ndarray, List[int]] = None,
                 settle_length: Union[Tensor, ndarray, List[float]] = None,
                 settle_unit: str = None,
                 head_atom: int = None,
                 tail_atom: int = None,
                 start_index: int = 0,
                 name: str = 'MOL',
                 template: Union[dict, str] = None,
                 length_unit: str = None,
                 **kwargs,
                 ):
        self._kwargs = get_arguments(locals(), kwargs)

        self._name = name
        self.length_unit = None

        self.atom_name = None
        if atom_name is not None:
            self.atom_name = np.array(atom_name, np.str_)
            if self.atom_name.ndim == 1:
                self.atom_name = np.expand_dims(self.atom_name, 0)
            if self.atom_name.ndim != 2:
                raise ValueError('The rank of "atom_name" must be 1 or 2!')

        settle = None
        if template is not None:
            template = get_template(template)
            if self._name is None:
                if len(template) == 1:
                    self._name = list(template.keys())[0]
                    template = template.get(self._name)
                else:
                    raise ValueError('The name cannot be None when the number of '
                                     'keys in template is larger than 1!')
            elif self._name not in template.keys():
                raise ValueError(f'Cannot found the key "{self._name}" in template.')

            template = template.get(self._name)

            atom_mass = np.array(template.get('atom_mass'), np.float32)
            atomic_number = np.array(template.get('atomic_number'), np.int32)

            atom_type = template.get('atom_type')
            if atom_type is not None:
                atom_type = np.array(atom_type, np.str_)

            atom_charge = template.get('atom_charge')
            if atom_charge is not None:
                atom_charge = np.array(atom_charge, np.float32)

            bonds = template.get('bonds')
            if bonds is not None:
                bonds = np.array(bonds, np.int32)

            self.length_unit = template.get('length_unit')
            settle: dict = template.get('settle')

            head_atom = template.get('head_atom')
            tail_atom = template.get('tail_atom')

            if self.atom_name is None:
                self.atom_name = np.array(template.get('atom_name'), np.str_).reshape(1, -1)
            else:
                atom_index = get_template_index(template, self.atom_name)
                atom_mass = atom_mass[atom_index]
                atomic_number = atomic_number[atom_index]

                if atom_type is not None:
                    atom_type = atom_type[atom_index]

                if atom_charge is not None:
                    atom_charge = atom_charge[atom_index]

                if bonds is not None:
                    bonds = self._get_bond(template, atom_index)

                serial_list: list = atom_index.reshape(-1).tolist()

                if head_atom is not None:
                    head_atom = serial_list.index(head_atom)

                if tail_atom is not None:
                    tail_atom = serial_list.index(tail_atom)

        if self.atom_name is None and atomic_number is None:
            raise ValueError('atom_name and atomic_number cannot both be None')

        if settle:
            settle_index = get_template_index(template, self.atom_name)
            settle_unit = settle.get('length_unit')
            distance = settle.get('distance')
            settle_length = np.array([[distance['OW-HW'], distance['HW-HW']]], np.float32)

        if atomic_number is not None:
            self.atomic_number = get_ms_array(atomic_number, ms.int32)
            if self.atomic_number.ndim == 1:
                self.atomic_number = F.expand_dims(self.atomic_number, 0)
            if self.atomic_number.ndim != 2:
                raise ValueError('The rank of "atomic_number" must be 1 or 2!')

        if self.atom_name is None:
            self.atom_name = np.array(elements[self.atomic_number.asnumpy()], np.str_)

        if atomic_number is None:
            atom_name_list = self.atom_name.reshape(-1).tolist()
            if set(atom_name_list) - element_set:
                self.atomic_number = msnp.ones(self.atom_name.shape, ms.int32)
            else:
                atomic_number = itemgetter(*atom_name_list)(element_dict)
                self.atomic_number = get_ms_array(atomic_number, ms.int32).reshape(self.atom_name.shape)

        if self.atomic_number.shape != self.atom_name.shape:
            if self.atomic_number.shape[-1] == self.atom_name.shape[-1]:
                if self.atomic_number.shape[0] == 1:
                    self.atomic_number = msnp.broadcast_to(self.atomic_number, self.atom_name.shape)
                elif self.atom_name.shape[0] == 1:
                    self.atom_name = msnp.broadcast_to(self.atom_name, self.atomic_number.shape)

            raise ValueError(f'The shape of "atomic_number" {self.atomic_number} does not match '
                             f'the shape of "atom_name" {self.atom_name}!')

        if atom_type is None:
            self.atom_type = self.atom_name.copy()
        else:
            self.atom_type = np.array(atom_type)
            if self.atom_type.ndim == 1:
                self.atom_type = np.expand_dims(self.atom_type, 0)
            if self.atom_type.shape != self.atom_name.shape:
                raise ValueError(f'The shape of "atom_type" {self.atom_type.shape} must be equal to '
                                 f'the shape of "atom_name" {self.atom_name.shape}!')

        self.num_atoms = self.atom_name.shape[-1]
        self.multi_system = self.atom_name.shape[0]

        self.start_index = get_integer(start_index)
        # (A'')
        self._index = Tensor(np.arange(self.num_atoms), ms.int32)
        self.system_index = self._index + start_index

        # (1,A') or (B,A')
        if atom_mass is None:
            if atomic_number is None:
                self.atom_mass = msnp.ones(
                    self.atom_name.shape, dtype=np.float32)
            else:
                self.atom_mass = get_ms_array(atomic_mass[self.atomic_number.asnumpy()], ms.float32)
        else:
            self.atom_mass = get_ms_array(atom_mass, ms.float32)
            if self.atom_mass.ndim == 1:
                self.atom_mass = F.expand_dims(self.atom_mass, 0)
            if self.atom_mass.ndim != 2:
                raise ValueError('The rank of "atom_mass" must be 1 or 2!')
            if self.atom_mass.shape[-1] != self.num_atoms:
                raise ValueError(f'The last dimension of atom_mass ({self.atom_mass.shape[-1]} must be equal to '
                                 f'the number of atoms ({self.num_atoms}!')
            if self.atom_mass.shape[0] > 1 and self.atom_mass.shape[0] != self.multi_system:
                raise ValueError(f'The first dimension of atom_mass ({self.atom_mass.shape[0]} does not match '
                                 f'the number of the number of system multi_system ({self.multi_system})!')

        # (B,A')
        self.atom_mask = F.logical_and(self.atomic_number > 0, self.atom_mass > 0)
        self.inv_mass = Tensor(np.where(self.atom_mask.asnumpy(),
                                        np.reciprocal(self.atom_mass.asnumpy()), 0))
        # (B,1)
        self.natom_tensor = msnp.sum(self.atom_mask, -1, keepdims=True)
        self.total_mass = msnp.sum(self.atom_mass, -1, keepdims=True)

        # (B,A')
        self.atom_charge = atom_charge
        if atom_charge is not None:
            self.atom_charge = get_ms_array(atom_charge, ms.float32)
            if self.atom_charge.ndim == 1:
                self.atom_charge = F.expand_dims(self.atom_charge, 0)
            if self.atom_charge.ndim != 2:
                raise ValueError('The rank of "atom_charge" must be 1 or 2!')
            if self.atom_charge.shape[-1] != self.num_atoms:
                raise ValueError(f'The last dimension of atom_charge ({self.atom_charge.shape[-1]} must be equal to '
                                 f'the num_atoms ({self.num_atoms})!')
            if self.atom_charge.shape[0] != self.multi_system and self.atom_charge.shape[0] != 1:
                raise ValueError(f'The first dimension of atom_charge ({self.atom_charge.shape[0]} must be equal to '
                                 f'the number of the number of system multi_system ({self.multi_system}) or 1!')

        # (B,C,2)
        self.bonds = bonds
        self.bond_mask = None
        if bonds is not None:
            self.bonds = get_ms_array(bonds, ms.int32)
            if self.bonds.shape[-1] != 2:
                raise ValueError('The last dimension of bond must 2!')
            if self.bonds.ndim == 2:
                self.bonds = F.expand_dims(self.bonds, 0)
            self.bond_mask = self.bonds < self.num_atoms

        # (B,1)
        self.head_atom = head_atom
        if head_atom is not None:
            self.head_atom = get_ms_array([head_atom,], ms.int32).reshape(-1, 1)
            if self.head_atom.shape[0] != self.multi_system and self.head_atom.shape[0] != 1:
                raise ValueError(f'The first dimension of head_atom ({self.head_atom.shape[0]} does not match '
                                 f'the number of system multi_system ({self.multi_system})!')
            if (self.head_atom >= self.num_atoms).any():
                raise ValueError(
                    'The value of head_atom has exceeds the number of atoms.')

        # (B, 3)
        self.settle_index = settle_index
        # (B, 2)
        self.settle_length = settle_length
        self.settle_unit = settle_unit
        if settle_index is not None:
            self.settle_index = get_ms_array(settle_index, ms.int32)
            if self.settle_index.shape[-1] != 3:
                raise ValueError('The last dimension of settle_index must 2!')
            if self.settle_index.ndim == 1:
                self.settle_index = F.expand_dims(self.settle_index, 0)

            if settle_length is None:
                raise ValueError('`settle_length` cannot be None when `settle_index` is given')
            self.settle_length = get_ms_array(settle_length, ms.float32)
            if self.settle_length.shape[-1] != 2:
                raise ValueError('The last dimension of settle_index must 2!')
            if self.settle_length.ndim == 1:
                self.settle_length = F.expand_dims(self.settle_length, 0)

        # (B,1)
        self.tail_atom = tail_atom
        if tail_atom is not None:
            self.tail_atom = get_ms_array(
                [tail_atom], ms.int32).reshape(-1, 1)
            if self.tail_atom.shape[0] != self.multi_system and self.tail_atom.shape[0] != 1:
                raise ValueError(f'The first dimension of tail_atom ({self.tail_atom.shape[0]}) does not match '
                                 f'the number of system multi_system ({self.multi_system})!')
            if (self.tail_atom >= self.num_atoms).any():
                raise ValueError(
                    'The value of tail_atom has exceeds the number of atoms.')

    @property
    def name(self) -> str:
        """
        Get the name of the residue.

        Returns:
            str, the name of the residue.
        """
        return str(self._name)

    @classmethod
    def _get_atom_mass(cls, template: dict, atom_index: ndarray = None) -> ndarray:
        """get atom mass from template and atom index"""
        atom_mass = np.array(template.get('atom_mass'), np.float32)
        if atom_index is not None:
            atom_mass = atom_mass[atom_index]
        return atom_mass

    @classmethod
    def _get_atomic_number(cls, template: dict, atom_index: ndarray = None) -> ndarray:
        """get atomic number from template and atom index"""
        atomic_number = np.array(template.get('atomic_number'), np.int32)
        if atom_index is not None:
            atomic_number = atomic_number[atom_index]
        return atomic_number

    @classmethod
    def _get_atom_type(cls, template: dict, atom_index: ndarray = None) -> ndarray:
        """get atom type from template and atom index"""
        atom_type = np.array(template.get('atom_type'), np.str_)
        if atom_index is not None:
            atom_type = atom_type[atom_index]
        return atom_type

    @classmethod
    def _get_atom_charge(cls, template: dict, atom_index: ndarray = None) -> ndarray:
        """get atom charge from template and atom index"""
        atom_charge = np.array(template['atom_charge'], np.float32)
        if atom_index is not None:
            atom_charge = atom_charge[atom_index]
        return atom_charge

    @classmethod
    def _get_bond(cls, template: dict, atom_index: ndarray = None) -> ndarray:
        """get bond from template and atom index"""
        bonds = template.get('bonds')
        if bonds is None:
            return None
        bonds = np.array(bonds, np.int32)
        if atom_index is not None:
            bond_list = bonds.reshape(-1).tolist()
            if atom_index.ndim == 2 and atom_index.shape[0] > 1:
                bond_ = []
                for serial in atom_index:
                    serial: list = serial.tolist()
                    b = np.array([serial.index(idx)
                                  for idx in bond_list]).reshape(bonds.shape)
                    bond_.append(b)
                bonds = np.stack(bond_, axis=0)
            else:
                serial: list = atom_index.reshape(-1).tolist()
                bond_list = np.array(bond_list).reshape((-1, 2))
                itsct = np.where(np.isin(bond_list, serial).sum(axis=-1) > 1)[0]
                bond_list = bond_list[itsct].reshape(-1)
                bonds = np.array([serial.index(idx) for idx in bond_list]).reshape((-1, 2))
        return bonds

    def build_atom_mass(self, template: dict):
        """
        According to the name of the atom, find the index of the atom in the template.
        Get atom mass of the atom with the index in the template and build it into the residue.

        Args:
            template(dict): Template for residue.
        """
        atom_index = get_template_index(template, self.atom_name)
        self.atom_mass = Tensor(self._get_atom_mass(template, atom_index), ms.float32)
        return self

    def build_atomic_number(self, template: dict):
        """
        According to the name of the atom, find the index of the atom in the template.
        Get atomic number of the atom with the index in the template and build it into the residue.

        Args:
            template(dict): Template for residue.
        """
        atom_index = get_template_index(template, self.atom_name)
        self.atomic_number = Tensor(self._get_atomic_number(template, atom_index), ms.int32)
        return self

    def build_atom_type(self, template: dict):
        """
        According to the name of the atom, find the index of the atom in the template.
        Get atom type of the atom with the index in the template and build it into the residue.

        Args:
            template(dict): Template for residue.
        """
        atom_index = get_template_index(template, self.atom_name)
        self.atom_type = self._get_atom_type(template, atom_index)
        return self

    def build_atom_charge(self, template: dict):
        """
        According to the name of the atom, find the index of the atom in the template.
        Get atom charge of the atom with the index in the template and build it into the residue.

        Args:
            template(dict): Template for residue.
        """
        atom_index = get_template_index(template, self.atom_name)
        self.atom_charge = Tensor(self._get_atom_charge(template, atom_index), ms.float32)
        return self

    def build_bond(self, template: dict):
        """
        According to the name of the atom, find the index of the atom in the template.
        Get bond of the atom with the index in the template and build it into the residue.

        Args:
            template(dict): Template for residue.
        """
        atom_index = get_template_index(template, self.atom_name)
        bonds = self._get_bond(template, atom_index)
        self.bonds = get_ms_array(bonds, ms.int32)
        return self

    def build_settle(self, template: dict):
        """
        According to the type of the atoms, find the indices and length for of SETTLE algorithm
        in the template. Get the indices and length in the template and build it into the residue.

        Args:
            template(dict): Template for residue.
        """


        atom_index = get_template_index(template, self.atom_name)
        self.bonds = Tensor(self._get_bond(template, atom_index), ms.int32)
        return self

    def add_atom(self,
                 atom_name: str = None,
                 atom_type: str = None,
                 atom_mass: float = None,
                 atom_charge: float = None,
                 atomic_number: str = None,
                 ):
        """
        Add an atom to the residue.

        Args:
            atom_name(str):     Atom name. Default: ``None``.
            atom_type(str):     Atom type. Default: ``None``.
            atom_mass(float):   Atom mass. Default: ``None``.
            atom_charge(float): Atom charge. Default: ``None``.
            atomic_number(str): Atomic number. Default: ``None``.
        """

        if atom_name is None and atomic_number is None:
            raise ValueError('atom_name and atomic_number cannot both be None')

        shape = (self.multi_system, 1)

        if atom_name is not None:
            atom_name = np.array(atom_name, np.str_)
            atom_name = np.broadcast_to(atom_name, shape)

        if atomic_number is not None:
            atomic_number = Tensor(atomic_number, ms.int32)
            atomic_number = msnp.broadcast_to(atomic_number, shape)

        if atom_name is None:
            atom_name = elements[atomic_number.asnumpy()]

        if atom_mass is None:
            if atomic_number is None:
                atom_mass = msnp.ones(atom_name.shape, dtype=np.float32)
            else:
                atom_mass = Tensor(
                    atomic_mass[atomic_number.asnumpy()], ms.float32)
        else:
            atom_mass = Tensor(atom_mass, ms.float32)
            atom_mass = np.broadcast_to(atom_mass, shape)

        if atomic_number is None:
            atom_name_list = atom_name.reshape(-1).tolist()
            if set(atom_name_list) - element_set:
                atomic_number = msnp.ones(atom_name.shape, ms.int32)
            else:
                atomic_number = itemgetter(*atom_name_list)(element_dict)
                atomic_number = Tensor(
                    atomic_number, ms.int32).reshape(atom_name.shape)

        if atomic_number.shape != atom_name.shape:
            if atomic_number.shape[-1] == atom_name.shape[-1]:
                if atomic_number.shape[0] == 1:
                    atomic_number = msnp.broadcast_to(
                        atomic_number, atom_name.shape)
                elif atom_name.shape[0] == 1:
                    atom_name = msnp.broadcast_to(
                        atom_name, atomic_number.shape)

            raise ValueError(f'The shape of "atomic_number" {atomic_number} does not match '
                             f'the shape of "atom_name" {atom_name}!')

        atom_mask = F.logical_and(atomic_number > 0, atom_mass > 0)
        inv_mass = Tensor(np.where(atom_mask.asnumpy(), np.reciprocal(atom_mass.asnumpy()), 0))

        if atom_type is None:
            atom_type = atom_name.copy()
        else:
            atom_type = np.array(atom_type)
            atom_type = np.broadcast_to(atom_type, shape)

        if atom_charge is not None:
            atom_charge = Tensor(atom_charge, ms.float32)
            atom_charge = np.broadcast_to(atom_charge, shape)

        self.atom_name = np.concatenate((self.atom_name, atom_name), axis=-1)
        self.atom_type = np.concatenate((self.atom_type, atom_type), axis=-1)
        self.atom_mass = F.concat((self.atom_mass, atom_mass), -1)
        self.atom_mask = F.concat((self.atom_mask, atom_mask), -1)
        self.atomic_number = F.concat((self.atomic_number, atomic_number), -1)
        self.inv_mass = F.concat((self.inv_mass, inv_mass), -1)
        if self.atom_charge is None and atom_charge is not None:
            self.atom_charge = msnp.zeros(
                (self.multi_system, self.num_atoms), ms.float32)
        if self.atom_charge is not None and atom_charge is None:
            atom_charge = msnp.zeros((self.multi_system, 1), ms.float32)
        if atom_charge is not None or self.atom_charge is not None:
            self.atom_charge = F.concat((self.atom_charge, atom_charge), -1)

        self.num_atoms = self.atom_name.shape[-1]
        self._index = msnp.arange(self.num_atoms)
        self.system_index = self._index + self.start_index
        self.natom_tensor = msnp.sum(self.atom_mask, -1, keepdims=True)
        self.total_mass = msnp.sum(self.atom_mass, -1, keepdims=True)

        return self

    def broadcast_multiplicity(self, multi_system: int):
        """
        Broadcast the information to the number of multiple system.

        Args:
            multi_system(int):  The number of multiple systems.
        """
        if multi_system <= 0:
            raise ValueError('multi_system must be larger than 0!')
        if self.multi_system > 1:
            raise ValueError(f'The current the number of system multi_system ({self.multi_system}) '
                             f'is larger than 1 and cannot be broadcast!')

        self.multi_system = multi_system
        self.atom_name = msnp.broadcast_to(self.atom_name, (self.multi_system, -1))
        self.atom_type = msnp.broadcast_to(self.atom_mass, (self.multi_system, -1))
        self.atomic_number = msnp.broadcast_to(self.atomic_number, (self.multi_system, -1))
        self.atom_mass = msnp.broadcast_to(self.atom_mass, (self.multi_system, -1))
        self.atom_mask = msnp.broadcast_to(self.atom_mask, (self.multi_system, -1))
        self.inv_mass = msnp.broadcast_to(self.inv_mass, (self.multi_system, -1))
        self.total_mass = msnp.broadcast_to(self.total_mass, (self.multi_system, -1))
        self.natom_tensor = msnp.broadcast_to(self.natom_tensor, (self.multi_system, -1))
        if self.atom_charge is not None:
            self.atom_charge = msnp.broadcast_to(self.atom_charge, (self.multi_system, -1))
        if self.bonds is not None:
            bond_shape = (self.multi_system,) + self.bonds.shape[1:]
            self.bonds = msnp.broadcast_to(self.bonds, bond_shape)
            self.bond_mask = msnp.broadcast_to(self.bond_mask, bond_shape)
        if self.head_atom is not None:
            self.head_atom = msnp.broadcast_to(
                self.head_atom, (self.multi_system, -1))
        if self.tail_atom is not None:
            self.tail_atom = msnp.broadcast_to(
                self.tail_atom, (self.multi_system, -1))

        return self

    def set_name(self, name: str):
        """
        Set residue name.

        Args:
            name(str):  Residue name.
        """
        self._name = name
        return self

    def set_start_index(self, start_index: int):
        """
        Set the start index.

        Args:
            start_index(int):   The start index.
        """
        if start_index < 0:
            raise ValueError('The start_index cannot be smaller than 0!')
        self.start_index = get_integer(start_index)
        index_shift = self.start_index - self.system_index[0]
        self.system_index += index_shift
        return self
