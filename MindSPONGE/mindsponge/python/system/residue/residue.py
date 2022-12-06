# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
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
from typing import Union
import numpy as np
from numpy import ndarray
import mindspore as ms
from mindspore import numpy as msnp
from mindspore import jit_class
from mindspore.ops import functional as F
from mindspore.common import Tensor

from ...function.functions import get_integer
from ...data.elements import elements, element_set, element_dict, atomic_mass
from ...data.template import get_template, get_template_index


@jit_class
class Residue:
    r"""
    Class for residue in molecule.

    Args:
        atom_name (list):               Atom name. Can be ndarray or list of str. Default: None.
        atom_type (list):               Atom type. Can be ndarray or list of str. Default: None.
        atom_mass (Tensor):             Tensor of shape (B, A). Data type is float.
                                        Atom mass. Default: None.
        atom_charge (Tensor):           Tensor of shape (B, A). Data type is float.
                                        Atom charge. Default: None.
        atomic_number (Tensor):         Tensor of shape (B, A). Data type is float.
                                        Atomic number. Default: None.
        bond (Tensor):                  Tensor of shape (B, b, 2) or (1, b, 2). Data type is int.
                                        Bond index. Default: None.
        head_atom (int):                Index of the head atom to connect with the previous residue.
                                        Default: None.
        tail_atom (int):                Index of the tail atom to connect with the next residue.
                                        Default: None.
        start_index (int):              The start index of the first atom in this residue.
        name (str):                     Name of the residue.
                                        Examples: 'SOL', 'CL'. Indicating water molecule and Na+ ion respectively.
                                        The residue that is not defined usually called 'MOL'.
                                        Default: 'MOL'.
        template (Union[dict, str]):    Template of residue.
                                        The key of the dict are base, template, the name of molecule and so on.
                                        The value of the dict is file name.
                                        Default: None.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        A:  Number of atoms.
        b:  Number of bonds.
    """

    def __init__(self,
                 atom_name: list = None,
                 atom_type: list = None,
                 atom_mass: Tensor = None,
                 atom_charge: Tensor = None,
                 atomic_number: Tensor = None,
                 bond: Tensor = None,
                 head_atom: int = None,
                 tail_atom: int = None,
                 start_index: int = 0,
                 name: str = 'MOL',
                 template: Union[dict, str] = None,
                 ):

        self._name = name

        self.atom_name = None
        if atom_name is not None:
            self.atom_name = np.array(atom_name, np.str_)
            if self.atom_name.ndim == 1:
                self.atom_name = np.expand_dims(self.atom_name, 0)
            if self.atom_name.ndim != 2:
                raise ValueError('The rank of "atom_name" must be 1 or 2!')

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
                raise ValueError('Cannot found the key "' + str(self._name) + ' in template."')

            template = template.get(self._name)

            atom_mass = np.array(template.get('atom_mass'), np.float32)
            atomic_number = np.array(template.get('atom_mass'), np.int32)

            atom_type = template.get('atom_type')
            if atom_type is not None:
                atom_type = np.array(atom_type, np.str_)

            atom_charge = template.get('atom_charge')
            if atom_charge is not None:
                atom_charge = np.array(atom_charge, np.float32)

            bond = template.get('bond')
            if bond is not None:
                bond = np.array(bond, np.int32)

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

                if bond is not None:
                    bond = self._get_bond(template, atom_index)

                serial_list: list = atom_index.reshape(-1).tolist()

                if head_atom is not None:
                    head_atom = serial_list.index(head_atom)

                if tail_atom is not None:
                    tail_atom = serial_list.index(tail_atom)

        if self.atom_name is None and atomic_number is None:
            raise ValueError('atom_name and atomic_number cannot both be None')

        if atomic_number is not None:
            self.atomic_number = Tensor(atomic_number, ms.int32)
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
                self.atomic_number = Tensor(atomic_number, ms.int32).reshape(self.atom_name.shape)

        if self.atomic_number.shape != self.atom_name.shape:
            if self.atomic_number.shape[-1] == self.atom_name.shape[-1]:
                if self.atomic_number.shape[0] == 1:
                    self.atomic_number = msnp.broadcast_to(self.atomic_number, self.atom_name.shape)
                elif self.atom_name.shape[0] == 1:
                    self.atom_name = msnp.broadcast_to(self.atom_name, self.atomic_number.shape)

            raise ValueError('The shape of "atomic_number" ' + str(self.atomic_number) +
                             ' does not match the shape of "atom_name" ' + str(self.atom_name) + '!')

        if atom_type is None:
            self.atom_type = self.atom_name.copy()
        else:
            self.atom_type = np.array(atom_type)
            if self.atom_type.ndim == 1:
                self.atom_type = np.expand_dims(self.atom_type, 0)
            if self.atom_type.shape != self.atom_name.shape:
                raise ValueError('The shape of "atom_type" ' + str(self.atom_type.shape) +
                                 ' must be equal to the shape of "atom_name" ' + str(self.atom_name.shape) + '!')

        self.num_atoms = self.atom_name.shape[-1]
        self.multi_system = self.atom_name.shape[0]

        self.start_index = get_integer(start_index)
        # (A'')
        self._index = msnp.arange(self.num_atoms)
        self.system_index = self._index + start_index

        # (1,A') or (B,A')
        if atom_mass is None:
            if atomic_number is None:
                self.atom_mass = msnp.ones(
                    self.atom_name.shape, dtype=np.float32)
            else:
                self.atom_mass = Tensor(
                    atomic_mass[self.atomic_number.asnumpy()], ms.float32)
        else:
            self.atom_mass = Tensor(atom_mass, ms.float32)
            if self.atom_mass.ndim == 1:
                self.atom_mass = F.expand_dims(self.atom_mass, 0)
            if self.atom_mass.ndim != 2:
                raise ValueError('The rank of "atom_mass" must be 1 or 2!')
            if self.atom_mass.shape[-1] != self.num_atoms:
                raise ValueError('The last dimension of atom_mass (' + str(self.atom_mass.shape[-1]) +
                                 ') must be equal to the number of atoms (' + str(self.num_atoms) + ')!')
            if self.atom_mass.shape[0] > 1 and self.atom_mass.shape[0] != self.multi_system:
                raise ValueError('The first dimension of atom_mass (' + str(self.atom_mass.shape[0]) +
                                 ') does not match the number of the number of system multi_system (' +
                                 str(self.multi_system) + ')!')

        # (B,A')
        self.atom_mask = F.logical_and(
            self.atomic_number > 0, self.atom_mass > 0)
        self.inv_mass = msnp.where(
            self.atom_mask, msnp.reciprocal(self.atom_mass), 0)
        # (B,1)
        self.natom_tensor = msnp.sum(
            F.cast(self.atom_mask, ms.float32), -1, keepdims=True)
        self.total_mass = msnp.sum(self.atom_mass, -1, keepdims=True)

        # (B,A')
        self.atom_charge = atom_charge
        if atom_charge is not None:
            self.atom_charge = Tensor(atom_charge, ms.float32)
            if self.atom_charge.ndim == 1:
                self.atom_charge = F.expand_dims(self.atom_charge, 0)
            if self.atom_charge.ndim != 2:
                raise ValueError('The rank of "atom_charge" must be 1 or 2!')
            if self.atom_charge.shape[-1] != self.num_atoms:
                raise ValueError('The last dimension of atom_charge (' + str(self.atom_charge.shape[-1]) +
                                 ') must be equal to the num_atoms (' + str(self.num_atoms) + ')!')
            if self.atom_charge.shape[0] != self.multi_system and self.atom_charge.shape[0] != 1:
                raise ValueError('The first dimension of atom_charge (' + str(self.atom_charge.shape[0]) +
                                 ') must be equal to 1 or the number of the number of system multi_system (' +
                                 str(self.multi_system) + ')!')

        # (B,C,2)
        self.bond = bond
        self.bond_mask = None
        if bond is not None:
            self.bond = Tensor(bond, ms.int32)
            if self.bond.shape[-1] != 2:
                raise ValueError('The last dimension of bond must 2!')
            if self.bond.ndim == 2:
                self.bond = F.expand_dims(self.bond, 0)
            self.bond_mask = self.bond < self.num_atoms

        # (B,1)
        self.head_atom = head_atom
        if head_atom is not None:
            self.head_atom = Tensor([head_atom,], ms.int32).reshape(-1, 1)
            if self.head_atom.shape[0] != self.multi_system and self.head_atom.shape[0] != 1:
                raise ValueError('The first dimension of head_atom (' + str(self.head_atom.shape[0]) +
                                 ') does not match the number of system multi_system (' + str(self.multi_system) + ')!')
            if (self.head_atom >= self.num_atoms).any():
                raise ValueError(
                    'The value of head_atom has exceeds the number of atoms.')

        # (B,1)
        self.tail_atom = tail_atom
        if tail_atom is not None:
            self.tail_atom = Tensor([tail_atom,], ms.int32).reshape(-1, 1)
            if self.tail_atom.shape[0] != self.multi_system and self.tail_atom.shape[0] != 1:
                raise ValueError('The first dimension of tail_atom (' + str(self.tail_atom.shape[0]) +
                                 ') does not match the number of system multi_system (' + str(self.multi_system) + ')!')
            if (self.tail_atom >= self.num_atoms).any():
                raise ValueError(
                    'The value of tail_atom has exceeds the number of atoms.')

    @property
    def name(self) -> str:
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
        bond = np.array(template.get('bond'))
        if atom_index is not None:
            bond_list = bond.reshape(-1).tolist()
            if atom_index.ndim == 2 and atom_index.shape[0] > 1:
                bond_ = []
                for serial in atom_index:
                    serial: list = serial.tolist()
                    b = np.array([serial.index(idx)
                                  for idx in bond_list]).reshape(bond.shape)
                    bond_.append(b)
                bond = np.stack(bond_, axis=0)
            else:
                serial: list = atom_index.reshape(-1).tolist()
                bond = np.array([serial.index(idx) for idx in bond_list]).reshape(bond.shape)
        return bond

    def build_atom_mass(self, template: dict):
        """
        This function is built to attach the mass of atom to the index of atom.

        Args:
            template (Union[dict, str]):    Template of residue.
                                            The key of the dict are base, template, the name of molecule and so on.
                                            The value of the dict is file name.
                                            Default: None.
        """
        atom_index = get_template_index(template, self.atom_name)
        self.atom_mass = Tensor(self._get_atom_mass(template, atom_index), ms.float32)
        return self

    def build_atomic_number(self, template: dict):
        """
        This function is built to attach the atomic number of atom to the index of atom.

        Args:
            template (Union[dict, str]):    Template of residue.
                                            The key of the dict are base, template, the name of molecule and so on.
                                            The value of the dict is file name.
                                            Default: None.
        """
        atom_index = get_template_index(template, self.atom_name)
        self.atomic_number = Tensor(self._get_atomic_number(template, atom_index), ms.int32)
        return self

    def build_atom_type(self, template: dict):
        """
        This function is built to attach the type of atom to the index of atom.

        Args:
            template (Union[dict, str]):    Template of residue.
                                            The key of the dict are base, template, the name of molecule and so on.
                                            The value of the dict is file name.
                                            Default: None.
        """
        atom_index = get_template_index(template, self.atom_name)
        self.atom_type = self._get_atom_type(template, atom_index)
        return self

    def build_atom_charge(self, template: dict):
        """
        This function is built to attach the chargre of atom to the index of atom.

        Args:
            template (Union[dict, str]):    Template of residue.
                                            The key of the dict are base, template, the name of molecule and so on.
                                            The value of the dict is file name.
                                            Default: None.
        """
        atom_index = get_template_index(template, self.atom_name)
        self.atom_charge = Tensor(self._get_atom_charge(template, atom_index), ms.float32)
        return self

    def build_bond(self, template: dict):
        """
        This function is built to attach the bonds of atom to the index of atom.

        Args:
            template (Union[dict, str]):    Template of residue.
                                            The key of the dict are base, template, the name of molecule and so on.
                                            The value of the dict is file name.
                                            Default: None.
        """
        atom_index = get_template_index(template, self.atom_name)
        self.bond = Tensor(self._get_bond(template, atom_index), ms.int32)
        return self

    def add_atom(self,
                 atom_name: str = None,
                 atom_type: str = None,
                 atom_mass: float = None,
                 atom_charge: float = None,
                 atomic_number: str = None,
                 ):
        """
        Set atom.

        Args:
            atom_name (Union[numpy.ndarray, list(str)]):    Atom name. Can be ndarray or list of str. Default: None.
            atom_type (Union[numpy.ndarray, list(str)]):    Atom type. Can be ndarray or list of str. Default: None.
            atom_mass (Tensor):                             Tensor of shape (B, A). Data type is float.
                                                            Atom mass. Default: None.
            atom_charge (Tensor):                           Tensor of shape (B, A). Data type is float.
                                                            Atom charge. Default: None.
            atomic_number (Tensor):                         Tensor of shape (B, A). Data type is float.
                                                            Atomic number. Default: None.
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

            raise ValueError('The shape of "atomic_number" '+str(atomic_number) +
                             ' does not match the shape of "atom_name" '+str(atom_name)+'!')

        atom_mask = F.logical_and(atomic_number > 0, atom_mass > 0)
        inv_mass = msnp.where(atom_mask, msnp.reciprocal(atom_mass), 0)

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
        self.natom_tensor = msnp.sum(
            F.cast(self.atom_mask, ms.int32), -1, keepdims=True)
        self.total_mass = msnp.sum(self.atom_mass, -1, keepdims=True)

        return self

    def broadcast_multiplicity(self, multi_system: int):
        """
        Broadcast the information to the number of multiple system.

        Args:
            multi_system (int):     Amount of multiple system.
        """
        if multi_system <= 0:
            raise ValueError('multi_system must be larger than 0!')
        if self.multi_system > 1:
            raise ValueError('The current the number of system multi_system ('+str(self.multi_system) +
                             ') is larger than 1 and cannot be broadcast!')

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
        if self.bond is not None:
            bond_shape = (self.multi_system,) + self.bond.shape[1:]
            self.bond = msnp.broadcast_to(self.bond, bond_shape)
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
        Set residue name of this residue.

        Args:
            name (str):             Name of the residue.
                                    Examples: 'SOL', 'CL'. Indicating water molecule and Na+ ion respectively.
                                    The residue that is not defined usually called 'MOL'.
                                    Default: 'MOL'.
        """
        self._name = name
        return self

    def set_start_index(self, start_index: int):
        """
        Set the start index of the first atom in this residue.

        Args:
            start_index (int):      The start index of the first atom in this residue.
        """
        if start_index < 0:
            raise ValueError('The start_index cannot be smaller than 0!')
        self.start_index = get_integer(start_index)
        index_shift = self.start_index - self.system_index[0]
        self.system_index += index_shift
        return self
