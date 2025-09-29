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
Molecule
"""

import copy
import itertools
from typing import Union, Tuple
import numpy as np
from numpy import ndarray
import mindspore as ms
from mindspore import Parameter
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.nn import Cell
from mindspore.common import Tensor
from mindspore import numpy as msnp

from ..residue import Residue
from ...data.template import get_molecule
from ...function import functions as func
from ...function.units import Units, global_units
from ...function.functions import get_ndarray


class Molecule(Cell):
    r"""
    Cell for molecular system.

    Args:
        atoms (list):                    Atoms in system. Can be list of str or int. Default: None.
        atom_name (list):                Atom name. Can be ndarray or list of str. Default: None.
        atom_type (list):                Atom type. Can be ndarray or list of str. Default: None.
        atom_mass (Tensor):              Tensor of shape (B, A). Data type is float.
                                         Atom mass. Default: None.
        atom_charge (Tensor):            Tensor of shape (B, A). Data type is float.
                                         Atom charge. Default: None.
        atomic_number (Tensor):          Tensor of shape (B, A). Data type is float.
                                         Atomic number. Default: None.
        bond (Tensor):                   Tensor of shape (B, b, 2) or (1, b, 2). Data type is int.
                                         Bond index. Default: None.
        coordinate (Tensor):             Tensor of shape (B, A, D) or (1, A, D). Data type is float.
                                         Position coordinates of atoms. Default: None.
        pbc_box (Tensor):                Tensor of shape (B, D) or (1, D). Data type is float.
                                         Box of periodic boundary condition. Default: None.
        template (Union[dict, str]):     Template of residue.
                                         The key of the dict are base, template, the name of molecule and so on.
                                         The value of the dict is file name.
                                         Default: None.
        residue (Union[Residue, list]):  Residue parameter. Default: None.
        length_unit (str):               Length unit for position coordinates. Default: None.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        A:  Number of atoms.
        b:  Number of bonds.
        D:  Dimension of the simulation system. Usually is 3.
    """

    def __init__(self,
                 atoms: list = None,
                 atom_name: list = None,
                 atom_type: list = None,
                 atom_mass: Tensor = None,
                 atom_charge: Tensor = None,
                 atomic_number: Tensor = None,
                 bond: Tensor = None,
                 coordinate: Tensor = None,
                 pbc_box: Tensor = None,
                 template: Union[dict, str] = None,
                 residue: Union[Residue, list] = None,
                 length_unit: str = None,
                 ):

        super().__init__()

        if length_unit is None:
            self.units = global_units
        else:
            self.units = Units(length_unit)

        if template is not None:
            molecule, template = get_molecule(template)
            residue: list = []
            for res in molecule.get('residue'):
                residue.append(Residue(name=res, template=template))
            if coordinate is None:
                coordinate = np.array(molecule.get('coordinate'), np.float32)
                coordinate *= self.units.convert_length_from(molecule.get('length_unit'))

        self.num_residue = 1
        if residue is None or not residue:
            if atoms is not None:
                atoms = get_ndarray(atoms)
                if np.issubdtype(atoms.dtype, np.integer):
                    if atomic_number is None:
                        atomic_number = atoms
                elif np.issubdtype(atoms.dtype, np.character):
                    if atom_name is None:
                        atom_name = atoms
                else:
                    raise TypeError(
                        'The dtype of atoms must be integer of character!')

            if atom_name is not None or atomic_number is not None:
                residue = Residue(
                    atom_name=atom_name,
                    atom_type=atom_type,
                    atom_mass=atom_mass,
                    atom_charge=atom_charge,
                    atomic_number=atomic_number,
                    bond=bond,
                )

        self.residue = None
        self.num_residue = 0
        if residue is not None:
            if isinstance(residue, list):
                self.residue = residue
            elif isinstance(residue, Residue):
                self.residue = [residue]
            else:
                raise ValueError(
                    'The type of residue must be Residue or list but got: '+str(type(residue)))

        # The number of multi_system of system
        self.multi_system = 1
        # A: number of atoms
        self.num_atoms = 0

        # (B,A)
        self.atom_name = None
        self.atom_type = None
        self.atom_mass = None
        self.atom_mask = None
        self.atomic_number = None
        self.inv_mass = None
        self.atom_charge = None

        # (B,R)
        self.residue_mass = None
        self.residue_name = None
        self.res_natom_tensor = None
        # (R)
        self.residue_pointer = None
        # (A)
        self.atom_resid = None
        self.image_index = None

        # (B,C,2)
        self.bond = None
        self.hydrogen_bond = None
        # (B,C): bond length for constraint
        self.bond_length = None

        # (B,A,D)
        self.coordinate = None
        # (B,D)
        self.pbc_box = None

        self.dimension = None
        self.num_walker = None
        self.degrees_of_freedom = None
        # (B,1)
        self.system_mass = None
        self.has_empty_atom = None
        self.system_natom = None

        self.use_pbc = False
        self.num_com = None
        self.image = None

        self.build_system()
        if self.residue is not None:
            self.build_space(coordinate, pbc_box)

    @property
    def length_unit(self):
        return self.units.length_unit

    def _check_pbc_box(self, pbc_box: Tensor):
        """check PBC box."""
        pbc_box = Tensor(pbc_box, ms.float32)
        if pbc_box.ndim == 1:
            pbc_box = F.expand_dims(pbc_box, 0)
        if pbc_box.ndim != 2:
            raise ValueError('The rank of pbc_box must be 1 or 2!')
        if pbc_box.shape[-1] != self.dimension:
            raise ValueError('The last dimension of "pbc_box" ('+str(pbc_box.shape[-1]) +
                             ') must be equal to the dimension of "coordinate" ('+str(self.dimension)+')!')
        if pbc_box.shape[0] > 1 and pbc_box.shape[0] != self.num_walker:
            raise ValueError('The first dimension of "pbc_box" ('+str(pbc_box.shape[0]) +
                             ') does not match the first dimension of "coordinate" ('+str(self.dimension)+')!')
        return Parameter(pbc_box, name='pbc_box', requires_grad=True)

    def move(self, shift: Tensor = None):
        """
        Move the coordinate of the system.

        Args:
            shift (Tensor):         Shift parameter. Default: None.
        """
        if shift is not None:
            self.update_coordinate(self.coordinate + Tensor(shift, ms.float32))
        return self

    def copy(self, shift: Tensor = None):
        """
        Return a Molecule that copy the parameters of this molecule.

        Args:
            shift (Tensor):         Shift parameter. Default: None.
        """
        coordinate = self.get_coordinate()
        if shift is not None:
            coordinate += Tensor(shift, ms.float32)
        return Molecule(
            residue=copy.deepcopy(self.residue),
            coordinate=coordinate,
            pbc_box=self.get_pbc_box(),
            length_unit=self.length_unit,
        )

    def add_residue(self, residue: Residue, coordinate: Tensor = None):
        """
        Add residue.

        Args:
            residue (Union[Residue, list]):  Residue parameter.
            coordinate (Tensor):             Tensor of shape (B, A, D) or (1, A, D). Data type is float.
                                             Position coordinates of atoms. Default: None.
        """
        if not isinstance(residue, list):
            if isinstance(residue, Residue):
                residue = [residue]
            else:
                raise TypeError('The type of residue must be Residue or list but got: ' +
                                str(type(residue)))

        self.residue.extend(copy.deepcopy(residue))
        self.build_system()
        if coordinate is None:
            natoms = 0
            for res in residue:
                natoms += res.num_atoms
            coordinate = msnp.ones((self.num_walker, natoms, self.dimension), ms.float32)

        coordinate = msnp.concatenate((self.coordinate, coordinate), axis=-2)
        self.build_space(coordinate, self.pbc_box)
        return self

    def append(self, system):
        """
        Append the system.

        Args:
            system (Molecule):    System parameter.
        """
        if not isinstance(system, Molecule):
            raise TypeError('For add, the type of system must be "Molecule" but got: ' +
                            str(type(system)))
        self.add_residue(system.residue, system.get_coordinate())
        return self

    def reduplicate(self, shift: Tensor):
        """
        Duplicate the system to double of the origin size.

        Args:
            shift (Tensor):         Shift parameter. Default: Tensor.
        """
        shift = Tensor(shift, ms.float32)
        self.residue.extend(copy.deepcopy(self.residue))
        self.build_system()
        coordinate = msnp.concatenate((self.coordinate, self.coordinate+shift), axis=-2)
        self.build_space(coordinate, self.pbc_box)
        return self

    def build_atom_type(self):
        """build atom type."""
        atom_type = ()
        for i in range(self.num_residue):
            atom_type += (self.residue[i].atom_type,)
        self.atom_type = np.concatenate(atom_type, axis=-1)
        return self

    def build_atom_charge(self):
        """build atom charge."""
        charges = []
        for i in range(self.num_residue):
            charges.append(self.residue[i].atom_charge is not None)

        if any(charges):
            atom_charge = ()
            for i in range(self.num_residue):
                if self.residue[i].atom_charge is None:
                    atom_charge += (msnp.zeros_like(self.residue[i].atom_mass),)
                else:
                    atom_charge += (self.residue[i].atom_charge,)
            self.atom_charge = msnp.concatenate(atom_charge, axis=-1)
        return self

    def build_system(self):
        """build the system by residues."""
        if self.residue is None:
            self.residue = None
            return self

        self.num_residue = len(self.residue)
        multi_system = []
        charges = []
        for i in range(self.num_residue):
            multi_system.append(self.residue[i].multi_system)
            charges.append(self.residue[i].atom_charge is not None)
        multi_system = list(set(multi_system))
        if len(multi_system) == 1:
            self.multi_system = multi_system[0]
        elif len(multi_system) == 2 and (multi_system[0] == 1 or multi_system[1] == 1):
            self.multi_system = max(multi_system)
        else:
            raise ValueError(
                'The multi_system of residues cannot be broadcast: '+str(multi_system))

        any_charge = any(charges)

        atom_name = ()
        atom_type = ()
        atom_mass = ()
        atom_mask = ()
        atom_charge = ()
        atomic_number = ()
        inv_mass = ()

        atom_resid = ()
        image_index = ()

        residue_mass = ()
        res_natom_tensor = ()

        bond = ()
        head_atom = None
        tail_atom = None

        pointer = 0
        residue_pointer = []
        residue_name = []

        for i in range(self.num_residue):
            if self.residue[i].multi_system != self.multi_system:
                self.residue[i].broadcast_multiplicity(self.multi_system)

            self.residue[i].set_start_index(pointer)
            residue_pointer.append(pointer)
            residue_name.append(self.residue[i].name)

            # (A')
            atom_resid += (msnp.full((self.residue[i].num_atoms,), i, ms.int32),)
            image_index += (msnp.full((self.residue[i].num_atoms,), pointer, ms.int32),)

            # (B,A')
            atom_name += (self.residue[i].atom_name,)
            atom_type += (self.residue[i].atom_type,)
            atom_mass += (self.residue[i].atom_mass,)
            atom_mask += (self.residue[i].atom_mask,)
            atomic_number += (self.residue[i].atomic_number,)
            inv_mass += (self.residue[i].inv_mass,)
            if any_charge:
                if self.residue[i].atom_charge is None:
                    atom_charge += (msnp.zeros_like(
                        self.residue[i].atom_mass),)
                else:
                    atom_charge += (self.residue[i].atom_charge,)

            # (B,1)
            residue_mass += (self.residue[i].total_mass,)
            res_natom_tensor += (self.residue[i].natom_tensor,)

            # (B,1)
            head_atom = self.residue_head(i)
            if head_atom is not None:
                if tail_atom is None:
                    print('Warrning! The head_atom of residue '+str(i)+' is not None' +
                          ' but the tail_atom of residue '+str(i-1)+' is None. ')
                else:
                    # (B,1,2)
                    connect = msnp.concatenate(
                        (F.expand_dims(tail_atom, -2), F.expand_dims(head_atom, -2)), axis=-1)
                    bond += (connect,)
            # (B,1,1)
            tail_atom = self.residue_tail(i)

            # (B,C',2)
            if self.residue[i].bond is not None:
                bond += (self.residue[i].bond + pointer,)

            pointer += self.residue[i].num_atoms

        self.num_atoms = pointer
        self.residue_pointer = Tensor(residue_pointer, ms.int32)
        self.residue_name = np.array(residue_name, np.str_)

        # (B,A)
        self.atom_name = np.concatenate(atom_name, axis=-1)
        self.atom_type = np.concatenate(atom_type, axis=-1)
        self.atom_mass = msnp.concatenate(atom_mass, axis=-1)
        self.atom_mask = msnp.concatenate(atom_mask, axis=-1)
        self.atomic_number = msnp.concatenate(atomic_number, axis=-1)
        self.inv_mass = msnp.concatenate(inv_mass, axis=-1)
        self.atom_charge = None
        if any_charge:
            self.atom_charge = msnp.concatenate(atom_charge, axis=-1)

        # (A)
        self.atom_resid = msnp.concatenate(atom_resid)
        self.image_index = msnp.concatenate(image_index)

        # (B,R)
        self.residue_mass = msnp.concatenate(residue_mass, axis=-1)
        self.res_natom_tensor = msnp.concatenate(res_natom_tensor, axis=-1)

        # (B,C,2)
        self.bond = None
        if bond:
            self.bond = msnp.concatenate(bond, -2)

        return self

    def build_space(self, coordinate: Tensor, pbc_box: Tensor = None):
        """
        Build coordinate and PBC box.

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D) or (1, A, D). Data type is float.
                                    Position coordinates of atoms.
            pbc_box (Tensor):       Tensor of shape (B, D) or (1, D). Data type is float.
                                    Box of periodic boundary condition. Default: None.
        """
        # (B,A,D)
        if coordinate is None:
            coordinate = np.random.uniform(0, self.units.length(
                1, 'nm'), size=(self.multi_system, self.num_atoms, 3))
            coordinate = Tensor(coordinate, ms.float32)
        coordinate = self._check_coordianate(coordinate)
        self.coordinate = Parameter(coordinate, name='coordinate')
        self.dimension = self.coordinate.shape[-1]
        self.num_walker = self.coordinate.shape[0]

        # (B,1)
        self.system_mass = msnp.sum(self.atom_mass, -1, keepdims=True)
        self.has_empty_atom = (not self.atom_mask.all())
        # (B,1) <- (B,A)
        self.system_natom = msnp.sum(F.cast(self.atom_mask, ms.float32), -1, keepdims=True)

        self.keep_prod = ops.ReduceProd(keep_dims=True)
        self.identity = ops.Identity()

        # (B,D)
        if pbc_box is None:
            self.pbc_box = None
            self.use_pbc = False
            self.num_com = self.dimension
            self.image = None
        else:
            self.use_pbc = True
            self.num_com = self.dimension
            pbc_box = Tensor(pbc_box, ms.float32)
            if pbc_box.ndim == 1:
                pbc_box = F.expand_dims(pbc_box, 0)
            if pbc_box.ndim != 2:
                raise ValueError('The rank of pbc_box must be 1 or 2!')
            if pbc_box.shape[-1] != self.dimension:
                raise ValueError('The last dimension of "pbc_box" ('+str(pbc_box.shape[-1]) +
                                 ') must be equal to the dimension of "coordinate" ('+str(self.dimension)+')!')
            if pbc_box.shape[0] > 1 and pbc_box.shape[0] != self.num_walker:
                raise ValueError('The first dimension of "pbc_box" ('+str(pbc_box.shape[0]) +
                                 ') does not match the first dimension of "coordinate" ('+str(self.dimension)+')!')
            self.pbc_box = Parameter(pbc_box, name='pbc_box')

            self.image = Parameter(msnp.zeros_like(self.coordinate, ms.int32), name='coordinate_image',
                                   requires_grad=False)
            self.update_image()

        self.degrees_of_freedom = self.dimension * self.num_atoms - self.num_com
        return self

    def set_bond_length(self, bond_length: Tensor):
        """
        Set bond length.

        Args:
            bond_length (Tensor):   Length of bond.
        """
        if self.bond is None:
            raise ValueError('Cannot setup bond_length because bond is None')
        bond_length = Tensor(bond_length, ms.float32)
        if bond_length.shape != self.bond.shape[:2]:
            raise ValueError('The shape of bond_length '+str(self.bond_length.shape) +
                             ' does not match the shape of bond '+str(self.bond.shape))
        self.bond_length = bond_length
        return self

    def residue_index(self, res_id: int) -> Tensor:
        """
        Get index of residue.

        Args:
            res_id (int):         Residue ID parameter.

        Returns:
            Tensor, the index of residue.
        """
        return self.residue[res_id].system_index

    def residue_bond(self, res_id: int) -> Tensor:
        """
        Get bond index of residue.

        Args:
            res_id (int):          Residue ID parameter.

        Returns:
            Tensor, the bond index of residue.
        """
        if self.residue[res_id].bond is None:
            return None
        return self.residue[res_id].bond + self.residue[res_id].start_index

    def residue_head(self, res_id: int) -> Tensor:
        """
        Get head index of residue.

        Args:
            res_id (int):        Residue ID parameter.

        Returns:
            Tensor, the head index of residue.
        """
        if self.residue[res_id].head_atom is None:
            return None
        return self.residue[res_id].head_atom + self.residue[res_id].start_index

    def residue_tail(self, res_id: int) -> Tensor:
        """
        Get tail index of residue.

        Args:
            res_id (int):     Residue ID parameter.

        Returns:
            Tensor, the tail index of residue.
        """
        if self.residue[res_id].tail_atom is None:
            return None
        return self.residue[res_id].tail_atom + self.residue[res_id].start_index

    def residue_coordinate(self, res_id: int) -> Tensor:
        """
        Get residue coordinate.

        Args:
            res_id (int):     Residue ID parameter.

        Returns:
            Tensor, the residue coordinate.
        """
        return F.gather_d(self.coordinate, -2, self.residue[res_id].system_index)

    def get_volume(self) -> Tensor:
        """
        get volume of system.

        Returns:
            Tensor, volume of system.
        """
        if self.pbc_box is None:
            return None
        return self.keep_prod(self.pbc_box, -1)

    def space_parameters(self) -> list:
        """
        get the parameter of space (coordinates and pbc box).

        Returns:
            list, a list of parameter of space.
        """
        if self.pbc_box is None:
            return [self.coordinate]
        return [self.coordinate, self.pbc_box]

    def trainable_params(self, recurse=True) -> list:
        """
        Args:
            recurse (bool, optional):      Recurse parameter. Default: True.

        Returns:
            list, a list of trainable_params.
        """
        return list(filter(lambda x: x.name.split('.')[-1] == 'coordinate', self.get_parameters(expand=recurse)))

    def _check_coordianate(self, coordinate: Tensor) -> Tensor:
        """
        check coordinate.

        Returns:
            Tensor, a Tensor of coordinate.
        """
        coordinate = Tensor(coordinate, ms.float32)
        if coordinate.ndim == 2:
            coordinate = F.expand_dims(coordinate, 0)
        if coordinate.ndim != 3:
            raise ValueError('The rank of "coordinate" must be 2 or 3!')
        if coordinate.shape[-2] != self.num_atoms:
            raise ValueError('The penultimate dimension of "coordinate" ('+str(coordinate.shape[-2]) +
                             ') must be equal to the number of atoms ('+str(self.num_atoms)+')!')
        if self.multi_system > 1 and coordinate.shape[0] != self.multi_system:
            raise ValueError('The first dimension of "coordinate" ('+str(coordinate.shape[0]) +
                             ') does not match the that of "atom_name" ('+str(self.multi_system)+')!')
        return coordinate

    def update_coordinate(self, coordinate: Tensor, success: bool = True) -> bool:
        """
        Update the parameter of coordinate.

        Args:
            coordinate (Tensor):        Tensor of shape (B, A, D) or (1, A, D). Data type is float.
                                        Position coordinates of atoms.
            success (bool, optional):   Success parameter. Default: True.

        Returns:
            bool, whether update the parameter of coordinate.
        """
        success = F.depend(success, F.assign(self.coordinate, coordinate))
        if self.pbc_box is not None:
            success = self.update_image(success=success)
        return success

    def set_coordianate(self, coordinate: Tensor):
        """
        Set the value of coordinate.

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D) or (1, A, D). Data type is float.
                                    Position coordinates of atoms. Default: None.
        """
        coordinate = self._check_coordianate(coordinate)
        if coordinate is not None and coordinate.shape == self.coordinate.shape:
            self.update_coordinate(coordinate)
        else:
            self.coordinate = Parameter(coordinate, name='coordinate')
            self.dimension = self.coordinate.shape[-1]
            self.num_walker = self.coordinate.shape[0]
        return self

    def update_pbc_box(self, pbc_box: Tensor, success: bool = True):
        """
        Update PBC box.

        Args:
            pbc_box (Tensor):           Tensor of shape (B, D) or (1, D). Data type is float.
                                        Box of periodic boundary condition. Default: None.
            success (bool, optional):   Success parameter. Default: True.

        Returns:
            bool, whether update PBC box.
        """
        success = F.depend(True, F.assign(self.pbc_box, pbc_box))
        if self.pbc_box is not None:
            success = self.update_image(success=success)
        return success

    def set_pbc_grad(self, grad_box: bool):
        """
        Set whether to calculate the gradient of PBC box.

        Args:
            grad_box (bool):        Whether to calculate the gradient of PBC box.
        """
        if self.pbc_box is not None:
            self.pbc_box.requires_grad = grad_box
        return self

    def set_pbc_box(self, pbc_box: Tensor = None):
        """
        Set PBC box.

        Args:
            pbc_box (Tensor):       Tensor of shape (B, D) or (1, D). Data type is float.
                                    Box of periodic boundary condition. Default: None.
        """
        if pbc_box is None:
            self.pbc_box = None
            self.use_pbc = False
            self.num_com = self.dimension
        else:
            self.use_pbc = True
            self.num_com = self.dimension * 2
            if self.pbc_box is None:
                self.pbc_box = self._check_pbc_box(pbc_box)
            else:
                if pbc_box.shape != self.pbc_box.shape:
                    raise ValueError('The shape of the new pbc_box '+str(pbc_box.shape) +
                                     'is not equal to the old one '+str(self.pbc_box)+'!')
                self.update_pbc_box(pbc_box)
        return self

    def repeat_box(self, lattices: list):
        """
        Repeat the system according to the lattices of PBC box.

        Args:
            lattices (list):        Lattices parameter.
        """
        if self.pbc_box is None:
            raise RuntimeError('repeat_box() cannot be used without pbc_box, '
                               'please use set_pbc_box() to set pbc_box first '
                               'before using this function.')

        if isinstance(lattices, Tensor):
            lattices = lattices.asnumpy()
        if isinstance(lattices, ndarray):
            lattices = lattices.tolist()
        if not isinstance(lattices, list):
            raise TypeError('The type of lattices must be list, ndarry or Tensor but got: ' +
                            str(type(lattices)))
        if len(lattices) != self.dimension:
            raise ValueError('The number of lattics ('+str(len(lattices))+') must be equal to '
                             'the dimension of system ('+str(self.dimension)+')')
        product_ = []
        for l in lattices:
            if l <= 0:
                raise ValueError('The number in lattices must larger than 0!')
            product_.append(list(range(l)))

        shift_num = tuple(itertools.product(*product_))[1:]
        if shift_num:
            shift_box = Tensor(shift_num, ms.float32) * self.pbc_box
            box = self.copy()
            coord = box.get_coordinate()
            coordinate = (coord,)
            for shift in shift_box:
                self.residue.extend(copy.deepcopy(box.residue))
                coordinate += (coord+shift,)

            self.build_system()
            coordinate = msnp.concatenate(coordinate, axis=-2)
            self.build_space(coordinate, self.pbc_box)
            new_box = Tensor(lattices, ms.int32) * self.pbc_box
            self.update_pbc_box(new_box)

        return self

    def coordinate_in_box(self, shift: float = 0) -> Tensor:
        """
        Get the coordinate in a whole PBC box.

        Args:
            shift (float):         Shift parameter. Default: 0.

        Returns:
            Tensor, the coordinate in a whole PBC box.
        """
        coordinate = self.identity(self.coordinate)
        pbc_box = self.identity(self.pbc_box)
        return func.displace_in_box(coordinate, pbc_box, shift)

    def calc_image(self, shift: float = 0) -> Tensor:
        """
        Calculate the image of coordinate.

        Args:
            shift (float):         Shift parameter. Default: 0.

        Returns:
            Tensor, a Tensor  of the image of coordinate.
        """
        coordinate = self.identity(self.coordinate)
        pbc_box = self.identity(self.pbc_box)
        image = func.periodic_image(coordinate, pbc_box, shift)
        if self.image_index is not None:
            image = image[:, self.image_index, :]
        return image

    def update_image(self, image: Tensor = None, success: bool = True) -> bool:
        """
        Update the image of coordinate.

        Args:
            image (Tensor):           Image parameter. Default: None.
            success (bool, optional): Success parameter. Default: True.

        Returns:
            bool.
        """
        if image is None:
            image = self.calc_image()
        return F.depend(success, F.assign(self.image, image))

    def set_length_unit(self, unit):
        """
        Set the length unit of system.

        Args:
            unit (Units):          Units of length and energy.
        """
        scale = self.units.convert_length_to(unit)
        coordinate = self.coordinate * scale
        self.update_coordinate(coordinate)
        if self.pbc_box is not None:
            pbc_box = self.pbc_box * scale
            self.update_pbc_box(pbc_box)
        self.units.set_length_unit(unit)
        return self

    def get_coordinate(self) -> Tensor:
        """
        get Tensor of coordinate.

        Returns:
            Tensor, a Tensor of coordinate.
        """
        return self.identity(self.coordinate)

    def get_pbc_box(self) -> Tensor:
        """
        get Tensor of PBC box.

        Returns:
            Tensor, a Tensor of PBC box.
        """
        if self.pbc_box is None:
            return None
        return self.identity(self.pbc_box)

    def construct(self) -> Tuple[Tensor, Tensor]:
        r"""
        Get space information of system.

        Returns:
            - coordinate (Tensor), Tensor of shape (B, A, D). Data type is float.
            - pbc_box (Tensor), Tensor of shape (B, D). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation.
            A:  Number of atoms.
            D:  Dimension of the simulation system. Usually is 3.
        """
        coordinate = self.identity(self.coordinate)
        pbc_box = None
        if self.pbc_box is not None:
            pbc_box = self.identity(self.pbc_box)
        return coordinate, pbc_box
