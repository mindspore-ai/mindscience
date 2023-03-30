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
Molecule
"""

import copy
import itertools
from typing import Union, List, Tuple
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
from ...colvar import Colvar
from ...colvar.atoms import AtomsBase
from ...colvar.atoms import get_atoms as _get_atoms
from ...function import functions as func
from ...function.units import Units, GLOBAL_UNITS
from ...function.functions import get_ms_array, get_ndarray


class Molecule(Cell):
    r"""Base class for molecular system, used as the "system module" in MindSPONGE.

        The `Molecule` Cell can represent a molecule or a system consisting of multiple molecules.

        The major components of the `Molecule` Cell is the `Residue` Cell. A `Molecule` Cell can
        contain multiple `Residue` Cells.

    Args:

        atoms (Union[List[Union[str, int]], ndarray]):
                            Array of atoms. The data in array can be str of atom name.
                            or int of atomic number. Defulat: None

        atom_name (Union[List[str], ndarray]):
                            Array of atom name with data type `str`. Defulat: None

        atom_type (Union[List[str], ndarray]):
                            Array of atom type with data type `str`. Defulat: None

        atom_mass (Union[Tensor, ndarray, List[float]]):
                            Array of atom mass of shape `(B, A)` with data type `float`. Defulat: None

        atom_charge (Union[Tensor, ndarray, List[float]]):
                            Array of atom charge of shape `(B, A)` with data type `float`. Defulat: None

        atomic_number (Union[Tensor, ndarray, List[float]]):
                            Array of atomic number of shape `(B, A)` with data type `int`. Defulat: None

        bond (Union[Tensor, ndarray, List[int]]):
                            Array of bond connection of shape `(B, b, 2)` with data type `int`. Defulat: None

        coordinate (Union[Tensor, ndarray, List[float]]):
                            Tensor of atomic coordinates :math:`R` of shape `(B, A, D)` with data type `float`.
                            Default: None

        pbc_box (Union[Tensor, ndarray, List[float]]):
                            Tensor of box size :math:`\vec{L}` of periodic boundary condition (PBC).
                            The shape of tensor is `(B, D)`, and the data type is `float`.
                            Default: None

        template (Union[dict, str, List[Union[dict, str]]]):
                            Template for molecule. It can be a `dict` in MindSPONGE template format
                            or a `str` for the filename of a MindSPONGE template file. If a `str` is given,
                            it will first look for a file with the same name in the current directory.
                            If the file does not exist, it will search in the built-in template directory
                            of MindSPONGE (`mindsponge.data.template`).
                            Default: None.

        length_unit (str):  Length unit. If `None` is given, the global length units will be used.
                            Default: None

    Supported Platforms:

        ``Ascend`` ``GPU``

    Symbols:

        B:  Batchsize, i.e. number of walkers in simulation

        A:  Number of atoms.

        b:  Number of bonds.

        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    def __init__(self,
                 atoms: Union[List[Union[str, int]], ndarray] = None,
                 atom_name: Union[List[str], ndarray] = None,
                 atom_type: Union[List[str], ndarray] = None,
                 atom_mass: Union[Tensor, ndarray, List[float]] = None,
                 atom_charge: Union[Tensor, ndarray, List[float]] = None,
                 atomic_number: Union[Tensor, ndarray, List[float]] = None,
                 bond: Union[Tensor, ndarray, List[int]] = None,
                 coordinate: Union[Tensor, ndarray, List[float]] = None,
                 pbc_box: Union[Tensor, ndarray, List[float]] = None,
                 template: Union[dict, str] = None,
                 residue: Union[Residue, List[Residue]] = None,
                 length_unit: str = None,
                 ):

        super().__init__()

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit

        self.units = Units(length_unit, GLOBAL_UNITS.energy_unit)

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
                raise ValueError(f'The type of residue must be Residue or list but got: {type(residue)}')

        # The number of multi_system of system
        self.multi_system = 1
        # A: number of atoms
        self.num_atoms = 0

        self.dimension = None
        self.num_walker = None
        self.degrees_of_freedom = None

        self.use_pbc = False
        self.num_com = None
        self.image = None

        # (B, A)
        self.atom_name = None
        self.atom_type = None
        self.atom_mass = None
        self.atom_mask = None
        self.atomic_number = None
        self.inv_mass = None
        self.atom_charge = None

        # (B, R)
        self.residue_mass = None
        self.residue_name = None
        self.res_natom_tensor = None

        # (R)
        self.residue_pointer = None

        # (A)
        self.atom_resid = None
        self.image_index = None

        # (B, C, 2)
        self.bond = None
        self.hydrogen_bond = None

        # (B, C): bond length for constraint
        self.bond_length = None

        # (B, A, D)
        self.coordinate: Parameter = None

        # (B, D)
        self.pbc_box: Parameter = None

        # (B,1)
        self.system_mass = None
        self.has_empty_atom = None
        self.system_natom = None

        self.build_system()
        if self.residue is not None:
            self.build_space(coordinate, pbc_box)

    @property
    def shape(self):
        return self.coordinate.shape

    @property
    def ndim(self):
        return self.coordinate.ndim

    @property
    def length_unit(self):
        return self.units.length_unit

    def convert_length_from(self, unit) -> float:
        """convert length from a specified units."""
        return self.units.convert_length_from(unit)

    def convert_length_to(self, unit) -> float:
        """convert length to a specified units."""
        return self.units.convert_length_to(unit)

    def move(self, shift: Tensor = None):
        """move the coordinate of the system"""
        if shift is not None:
            self.update_coordinate(self.coordinate + Tensor(shift, ms.float32))
        return self

    def copy(self, shift: Tensor = None):
        """return a Molecule that copy the parameters of this molecule"""
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
        """add residue"""
        if not isinstance(residue, list):
            if isinstance(residue, Residue):
                residue = [residue]
            else:
                raise TypeError(f'The type of residue must be Residue or list but got: {type(residue)}')

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
        """append the system"""
        if not isinstance(system, Molecule):
            raise TypeError(f'For append, the type of system must be "Molecule" but got: {type(system)}')
        self.add_residue(system.residue, system.get_coordinate())
        return self

    def reduplicate(self, shift: Tensor):
        """duplicate the system to double of the origin size"""
        shift = Tensor(shift, ms.float32)
        self.residue.extend(copy.deepcopy(self.residue))
        self.build_system()
        coordinate = msnp.concatenate((self.coordinate, self.coordinate+shift), axis=-2)
        self.build_space(coordinate, self.pbc_box)
        return self

    def build_atom_type(self):
        """build atom type"""
        atom_type = ()
        for i in range(self.num_residue):
            atom_type += (self.residue[i].atom_type,)
        self.atom_type = np.concatenate(atom_type, axis=-1)
        return self

    def build_atom_charge(self):
        """build atom charge"""
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
        """build the system by residues"""
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
            raise ValueError(f'The multi_system of residues cannot be broadcast: {multi_system}')

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

            # (B, A')
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

            # (B, 1)
            residue_mass += (self.residue[i].total_mass,)
            res_natom_tensor += (self.residue[i].natom_tensor,)

            # (B, 1)
            head_atom = self.residue_head(i)
            if head_atom is not None:
                if tail_atom is None:
                    print(f'Warrning! The head_atom of residue {i} is not None but '
                          f'the tail_atom of residue {i-1} is None.')
                else:
                    # (B, 1, 2)
                    connect = msnp.concatenate(
                        (F.expand_dims(tail_atom, -2), F.expand_dims(head_atom, -2)), axis=-1)
                    bond += (connect,)
            # (B, 1, 1)
            tail_atom = self.residue_tail(i)

            # (B, C', 2)
            if self.residue[i].bond is not None:
                bond += (self.residue[i].bond + pointer,)

            pointer += self.residue[i].num_atoms

        self.num_atoms = pointer
        self.residue_pointer = Tensor(residue_pointer, ms.int32)
        self.residue_name = np.array(residue_name, np.str_)

        # (B, A)
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

        # (B, R)
        self.residue_mass = msnp.concatenate(residue_mass, axis=-1)
        self.res_natom_tensor = msnp.concatenate(res_natom_tensor, axis=-1)

        # (B, C, 2)
        self.bond = None
        if bond:
            self.bond = msnp.concatenate(bond, -2)

        return self

    def build_space(self, coordinate: Tensor, pbc_box: Tensor = None):
        """build coordinate and PBC box"""
        # (B, A, D)
        if coordinate is None:
            coordinate = np.random.uniform(0, self.units.length(
                1, 'nm'), size=(self.multi_system, self.num_atoms, 3))
            coordinate = Tensor(coordinate, ms.float32)
        coordinate = self._check_coordianate(coordinate)
        self.coordinate = Parameter(coordinate, name='coordinate')
        self.dimension = self.coordinate.shape[-1]
        self.num_walker = self.coordinate.shape[0]

        # (B, 1)
        self.system_mass = msnp.sum(self.atom_mass, -1, keepdims=True)
        self.has_empty_atom = (not self.atom_mask.all())
        # (B, 1) <- (B, A)
        self.system_natom = msnp.sum(F.cast(self.atom_mask, ms.float32), -1, keepdims=True)

        self.keep_prod = ops.ReduceProd(True)
        self.identity = ops.Identity()

        # (B, D)
        if pbc_box is None:
            self.pbc_box = None
            self.use_pbc = False
            self.num_com = self.dimension
            self.image = None
        else:
            self.use_pbc = True
            self.num_com = self.dimension
            pbc_box = get_ms_array(pbc_box, ms.float32)
            if pbc_box.ndim == 1:
                pbc_box = F.expand_dims(pbc_box, 0)
            if pbc_box.ndim != 2:
                raise ValueError('The rank of pbc_box must be 1 or 2!')
            if pbc_box.shape[-1] != self.dimension:
                raise ValueError(f'The last dimension of "pbc_box" ({pbc_box.shape[-1]}) must be equal to '
                                 f'the dimension of "coordinate" ({self.dimension})!')
            if pbc_box.shape[0] > 1 and pbc_box.shape[0] != self.num_walker:
                raise ValueError(f'The first dimension of "pbc_box" ({pbc_box.shape[0]}) does not match '
                                 f'the first dimension of "coordinate" ({self.dimension})!')
            self.pbc_box = Parameter(pbc_box, name='pbc_box')

            self.image = Parameter(msnp.zeros_like(self.coordinate, ms.int32), name='coordinate_image',
                                   requires_grad=False)
            self.update_image()

        self.degrees_of_freedom = self.dimension * self.num_atoms - self.num_com
        return self

    def set_bond_length(self, bond_length: Tensor):
        """set bond length"""
        if self.bond is None:
            raise ValueError('Cannot setup bond_length because bond is None')
        bond_length = Tensor(bond_length, ms.float32)
        if bond_length.shape != self.bond.shape[:2]:
            raise ValueError(f'The shape of bond_length {self.bond_length.shape} does not match '
                             f'the shape of bond {self.bond.shape}')
        self.bond_length = bond_length
        return self

    def residue_index(self, res_id: int) -> Tensor:
        """get index of residue"""
        return self.residue[res_id].system_index

    def residue_bond(self, res_id: int) -> Tensor:
        """get bond index of residue"""
        if self.residue[res_id].bond is None:
            return None
        return self.residue[res_id].bond + self.residue[res_id].start_index

    def residue_head(self, res_id: int) -> Tensor:
        """get head index of residue"""
        if self.residue[res_id].head_atom is None:
            return None
        return self.residue[res_id].head_atom + self.residue[res_id].start_index

    def residue_tail(self, res_id: int) -> Tensor:
        """get tail index of residue"""
        if self.residue[res_id].tail_atom is None:
            return None
        return self.residue[res_id].tail_atom + self.residue[res_id].start_index

    def residue_coordinate(self, res_id: int) -> Tensor:
        """get residue coordinate"""
        return F.gather_d(self.coordinate, -2, self.residue[res_id].system_index)

    def get_volume(self) -> Tensor:
        """get volume of system"""
        if self.pbc_box is None:
            return None
        return self.keep_prod(self.pbc_box, -1)

    def space_parameters(self) -> list:
        """get the parameter of space (coordinates and pbc box)"""
        if self.pbc_box is None:
            return [self.coordinate]
        return [self.coordinate, self.pbc_box]

    def trainable_params(self, recurse=True) -> list:
        return list(filter(lambda x: x.name.split('.')[-1] == 'coordinate', self.get_parameters(expand=recurse)))

    def _check_coordianate(self, coordinate: Tensor) -> Tensor:
        """check coordinate"""
        coordinate = Tensor(coordinate, ms.float32)
        if coordinate.ndim == 2:
            coordinate = F.expand_dims(coordinate, 0)
        if coordinate.ndim != 3:
            raise ValueError('The rank of "coordinate" must be 2 or 3!')
        if coordinate.shape[-2] != self.num_atoms:
            raise ValueError(f'The penultimate dimension of "coordinate" ({coordinate.shape[-2]}) must be equal to '
                             f'the number of atoms ({self.num_atoms})!')
        if self.multi_system > 1 and coordinate.shape[0] != self.multi_system:
            raise ValueError(f'The first dimension of "coordinate" ({coordinate.shape[0]}) does not match '
                             f'that of "atom_name" ({self.multi_system})!')
        return coordinate

    def update_coordinate(self, coordinate: Tensor) -> Tensor:
        """update the parameter of coordinate"""
        coordinate = F.assign(self.coordinate, coordinate)
        if self.pbc_box is None:
            return coordinate
        return F.depend(coordinate, self.update_image())

    def set_coordianate(self, coordinate: Tensor) -> Tensor:
        """set the value of coordinate"""
        coordinate = self._check_coordianate(coordinate)
        if coordinate is not None and coordinate.shape == self.coordinate.shape:
            return self.update_coordinate(coordinate)

        self.coordinate = Parameter(coordinate, name='coordinate')
        self.dimension = self.coordinate.shape[-1]
        return self.identity(coordinate)

    def update_pbc_box(self, pbc_box: Tensor) -> Tensor:
        """update PBC box"""
        pbc_box = F.assign(self.pbc_box, pbc_box)
        return F.depend(pbc_box, self.update_image())

    def set_pbc_grad(self, grad_box: bool) -> bool:
        """set whether to calculate the gradient of PBC box"""
        if self.pbc_box is None:
            return grad_box
        self.pbc_box.requires_grad = grad_box
        return self.pbc_box.requires_grad

    def set_pbc_box(self, pbc_box: Tensor = None) -> Tensor:
        """set PBC box"""
        if pbc_box is None:
            self.pbc_box = None
            self.use_pbc = False
            self.num_com = self.dimension
        else:
            self.use_pbc = True
            self.num_com = self.dimension * 2
            if self.pbc_box is None:
                pbc_box = Tensor(pbc_box, ms.float32)
                if pbc_box.ndim == 1:
                    pbc_box = F.expand_dims(pbc_box, 0)
                if pbc_box.ndim != 2:
                    raise ValueError('The rank of pbc_box must be 1 or 2!')
                if pbc_box.shape[-1] != self.dimension:
                    raise ValueError(f'The last dimension of "pbc_box" ({pbc_box.shape[-1]}) must be equal to '
                                     f'the dimension of "coordinate" ({self.dimension})!')
                if pbc_box.shape[0] > 1 and pbc_box.shape[0] != self.num_walker:
                    raise ValueError(f'The first dimension of "pbc_box" ({pbc_box.shape[0]}) does not match '
                                     f'the first dimension of "coordinate" ({self.dimension})!')
                self.pbc_box = Parameter(pbc_box, name='pbc_box', requires_grad=True)
            else:
                if pbc_box.shape != self.pbc_box.shape:
                    raise ValueError(f'The shape of the new pbc_box {pbc_box.shape} is not equal to '
                                     f'the old one {self.pbc_box.shape}!')
                self.update_pbc_box(pbc_box)
        return self.pbc_box

    def repeat_box(self, lattices: list):
        """repeat the system according to the lattices of PBC box"""
        if self.pbc_box is None:
            raise RuntimeError('repeat_box() cannot be used without pbc_box, '
                               'please use set_pbc_box() to set pbc_box first '
                               'before using this function.')

        if isinstance(lattices, Tensor):
            lattices = lattices.asnumpy()
        if isinstance(lattices, ndarray):
            lattices = lattices.tolist()
        if not isinstance(lattices, list):
            raise TypeError(f'The type of lattices must be list, ndarry or Tensor but got: {type(lattices)}')
        if len(lattices) != self.dimension:
            raise ValueError(f'The number of lattics ({len(lattices)}) must be equal to '
                             f'the dimension of system ({self.dimension})')
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

    def coordinate_in_pbc(self, shift: float = 0) -> Tensor:
        """get the coordinate in a whole PBC box"""
        coordinate = self.identity(self.coordinate)
        pbc_box = self.identity(self.pbc_box)
        return func.coordinate_in_pbc(coordinate, pbc_box, shift)

    def calc_image(self, shift: float = 0) -> Tensor:
        """calculate the image of coordinate"""
        coordinate = self.identity(self.coordinate)
        pbc_box = self.identity(self.pbc_box)
        image = func.pbc_image(coordinate, pbc_box, shift)
        if self.image_index is not None:
            image = image[:, self.image_index, :]
        return image

    def update_image(self, image: Tensor = None) -> bool:
        """update the image of coordinate"""
        if image is None:
            image = self.calc_image()
        return F.assign(self.image, image)

    def set_length_unit(self, unit):
        """set the length unit of system"""
        scale = self.units.convert_length_to(unit)
        coordinate = self.coordinate * scale
        self.update_coordinate(coordinate)
        if self.pbc_box is not None:
            pbc_box = self.pbc_box * scale
            self.update_pbc_box(pbc_box)
        self.units.set_length_unit(unit)
        return self

    def calc_colvar(self, colvar: Colvar) -> Tensor:
        """calculate the value of specific collective variables in the system"""
        coordinate = self.identity(self.coordinate)
        pbc_box = None if self.pbc_box is None else self.identity(self.pbc_box)
        return colvar(coordinate, pbc_box)

    def get_atoms(self, atoms: Union[Tensor, Parameter, ndarray, str, list, tuple]) -> AtomsBase:
        """get Atoms from the system"""
        try:
            atoms = _get_atoms(atoms)
        except TypeError:
            #TODO
            pass
        return atoms

    def get_coordinate(self, atoms: AtomsBase = None) -> Tensor:
        """get Tensor of coordinate"""
        coordinate = self.identity(self.coordinate)
        if atoms is None:
            return coordinate
        pbc_box = None if self.pbc_box is None else self.identity(self.pbc_box)
        return atoms(coordinate, pbc_box)

    def get_pbc_box(self) -> Tensor:
        """get Tensor of PBC box"""
        if self.pbc_box is None:
            return None
        return self.identity(self.pbc_box)

    def construct(self) -> Tuple[Tensor, Tensor]:
        r"""Get space information of system.

        Returns:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """
        coordinate = self.identity(self.coordinate)
        pbc_box = None
        if self.pbc_box is not None:
            pbc_box = self.identity(self.pbc_box)
        return coordinate, pbc_box
