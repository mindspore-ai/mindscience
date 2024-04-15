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

from ..modelling.pdb_generator import gen_pdb
from ..modelling.mol2_parser import mol2parser
from ..modelling.hadder import read_pdb
from ..residue.residue import Residue
from ..residue.amino import AminoAcid
from ...data.template import get_molecule, get_template
from ...colvar import Colvar
from ...colvar.atoms import AtomsBase
from ...colvar.atoms import get_atoms as _get_atoms
from ...function import functions as func
from ...function.units import Units, GLOBAL_UNITS
from ...function.functions import get_ms_array, get_ndarray, get_arguments
from ...function.functions import keepdims_prod, bonds_in
from ...function import length_convert

RESIDUE_NAMES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HID', 'HIS',
                 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
N_RESIDUE_NAMES = ['N' + res for res in RESIDUE_NAMES]
C_RESIDUE_NAMES = ['C' + res for res in RESIDUE_NAMES]

RESIDUE_NAMES += N_RESIDUE_NAMES
RESIDUE_NAMES += C_RESIDUE_NAMES

# mol/L
_DENSITY = 55
# NW/A^3
DENSITY = _DENSITY * 6.022E23 / 1e03 ** 9
_AVGDIS = DENSITY ** (-1 / 3)
AVGDIS = _AVGDIS * 1.


class Molecule(Cell):
    r"""
    Base class for molecular system, used as the "system module" in MindSPONGE.
    The `Molecule` Cell can represent a molecule or a system consisting of multiple molecules.
    The major components of the `Molecule` Cell is the `Residue` Cell. A `Molecule` Cell can
    contain multiple `Residue` Cells.

    Args:
        atoms(Union[List[Union[str, int]], ndarray]):       Array of atoms. The data in array can be str of atom
                                                            name or int of atomic number. Defulat: ``None``
        atom_name(Union[List[str], ndarray]):               Array of atom name with data type `str`. Defulat: ``None``
        atom_type(Union[List[str], ndarray]):               Array of atom type with data type `str`. Defulat: None
        atom_mass(Union[Tensor, ndarray, List[float]]):     Array of atom mass of shape :math:`(B, A)` with data type
                                                            `float` where B represents the batchsize, i.e. the number
                                                            of walker in the system, A represents the number of atoms.
                                                            Defulat: ``None``
        atom_charge(Union[Tensor, ndarray, List[float]]):   Array of atom charge of shape :math:`(B, A)` with data type
                                                            `float`. Defulat: ``None``
        atomic_number(Union[Tensor, ndarray, List[float]]): Array of atomic number of shape :math:`(B, A)` with data
                                                            type `int`. Defulat: ``None``
        bond(Union[Tensor, ndarray, List[int]]):            Array of bond connection of shape :math:`(B, b, 2)` with
                                                            data type `int` where b represents the number of bonds.
                                                            Defulat: ``None``
        coordinate(Union[Tensor, ndarray, List[float]]):    Tensor of atomic coordinates :math:`R` of shape
                                                            :math:`(B, A, D)` with data type `float` where D represents
                                                            the spatial dimension of the simulation system,
                                                            usually is 3. Default: ``None``
        pbc_box(Union[Tensor, ndarray, List[float]]):       Tensor of box size :math:`\vec{L}` of periodic boundary
                                                            condition (PBC). The shape of tensor is :math:`(B, D)`
                                                            and the data type is `float`. Default: ``None``
        template(Union[dict, str, List[Union[dict, str]]]): Template for molecule. It can be a `dict` in MindSPONGE
                                                            template format or a `str` for the filename of a
                                                            MindSPONGE template file. If a `str` is given,
                                                            it will first look for a file with the same name in the
                                                            current directory. If the file does not exist, it will
                                                            search in the built-in template directory of
                                                            MindSPONGE (`mindsponge.data.template`).
                                                            Default: ``None``.
        residue(Union[Residue, List[Residue]]):             Residue or a list of residues. If template is not ``None``,
                                                            only the residues in the template will be used.
                                                            Default: ``None``.
        length_unit(str):                                   Length unit. If ``None`` is given, the global length
                                                            units will be used. Default: ``None``
        kwargs(dict):                                       Other parameters for extension

    Outputs:
        - coordinate, Tensor of shape `(B, A, D)`. Data type is float.
        - pbc_box, Tensor of shape `(B, D)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from sponge import Molecule
        >>> system = Molecule(atoms=['O', 'H', 'H'],
        ...                   coordinate=[[0, 0, 0], [0.1, 0, 0], [-0.0333, 0.0943, 0]],
        ...                   bonds=[[[0, 1], [0, 2]]])
        >>> print ('The number of atoms in the system is: ', system.num_atoms)
        The number of atoms in the system is:  3
        >>> print ('All the atom names in the system are: ', system.atom_name)
        All the atom names in the system are:  [['O' 'H' 'H']]
        >>> print ('The coordinates of atoms are: \n{}'.format(system.coordinate.asnumpy()))
        The coordinates of atoms are:
        [[[ 0.      0.      0.    ]
          [ 0.1     0.      0.    ]
          [-0.0333  0.0943  0.    ]]]
        >>> system = Molecule(template='water.spce.yaml')
        >>> print ('The number of atoms in the system is: ', system.num_atoms)
        The number of atoms in the system is:  3
        >>> print ('All the atom names in the system are: ', system.atom_name)
        All the atom names in the system are:  [['O' 'H1' 'H2']]
        >>> print ('The coordinates of atoms are: \n{}'.format(system.coordinate.asnumpy()))
        The coordinates of atoms are:
        [[[ 0.          0.          0.        ]
          [ 0.08164904  0.0577359   0.        ]
          [-0.08164904  0.0577359   0.        ]]]

    """

    def __init__(self,
                 atoms: Union[List[Union[str, int]], ndarray] = None,
                 atom_name: Union[List[str], ndarray] = None,
                 atom_type: Union[List[str], ndarray] = None,
                 atom_mass: Union[Tensor, ndarray, List[float]] = None,
                 atom_charge: Union[Tensor, ndarray, List[float]] = None,
                 atomic_number: Union[Tensor, ndarray, List[float]] = None,
                 bonds: Union[Tensor, ndarray, List[int]] = None,
                 coordinate: Union[Tensor, ndarray, List[float]] = None,
                 pbc_box: Union[Tensor, ndarray, List[float]] = None,
                 template: Union[dict, str, List[Union[dict, str]]] = None,
                 residue: Union[Residue, List[Residue]] = None,
                 length_unit: str = None,
                 **kwargs,
                 ):

        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit

        self.units = Units(length_unit, GLOBAL_UNITS.energy_unit)
        self.template = template

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
                    bonds=bonds,
                )

        self.residue: List[Residue] = None
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
        self.atom_name: ndarray = None
        self.atom_type: ndarray = None
        self.atom_mass: Tensor = None
        self.atom_mask: Tensor = None
        self.atomic_number: Tensor = None
        self.inv_mass: Tensor = None
        self.atom_charge: Parameter = None

        # (B, R)
        self.residue_mass = None
        self.residue_name = None
        self.res_natom_tensor = None

        # (R)
        self.residue_pointer = None

        # (A)
        self.atom_resid = None
        self.image_index = None

        self.multi_bonds = False

        # (B, C, 2)
        self.bonds: Tensor = None
        self.h_bonds: Tensor = None

        # (B, S, 3)
        self.settle_index: Tensor = None
        self.remaining_index: Tensor = None
        # (B, S, 2)
        self.settle_length: Tensor = None
        self.force_settle = False

        self.angles: Tensor = None
        self.dihedrals: Tensor = None
        self.improper_dihedrals: Tensor = None

        self.angle_vertices = None
        self.improper_axis_atoms = None

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
        r"""
        Shape of atomic coordinate.

        Returns:
            Tuple, atomic coordinate.
        """
        return self.coordinate.shape

    @property
    def ndim(self):
        r"""
        Ndim of atomic coordinate.

        Returns:
            int, number of dims of atomic coordinate.
        """
        return self.coordinate.ndim

    @property
    def length_unit(self):
        r"""
        Length unit.

        Returns:
            str, length unit.
        """
        return self.units.length_unit

    @property
    def heavy_atom_mask(self):
        r"""
        mask for heavy (non-hydrogen) atoms.

        Returns:
            Tensor, mask for heavy atoms.
        """
        return msnp.where(self.atomic_number[0] > 1, 0, 1)

    def convert_length_from(self, unit) -> float:
        """
        Convert length from a specified units.

        Args:
            unit(Union[str, Units, Length, float, int]):    Length unit.

        Returns:
            float, length according to a specified units.
        """
        return self.units.convert_length_from(unit)

    def convert_length_to(self, unit) -> float:
        """
        Convert length to a specified units.

        Args:
            unit(Union[str, Units, Length, float, int]):    Length unit.

        Returns:
            float, length according to a specified units.
        """
        return self.units.convert_length_to(unit)

    def move(self, shift: Tensor = None):
        """
        Move the coordinate of the system.

        Args:
            shift(Tensor): The displacement distance of the system. Default: ``None``.
        """
        if shift is not None:
            self.update_coordinate(self.coordinate + Tensor(shift, ms.float32))
        return self

    def copy(self, shift: Tensor = None):
        """
        Return a Molecule that copy the parameters of this molecule.

        Args:
            shift(Tensor): The displacement distance of the system. Default: ``None``.

        Returns:
            class, class Molecule that copy the parameters of this molecule.
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
        Add residue to this molecule system.

        Args:
            residue(class): a Residue class of the residue added in the system.
            coordinate(Tensor): The coordinate of the input residue. Default: ``None``.
        """
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
        """
        Append a system to this molecule system.

        Args:
            system(Molecule): Another molecule system that will be added to this molecule system.
        """
        if not isinstance(system, Molecule):
            raise TypeError(f'For append, the type of system must be "Molecule" but got: {type(system)}')
        self.add_residue(system.residue, system.get_coordinate())
        return self

    def reduplicate(self, shift: Tensor):
        """
        Duplicate the system to double of the origin size.

        Args:
            shift(Tensor):  The distance moved from the origin system.
        """
        shift = Tensor(shift, ms.float32)
        self.residue.extend(copy.deepcopy(self.residue))
        self.build_system()
        coordinate = msnp.concatenate((self.coordinate, self.coordinate + shift), axis=-2)
        self.build_space(coordinate, self.pbc_box)
        return self

    def build_atom_type(self):
        """Build atom type."""
        atom_type = ()
        for i in range(self.num_residue):
            atom_type += (self.residue[i].atom_type,)
        self.atom_type = np.concatenate(atom_type, axis=-1)
        return self

    def build_atom_charge(self):
        """Build atom charge."""
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
            atom_charge = msnp.concatenate(atom_charge, axis=-1)
            self.set_atom_charge(atom_charge)
        return self

    def build_system(self):
        """Build the system by residues."""
        if self.residue is None:
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

        bonds = ()
        head_atom = None
        tail_atom = None

        pointer = 0
        residue_pointer = []
        residue_name = []
        res_names = []

        settle_index = []
        settle_length = []

        for i in range(self.num_residue):
            if self.residue[i].multi_system != self.multi_system:
                self.residue[i].broadcast_multiplicity(self.multi_system)

            self.residue[i].set_start_index(pointer)
            residue_pointer.append(pointer)
            residue_name.append(self.residue[i].name)
            res_names.extend([self.residue[i].name] * self.residue[i].num_atoms)

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
                          f'the tail_atom of residue {i - 1} is None.')
                else:
                    # (B, 1, 2)
                    connect = msnp.concatenate(
                        (F.expand_dims(tail_atom, -2), F.expand_dims(head_atom, -2)), axis=-1)
                    bonds += (connect,)
            # (B, 1, 1)
            tail_atom = self.residue_tail(i)

            # (B, C', 2)
            if self.residue[i].bonds is not None:
                bonds += (self.residue[i].bonds + pointer,)

            if self.residue[i].settle_index is not None:
                settle_index.append(self.residue[i].settle_index + pointer)
                settle_length.append(self.residue[i].settle_length *
                                     self.units.convert_length_from(self.residue[i].settle_unit))

            pointer += self.residue[i].num_atoms

        self.num_atoms = pointer
        self.residue_pointer = Tensor(residue_pointer, ms.int32)
        self.residue_name = np.array(residue_name, np.str_)
        self.res_names = np.array(res_names, np.str_)

        # (B, A)
        self.atom_name = np.concatenate(atom_name, axis=-1)
        self.atom_type = np.concatenate(atom_type, axis=-1)
        self.atom_mass = msnp.concatenate(atom_mass, axis=-1)
        new_atom_mask = []
        for small_t in atom_mask:
            new_atom_mask.append(ops.Cast()(small_t, ms.int32))
        self.atom_mask = msnp.concatenate(new_atom_mask, axis=-1).bool()
        self.atomic_number = msnp.concatenate(atomic_number, axis=-1)
        self.inv_mass = msnp.concatenate(inv_mass, axis=-1)
        self.atom_charge = None
        if any_charge:
            atom_charge = msnp.concatenate(atom_charge, axis=-1)
            self.set_atom_charge(atom_charge)

        # (A)
        self.atom_resid = msnp.concatenate(atom_resid)
        self.image_index = msnp.concatenate(image_index)

        # (B, R)
        self.residue_mass = msnp.concatenate(residue_mass, axis=-1)
        self.res_natom_tensor = msnp.concatenate(res_natom_tensor, axis=-1)

        # (B, C, 2)
        self.bonds = None
        if bonds:
            self.bonds = msnp.concatenate(bonds, -2)
            self.build_h_bonds()
            self.build_angle()
            self.build_dihedrals()
            self.build_improper()

        self.settle_index = None
        self.remaining_index = None
        self.settle_length = None
        if settle_index:
            self.settle_index = msnp.stack(settle_index, -2)
            self.settle_length = msnp.stack(settle_length, -2)
            settle_bond_1 = self.settle_index[..., :2]
            settle_bond_2 = self.settle_index[..., ::2]
            if self.bonds is None:
                self.remaining_index = None
            else:
                remaining_index = msnp.where(bonds_in(self.bonds, settle_bond_1) + bonds_in(self.bonds, settle_bond_2),
                                             0, 1)
                self.remaining_index = ops.nonzero(remaining_index)
                if self.remaining_index.size == 0:
                    self.remaining_index = None
                else:
                    self.remaining_index = self.remaining_index[..., 1]
        else:
            if self.bonds is None:
                self.remaining_index = None
            else:
                remaining_index = ops.ones(self.bonds.shape[:-1], ms.int32)
                self.remaining_index = ops.nonzero(remaining_index)[..., 1]

        return self


    def build_space(self, coordinate: Tensor, pbc_box: Tensor = None):
        """
        Build coordinate and PBC box.

        Args:
            coordinate(Tensor): The initial coordinate of system. If it's None, the system will
                                generate a random coordinate as its initial coordinate.
            pbc_box(Tensor):    The initial pbc_box of the system. If it's None, the system won't use pbc_box.
                                Default:None
        """
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

    def build_h_bonds(self):
        """
        build bonds with hydrogen atom.
        """

        bonds = self.bonds[0].asnumpy()
        atom_type = self.atom_type[0]
        htypes = np.array(["H", "HC", "H1", "HS", "H5", "H4", "HP", "HA", "HO", "HW", "H0", "H2", "H3", "HZ",
                           "HN", "HX"], np.str_)

        hatoms = np.where(np.isin(atom_type, htypes))[0]
        bonds_with_h = np.where(np.isin(bonds, hatoms).sum(axis=-1))[0]
        # non_hbonds = np.where(np.isin(bonds, hatoms).sum(axis=-1) == 0)[0]

        self.h_bonds = Tensor(bonds_with_h, ms.int32)

        return self

    def build_angle(self):
        r"""build angles for the system"""
        if self.bonds is None:
            return self

        def _construct_angles(bonds, angle_bonds):
            for idx in self.angle_vertices:
                this_bonds = bonds[np.where(angle_bonds == idx)[0]]
                flatten_bonds = this_bonds.flatten()
                this_idx = np.delete(flatten_bonds, np.where(flatten_bonds == idx))
                yield this_idx

        def _combinations(bonds, angle_bonds):
            """
            Get all the combinations of 3 atoms.

            Args:
                bonds (ndarray):            Array of bonds.
                angle_bonds (ndarray):  Array of bonds for angles.

            Returns:
                np.ndarray, angles.
            """
            this_idx = _construct_angles(bonds, angle_bonds)
            id_selections = [
                [[0, 1]],
                [[0, 1], [1, 2], [0, 2]],
                [[0, 1], [1, 2], [2, 3], [0, 2], [0, 3], [1, 3]],
            ]
            angles = None
            counter = 0
            for idx in this_idx:
                selections = id_selections[idx.size - 2]
                for selection in selections:
                    if angles is None:
                        angles = np.insert(idx[selection], 1, self.angle_vertices[counter])[
                            None, :]
                    else:
                        angles = np.append(
                            angles,
                            np.insert(idx[selection], 1, self.angle_vertices[counter])[
                                None, :],
                            axis=0,
                        )
                counter += 1
            return angles

        bonds = self.bonds[0].asnumpy()
        self.angle_vertices = np.where(np.bincount(bonds.flatten()) > 1)[0]
        angle_index = np.where(np.sum(np.isin(bonds, self.angle_vertices), axis=1) > 0)[0]
        angle_bonds = bonds[angle_index]

        angles = _combinations(bonds, angle_bonds)

        if angles is None:
            self.angles = None
        else:
            self.angles = Tensor(angles, ms.int32)

        return self

    def build_dihedrals(self):
        r"""build dihedral angles for the system"""
        if self.bonds is None:
            return self

        def _trans_dangles(dangles, middle_id):
            """
            Construct the dihedrals.

            Args:
                dangles (ndarray):      Array of dangles.
                middle_id (ndarray):    Array of middle IDs.

            Returns:
                np.ndarray, dihedrals.
            """
            left_id = np.isin(dangles[:, 0], middle_id[0])
            left_ele = dangles[:, 2][left_id]
            left_id = np.isin(dangles[:, 2], middle_id[0])
            left_ele = np.append(left_ele, dangles[:, 0][left_id])
            right_id = np.isin(dangles[:, 1], middle_id[0])
            right_ele = np.unique(dangles[right_id])
            right_ele = right_ele[np.where(
                np.isin(right_ele, middle_id, invert=True))[0]]
            sides = itertools.product(right_ele, left_ele)
            sides_array = np.array(list(sides))

            if sides_array.size == 0:
                return sides_array

            sides = sides_array[np.where(
                sides_array[:, 0] != sides_array[:, 1])[0]]
            left = np.append(
                sides[:, 0].reshape(sides.shape[0], 1),
                np.broadcast_to(middle_id, (sides.shape[0],) + middle_id.shape),
                axis=1,
            )
            dihedrals = np.append(
                left, sides[:, 1].reshape(sides.shape[0], 1), axis=1)
            return dihedrals

        def _get_dihedrals(angles, dihedral_middle_id):
            """
            Get the dihedrals indexes.

            Args:
                angles (ndarray):               Array of angles.
                dihedral_middle_id (ndarray):   Array of dihedrals middle indexes.

            Returns:
                np.ndarray, dihedrals.
            """
            dihedrals = None
            for i in range(dihedral_middle_id.shape[0]):
                dangles = angles[
                    np.where(
                        (
                            np.isin(angles, dihedral_middle_id[i]).sum(axis=1)
                            * np.isin(angles[:, 1], dihedral_middle_id[i])
                        )
                        > 1
                    )[0]
                ]
                this_sides = _trans_dangles(dangles, dihedral_middle_id[i])
                if this_sides.size == 0:
                    continue
                if dihedrals is None:
                    dihedrals = this_sides
                else:
                    dihedrals = np.append(dihedrals, this_sides, axis=0)

            return dihedrals

        bonds = self.bonds[0].asnumpy()
        angles = self.angles.asnumpy()

        dihedral_middle_id = bonds[np.where(np.isin(bonds, self.angle_vertices).sum(axis=1) == 2)[0]]

        dihedrals = _get_dihedrals(angles, dihedral_middle_id)

        if dihedrals is None:
            self.dihedrals = None
        else:
            self.dihedrals = Tensor(dihedrals, ms.int32)

        return self

    def build_improper(self):
        r"""Build improper dihedral angles for the system"""
        if self.bonds is None:
            return self

        def _check_improper(bonds, core_id):
            """
            Check if there are same improper dihedrals.

            Args:
                bonds (ndarray):    Array of bonds.
                core_id (ndarray):  Array of core indexes.

            Returns:
                int, core id of same improper dihedrals.
            """
            # pylint: disable=pointless-statement
            checked_core_id = core_id.copy()
            bonds_hash = [hash(tuple(x)) for x in bonds]
            for i in range(core_id.shape[0]):
                ids_for_idihedral = np.where(
                    np.sum(np.isin(bonds, core_id[i]), axis=1) > 0
                )[0]
                bonds_for_idihedral = bonds[ids_for_idihedral]
                uniques = np.unique(bonds_for_idihedral.flatten())
                uniques = np.delete(uniques, np.where(uniques == core_id[i])[0])
                uniques_product = np.array(list(itertools.product(uniques, uniques)))
                uniques_hash = np.array([hash(tuple(x)) for x in itertools.product(uniques, uniques)])
                excludes = np.isin(uniques_hash, bonds_hash)
                exclude_size = np.unique(uniques_product[excludes]).size
                # Exclude condition
                if uniques.shape[0] - excludes.sum() <= 2 or exclude_size > 3:
                    checked_core_id[i] == -1
            return checked_core_id[np.where(checked_core_id > -1)[0]]

        def _get_improper(bonds, core_id):
            """
            Get the improper dihedrals indexes.

            Args:
                bonds (ndarray):    Array of bonds.
                core_id (ndarray):  Array of core indexes.

            Returns:
                - improper (np.ndarray).
                - new_id (np.ndarray).
            """
            improper = None
            new_id = None
            for i in range(core_id.shape[0]):
                ids_for_idihedral = np.where(
                    np.sum(np.isin(bonds, core_id[i]), axis=1) > 0
                )[0]
                bonds_for_idihedral = bonds[ids_for_idihedral]
                if bonds_for_idihedral.shape[0] == 3:
                    idihedral = np.unique(bonds_for_idihedral.flatten())[None, :]
                    if improper is None:
                        improper = idihedral
                        new_id = core_id[i]
                    else:
                        improper = np.append(improper, idihedral, axis=0)
                        new_id = np.append(new_id, core_id[i])
                else:
                    # Only SP2 is considered.
                    continue
            return improper, new_id

        bonds = self.bonds[0].asnumpy()

        core_id = np.where(np.bincount(bonds.flatten()) > 2)[0]

        checked_core_id = _check_improper(bonds, core_id)
        improper, third_id = _get_improper(bonds, checked_core_id)

        if improper is None:
            self.improper_dihedrals = None
            self.improper_axis_atoms = None
        else:
            self.improper_dihedrals = Tensor(improper, ms.int32)
            self.improper_axis_atoms = Tensor(third_id, ms.int32)

        return self

    def set_atom_charge(self, atom_charge: Tensor):
        """
        set atom charge

        Args:
            atom_charge(Tensor): Atom charge.
        """
        if atom_charge is None:
            self.atom_charge = atom_charge
        elif self.atom_charge is None:
            self.atom_charge = Parameter(atom_charge, name='atom_charge', requires_grad=False)
        else:
            F.assign(self.atom_charge, atom_charge)
        return self

    def set_bond_length(self, bond_length: Tensor):
        """
        Set bond length.

        Args:
            bond_length(Tensor):    Set the bond length of the system.
        """
        if self.bonds is None:
            raise ValueError('Cannot setup bond_length because bond is None')
        bond_length = Tensor(bond_length, ms.float32)
        if bond_length.shape != self.bonds.shape[:2]:
            raise ValueError(f'The shape of bond_length {self.bond_length.shape} does not match '
                             f'the shape of bond {self.bonds.shape}')
        self.bond_length = bond_length
        return self

    def residue_index(self, res_id: int) -> Tensor:
        """
        Get index of residue.

        Args:
            res_id(int):    Residue index.

        Returns:
            Tensor, the system index of the residue.
        """
        return self.residue[res_id].system_index

    def residue_bond(self, res_id: int) -> Tensor:
        """
        Get bond index of residue.

        Args:
            res_id(int):    Residue index.

        Returns:
            Tensor, the bond index of residue.
        """
        if self.residue[res_id].bonds is None:
            return None
        return self.residue[res_id].bonds + self.residue[res_id].start_index

    def residue_head(self, res_id: int) -> Tensor:
        """
        Get head index of residue.

        Args:
            res_id(int):    Residue index.

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
            res_id(int):    Residue index.

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
            res_id(int):    Residue index.

        Returns:
            Tensor, residue coordinate in the system.
        """
        return F.gather_d(self.coordinate, -2, self.residue[res_id].system_index)

    def get_volume(self) -> Tensor:
        """
        Get volume of system.

        Returns:
            Tensor, the volume of the system. If pbc_box is not used, the volume is None.
        """
        if self.pbc_box is None:
            return None
        return keepdims_prod(self.pbc_box, -1)

    def space_parameters(self) -> list:
        """
        Get the parameter of space (coordinates and pbc box).

        Returns:
            list[Tensor], coordinate and pbc_box. If pbc_box is not used, it will only return coordinate.
        """
        if self.pbc_box is None:
            return [self.coordinate]
        return [self.coordinate, self.pbc_box]

    def trainable_params(self, recurse=True) -> list:
        """
        Trainable parameters.

        Args:
            recurse(bool):  If true, yields parameters of this cell and all subcells. Otherwise, only yield parameters
                            that are direct members of this cell. Default: ``True``.

        Returns:
            list, all trainable system parameters.
        """
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
        """
        Update the parameter of coordinate.

        Args:
            coordinate(Tensor): Coordinates used to update system coordinates.

        Returns:
            Tensor, updated coordinate.
        """
        coordinate = F.assign(self.coordinate, coordinate)
        if self.pbc_box is None:
            return coordinate
        return F.depend(coordinate, self.update_image())

    def set_coordianate(self, coordinate: Tensor) -> Tensor:
        """
        Set the value of coordinate.

        Args:
            coordinate(Tensor): Coordinates used to set system coordinates.

        Returns:
            Tensor, the coordinate of the system.
        """
        coordinate = self._check_coordianate(coordinate)
        if coordinate is not None and coordinate.shape == self.coordinate.shape:
            return self.update_coordinate(coordinate)

        self.coordinate = Parameter(coordinate, name='coordinate')
        self.dimension = self.coordinate.shape[-1]
        return self.identity(coordinate)

    def update_pbc_box(self, pbc_box: Tensor) -> Tensor:
        """
        Update PBC box

        Args:
            pbc_box(Tensor):    PBC box used to update the system PBC box.

        Returns:
            Tensor, updated system PBC box.
        """
        pbc_box = F.assign(self.pbc_box, pbc_box)
        return F.depend(pbc_box, self.update_image())

    def set_pbc_grad(self, grad_box: bool) -> bool:
        """
        Set whether to calculate the gradient of PBC box.

        Args:
            grad_box(bool): Whether to calculate the gradient of PBC box.

        Returns:
            bool, whether to calculate the gradient of PBC box.
        """
        if self.pbc_box is None:
            return grad_box
        self.pbc_box.requires_grad = grad_box
        return self.pbc_box.requires_grad

    def set_pbc_box(self, pbc_box: Tensor = None) -> Tensor:
        """
        Set PBC box.

        Args:
            pbc_box(Tensor):    Set the PBC box of the system. If it's None, the system won't use PBC box.
                                Default: ``None``.

        Returns:
            Tensor, system PBC box.
        """
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
        """
        Repeat the system according to the lattices of PBC box.

        Args:
            lattices(list): Lattices of PBC box.
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
                coordinate += (coord + shift,)

            self.build_system()
            coordinate = msnp.concatenate(coordinate, axis=-2)
            self.build_space(coordinate, self.pbc_box)
            new_box = Tensor(lattices, ms.int32) * self.pbc_box
            self.update_pbc_box(new_box)

        return self

    def coordinate_in_pbc(self, shift: float = 0) -> Tensor:
        """
        Get the coordinate in a whole PBC box.

        Args:
            shift(float):   Offset ratio relative to box size. Default: 0

        Returns:
            Tensor, the coordinate in the PBC box. Shape `(B, ..., D)`. Data type is float.
        """
        coordinate = self.identity(self.coordinate)
        pbc_box = self.identity(self.pbc_box)
        return func.coordinate_in_pbc(coordinate, pbc_box, shift)

    def calc_image(self, shift: float = 0) -> Tensor:
        r"""
        Calculate the image of coordinate.

        Args:
            shift(float):   Offset ratio :math:`c` relative to box size :math:`\vec{L}`.
                Default: ``0`` .

        Returns:
            Tensor, the image of coordinate.
        """
        coordinate = self.identity(self.coordinate)
        pbc_box = self.identity(self.pbc_box)
        image = func.pbc_image(coordinate, pbc_box, shift)
        if self.image_index is not None:
            image = image[:, self.image_index, :]
        return image

    def update_image(self, image: Tensor = None) -> bool:
        """
        Update the image of coordinate.

        Args:
            image(Tensor):  The image of coordinate used to update the image of system coordinate. Default: ``None``.

        Returns:
            bool, whether successfully update the image of coordinate.
        """
        if image is None:
            image = self.calc_image()
        return F.assign(self.image, image)

    def set_length_unit(self, unit):
        """
        Set the length unit of system.

        Args:
            unit(Union[str, Units, Length, float, int]):    Length unit.
        """
        scale = self.units.convert_length_to(unit)
        coordinate = self.coordinate * scale
        self.update_coordinate(coordinate)
        if self.pbc_box is not None:
            pbc_box = self.pbc_box * scale
            self.update_pbc_box(pbc_box)
        self.units.set_length_unit(unit)
        return self

    def calc_colvar(self, colvar: Colvar) -> Tensor:
        """
        Calculate the value of specific collective variables in the system.

        Args:
            colvar(Colvar):  Base class for generalized collective variables (CVs) :math:`s(R)`.

        Returns:
            Tensor, the value of a collective variables :math:`s(R)`.
        """
        coordinate = self.identity(self.coordinate)
        pbc_box = None if self.pbc_box is None else self.identity(self.pbc_box)
        return colvar(coordinate, pbc_box)

    def get_atoms(self, atoms: Union[Tensor, Parameter, ndarray, str, list, tuple]) -> AtomsBase:
        """
        Get atoms from the system.

        Args:
            atoms(Union[Tensor, Parameter, ndarray, str, list, tuple]): List of atoms.

        Returns:
            class, atoms or groups of atoms.
        """
        try:
            atoms = _get_atoms(atoms)
        except TypeError:
            # TODO
            pass
        return atoms

    def get_coordinate(self, atoms: AtomsBase = None) -> Tensor:
        """
        Get Tensor of coordinate.

        Args:
            atoms(class):   Base class for specific atoms group, used as the "atoms group module" in MindSPONGE.
                            Default: ``None``.

        Returns:
            Tensor. Coordinate. Data type is float.
        """
        coordinate = self.identity(self.coordinate)
        if atoms is None:
            return coordinate
        pbc_box = None if self.pbc_box is None else self.identity(self.pbc_box)
        return atoms(coordinate, pbc_box)

    def get_pbc_box(self) -> Tensor:
        """
        Get Tensor of PBC box.

        Returns:
            Tensor, PBC box
        """
        if self.pbc_box is None:
            return None
        return self.identity(self.pbc_box)

    def fill_water(self, edge: float = None, gap: float = None, box: ndarray = None, pdb_out: str = None,
                   template: str = None):
        """ The inner function in Molecule class to add water in a given box.

        Args:
            edge(float): The water edge around the system.
            gap(float): The minimum gap between system atoms and water atoms.
            box(Tensor): The pbc box we want, default to be None.
            pdb_out(str): The string format pdb file name to store the information of system after filling water.
            template(str): The supplemental template of the water molecules filled.

        Returns:
            new_pbc_box(Tensor), this function will return a pbc_box after filling water.
        """
        if template is None:
            raise ValueError('The template when filling water must be given!')

        atom_names = self.atom_name[0]
        res_names = self.res_names

        if isinstance(self.atom_resid, Tensor):
            res_ids = self.atom_resid.asnumpy()
        else:
            res_ids = self.atom_resid

        if isinstance(self.coordinate, Tensor):
            crds = self.coordinate[0].asnumpy()
        else:
            crds = self.coordinate[0]

        crds *= length_convert(self.length_unit, 'A')
        edge *= length_convert(self.length_unit, 'A')

        if gap is None:
            gap = 4.0
        else:
            gap *= length_convert(self.length_unit, 'A')

        min_x = crds[:, 0].min()
        min_y = crds[:, 1].min()
        min_z = crds[:, 2].min()

        max_x = crds[:, 0].max()
        max_y = crds[:, 1].max()
        max_z = crds[:, 2].max()

        size_x = max_x - min_x
        size_y = max_y - min_y
        size_z = max_z - min_z

        if box is None:
            box = np.array([size_x + 2 * edge,
                            size_y + 2 * edge,
                            size_z + 2 * edge], np.float32)
        else:
            box = box * length_convert(self.length_unit, 'A')
        final_box = box + 0.5 * AVGDIS
        print('[MindSPONGE] The box size used when filling water is: {}'.format(final_box *
                                                                                self.units.convert_length_from('A')))

        if (final_box < size_x).any():
            raise ValueError("Please given a larger box for adding water molecules.")

        origin_x = min_x - edge * 0.9
        origin_y = min_y - edge * 0.9
        origin_z = min_z - edge * 0.9

        num_waters = np.ceil(box / AVGDIS)
        total_waters = np.product(num_waters)
        print('[MindSPONGE] The edge gap along x axis is {}.'.format((box[0] - size_x) *
                                                                     self.units.convert_length_from('A') / 2))
        print('[MindSPONGE] The edge gap along y axis is {}.'.format((box[1] - size_y) *
                                                                     self.units.convert_length_from('A') / 2))
        print('[MindSPONGE] The edge gap along z axis is {}.'.format((box[2] - size_z) *
                                                                     self.units.convert_length_from('A') / 2))

        o_x = origin_x + (np.arange(total_waters) % num_waters[0]) * AVGDIS
        o_y = origin_y + ((np.arange(total_waters) // num_waters[0]) % num_waters[1]) * AVGDIS
        o_z = origin_z + ((np.arange(total_waters) // np.product(num_waters[:2])) % num_waters[2]) * AVGDIS

        o_crd = np.concatenate((o_x[:, None], o_y[:, None], o_z[:, None]), axis=-1)

        dis = np.linalg.norm(crds - o_crd[:, None, :], axis=-1)
        filt = np.where((dis <= gap).sum(axis=-1) > 0)

        o_crd = np.delete(o_crd, filt, axis=0)
        print('[MindSPONGE] Totally {} waters is added to the system!'.format(o_crd.shape[0]))
        h1_crd = o_crd + np.array([0.079079641, 0.061207927, 0.0], np.float32) * 10
        h2_crd = o_crd + np.array([-0.079079641, 0.061207927, 0.0], np.float32) * 10
        water_crd = np.hstack((o_crd, h1_crd, h2_crd)).reshape((-1, 3))
        water_names = np.array(['O', 'H1', 'H2'] * o_crd.shape[0], np.str_)
        water_res = np.array(['WAT'] * water_crd.shape[0], np.str_)
        water_resid = np.concatenate((np.arange(o_crd.shape[0])[:, None],
                                      np.arange(o_crd.shape[0])[:, None],
                                      np.arange(o_crd.shape[0])[:, None]), axis=-1).reshape((-1))

        new_crd = np.vstack((crds, water_crd))
        atom_names = np.concatenate((atom_names, water_names))
        res_names = np.concatenate((res_names, water_res))
        res_ids = np.concatenate((res_ids, max(res_ids) + 1 + water_resid))
        new_pbc_box = final_box * self.units.convert_length_from('A')

        self.init_resname = res_names
        self.init_resid = res_ids
        num_residue = self.num_residue + o_crd.shape[0]
        residue_name = np.concatenate((self.residue_name, np.array(['WAT'] * o_crd.shape[0], np.str_)), axis=0)
        residue_pointer = np.concatenate((self.residue_pointer.asnumpy(),
                                          np.arange(o_crd.shape[0]) * 3 + self.num_atoms), axis=0)

        residue_pointer = np.append(residue_pointer, len(atom_names))

        if isinstance(self.template, str):
            self.template = [self.template]
        self.template.append(template)
        _template = get_template(self.template)

        self.residue = []
        for i in range(num_residue):
            name = residue_name[i]
            atom_name = atom_names[residue_pointer[i]: residue_pointer[i + 1]][None, :]
            if name in RESIDUE_NAMES:
                residue = AminoAcid(name=name, template=_template, atom_name=atom_name,
                                    length_unit=self.units.length_unit)
            else:
                residue = Residue(name=name, template=_template, atom_name=atom_name,
                                  length_unit=self.units.length_unit)
            self.residue.append(residue)

        coordinate = new_crd * self.units.convert_length_from('A')

        self.build_system()
        self.build_space(coordinate, new_pbc_box)

        if pdb_out is not None:
            gen_pdb(new_crd[None, :], atom_names, res_names, res_ids, pdb_name=pdb_out)

        print('[MindSPONGE] Adding water molecule task finished!')

        return self

    def construct(self) -> Tuple[Tensor, Tensor]:
        r"""Get space information of system.

        Returns:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.

        Note:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """
        coordinate = self.identity(self.coordinate)
        pbc_box = None
        if self.pbc_box is not None:
            pbc_box = self.identity(self.pbc_box)
        return coordinate, pbc_box


class MoleculeFromMol2(Molecule):
    r"""
    Base class for molecular system, used as the "system module" in MindSPONGE.
    The `Molecule` Cell can represent a molecule or a system consisting of multiple molecules.
    The major components of the `Molecule` Cell is the `Residue` Cell. A `Molecule` Cell can
    contain multiple `Residue` Cells.

    Args:
        mol2_name(str): The string format mol2 file name.

    Outputs:
        - coordinate, Tensor of shape `(B, A, D)`. Data type is float.
        - pbc_box, Tensor of shape `(B, D)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation
        A:  Number of atoms.
        b:  Number of bonds.
        D:  Spatial dimension of the simulation system. Usually is 3.
    """

    def __init__(self, mol2_name: str, pbc_box: Union[Tensor, ndarray, List[float]] = None, length_unit: str = None,
                 **kwargs):
        mol2_obj = mol2parser(mol2_name)
        super().__init__(atoms=mol2_obj.atom_types,
                         bonds=mol2_obj.bond_indexes,
                         atom_charge=mol2_obj.charges,
                         coordinate=mol2_obj.crds,
                         pbc_box=pbc_box,
                         length_unit=length_unit)
        self.build_space(coordinate=mol2_obj.crds, pbc_box=pbc_box)


class _MoleculeFromPDB(Molecule):
    r"""
    Base class for molecular system, used as the "system module" in MindSPONGE.
    The `Molecule` Cell can represent a molecule or a system consisting of multiple molecules.
    The major components of the `Molecule` Cell is the `Residue` Cell. A `Molecule` Cell can
    contain multiple `Residue` Cells.

    Args:
        pdb_name(str): The string format pdb file name.

    Outputs:
        - coordinate, Tensor of shape `(B, A, D)`. Data type is float.
        - pbc_box, Tensor of shape `(B, D)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation
        A:  Number of atoms.
        b:  Number of bonds.
        D:  Spatial dimension of the simulation system. Usually is 3.
    """

    def __init__(self, pdb_name: str, pbc_box: Union[Tensor, ndarray, List[float]] = None, length_unit: str = None,
                 template: Union[dict, str, List[Union[dict, str]], Tuple[Union[dict, str]]] = None,
                 rebuild_hydrogen: bool = False):
        pdb_obj = read_pdb(pdb_name, rebuild_hydrogen=rebuild_hydrogen)
        super().__init__(atoms=pdb_obj.flatten_atoms,
                         coordinate=pdb_obj.flatten_crds,
                         pbc_box=pbc_box,
                         length_unit=length_unit)

        residue_name = pdb_obj.res_names
        residue_names = np.array(RESIDUE_NAMES, np.str_)
        residue_pointer = pdb_obj.res_pointer
        flatten_atoms = pdb_obj.flatten_atoms
        flatten_crds = pdb_obj.flatten_crds
        init_res_names = pdb_obj.init_res_names
        init_res_ids = pdb_obj.init_res_ids
        residue_in_amino = np.where(np.isin(residue_name, residue_names))[0]
        residue_name[residue_in_amino.min()] = 'N' + residue_name[residue_in_amino.min()]
        residue_name[residue_in_amino.max()] = 'C' + residue_name[residue_in_amino.max()]

        self.init_resname = init_res_names
        self.init_resid = init_res_ids
        num_residue = len(residue_name)
        residue_pointer = np.append(residue_pointer, len(flatten_atoms))

        self.template = template
        template = get_template(template)
        self.residue = []
        for i in range(num_residue):
            name = residue_name[i]
            atom_name = flatten_atoms[residue_pointer[i]: residue_pointer[i + 1]][None, :]
            if name in RESIDUE_NAMES:
                residue = AminoAcid(name=name, template=template, atom_name=atom_name,
                                    length_unit=self.units.length_unit)
            else:
                residue = Residue(name=name, template=template, atom_name=atom_name,
                                  length_unit=self.units.length_unit)
            self.residue.append(residue)

        coordinate = flatten_crds * self.units.convert_length_from('A')

        self.build_system()
        self.build_space(coordinate, pbc_box)
