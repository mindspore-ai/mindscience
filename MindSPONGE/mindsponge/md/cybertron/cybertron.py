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
"""cybertron"""

import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from .units import units
from .base import Types2FullConnectNeighbors
from .readouts import Readout, LongeRangeReadout
from .readouts import AtomwiseReadout, GraphReadout
from .neighbors import Distances


class Cybertron(nn.Cell):
    """Cybertron: An architecture to perform deep molecular model for molecular modeling.

    Args:
        model       (nn.Cell):          Deep molecular model
        dim_output  (int):              Output dimension of the predictions
        unit_dis    (str):              Unit of input distance
        unit_energy (str):              Unit of output energy
        readout     (readouts.Readout): Readout function

    """

    def __init__(self, model, dim_output=1, unit_dis='nm', unit_energy=None, readout='atomwise', max_atoms_number=0,
                 atom_types=None, bond_types=None, pbcbox=None, full_connect=False, cut_shape=False,):
        super().__init__()

        self.model = model
        self.dim_output = dim_output
        self.cut_shape = cut_shape

        self.unit_dis = unit_dis
        self.unit_energy = unit_energy

        self.dis_scale = units.length_convert_from(unit_dis)
        activation = self.model.activation

        self.molsum = P.ReduceSum(keep_dims=True)

        self.atom_mask = None
        self.atom_types = None
        if atom_types is None:
            self.fixed_atoms = False
            self.num_atoms = 0
        else:
            self.fixed_atoms = True
            self.model.set_fixed_atoms(True)

            if len(atom_types.shape) == 1:
                self.num_atoms = len(atom_types)
            elif len(atom_types.shape) == 2:
                self.num_atoms = len(atom_types[0])

            if self.num_atoms <= 0:
                raise ValueError(
                    "The 'num_atoms' cannot be 0 " +
                    "'atom_types' is not 'None' in MolCalculator!")

            if not isinstance(atom_types, Tensor):
                atom_types = Tensor(atom_types, ms.int32)

            self.atom_types = atom_types
            self.atom_mask = F.expand_dims(atom_types, -1) > 0
            if self.atom_mask.all():
                self.atom_mask = None

            atoms_number = F.cast(atom_types > 0, ms.float32)
            self.atoms_number = self.molsum(atoms_number, -1)

        self.pbcbox = None
        self.use_fixed_box = False
        if pbcbox is not None:
            if isinstance(pbcbox, (list, tuple)):
                pbcbox = Tensor(pbcbox, ms.float32)
            if not isinstance(pbcbox, Tensor):
                raise TypeError(
                    "Unsupported pbcbox type '{}'.".format(
                        type(pbcbox)))
            if len(pbcbox.shape) == 1:
                pbcbox = F.expand_dims(pbcbox, 0)
            if len(pbcbox.shape) != 2:
                raise ValueError(
                    "The length of shape of pbcbox must be 1 or 2")
            if pbcbox.shape[-1] != 3:
                raise ValueError("The last dimension of pbcbox must be 3")
            if pbcbox.shape[0] != 1:
                raise ValueError("The first dimension of pbcbox must be 1")
            self.pbcbox = pbcbox
            self.use_fixed_box = True

        self.use_bonds = self.model.use_bonds
        self.fixed_bonds = False
        self.bonds = None
        if bond_types is not None:
            self.bonds = bond_types
            self.bond_mask = (bond_types > 0)
            self.fixed_bonds = True

        self.cutoff = self.model.cutoff

        self.use_distances = self.model.use_distances

        self.full_connect = full_connect

        if self.fixed_bonds and (not self.use_distances):
            raise ValueError(
                '"fixed_bonds" cannot be used without using distances')

        self.neighbors = None
        self.mask = None
        self.fc_neighbors = None
        if self.full_connect:
            if self.fixed_atoms:
                self.fc_neighbors = Types2FullConnectNeighbors(self.num_atoms)
                self.neighbors = self.fc_neighbors.get_full_neighbors()
            else:
                if max_atoms_number <= 0:
                    raise ValueError(
                        "The 'max_atoms_num' cannot be 0 " +
                        "when the 'full_connect' flag is 'True' and " +
                        "'atom_types' is 'None' in MolCalculator!")
                self.fc_neighbors = Types2FullConnectNeighbors(
                    max_atoms_number)
        self.max_atoms_number = max_atoms_number

        if self.fixed_atoms and self.full_connect:
            fixed_neigh = True
            self.distances = Distances(True, long_dis=self.cutoff * 10)
            self.model.set_fixed_neighbors(True)
        else:
            fixed_neigh = False
            self.distances = Distances(False, long_dis=self.cutoff * 10)
        self.fixed_neigh = fixed_neigh

        self.multi_readouts = False
        self.num_readout = 1

        dim_feature = self.model.dim_feature
        n_interactions = self.model.n_interactions

        if isinstance(readout, (tuple, list)):
            self.num_readout = len(readout)
            if self.num_readout == 1:
                readout = readout[0]
            else:
                self.multi_readouts = True

        if self.multi_readouts:
            readouts = []
            for i in range(self.num_readout):
                readouts.append(self._get_readout(readout[i],
                                                  n_in=dim_feature,
                                                  n_out=dim_output,
                                                  activation=activation,
                                                  unit_energy=unit_energy,
                                                  ))
            self.readout = nn.CellList(readouts)
        else:
            self.readout = self._get_readout(readout,
                                             n_in=dim_feature,
                                             n_out=dim_output,
                                             activation=activation,
                                             unit_energy=unit_energy,
                                             )

        self.output_scale = 1
        self.calc_far = False
        read_all_interactions = False
        self.dim_output = 0
        if self.multi_readouts:
            read_all_interactions = False
            self.output_scale = []
            for i in range(self.num_readout):
                self.dim_output += self.readout[i].total_out
                if unit_energy is not None and self.readout[i].output_is_energy:
                    unit_energy = units.check_energy_unit(unit_energy)
                    self.output_scale.append(
                        units.energy_convert_to(unit_energy))
                else:
                    self.output_scale.append(1)

                if isinstance(self.readout[i], LongeRangeReadout):
                    self.calc_far = True
                    self.readout[i].set_fixed_neighbors(fixed_neigh)
                if self.readout[i].read_all_interactions:
                    read_all_interactions = False
                    if self.readout[i].interaction_decoders is not None and\
                            self.readout[i].n_interactions != n_interactions:
                        raise ValueError(
                            'The n_interactions in model readouts are not equal')
                if self.readout[i].n_in != dim_feature:
                    raise ValueError(
                        'n_in in readouts is not equal to dim_feature')
        else:
            self.dim_output = self.readout.total_out

            if unit_energy is not None and self.readout.output_is_energy:
                unit_energy = units.check_energy_unit(unit_energy)
                self.output_scale = units.energy_convert_to(unit_energy)
            else:
                self.output_scale = 1

            if isinstance(self.readout, LongeRangeReadout):
                self.calc_far = True
                self.readout.set_fixed_neighbors(fixed_neigh)

            if self.readout.read_all_interactions:
                read_all_interactions = True
                if self.readout.interaction_decoders is not None and self.readout.n_interactions != n_interactions:
                    raise ValueError(
                        'The n_interactions in model readouts are not equal')

            if self.readout.n_in != dim_feature:
                raise ValueError(
                    'n_in in readouts is not equal to dim_feature')

        self.unit_energy = unit_energy

        self.model.read_all_interactions = read_all_interactions

        self.ones = P.Ones()
        self.reduceany = P.ReduceAny(keep_dims=True)
        self.reducesum = P.ReduceSum(keep_dims=False)
        self.reducemax = P.ReduceMax()
        self.reducemean = P.ReduceMean(keep_dims=False)
        self.concat = P.Concat(-1)

    def _get_readout(self, readout, n_in, n_out, activation, unit_energy,):
        """get readout"""
        if readout is None or isinstance(readout, Readout):
            return readout
        if isinstance(readout, str):
            if readout.lower() == 'atom' or readout.lower() == 'atomwise':
                readout = AtomwiseReadout
            elif readout.lower() == 'graph' or readout.lower() == 'set2set':
                readout = GraphReadout
            else:
                raise ValueError("Unsupported Readout type" + readout.lower())

            return readout(
                n_in=n_in,
                n_out=n_out,
                activation=activation,
                unit_energy=unit_energy,
            )

        raise TypeError("Unsupported Readout type '{}'.".format(type(readout)))

    def print_info(self):
        """print info"""
        print("================================================================================")
        print("Cybertron Engine, Ride-on!")
        print('---with input distance unit: ' + self.unit_dis)
        print('---with input distance unit: ' + self.unit_dis)
        if self.fixed_atoms:
            print('---with fixed atoms: ' + str(self.atom_types[0]))
        if self.full_connect:
            print('---using full connected neighbors')
        if self.use_bonds and self.fixed_bonds:
            print('---using fixed bond connection:')
            for b in self.bonds[0]:
                print('------' + str(b.asnumpy()))
            print('---with fixed bond mask:')
            for m in self.bond_mask[0]:
                print('------' + str(m.asnumpy()))
        self.model.print_info()

        if self.multi_readouts:
            print("---with multiple readouts: ")
            for i in range(self.num_readout):
                print("---" + str(i + 1) +
                      (". " + self.readout[i].name + " readout"))
        else:
            print("---with readout type: " + self.readout.name)
            self.readout.print_info()

        if self.unit_energy is not None:
            print("---with output units: " + str(self.unit_energy))
            print("---with output scale: " + str(self.output_scale))
        print("---with total output dimension: " + str(self.dim_output))
        print("================================================================================")

    def construct(self,
                  positions=None,
                  atom_types=None,
                  pbcbox=None,
                  neighbors=None,
                  neighbor_mask=None,
                  bonds=None,
                  bond_mask=None,
                  ):
        """Compute the properties of the molecules.

        Args:
            positions     (mindspore.Tensor[float], [B, A, 3]): Cartesian coordinates for each atom.
            atom_types    (mindspore.Tensor[int],   [B, A]):    Types (nuclear charge) of input atoms.
                                                                If the attribute "self.atom_types" have been set and
                                                                atom_types is not given here,
                                                                atom_types = self.atom_types
            neighbors     (mindspore.Tensor[int],   [B, A, N]): Indices of other near neighbor atoms around a atom
            neighbor_mask (mindspore.Tensor[bool],  [B, A, N]): Mask for neighbors
            bonds         (mindspore.Tensor[int],   [B, A, N]): Types (ID) of bond connected with two atoms
            bond_mask     (mindspore.Tensor[bool],  [B, A, N]): Mask for bonds

            B:  Batch size, usually the number of input molecules or frames
            A:  Number of input atoms, usually the number of atoms in one molecule or frame
            N:  Number of other nearest neighbor atoms around a atom
            O:  Output dimension of the predicted properties

        Returns:
            properties mindspore.Tensor[float], [B,A,O]: prediction for the properties of the molecules

        """

        atom_mask = None
        atoms_number = None
        if atom_types is None:
            if self.fixed_atoms:
                atom_types = self.atom_types
                atom_mask = self.atom_mask
                atoms_number = self.atoms_number
                if self.full_connect:
                    neighbors = self.neighbors
                    neighbor_mask = None
            else:
                # raise ValueError('atom_types is miss')
                return None
        else:
            atom_mask = F.expand_dims(atom_types, -1) > 0
            atoms_number = F.cast(atom_types > 0, ms.float32)
            atoms_number = self.molsum(atoms_number, -1)

        if pbcbox is None and self.use_fixed_box:
            pbcbox = self.pbcbox

        if self.use_bonds:
            if bonds is None:
                if self.fixed_bonds:
                    exones = self.ones((positions.shape[0], 1, 1), ms.int32)
                    bonds = exones * self.bonds
                    bond_mask = exones * self.bond_mask
                else:
                    # raise ValueError('bonds is miss')
                    return None
            if bond_mask is None:
                bond_mask = (bonds > 0)

        if neighbors is None:
            if self.full_connect:
                neighbors, neighbor_mask = self.fc_neighbors(atom_types)
                if self.cut_shape:
                    atypes = F.cast(atom_types > 0, positions.dtype)
                    anum = self.reducesum(atypes, -1)
                    nmax = self.reducemax(anum)
                    nmax = F.cast(nmax, ms.int32)
                    nmax0 = int(nmax.asnumpy())
                    nmax1 = nmax0 - 1

                    atom_types = atom_types[:, :nmax0]
                    positions = positions[:, :nmax0, :]
                    neighbors = neighbors[:, :nmax0, :nmax1]
                    neighbor_mask = neighbor_mask[:, :nmax0, :nmax1]
            else:
                # raise ValueError('neighbors is miss')
                return None

        if self.use_distances:
            r_ij = self.distances(
                positions,
                neighbors,
                neighbor_mask,
                pbcbox) * self.dis_scale
        else:
            r_ij = 1
            neighbor_mask = bond_mask

        x, xlist = self.model(r_ij, atom_types, atom_mask,
                              neighbors, neighbor_mask, bonds, bond_mask)

        if self.readout is None:
            return x

        if self.multi_readouts:
            ytuple = ()
            for i in range(self.num_readout):
                yi = self.readout[i](
                    x,
                    xlist,
                    atom_types,
                    atom_mask,
                    atoms_number)
                if self.unit_energy is not None:
                    yi = yi * self.output_scale[i]
                ytuple = ytuple + (yi,)
            y = self.concat(ytuple)
        else:
            y = self.readout(
                x,
                xlist,
                atom_types,
                atom_mask,
                atoms_number)
            if self.unit_energy is not None:
                y = y * self.output_scale

        return y


class CybertronFF(Cybertron):
    """CybertronFF"""
    def __init__(self, model, dim_output=1, unit_dis='nm', unit_energy=None, readout='atomwise', max_atoms_number=0,
                 atom_types=None, bond_types=None, full_connect=False, pbcbox=None, cut_shape=False,):
        super().__init__(
            model=model,
            dim_output=dim_output,
            unit_dis=unit_dis,
            unit_energy=unit_energy,
            readout=readout,
            max_atoms_number=max_atoms_number,
            atom_types=atom_types,
            bond_types=bond_types,
            full_connect=full_connect,
            pbcbox=pbcbox,
            cut_shape=cut_shape,
        )

    def construct(self,
                  positions=None,
                  atom_types=None,
                  pbcbox=None,
                  neighbors=None,
                  neighbor_mask=None,
                  bonds=None,
                  bond_mask=None
                  ):
        if self.full_connect and self.atom_types is not None:
            atom_types = self.atom_types

        if self.use_fixed_box:
            pbcbox = self.pbcbox

        return super().construct(
            positions=positions,
            atom_types=atom_types,
            pbcbox=pbcbox,
            neighbors=neighbors,
            neighbor_mask=neighbor_mask,
            bonds=bonds,
            bond_mask=bond_mask
        )
