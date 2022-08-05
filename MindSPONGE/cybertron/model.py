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
Deep molecular model
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as msnp
from mindspore.nn import Cell
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.initializer import Normal

from mindsponge import global_units
from mindsponge.data import set_class_into_hyper_param, set_hyper_parameter
from mindsponge.data import get_hyper_string, get_hyper_parameter, get_class_parameters
from mindsponge.function import Units, Length
from mindsponge.function import get_integer

from .block import Residual, Dense
from .interaction import SchNetInteraction
from .interaction import PhysNetModule
from .interaction import NeuralInteractionUnit
from .base import GraphNorm
from .filter import ResFilter, DenseFilter
from .cutoff import Cutoff, get_cutoff
from .rbf import get_rbf
from .activation import get_activation

__all__ = [
    "MolecularModel",
    "SchNet",
    "PhysNet",
    "MolCT",
]


class MolecularModel(Cell):
    r"""Basic class for graph neural network (GNN) based deep molecular model

    Reference:

        Zhang, J.; Lei, Y.-K.; Zhang, Z.; Chang, J.; Li, M.; Han, X.; Yang, L.; Yang, Y. I.; Gao, Y. Q.
        A Perspective on Deep Learning for Molecular Modeling and Simulations [J].
        The Journal of Physical Chemistry A, 2020, 124(34): 6745-6763.

    Args:

        dim_feature (int):          Dimension of atomic representation. Default: 128

        n_interaction (int):        Number of interaction layers. Default: 3

        activation (Cell):          Activation function. Default: None

        cutoff (Length):            Cutoff distance. Default: Length(1, 'nm')

        cutoff_fn (Cell):           Cutoff function. Default: None

        rbf (Cell):                 Radical baiss function. Default: None

        r_self (Length)             Distance of atomic self-interaction. Default: None

        coupled_interaction (bool): Whether to use coupled (shared) interaction layer. Default: False

        use_distance (bool):        Whether to use distance between atoms. Default: True

        use_bond (bool):            Whether to use bond information. Default: False

        use_graph_norm (bool):      Whether to use graph normalization. Default: False

        public_dis_filter (bool):   Whether to use public (shared) filter for distance. Default: False

        public_bond_filter (bool):  Whether to use public (shared) filter for bond. Default: False

        num_atom_types (int):       Maximum number of atomic types. Default: 64

        num_bond_types (int):       Maximum number of bond types. Default: 16

        length_unit (bool):         Unit of position coordinates. Default: 'nm'

        hyper_param (dict):         Hyperparameter for molecular model. Default: None

    Symbols:

        B:  Number of simulation walker

        A:  Number of atoms in system

        N:  Number of neighbour atoms

        D:  Dimension of position coordinates, usually is 3

        K:  Number of basis functions in RBF

        F:  Feature dimension of representation

    """

    def __init__(self,
                 dim_feature: int = 128,
                 n_interaction: int = 3,
                 activation: Cell = None,
                 cutoff: Length = Length(1, 'nm'),
                 cutoff_fn: Cutoff = None,
                 rbf: Cell = None,
                 r_self: Length = None,
                 coupled_interaction: bool = False,
                 use_distance: bool = True,
                 use_bond: bool = False,
                 use_graph_norm: bool = False,
                 public_dis_filter: bool = False,
                 public_bond_filter: bool = False,
                 num_atom_types: int = 64,
                 num_bond_types: int = 16,
                 length_unit: bool = 'nm',
                 hyper_param: dict = None,
                 ):

        super().__init__()

        self.network_name = 'MolecularModel'

        if hyper_param is not None:
            num_atom_types = get_hyper_parameter(hyper_param, 'num_atom_types')
            num_bond_types = get_hyper_parameter(hyper_param, 'num_bond_types')
            dim_feature = get_hyper_parameter(hyper_param, 'dim_feature')
            n_interaction = get_hyper_parameter(hyper_param, 'n_interaction')
            activation = get_class_parameters(hyper_param, 'activation')
            cutoff = get_hyper_parameter(hyper_param, 'cutoff')
            cutoff_fn = get_class_parameters(hyper_param, 'cutoff_fn')
            rbf = get_class_parameters(hyper_param, 'rbf')
            r_self = get_hyper_parameter(hyper_param, 'r_self')
            coupled_interaction = get_hyper_parameter(
                hyper_param, 'coupled_interaction')
            use_distance = get_hyper_parameter(hyper_param, 'use_distance')
            use_bond = get_hyper_parameter(hyper_param, 'use_bond')
            public_dis_filter = get_hyper_parameter(
                hyper_param, 'public_dis_filter')
            public_bond_filter = get_hyper_parameter(
                hyper_param, 'public_bond_filter')
            use_graph_norm = get_hyper_parameter(hyper_param, 'use_graph_norm')
            length_unit = get_hyper_string(hyper_param, 'length_unit')

        if length_unit is None:
            self.units = global_units
        else:
            self.units = Units(length_unit)
        self.length_unit = self.units.length_unit

        self.num_atom_types = get_integer(num_atom_types)
        self.num_bond_types = get_integer(num_bond_types)
        self.dim_feature = get_integer(dim_feature)
        self.n_interaction = get_integer(n_interaction)
        self.r_self = r_self
        self.coupled_interaction = Tensor(coupled_interaction, ms.bool_)
        self.use_distance = self.broadcast_to_interactions(
            use_distance, 'use_distance')
        self.use_bond = self.broadcast_to_interactions(use_bond, 'use_bond')
        self.public_dis_filter = Tensor(public_dis_filter, ms.bool_)
        self.public_bond_filter = Tensor(public_bond_filter, ms.bool_)
        self.use_graph_norm = Tensor(use_graph_norm, ms.bool_)

        self.activation = get_activation(activation)

        self.cutoff = None
        self.cutoff_fn = None
        self.rbf = None
        self.atom_embedding = None
        if self.use_distance.any():
            self.atom_embedding = nn.Embedding(
                self.num_atom_types, self.dim_feature, use_one_hot=True, embedding_table=Normal(1.0))
            self.cutoff = self.get_length(cutoff)
            self.cutoff_fn = get_cutoff(cutoff_fn, self.cutoff)
            self.rbf = get_rbf(rbf, self.cutoff, length_unit=self.length_unit)

        self.r_self_ex = None
        if self.r_self is not None:
            self.r_self = self.get_length(self.r_self)
            self.r_self_ex = F.expand_dims(self.r_self, 0)

        self.bond_embedding = None
        if self.use_bond.any():
            self.bond_embedding = nn.Embedding(
                self.num_bond_types, self.dim_feature, use_one_hot=True, embedding_table=Normal(1.0))

        self.num_basis = self.rbf.num_basis

        self.interactions = None
        self.interaction_typenames = []

        self.calc_distance = self.use_distance.any()
        self.calc_bond = self.use_bond.any()

        self.use_pub_norm = False

        if self.use_graph_norm:
            if self.use_pub_norm:
                self.graph_norm = nn.CellList(
                    [GraphNorm(dim_feature) * self.n_interaction]
                )
            else:
                self.graph_norm = nn.CellList(
                    [GraphNorm(dim_feature) for _ in range(self.n_interaction)]
                )
        else:
            self.graph_norm = None

        self.zeros = P.Zeros()
        self.ones = P.Ones()
        self.concat = P.Concat(-1)

        self.hyper_param = dict()
        self.hyper_types = {
            'num_atom_types': 'int',
            'num_bond_types': 'int',
            'dim_feature': 'int',
            'n_interaction': 'int',
            'activation': 'Cell',
            'cutoff': 'float',
            'cutoff_fn': 'Cell',
            'rbf': 'Cell',
            'r_self': 'float',
            'coupled_interaction': 'bool',
            'use_distance': 'bool',
            'use_bond': 'bool',
            'public_dis_filter': 'bool',
            'public_bond_filter': 'bool',
            'use_graph_norm': 'bool',
            'length_unit': 'str',
        }

    def set_hyper_param(self):
        """set hyperparameters"""
        set_hyper_parameter(self.hyper_param, 'name', self.cls_name)
        set_class_into_hyper_param(self.hyper_param, self.hyper_types, self)
        return self

    def get_length(self, length, unit=None):
        """get length value according to unit"""
        if isinstance(length, Length):
            if unit is None:
                unit = self.units
            return Tensor(length(unit), ms.float32)
        return Tensor(length, ms.float32)

    def broadcast_to_interactions(self, value, name: str):
        """return the broad cast value as the interaction layers"""
        tensor = Tensor(value)
        size = tensor.size
        if self.coupled_interaction:
            if size > 1:
                raise ValueError(
                    'The size of "'+name+'" must be 1 when "coupled_interaction" is "True"')
        else:
            if size  not in (self.n_interaction, 1):
                raise ValueError('"The size of "'+name+'" ('+str(size) +
                                 ') must be equal to "n_interaction" ('+str(self.n_interaction)+')!')
            tensor = msnp.broadcast_to(tensor, (self.n_interaction,))
        return tensor

    def print_info(self, num_retraction: int = 3, num_gap: int = 3, char: str = '-'):
        """print the information of molecular model"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+' Deep molecular model: ', self.network_name)
        print('-'*80)
        print(ret+gap+' Length unit: ' + self.units.length_unit_name)
        print(ret+gap+' Atom embedding size: ' + str(self.num_atom_types))
        print(ret+gap+' Cutoff distance: ' +
              str(self.cutoff) + ' ' + self.length_unit)
        print(ret+gap+' Radical basis function (RBF): ' + str(self.rbf.cls_name))
        self.rbf.print_info(num_retraction=num_retraction +
                            num_gap, num_gap=num_gap, char=char)
        print(ret+gap+' Calculate distance: ' +
              ('Yes' if self.calc_distance else 'No'))
        print(ret+gap+' Calculate bond: ' +
              ('Yes' if self.calc_bond else 'No'))
        print(ret+gap+' Feature dimension: ' + str(self.dim_feature))
        print('-'*80)
        if self.coupled_interaction:
            print(ret+gap+' Using coupled interaction with ' +
                  str(self.n_interaction)+' layers:')
            print('-'*80)
            print(ret+gap+gap+' '+self.interactions[0].name)
            self.interactions[0].print_info(
                num_retraction=num_retraction+num_gap, num_gap=num_gap, char=char)
        else:
            print(ret+gap+' Using '+str(self.n_interaction) +
                  ' independent interaction layers:')
            print('-'*80)
            for i, inter in enumerate(self.interactions):
                print(ret+gap+' '+str(i)+'. '+inter.name)
                inter.print_info(num_retraction=num_retraction +
                                 num_gap, num_gap=num_gap, char=char)

    def _get_self_interaction(self, atom_mask):
        """get the distance of atomic self-interaction"""
        # (B,A,1)
        r_ii = msnp.full_like(atom_mask, self.r_self)
        r_ii = msnp.where(atom_mask, r_ii, 5e4)
        c_ii = F.ones_like(r_ii) * atom_mask
        return r_ii, c_ii

    def _calc_cutoffs(self, r_ij=1, neighbour_mask=None, atom_mask=None, bond_mask=None):
        """calculate cutoff distance"""
        if self.calc_distance:
            if self.cutoff_fn is None:
                return F.ones_like(r_ij), neighbour_mask
            return self.cutoff_fn(r_ij, neighbour_mask)

        mask = None
        if bond_mask is not None:
            mask = self.concat((atom_mask, bond_mask))
        return F.cast(mask > 0, ms.float32), mask

    def _get_self_cutoff(self, atom_mask):
        """get the cutoff distance of atom itself"""
        return F.cast(atom_mask, ms.float32)

    def _get_rbf(self, dis):
        """get radical basis function"""
        if self.rbf is None:
            rbf = F.expand_dims(dis, -1)
        else:
            rbf = self.rbf(dis)

        return rbf

    def construct(self,
                  r_ij: Tensor = 1,
                  atom_types: Tensor = None,
                  atom_mask: Tensor = None,
                  neighbours: Tensor = None,
                  neighbour_mask: Tensor = None,
                  bonds: Tensor = None,
                  bond_mask: Tensor = None,
                  ):
        """Compute the representation of atoms.

        Args:
            r_ij (Tensor):              Tensor of shape (B, A, N). Data type is float
                                        Distances between atoms.
            atom_types (Tensor):        Tensor of shape (B, A). Data type is int
                                        Atomic number.
            atom_mask (Tensor):         Tensor of shape (B, A). Data type is bool
                                        Mask of atomic number
            neighbours (Tensor):        Tensor of shape (B, A, N). Data type is int
                                        Neighbour index.
            neighbour_mask (Tensor):    Tensor of shape (B, A, N). Data type is bool
                                        Nask of neighbour index.

        Returns:
            representation: (Tensor)    Tensor of shape (B, A, F). Data type is float

        Symbols:

            B:  Batch size.
            A:  Number of atoms in system.
            N:  Number of neighbour atoms.
            D:  Dimension of position coordinates, usually is 3.
            F:  Feature dimension of representation.

        """

        bsize = r_ij.shape[0] if self.calc_distance else bonds.shape[0]

        # (B,A) -> (B,A,1)
        atom_mask = F.expand_dims(atom_mask, -1)

        e = self.atom_embedding(atom_types)
        if atom_types.shape[0] != bsize:
            e = msnp.broadcast_to(e, (bsize,)+e.shape[1:])
            atom_mask = msnp.broadcast_to(
                atom_mask, (bsize,)+atom_mask.shape[1:])

        if self.calc_distance:
            nbatch = r_ij.shape[0]
            natoms = r_ij.shape[1]

            f_ij = self._get_rbf(r_ij)
            f_ii = 0 if self.r_self is None else self._get_rbf(self.r_self_ex)
        else:
            f_ii = 1
            f_ij = 1
            nbatch = bonds.shape[0]
            natoms = bonds.shape[1]

        if self.calc_bond:
            b_ii = self.zeros((nbatch, natoms), ms.int32)
            b_ii = self.bond_embedding(b_ii)

            b_ij = self.bond_embedding(bonds)

            if bond_mask is not None:
                b_ij = b_ij * F.expand_dims(bond_mask, -1)
        else:
            b_ii = 0
            b_ij = 0

        # apply cutoff
        c_ij, mask = self._calc_cutoffs(
            r_ij, neighbour_mask, atom_mask, bond_mask)
        c_ii = None if self.r_self is None else self._get_self_cutoff(
            atom_mask)

        # continuous-filter convolution interaction block followed by Dense layer
        x = e
        n_interaction = len(self.interactions)
        xlist = []
        for i in range(n_interaction):
            x = self.interactions[i](
                x, f_ij, b_ij, c_ij, neighbours, mask, e, f_ii, b_ii, c_ii, atom_mask)
            if self.use_graph_norm:
                x = self.graph_norm[i](x)
            xlist.append(x)
        return x, xlist


class SchNet(MolecularModel):
    r"""SchNet Model.

    Reference:

        Schütt, K. T.; Sauceda, H. E.; Kindermans, P.-J.; Tkatchenko, A.; Müller, K.-R.
        Schnet - a Deep Learning Architecture for Molecules and Materials [J].
        The Journal of Chemical Physics, 2018, 148(24): 241722.

    Args:

        dim_feature (int):          Dimension of atomic representation. Default: 64

        dim_filter (int):           Dimension of filter network. Default: 64

        n_interaction (int):        Number of interaction layers. Default: 3

        activation (Cell):          Activation function. Default: 'ssp'

        cutoff (Length):            Cutoff distance. Default: Length(1, 'nm')

        cutoff_fn (Cell):           Cutoff function. Default: 'cosine'

        rbf (Cell):                 Radical baiss function. Default: 'gaussian'

        normalize_filter (bool):    Whether to normalize the filter network. Default: False

        coupled_interaction (bool): Whether to use coupled (shared) interaction layer. Default: False

        use_graph_norm (bool):      Whether to use graph normalization. Default: False

        public_dis_filter (bool):   Whether to use public (shared) filter for distance. Default: False

        num_atom_types (int):       Maximum number of atomic types. Default: 64

        length_unit (bool):         Unit of position coordinates. Default: 'nm'

        hyper_param (dict):         Hyperparameter for molecular model. Default: None

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        N:  Number of neighbour atoms.

        D:  Dimension of position coordinates, usually is 3.

        K:  Number of basis functions in RBF.

        F:  Feature dimension of representation.

    """

    def __init__(self,
                 dim_feature: int = 64,
                 dim_filter: int = 64,
                 n_interaction: int = 3,
                 activation: Cell = 'ssp',
                 cutoff: float = Length(1, 'nm'),
                 cutoff_fn: Cell = 'cosine',
                 rbf: Cell = 'gaussian',
                 normalize_filter: bool = False,
                 coupled_interaction: bool = False,
                 use_graph_norm: bool = False,
                 public_dis_filter: bool = False,
                 num_atom_types: int = 64,
                 length_unit: str = 'nm',
                 hyper_param: dict = None,
                 ):

        super().__init__(
            dim_feature=dim_feature,
            n_interaction=n_interaction,
            activation=activation,
            cutoff=cutoff,
            cutoff_fn=cutoff_fn,
            rbf=rbf,
            r_self=None,
            coupled_interaction=coupled_interaction,
            use_distance=True,
            use_bond=False,
            use_graph_norm=use_graph_norm,
            public_dis_filter=public_dis_filter,
            num_atom_types=num_atom_types,
            length_unit=length_unit,
            hyper_param=hyper_param,
        )
        self.reg_key = 'schnet'
        self.network_name = 'SchNet'

        if hyper_param is not None:
            dim_filter = get_hyper_parameter(hyper_param, 'dim_filter')
            normalize_filter = get_hyper_parameter(
                hyper_param, 'normalize_filter')

        if self.calc_bond:
            raise ValueError('SchNet cannot supported bond information!')

        self.dim_filter = self.broadcast_to_interactions(
            dim_filter, 'dim_filter')
        self.normalize_filter = self.broadcast_to_interactions(
            normalize_filter, 'normalize_filter')

        self.set_hyper_param()

        self.filter = None
        if self.public_dis_filter and (not self.coupled_interaction):
            self.filter = DenseFilter(
                self.num_basis, self.dim_filter, self.activation)
        # block for computing interaction
        if self.coupled_interaction:
            # use the same SchNetInteraction instance (hence the same weights)
            self.interactions = nn.CellList(
                [
                    SchNetInteraction(
                        dim_feature=self.dim_feature,
                        dim_filter=self.dim_filter,
                        activation=self.activation,
                        dis_filter=DenseFilter(
                            self.num_basis, self.dim_filter, self.activation),
                        normalize_filter=self.normalize_filter,
                    )
                ]
                * self.n_interaction
            )
        else:
            # use one SchNetInteraction instance for each interaction
            self.interactions = nn.CellList(
                [
                    SchNetInteraction(
                        dim_feature=self.dim_feature,
                        dim_filter=self.dim_filter[i],
                        activation=self.activation,
                        dis_filter=self.filter if self.public_dis_filter
                        else DenseFilter(self.num_basis, self.dim_filter[i], self.activation),
                        normalize_filter=self.normalize_filter[i],
                    )
                    for i in range(self.n_interaction)
                ]
            )

    def set_hyper_param(self):
        """set hyperparameters"""
        super().set_hyper_param()
        set_hyper_parameter(self.hyper_param, 'dim_filter', self.dim_filter)
        set_hyper_parameter(
            self.hyper_param, 'normalize_filter', self.normalize_filter)
        return self


class PhysNet(MolecularModel):
    r"""PhysNet Model

    Reference:

        Unke, O. T. and Meuwly, M.,
        PhysNet: A neural network for predicting energyies, forces, dipole moments, and partial charges [J].
        The Journal of Chemical Theory and Computation, 2019, 15(6): 3678-3693.

    Args:

        dim_feature (int):          Dimension of atomic representation. Default: 128

        n_interaction (int):        Number of interaction layers. Default: 5

        activation (Cell):          Activation function. Default: 'ssp'

        cutoff (Length):            Cutoff distance. Default: Length(1, 'nm')

        cutoff_fn (Cell):           Cutoff function. Default: 'smooth'

        rbf (Cell):                 Radical baiss function. Default: 'log_gaussian'

        coupled_interaction (bool): Whether to use coupled (shared) interaction layer. Default: False

        use_graph_norm (bool):      Whether to use graph normalization. Default: False

        public_dis_filter (bool):   Whether to use public (shared) filter for distance. Default: False

        num_atom_types (int):       Maximum number of atomic types. Default: 64

        n_inter_residual (int):     Number of blocks in the inside pre-activation residual block. Default: 3

        n_outer_residual (int):     Number of blocks in the outside pre-activation residual block. Default: 2

        length_unit (bool):         Unit of position coordinates. Default: 'nm'

        hyper_param (dict):         Hyperparameter for molecular model. Default: None

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        N:  Number of neighbour atoms.

        D:  Dimension of position coordinates, usually is 3.

        K:  Number of basis functions in RBF.

        F:  Feature dimension of representation.

    """

    def __init__(self,
                 dim_feature: int = 128,
                 n_interaction: int = 5,
                 activation: Cell = 'ssp',
                 cutoff: float = Length(1, 'nm'),
                 cutoff_fn: Cell = 'smooth',
                 rbf: Cell = 'log_gaussian',
                 coupled_interaction: bool = False,
                 use_graph_norm: bool = False,
                 public_dis_filter: bool = False,
                 num_atom_types: int = 64,
                 n_inter_residual: int = 3,
                 n_outer_residual: int = 2,
                 length_unit: str = 'nm',
                 hyper_param: dict = None,
                 ):

        super().__init__(
            dim_feature=dim_feature,
            n_interaction=n_interaction,
            activation=activation,
            cutoff=cutoff,
            cutoff_fn=cutoff_fn,
            rbf=rbf,
            r_self=None,
            coupled_interaction=coupled_interaction,
            use_distance=True,
            use_bond=False,
            use_graph_norm=use_graph_norm,
            public_dis_filter=public_dis_filter,
            num_atom_types=num_atom_types,
            length_unit=length_unit,
            hyper_param=hyper_param,
        )

        self.reg_key = 'physnet'
        self.network_name = 'PhysNet'

        if hyper_param is not None:
            n_inter_residual = get_hyper_parameter(
                hyper_param, 'n_inter_residual')
            n_outer_residual = get_hyper_parameter(
                hyper_param, 'n_outer_residual')

        self.n_inter_residual = get_integer(n_inter_residual)
        self.n_outer_residual = get_integer(n_outer_residual)

        self.set_hyper_param()

        self.filter = None
        if self.public_dis_filter and (not self.coupled_interaction):
            self.filter = Dense(self.num_basis, self.dim_feature,
                                has_bias=False, activation=None)

        # block for computing interaction
        if self.coupled_interaction:
            self.interaction_typenames = ['D0',] * self.n_interaction
            # use the same SchNetInteraction instance (hence the same weights)
            self.interactions = nn.CellList(
                [
                    PhysNetModule(
                        dis_filter=Dense(
                            self.num_basis, self.dim_feature, has_bias=False, activation=None),
                        dim_feature=self.dim_feature,
                        activation=self.activation,
                        n_inter_residual=self.n_inter_residual,
                        n_outer_residual=self.n_outer_residual,
                    )
                ]
                * self.n_interaction
            )
        else:
            self.interaction_typenames = [
                'D' + str(i) for i in range(self.n_interaction)]
            # use one SchNetInteraction instance for each interaction
            self.interactions = nn.CellList(
                [
                    PhysNetModule(
                        dis_filter=self.filter if self.public_dis_filter
                        else Dense(self.num_basis, self.dim_feature, has_bias=False, activation=None),
                        dim_feature=self.dim_feature,
                        activation=self.activation,
                        n_inter_residual=self.n_inter_residual,
                        n_outer_residual=self.n_outer_residual,
                    )
                    for _ in range(self.n_interaction)
                ]
            )

        self.readout = None

    def set_hyper_param(self):
        """set hyperparameters"""
        super().set_hyper_param()
        set_hyper_parameter(
            self.hyper_param, 'n_inter_residual', self.n_inter_residual)
        set_hyper_parameter(
            self.hyper_param, 'n_outer_residual', self.n_outer_residual)
        return self


class MolCT(MolecularModel):
    r"""Molecular Configuration Transformer (MolCT) Model

    Reference:

        Zhang, J.; Zhou, Y.; Lei, Y.-K.; Yang, Y. I.; Gao, Y. Q.,
        Molecular CT: unifying geometry and representation learning for molecules at different scales [J/OL].
        arXiv preprint, 2020: arXiv:2012.11816 [2020-12-22]. https://arxiv.org/abs/2012.11816

    Args:

        dim_feature (int):          Dimension of atomic representation. Default: 128

        n_interaction (int):        Number of interaction layers. Default: 3

        n_heads (int):              Number of heads in multi-head attention. Default: 8

        max_cycles (int):           Maximum number of cycles of the adapative computation time (ACT).
                                    Default: 10

        activation (Cell):          Activation function. Default: 'swish'

        cutoff (Length):            Cutoff distance. Default: Length(1, 'nm')

        cutoff_fn (Cell):           Cutoff function. Default: 'smooth'

        rbf (Cell):                 Radical baiss function. Default: 'log_gaussian'

        r_self (Length)             Distance of atomic self-interaction. Default: Length(0.05, 'nm')

        coupled_interaction (bool): Whether to use coupled (shared) interaction layer. Default: False

        use_distance (bool):        Whether to use distance between atoms. Default: True

        use_bond (bool):            Whether to use bond information. Default: False

        public_dis_filter (bool):   Whether to use public (shared) filter for distance. Default: False

        public_bond_filter (bool):  Whether to use public (shared) filter for bond. Default: False

        num_atom_types (int):       Maximum number of atomic types. Default: 64

        num_bond_types (int):       Maximum number of bond types. Default: 16

        act_threshold (float):      Threshold of adapative computation time. Default: 0.9

        fixed_cycles (bool):        Whether to use the fixed cycle number to do ACT. Default: False

        use_feed_forward (bool):    Whether to use feed forward after multi-head attention. Default: False

        length_unit (bool):         Unit of position coordinates. Default: 'nm'

        hyper_param (dict):         Hyperparameter for molecular model. Default: None

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        N:  Number of neighbour atoms.

        D:  Dimension of position coordinates, usually is 3.

        K:  Number of basis functions in RBF.

        F:  Feature dimension of representation.


    """

    def __init__(self,
                 dim_feature: int = 128,
                 n_interaction: int = 3,
                 n_heads: int = 8,
                 max_cycles: int = 10,
                 activation: Cell = 'swish',
                 cutoff: Length = Length(1, 'nm'),
                 cutoff_fn: Cell = 'smooth',
                 rbf: Cell = 'log_gaussian',
                 r_self: Length = Length(0.05, 'nm'),
                 coupled_interaction: bool = False,
                 use_distance: bool = True,
                 use_bond: bool = False,
                 public_dis_filter: bool = True,
                 public_bond_filter: bool = True,
                 num_atom_types: int = 64,
                 num_bond_types: int = 16,
                 act_threshold: float = 0.9,
                 fixed_cycles: bool = False,
                 use_feed_forward: bool = False,
                 length_unit: str = 'nm',
                 hyper_param: dict = None,
                 ):

        super().__init__(
            dim_feature=dim_feature,
            n_interaction=n_interaction,
            activation=activation,
            cutoff=cutoff,
            cutoff_fn=cutoff_fn,
            rbf=rbf,
            r_self=r_self,
            coupled_interaction=coupled_interaction,
            use_distance=use_distance,
            use_bond=use_bond,
            public_dis_filter=public_dis_filter,
            public_bond_filter=public_bond_filter,
            use_graph_norm=False,
            num_atom_types=num_atom_types,
            num_bond_types=num_bond_types,
            length_unit=length_unit,
            hyper_param=hyper_param,
        )

        self.reg_key = 'molct'
        self.network_name = 'MolCT'

        if hyper_param is not None:
            n_heads = get_hyper_parameter(hyper_param, 'n_heads')
            max_cycles = get_hyper_parameter(hyper_param, 'max_cycles')
            use_feed_forward = get_hyper_parameter(
                hyper_param, 'use_feed_forward')
            fixed_cycles = get_hyper_parameter(hyper_param, 'fixed_cycles')
            act_threshold = get_hyper_parameter(hyper_param, 'act_threshold')

        if self.r_self is None:
            raise ValueError('"r_self" cannot be "None" at MolCT.')
        self.self_dis_tensor = F.expand_dims(self.r_self, 0)

        self.n_heads = self.broadcast_to_interactions(n_heads, 'n_heads')
        self.max_cycles = self.broadcast_to_interactions(
            max_cycles, 'max_cycles')
        self.use_feed_forward = self.broadcast_to_interactions(
            use_feed_forward, 'use_feed_forward')
        self.fixed_cycles = self.broadcast_to_interactions(
            fixed_cycles, 'fixed_cycles')
        self.act_threshold = self.broadcast_to_interactions(
            act_threshold, 'act_threshold')

        self.set_hyper_param()

        self.dis_filter = None
        if self.calc_distance and self.public_dis_filter and (not self.coupled_interaction):
            self.dis_filter = ResFilter(
                self.num_basis, self.dim_feature, self.activation)

        self.bond_embedding = None
        self.bond_filter = None
        if self.calc_bond:
            if self.calc_bond and self.public_bond_filter and (not self.coupled_interaction):
                self.bond_filter = Residual(
                    self.dim_feature, activation=self.activation)

        if self.coupled_interaction:
            self.interactions = nn.CellList(
                [
                    NeuralInteractionUnit(
                        dim_feature=self.dim_feature,
                        n_heads=self.n_heads,
                        max_cycles=self.max_cycles,
                        activation=self.activation,
                        dis_filter=(ResFilter(self.num_basis, self.dim_feature,
                                              self.activation) if self.use_distance else None),
                        bond_filter=(Residual(
                            self.dim_feature, activation=self.activation) if self.use_bond else None),
                        use_feed_forward=self.use_feed_forward,
                        fixed_cycles=self.fixed_cycles,
                        act_threshold=self.act_threshold,
                    )
                ]
                * self.n_interaction
            )
        else:
            interaction_list = []
            for i in range(self.n_interaction):
                dis_filter = None
                if self.use_distance[i]:
                    if self.public_dis_filter:
                        dis_filter = self.dis_filter
                    else:
                        dis_filter = ResFilter(
                            self.num_basis, self.dim_feature, self.activation)
                bond_filter = None
                if self.use_bond[i]:
                    if self.public_bond_filter:
                        bond_filter = self.bond_filter
                    else:
                        bond_filter = Residual(
                            self.dim_feature, activation=self.activation)

                interaction_list.append(
                    NeuralInteractionUnit(
                        dim_feature=self.dim_feature,
                        n_heads=self.n_heads[i],
                        max_cycles=self.max_cycles[i],
                        activation=self.activation,
                        dis_filter=dis_filter,
                        bond_filter=bond_filter,
                        use_feed_forward=self.use_feed_forward[i],
                        fixed_cycles=self.fixed_cycles[i],
                        act_threshold=self.act_threshold[i],
                    )
                )
            self.interactions = nn.CellList(interaction_list)

    def set_hyper_param(self):
        """set hyperparameters"""
        super().set_hyper_param()
        set_hyper_parameter(self.hyper_param, 'n_heads', self.n_heads)
        set_hyper_parameter(self.hyper_param, 'max_cycles', self.max_cycles)
        set_hyper_parameter(
            self.hyper_param, 'use_feed_forward', self.use_feed_forward)
        set_hyper_parameter(
            self.hyper_param, 'fixed_cycles', self.fixed_cycles)
        set_hyper_parameter(
            self.hyper_param, 'act_threshold', self.act_threshold)
        return self


_MOLECULAR_MODEL_BY_KEY = {
    'molct': MolCT,
    'schnet': SchNet,
    'physnet': PhysNet
}

_MOLECULAR_MODEL_BY_NAME = {
    model.__name__: model for model in _MOLECULAR_MODEL_BY_KEY.values()}


def get_molecular_model(model, length_unit=None) -> MolecularModel:
    """get molecular model"""
    if isinstance(model, MolecularModel):
        return model
    if model is None:
        return None

    hyper_param = None
    if isinstance(model, dict):
        if 'name' not in model.keys():
            raise KeyError('Cannot find the key "name" in model dict!')
        hyper_param = model
        model = get_hyper_string(hyper_param, 'name')

    if isinstance(model, str):
        if model.lower() == 'none':
            return None
        if model.lower() in _MOLECULAR_MODEL_BY_KEY.keys():
            return _MOLECULAR_MODEL_BY_KEY[model.lower()](
                length_unit=length_unit,
                hyper_param=hyper_param,
            )
        if model in _MOLECULAR_MODEL_BY_NAME.keys():
            return _MOLECULAR_MODEL_BY_NAME[model](
                length_unit=length_unit,
                hyper_param=hyper_param,
            )
        raise ValueError("The MolecularModel corresponding to '{}' was not found.".format(model))
    raise TypeError("Unsupported MolecularModel type '{}'.".format(type(model)))
