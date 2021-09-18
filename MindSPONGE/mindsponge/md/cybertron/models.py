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
"""models"""

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.initializer import Normal

from .units import units
from .blocks import Dense, Residual
from .interactions import SchNetInteraction
from .interactions import PhysNetModule
from .interactions import NeuralInteractionUnit
from .base import ResFilter, GraphNorm
from .cutoff import get_cutoff
from .rbf import GaussianSmearing, LogGaussianDistribution
from .activations import ShiftedSoftplus, Swish

__all__ = [
    "DeepGraphMolecularModel",
    "SchNet",
    "PhysNet",
    "MolCT",
]


class DeepGraphMolecularModel(nn.Cell):
    r"""Basic class for graph neural network (GNN) based deep molecular model

    Args:
        num_elements (int): maximum number of atomic types
        num_rbf (int): number of the serial of radical basis functions (RBF)
        dim_feature (int): dimension of the vectors for atomic embedding
        atom_types (ms.Tensor[int], optional): atomic index
        rbf_function(nn.Cell, optional): the algorithm to calculate RBF
        cutoff_network (nn.Cell, optional): the algorithm to calculate cutoff.

    """

    def __init__(
            self,
            num_elements,
            min_rbf_dis,
            max_rbf_dis,
            num_rbf,
            dim_feature,
            n_interactions,
            interactions=None,
            unit_length='nm',
            activation=None,
            rbf_sigma=None,
            trainable_rbf=False,
            rbf_function=None,
            cutoff=None,
            cutoff_network=None,
            rescale_rbf=False,
            use_distances=True,
            use_public_filter=False,
            use_graph_norm=False,
            dis_filter=None
    ):
        super().__init__()
        self.num_elements = num_elements
        self.dim_feature = dim_feature
        self.num_rbf = num_rbf
        self.rbf_function = rbf_function
        self.rescale_rbf = rescale_rbf

        self.activation = activation

        self.interaction_types = interactions
        if isinstance(interactions, list):
            self.n_interactions = len(interactions)
        else:
            self.n_interactions = n_interactions

        self.unit_length = unit_length
        units.set_length_unit(self.unit_length)

        self.use_distances = use_distances
        self.use_bonds = False

        self.network_name = 'DeepGraphMolecularModel'

        self.read_all_interactions = False

        # make a lookup table to store embeddings for each element (up to atomic
        # number max_z) each of which is a vector of size dim_feature
        self.atom_embedding = nn.Embedding(
            num_elements,
            dim_feature,
            use_one_hot=True,
            embedding_table=Normal(1.0))

        self.bond_embedding = [None,]
        self.bond_filter = [None,]

        self.use_public_filter = use_public_filter

        self.dis_filter = dis_filter

        self.fixed_atoms = False

        # layer for expanding interatomic distances in a basis
        if rbf_function is not None:
            self.rbf_function = rbf_function(
                d_min=min_rbf_dis,
                d_max=max_rbf_dis,
                num_rbf=num_rbf,
                sigma=rbf_sigma,
                trainable=trainable_rbf)
        else:
            self.rbf_function = None

        self.cutoff_network = None
        self.cutoff = None
        if cutoff_network is not None:
            if cutoff is None:
                self.cutoff = max_rbf_dis
            else:
                self.cutoff = cutoff
            self.cutoff_network = get_cutoff(
                cutoff_network,
                r_max=self.cutoff,
                return_mask=True,
                reverse=False)

        self.interactions = [None,]

        self.interaction_typenames = []

        self.use_graph_norm = use_graph_norm
        self.use_pub_norm = False

        if self.use_graph_norm:
            if self.use_pub_norm:
                self.graph_norm = nn.CellList(
                    [GraphNorm(dim_feature) * self.n_interactions]
                )
            else:
                self.graph_norm = nn.CellList(
                    [GraphNorm(dim_feature) for _ in range(self.n_interactions)]
                )
        else:
            self.graph_norm = None

        self.decoder = 'halve'
        self.merge_method = None
        self.far_type = None

        self.zeros = P.Zeros()
        self.ones = P.Ones()

    def print_info(self):
        """print info"""
        print('---with GNN-based deep molecular model: ', self.network_name)
        print('------with atom embedding size: ' + str(self.num_elements))
        print('------with cutoff distance: ' +
              str(self.cutoff) + ' ' + self.unit_length)
        print('------with number of RBF functions: ' + str(self.num_rbf))
        print('------with bond connection: ' +
              ('Yes' if self.use_bonds else 'No'))
        print('------with feature dimension: ' + str(self.dim_feature))
        print('------with interaction layers:')
        for i, inter in enumerate(self.interactions):
            print('------' + str(i + 1) + '. ' + inter.name +
                  '(' + self.interaction_typenames[i] + ')')
            inter.print_info()
        print('------with total layers: ' + str(len(self.interactions)))
        print('------output all interaction layers: ' +
              ('Yes'if self.read_all_interactions else 'No'))

    def set_fixed_atoms(self, fixed_atoms=True):
        self.fixed_atoms = fixed_atoms

    def set_fixed_neighbors(self, flag=True):
        for interaction in self.interactions:
            interaction.set_fixed_neighbors(flag)

    def _calc_cutoffs(
            self,
            r_ij=1,
            neighbor_mask=None,
            bonds=None,
            bond_mask=None,
            atom_mask=None):
        """_calc_cutoffs"""

        self.bonds_t = bonds
        self.bond_mask_t = bond_mask
        self.atom_mask_t = atom_mask

        if self.cutoff_network is None:
            return F.ones_like(r_ij), neighbor_mask
        return self.cutoff_network(r_ij, neighbor_mask)

    def _get_rbf(self, dis):
        """_get_rbf"""
        # expand interatomic distances (for example, Gaussian smearing)
        if self.rbf_function is None:
            rbf = F.expand_dims(dis, -1)
        else:
            rbf = self.rbf_function(dis)

        if self.rescale_rbf:
            rbf = rbf * 2.0 - 1.0

        if self.dis_filter is not None:
            return self.dis_filter(rbf)
        return rbf

    def _get_self_rbf(self):
        return 0

    def construct(
            self,
            r_ij=1,
            atom_types=None,
            atom_mask=None,
            neighbors=None,
            neighbor_mask=None,
            bonds=None,
            bond_mask=None):
        """Compute interaction output.

        Args:
            r_ij (ms.Tensor[float], [B, A, N]): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (ms.Tensor[int]): indices of neighbors of (N_b, N_a, N_nbh) shape.
            neighbor_mask (ms.Tensor[bool], optional): mask to filter out non-existing neighbors
                introduced via padding.
            atom_types (ms.Tensor[int], optional): atomic index

        Returns:
            torch.Tensor: block output with (N_b, N_a, N_basis) shape.

        """

        if self.fixed_atoms:
            exones = self.ones((r_ij.shape[0], 1, 1), r_ij.dtype)
            e = exones * self.atom_embedding(atom_types)
            if atom_mask is not None:
                atom_mask = (exones * atom_mask) > 0
        else:
            e = self.atom_embedding(atom_types)

        if self.use_distances:
            f_ij = self._get_rbf(r_ij)
            f_ii = self._get_self_rbf()
        else:
            f_ii = 1
            f_ij = 1

        b_ii = 0
        b_ij = 0

        # apply cutoff
        c_ij, mask = self._calc_cutoffs(
            r_ij, neighbor_mask, bonds, bond_mask, atom_mask)

        # continuous-filter convolution interaction block followed by Dense
        # layer
        x = e
        n_interactions = len(self.interactions)
        xlist = []
        for i in range(n_interactions):
            x = self.interactions[i](
                x, e, f_ii, f_ij, b_ii, b_ij, c_ij, neighbors, mask)

            if self.use_graph_norm:
                x = self.graph_norm[i](x)

            if self.read_all_interactions:
                xlist.append(x)

        if self.read_all_interactions:
            return x, xlist
        return x, None


class SchNet(DeepGraphMolecularModel):
    r"""SchNet Model.

    References:
        Schütt, K. T.; Sauceda, H. E.; Kindermans, P.-J.; Tkatchenko, A.; Müller K.-R.,
        SchNet - a deep learning architecture for molceules and materials.
        The Journal of Chemical Physics 148 (24), 241722. 2018.

    Args:

        num_elements (int): maximum number of atomic types
        num_rbf (int): number of the serial of radical basis functions (RBF)
        dim_feature (int): dimension of the vectors for atomic embedding
        dim_filter (int): dimension of the vectors for filters used in continuous-filter convolution.
        n_interactions (int, optional): number of interaction blocks.
        max_distance (float): the maximum distance to calculate RBF.
        atom_types (ms.Tensor[int], optional): atomic index
        rbf_function(nn.Cell, optional): the algorithm to calculate RBF
        cutoff_network (nn.Cell, optional): the algorithm to calculate cutoff.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.
        coupled_interactions (bool, optional): if True, share the weights across
            interaction blocks and filter-generating networks.
        trainable_gaussians (bool, optional): If True, widths and offset of Gaussian
            functions are adjusted during training process.

    """

    def __init__(
            self,
            num_elements=100,
            dim_feature=64,
            min_rbf_dis=0.02,
            max_rbf_dis=0.5,
            num_rbf=32,
            dim_filter=64,
            n_interactions=3,
            activation=ShiftedSoftplus(),
            unit_length='nm',
            rbf_sigma=None,
            rbf_function=GaussianSmearing,
            cutoff=None,
            cutoff_network='cosine',
            normalize_filter=False,
            coupled_interactions=False,
            trainable_rbf=False,
            use_graph_norm=False,
        ):
        super().__init__(
            num_elements=num_elements,
            dim_feature=dim_feature,
            min_rbf_dis=min_rbf_dis,
            max_rbf_dis=max_rbf_dis,
            num_rbf=num_rbf,
            n_interactions=n_interactions,
            activation=activation,
            unit_length=unit_length,
            rbf_sigma=rbf_sigma,
            rbf_function=rbf_function,
            cutoff=cutoff,
            cutoff_network=cutoff_network,
            rescale_rbf=False,
            use_public_filter=False,
            use_graph_norm=use_graph_norm,
            trainable_rbf=trainable_rbf,
        )
        self.network_name = 'SchNet'

        # block for computing interaction
        if coupled_interactions:
            self.interaction_typenames = ['D0',] * self.n_interactions
            # use the same SchNetInteraction instance (hence the same weights)
            self.interactions = nn.CellList(
                [
                    SchNetInteraction(
                        dim_feature=dim_feature,
                        num_rbf=num_rbf,
                        dim_filter=dim_filter,
                        activation=self.activation,
                        normalize_filter=normalize_filter,
                    )
                ]
                * self.n_interactions
            )
        else:
            self.interaction_typenames = [
                'D' + str(i) for i in range(self.n_interactions)]
            # use one SchNetInteraction instance for each interaction
            self.interactions = nn.CellList(
                [
                    SchNetInteraction(
                        dim_feature=dim_feature,
                        num_rbf=num_rbf,
                        dim_filter=dim_filter,
                        activation=self.activation,
                        normalize_filter=normalize_filter,
                    )
                    for _ in range(self.n_interactions)
                ]
            )


class PhysNet(DeepGraphMolecularModel):
    r"""PhysNet Model

    References:
        Unke, O. T. and Meuwly, M.,
        PhysNet: A neural network for predicting energyies, forces, dipole moments, and partial charges.
        The Journal of Chemical Theory and Computation 2019, 15(6), 3678-3693.

    Args:

        num_elements (int): maximum number of atomic types
        num_rbf (int): number of the serial of radical basis functions (RBF)
        dim_feature (int): dimension of the vectors for atomic embedding
        dim_filter (int): dimension of the vectors for filters used in continuous-filter convolution.
        n_interactions (int, optional): number of interaction blocks.
        max_distance (float): the maximum distance to calculate RBF.
        atom_types (ms.Tensor[int], optional): atomic index
        rbf_function(nn.Cell, optional): the algorithm to calculate RBF
        cutoff_network (nn.Cell, optional): the algorithm to calculate cutoff.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.
        coupled_interactions (bool, optional): if True, share the weights across
            interaction blocks and filter-generating networks.
        trainable_gaussians (bool, optional): If True, widths and offset of Gaussian
            functions are adjusted during training process.

    """

    def __init__(
            self,
            num_elements=100,
            min_rbf_dis=0.02,
            max_rbf_dis=1,
            num_rbf=64,
            dim_feature=128,
            n_interactions=5,
            n_inter_residual=3,
            n_outer_residual=2,
            unit_length='nm',
            activation=ShiftedSoftplus(),
            rbf_sigma=None,
            rbf_function=GaussianSmearing,
            cutoff=None,
            cutoff_network='smooth',
            use_graph_norm=False,
            coupled_interactions=False,
            trainable_rbf=False,
        ):
        super().__init__(
            num_elements=num_elements,
            dim_feature=dim_feature,
            min_rbf_dis=min_rbf_dis,
            max_rbf_dis=max_rbf_dis,
            num_rbf=num_rbf,
            n_interactions=n_interactions,
            activation=activation,
            rbf_sigma=rbf_sigma,
            unit_length=unit_length,
            rbf_function=rbf_function,
            cutoff=cutoff,
            cutoff_network=cutoff_network,
            rescale_rbf=False,
            use_graph_norm=use_graph_norm,
            use_public_filter=False,
            trainable_rbf=trainable_rbf,
        )
        self.network_name = 'PhysNet'

        # block for computing interaction
        if coupled_interactions:
            self.interaction_typenames = ['D0',] * self.n_interactions
            # use the same SchNetInteraction instance (hence the same weights)
            self.interactions = nn.CellList(
                [
                    PhysNetModule(
                        num_rbf=num_rbf,
                        dim_feature=dim_feature,
                        activation=self.activation,
                        n_inter_residual=n_inter_residual,
                        n_outer_residual=n_outer_residual,
                    )
                ]
                * self.n_interactions
            )
        else:
            self.interaction_typenames = [
                'D' + str(i) for i in range(self.n_interactions)]
            # use one SchNetInteraction instance for each interaction
            self.interactions = nn.CellList(
                [
                    PhysNetModule(
                        num_rbf=num_rbf,
                        dim_feature=dim_feature,
                        activation=self.activation,
                        n_inter_residual=n_inter_residual,
                        n_outer_residual=n_outer_residual,
                    )
                    for _ in range(self.n_interactions)
                ]
            )

        self.readout = None

    def set_fixed_neighbors(self, flag=True):
        for interaction in self.interactions:
            interaction.set_fixed_neighbors(flag)


class MolCT(DeepGraphMolecularModel):
    r"""Molecular Configuration Transformer (MolCT) Model

    References:
        Zhang, J.; Zhou, Y.; Lei, Y.-K.; Yang, Y. I.; Gao, Y. Q.,
        Molecular CT: unifying geometry and representation learning for molecules at different scales
        ArXiv: 2012.11816

    Args:



    """

    def __init__(
            self,
            num_elements=100,
            min_rbf_dis=0.05,
            max_rbf_dis=1,
            num_rbf=32,
            dim_feature=64,
            n_interactions=3,
            interactions=None,
            n_heads=8,
            max_cycles=10,
            activation=Swish(),
            unit_length='nm',
            self_dis=None,
            rbf_sigma=None,
            rbf_function=LogGaussianDistribution,
            cutoff=None,
            cutoff_network='smooth',
            use_distances=True,
            use_bonds=False,
            num_bond_types=16,
            public_dis_filter=True,
            public_bond_filter=True,
            use_feed_forward=False,
            trainable_gaussians=False,
            use_pondering=True,
            fixed_cycles=False,
            rescale_rbf=True,
            use_graph_norm=False,
            use_time_embedding=True,
            coupled_interactions=False,
            use_mcr=False,
            debug=False,
        ):
        super().__init__(
            num_elements=num_elements,
            dim_feature=dim_feature,
            min_rbf_dis=min_rbf_dis,
            max_rbf_dis=max_rbf_dis,
            n_interactions=n_interactions,
            interactions=interactions,
            activation=activation,
            num_rbf=num_rbf,
            unit_length=unit_length,
            rbf_sigma=rbf_sigma,
            rbf_function=rbf_function,
            cutoff=cutoff,
            cutoff_network=cutoff_network,
            rescale_rbf=rescale_rbf,
            use_graph_norm=use_graph_norm,
            use_public_filter=public_dis_filter,
        )
        self.network_name = 'MolCT'
        self.max_distance = max_rbf_dis
        self.min_distance = min_rbf_dis
        self.use_distances = use_distances
        self.trainable_gaussians = trainable_gaussians
        self.use_mcr = use_mcr
        self.debug = debug

        if self_dis is None:
            self.self_dis = self.min_distance
        else:
            self.self_dis = self_dis

        self.self_dis_tensor = Tensor([self.self_dis], ms.float32)

        self.n_heads = n_heads

        if use_time_embedding:
            time_embedding = self._get_time_signal(max_cycles, dim_feature)
        else:
            time_embedding = [0 for _ in range(max_cycles)]

        self.use_bonds = use_bonds

        use_dis_inter = False
        use_bond_inter = False
        use_mix_inter = False
        if self.interaction_types is not None:
            self.use_distances = False
            self.use_bonds = False
            for itype in self.interaction_types:
                if itype == 'dis':
                    use_dis_inter = True
                    self.use_distances = True
                elif itype == 'bond':
                    use_bond_inter = True
                    self.use_bonds = True
                elif itype == 'mix':
                    use_mix_inter = True
                    self.use_distances = True
                    self.use_bonds = True
                else:
                    raise ValueError(
                        '"interactions" must be "dis", "bond" or "mix"')
        else:
            if self.use_distances and self.use_bonds:
                use_mix_inter = True
            elif self.use_distances:
                use_dis_inter = True
            elif self.use_bonds:
                use_bond_inter = True
            else:
                raise ValueError(
                    '"use_bonds" and "use_distances" cannot be both "False"!')

        inter_bond_filter = False
        if self.use_bonds:
            self.bond_embedding = nn.Embedding(
                num_bond_types,
                dim_feature,
                use_one_hot=True,
                embedding_table=Normal(1.0))
            if public_bond_filter:
                self.bond_filter = Residual(dim_feature, activation=activation)
            else:
                inter_bond_filter = True

        inter_dis_filter = False
        if self.use_distances:
            if self.use_public_filter:
                self.dis_filter = ResFilter(
                    num_rbf, dim_feature, self.activation)
                # self.dis_filter = Filter(num_rbf,dim_feature,None)
                # self.dis_filter = Dense(num_rbf,dim_feature,has_bias=True,activation=None)
            else:
                self.dis_filter = Dense(
                    num_rbf, dim_feature, has_bias=True, activation=None)
                inter_dis_filter = True

        interaction_list = []
        if coupled_interactions:
            if use_dis_inter:
                self.dis_interaction = NeuralInteractionUnit(
                    dim_feature=dim_feature,
                    num_rbf=num_rbf,
                    n_heads=n_heads,
                    activation=self.activation,
                    max_cycles=max_cycles,
                    time_embedding=time_embedding,
                    use_pondering=use_pondering,
                    use_distances=True,
                    use_bonds=False,
                    use_dis_filter=inter_dis_filter,
                    use_bond_filter=False,
                    fixed_cycles=fixed_cycles,
                    use_feed_forward=use_feed_forward,
                )
            else:
                self.dis_interaction = None

            if use_bond_inter:
                self.bond_interaction = NeuralInteractionUnit(
                    dim_feature=dim_feature,
                    num_rbf=num_rbf,
                    n_heads=n_heads,
                    activation=self.activation,
                    max_cycles=max_cycles,
                    time_embedding=time_embedding,
                    use_pondering=use_pondering,
                    use_distances=False,
                    use_bonds=True,
                    use_dis_filter=False,
                    use_bond_filter=inter_bond_filter,
                    fixed_cycles=fixed_cycles,
                    use_feed_forward=use_feed_forward,
                )
            else:
                self.bond_interaction = None

            if use_mix_inter:
                self.mix_interaction = NeuralInteractionUnit(
                    dim_feature=dim_feature,
                    num_rbf=num_rbf,
                    n_heads=n_heads,
                    activation=self.activation,
                    max_cycles=max_cycles,
                    time_embedding=time_embedding,
                    use_pondering=use_pondering,
                    use_distances=True,
                    use_bonds=True,
                    use_dis_filter=inter_dis_filter,
                    use_bond_filter=inter_bond_filter,
                    fixed_cycles=fixed_cycles,
                    use_feed_forward=use_feed_forward,
                )
            else:
                self.mix_interaction = None

            if self.interaction_types is not None:
                for inter in self.interaction_types:
                    if inter == 'dis':
                        interaction_list.append(self.dis_interaction)
                        self.interaction_typenames.append('D0')
                    elif inter == 'bond':
                        interaction_list.append(self.bond_interaction)
                        self.interaction_typenames.append('B0')
                    else:
                        interaction_list.append(self.mix_interaction)
                        self.interaction_typenames.append('M0')
            else:
                if use_dis_inter:
                    interaction_list = [
                        self.dis_interaction * self.n_interactions]
                    self.interaction_typenames = ['D0',] * self.n_interactions
                elif use_bond_inter:
                    interaction_list = [
                        self.bond_interaction * self.n_interactions]
                    self.interaction_typenames = ['B0',] * self.n_interactions
                else:
                    interaction_list = [
                        self.mix_interaction * self.n_interactions]
                    self.interaction_typenames = ['M0',] * self.n_interactions
        else:
            if self.interaction_types is not None:
                did = 0
                bid = 0
                mid = 0
                for inter in self.interaction_types:
                    use_distances = False
                    use_bonds = False
                    use_dis_filter = False
                    use_bond_filter = False
                    if inter == 'dis':
                        use_distances = True
                        use_dis_filter = inter_dis_filter
                        self.interaction_typenames.append('D' + str(did))
                        did += 1
                    elif inter == 'bond':
                        use_bonds = True
                        self.interaction_typenames.append('B' + str(bid))
                        use_bond_filter = inter_bond_filter
                        bid += 1
                    elif inter == 'mix':
                        use_distances = True
                        use_bonds = True
                        use_dis_filter = inter_dis_filter
                        use_bond_filter = inter_bond_filter
                        self.interaction_typenames.append('M' + str(mid))
                        mid += 1

                    interaction_list.append(
                        NeuralInteractionUnit(
                            dim_feature=dim_feature,
                            num_rbf=num_rbf,
                            n_heads=n_heads,
                            activation=self.activation,
                            max_cycles=max_cycles,
                            time_embedding=time_embedding,
                            use_pondering=use_pondering,
                            use_distances=use_distances,
                            use_bonds=use_bonds,
                            use_dis_filter=use_dis_filter,
                            use_bond_filter=use_bond_filter,
                            fixed_cycles=fixed_cycles,
                            use_feed_forward=use_feed_forward,
                        )
                    )
            else:
                if use_dis_inter:
                    t = 'D'
                elif use_bond_inter:
                    t = 'B'
                else:
                    t = 'M'

                self.interaction_typenames = [
                    t + str(i) for i in range(self.n_interactions)]

                interaction_list = [
                    NeuralInteractionUnit(
                        dim_feature=dim_feature,
                        num_rbf=num_rbf,
                        n_heads=n_heads,
                        activation=self.activation,
                        max_cycles=max_cycles,
                        time_embedding=time_embedding,
                        use_pondering=use_pondering,
                        use_distances=self.use_distances,
                        use_bonds=self.use_bonds,
                        use_dis_filter=inter_dis_filter,
                        use_bond_filter=inter_bond_filter,
                        fixed_cycles=fixed_cycles,
                        use_feed_forward=use_feed_forward,
                    )
                    for i in range(self.n_interactions)
                ]

        self.n_interactions = len(interaction_list)
        self.interactions = nn.CellList(interaction_list)

        self.lmax_label = []
        for i in range(n_interactions):
            self.lmax_label.append('l' + str(i) + '_cycles')

        self.fill = P.Fill()
        self.concat = P.Concat(-1)
        self.reducesum = P.ReduceSum()
        self.reducemax = P.ReduceMax()
        self.tensor_summary = P.TensorSummary()
        self.scalar_summary = P.ScalarSummary()

    def set_fixed_neighbors(self, flag=True):
        for interaction in self.interactions:
            interaction.set_fixed_neighbors(flag)

    def _calc_cutoffs(
            self,
            r_ij=1,
            neighbor_mask=None,
            bonds=None,
            bond_mask=None,
            atom_mask=None):
        mask = None

        if self.use_distances:
            if neighbor_mask is not None:
                mask = self.concat((atom_mask, neighbor_mask))

            if self.cutoff_network is None:
                new_shape = (r_ij.shape[0], r_ij.shape[1] + 1, r_ij.shape[2])
                return self.fill(r_ij.dtype, new_shape, 1.0), mask
            rii_shape = r_ij.shape[:-1] + (1,)
            r_ii = self.fill(r_ij.dtype, rii_shape, self.self_dis)
            if atom_mask is not None:
                r_large = F.ones_like(r_ii) * 5e4
                r_ii = F.select(atom_mask, r_ii, r_large)
            # [B, A, N']
            r_ij = self.concat((r_ii, r_ij))

            return self.cutoff_network(r_ij, mask)
        if bond_mask is not None:
            mask = self.concat((atom_mask, bond_mask))
        return F.cast(mask > 0, ms.float32), mask

    def _get_self_rbf(self):
        f_ii = self._get_rbf(self.self_dis_tensor)
        return f_ii

    def _get_time_signal(
            self,
            length,
            channels,
            min_timescale=1.0,
            max_timescale=1.0e4):
        """
        Generates a [1, length, channels] timing signal consisting of sinusoids
        Adapted from:
        https://github.com/andreamad8/Universal-Transformer-Pytorch/blob/master/models/common_layer.py
        """
        position = np.arange(length)
        num_timescales = channels // 2
        log_timescale_increment = (np.log(
            float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
        inv_timescales = min_timescale * \
            np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
        scaled_time = np.expand_dims(
            position, 1) * np.expand_dims(inv_timescales, 0)

        signal = np.concatenate(
            [np.sin(scaled_time), np.cos(scaled_time)], axis=1)
        signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                        'constant', constant_values=[0.0, 0.0])

        return Tensor(signal, ms.float32)
