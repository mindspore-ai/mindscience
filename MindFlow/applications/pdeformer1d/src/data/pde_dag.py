# Copyright 2023 Huawei Technologies Co., Ltd
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
r"""
Representing the PDE in the form of a computational graph, which includes both
the symbolic and numeric information inherent in a PDE. This essentially
constructs a directed acyclic graph (DAG).
"""
from functools import lru_cache
from typing import Tuple, List, Optional

import numpy as np
from numpy.typing import NDArray
import networkx as nx
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from .env import USE_GLOBAL_NODE, int_dtype, float_dtype

# 'uf': unknown field; 'bv_l'/'bv_r': boundary values at left/right;
# 'ic': initial condition; 'cf': coefficient field; 'vc': varying coefficient
VAR_NODE_TYPES = ['uf']
COEF_NODE_TYPES = ['coef', 'bv_l', 'bv_r']
FUNCTION_NODE_TYPES = ['ic', 'cf']
OPERATOR_NODE_TYPES = ['add', 'mul', 'eq0',
                       'dt', 'dx', 'neg', 'square', 'exp', 'sin', 'cos']
RESERVED_NODE_TYPES = ['vc'] + [f'Reserved{i}' for i in range(15)]
FUNCTION_BRANCH_NODE_TYPES = [f'Branch{i}' for i in range(32)]
INR_NODE_TYPES = [f'Mod{i}' for i in range(16)]
# During batch training, we may need to add 'pad' nodes (with index 0) to
# make sure all graphs in a batch have the same number of nodes.
DAG_NODE_TYPES = (['pad'] + VAR_NODE_TYPES + COEF_NODE_TYPES + FUNCTION_NODE_TYPES
                  + OPERATOR_NODE_TYPES + RESERVED_NODE_TYPES
                  + FUNCTION_BRANCH_NODE_TYPES + INR_NODE_TYPES)
NODE_TYPE_DICT = {t: i for (i, t) in enumerate(DAG_NODE_TYPES)}


class ModNodeSwapper:
    r"""
    In a pde_dag, the first inr nodes correspond to the first uf node (i.e. the
    first variable/component of the PDE). When we want to let them correspond
    to another uf node, this class will help us to reorder the nodes.
    Specifically, we swap (a) the first inr nodes and (b) inr nodes
    corresponding to the component with index `idx_var`. Note that we only have
    to modify `spatial_pos`, swapping its rows and columns, and generate
    `attn_bias` accordingly. All the remaining pde_dag data are left unchanged.
    """

    def __init__(self, uf_num_mod: int, n_vars: int) -> None:
        inds_old = np.arange(uf_num_mod)  # eg. [0,..,4] when uf_num_mod==5
        inds_old = np.expand_dims(inds_old, axis=0)  # [[0,..,4]]
        inds_old = np.repeat(inds_old, n_vars, axis=0)  # [[0,..,4],..,[0,..4]]
        # inds_old: [[0,..,4],..,[0,..,4]]; inds_new: [[0,..,4],..,[10,..,14]]
        inds_new = np.arange(uf_num_mod * n_vars).reshape(
            (n_vars, uf_num_mod))
        # Shape is [n_vars, 2 * uf_num_mod].
        self.inds_l = np.concatenate([inds_old, inds_new], axis=-1)
        self.inds_r = np.concatenate([inds_new, inds_old], axis=-1)

    def apply_(self, spatial_pos: NDArray[int], idx_var: int,
               node_type: Optional[NDArray[int]] = None) -> None:
        r"""
        Swap the modulation nodes in `spatial_pos`, making those corresponding
        to the unknown field (uf) with index `idx_var` moved to the first
        places. Note that the operation is done in-place, i.e. the changes are
        applied directly to the `spatial_pos` NumPy array.
        """
        if idx_var == 0:
            return  # nothing to do
        inds_l = self.inds_l[idx_var]  # [2 * uf_num_mod]
        inds_r = self.inds_r[idx_var]  # [2 * uf_num_mod]
        spatial_pos[inds_l] = spatial_pos[inds_r]
        spatial_pos[:, inds_l] = spatial_pos[:, inds_r]
        if node_type is not None:  # check node type is correct
            inds_r0 = inds_r[0]
            if inds_l[0] != 0 or node_type[inds_r0, 0] != node_type[0, 0]:
                raise RuntimeError("'ModNodeSwapper' not working as expected.")


class PDEAsDAG:
    r"""
    Represent a PDE (partial differential equation) in the form of a DAG
    (directed acyclic graph), including PDE nodes and auxiliary nodes. The
    attributes shall include the inputs to PDEformer, eg. 'node_type',
    'node_scalar', 'attn_bias' and 'spatial_pos'.

    Args:
        config (DictConfig): PDEformer configuration.
        node_list (List[Tuple[Str]]): PDE nodes in the DAG. Each element in the
            list takes the form ``(node_name, node_type,
                predecessor_node_1_name, predecessor_node_2_name, ...)``.
            Please make sure that all the nodes involved have distinct names.
        node_scalar (List[float], Optional): Scalar values assigned to the
            nodes in the DAG. The length should be equal to that of
            `node_list`.
        node_function (List[NDArray[float]], Optional): Function values of the
            initial condition (IC) $g(x)$, coefficient field (CF) $c(x)$,
            varying coefficient (VC) $c(t)$, etc. The length should be equal to
            the number of nodes with type 'ic', 'cf' or 'vc' in `node_list`.
            Each array in the list stores the values of the form
            :math:`\{(x_i, s(x_i))\}`, and should have the same shape
            :math:`(num\_pts, 2)`.

    Attributes:
        node_type (NDArray[int]): The type number of each node, shape
            :math:`(n\_node, 1)`.
        node_scalar (NDArray[float]): The scalar value of each node, shape
            :math:`(num\_scalar, 1)`.
        node_function (NDArray[float]): The function value of each node,
            shape :math:`(num\_function, num\_pts, 2)`.
        in_degree (NDArray[int]): The in-degree of each node, shape :math:`(n\_node)`.
        out_degree (NDArray[int]): The out-degree of each node, shape :math:`(n\_node)`.
        attn_bias (NDArray[float]): The attention bias matrix of the graph,
            shape :math:`(n\_node, n\_node)`.
        spatial_pos (NDArray[int]): The spatial position (shortest path length)
            from each node to each other node, shape :math:`(n\_node, n\_node)`.

    Here, :math:`num\_scalar` is given by the configuration option
    `config.data.pde_dag.max_n_scalar_nodes`, and :math:`num\_function` is
    `config.data.pde_dag.max_n_function_nodes`. Denote :math:`N` to be the
    number of branches (`config.model.function_encoder.num_branches`), we shall
    have :math:`n\_node = num\_scalar + num\_function \times N`.

    Note that PDEformer by default predicts only the first variable (i.e. the
    first component) of the PDE. If the user wants to get the prediction of the
    variable indexed by :math:`i` (count starting from 0), consider using
        ``spatial_pos, attn_bias = pde_dag.get_spatial_pos_attn_bias(i)``
    instead of the direct version `pde_dag.spatial_pos` or `pde_dag.attn_bias`
    as the inputs to PDEformer. (This makes no difference if :math:`i=0`.)

    Examples:
        >>> # u_t + u = 0
        >>> pde_nodes = [('u', 'uf'),
        ...              ('u_t', 'dt', 'u'),
        ...              ('u_t+u', 'add', 'u_t', 'u'),
        ...              ('my_eqn', 'eq0', 'u_t+u')]
        >>> pde_dag = PDEAsDAG(config, pde_nodes)
        >>> pde_dag.plot()  # plot the resulting DAG
    """

    def __init__(self,
                 config: DictConfig,
                 node_list: List[Tuple[str]],
                 node_scalar: List[float] = None,
                 node_function: List[np.ndarray] = None) -> None:
        num_spatial = config.model.graphormer.num_spatial
        disconn_attn_bias = float(config.data.pde_dag.disconn_attn_bias)
        # Number of INR modulation nodes for each uf node.
        uf_num_mod = config.model.inr.num_layers - 1
        function_num_branches = config.model.function_encoder.num_branches
        max_n_scalar_nodes = config.data.pde_dag.max_n_scalar_nodes
        max_n_function_nodes = config.data.pde_dag.max_n_function_nodes

        # process node_list to build DAG
        dag, n_vars, n_functions, n_scalar_pad = self._build_dag(
            node_list, uf_num_mod, function_num_branches, max_n_scalar_nodes)

        n_functions_to_add = max_n_function_nodes - n_functions
        if n_functions_to_add < 0:
            raise ValueError(
                "The number of function nodes involved in the PDE computational "
                f"graph ({n_functions}) exceeds max_n_function_nodes set in "
                f"config ({max_n_function_nodes})!")
        pad_len = n_functions_to_add * function_num_branches

        # node_type
        node_type = np.array([attr_dict['typeid']
                              for node, attr_dict in dag.nodes.data()])
        padding_mask = node_type == 0  # [n_dag_node]
        node_type = np.pad(node_type, (0, pad_len)).astype(int_dtype)
        self.node_type = node_type[:, np.newaxis]  # [n_node, 1]

        # node_scalar
        if node_scalar is None:
            node_scalar = np.zeros([max_n_scalar_nodes], dtype=float_dtype)
        else:
            if len(node_scalar) != len(node_list):
                raise ValueError(
                    f"The length of 'node_scalar' ({len(node_scalar)}) should "
                    f"be equal to that of 'node_list' ({len(node_list)})!")
            node_scalar = np.array(node_scalar, dtype=float_dtype)
            n_mod_nodes = n_vars * uf_num_mod
            # [n_raw_nodes] -> [max_n_scalar_nodes]
            node_scalar = np.pad(node_scalar, (n_mod_nodes, n_scalar_pad))
        # Shape is [max_n_scalar_nodes, 1].
        self.node_scalar = node_scalar[:, np.newaxis]

        # node_function
        if node_function is None:
            n_pts = 1
            node_function = np.zeros([max_n_function_nodes, n_pts, 2],
                                     dtype=float_dtype)
        else:
            if len(node_function) != n_functions:
                raise ValueError(
                    f"The length of 'node_function' ({len(node_function)}) should be "
                    "equal to the number of function nodes involved in the PDE "
                    f"computational graph ({n_functions})!")
            node_function = np.array(node_function, dtype=float_dtype)
            node_function = np.pad(
                node_function, ((0, n_functions_to_add), (0, 0), (0, 0)))
        # Shape is [max_n_function_nodes, n_pts, 2].
        self.node_function = node_function

        # spatial_pos, attn_bias
        shortest_path_len = nx.floyd_warshall_numpy(dag)
        # Need +1 because value 0 is reserved for padded nodes.
        spatial_pos = 1 + shortest_path_len.clip(
            0, num_spatial - 2).astype(int_dtype)
        spatial_pos[padding_mask] = 0
        spatial_pos[:, padding_mask] = 0
        self.spatial_pos = np.pad(spatial_pos, ((0, pad_len), (0, pad_len)))
        self.attn_bias = self.get_attn_bias(
            self.node_type, self.spatial_pos, disconn_attn_bias)
        if n_vars > 1:
            self.mod_node_swapper = ModNodeSwapper(uf_num_mod, n_vars)
            self.disconn_attn_bias = disconn_attn_bias

        # in_degree, out_degree
        # Need +1 because value 0 is reserved for padded nodes.
        in_degree = 1 + np.array([d for node, d in dag.in_degree()])
        in_degree[padding_mask] = 0
        self.in_degree = np.pad(in_degree, (0, pad_len)).astype(int_dtype)
        out_degree = 1 + np.array([d for node, d in dag.out_degree()])
        out_degree[padding_mask] = 0
        self.out_degree = np.pad(out_degree, (0, pad_len)).astype(int_dtype)

        self._dag = dag
        self.n_vars = n_vars

        # validate config
        if len(DAG_NODE_TYPES) > config.model.graphormer.num_node_type:
            raise ValueError("'num_node_type' is too small.")
        if self.in_degree.max() >= config.model.graphormer.num_in_degree:
            raise ValueError("'num_in_degree' is too small.")
        if self.out_degree.max() >= config.model.graphormer.num_out_degree:
            raise ValueError("'num_out_degree' is too small.")

    @staticmethod
    def _build_dag(node_list: List[Tuple[str]],
                   uf_num_mod: int,
                   function_num_branches: int,
                   max_n_scalar_nodes: int) -> Tuple:
        r"""Build DAG from the given node list."""
        mod_node_list = []
        function_branch_node_list = []
        edge_list = []
        n_vars = 0
        n_functions = 0
        for node, type_, *predecessors in node_list:
            edge_list.extend([(node_p, node) for node_p in predecessors])
            if type_ in VAR_NODE_TYPES:
                n_vars += 1
                for j in range(uf_num_mod):
                    node_new = f'{node}:Mod{j}'
                    mod_node_list.append((node_new, f'Mod{j}'))
                    edge_list.append((node_new, node))
            elif type_ in ['ic', 'cf', 'vc']:
                n_functions += 1
                for j in range(function_num_branches):
                    node_new = f'{node}:Branch{j}'
                    function_branch_node_list.append((node_new, f'Branch{j}'))
                    if type_ == 'ic':
                        edge_list.append((node, node_new))
                    else:
                        edge_list.append((node_new, node))

        if n_vars < 1:
            raise ValueError("There should be at least one 'uf' node in the PDE.")

        # pad scalar nodes
        dag_node_list = mod_node_list + node_list
        n_scalar_pad = max_n_scalar_nodes - len(dag_node_list)
        if n_scalar_pad < 0:
            raise ValueError(
                f"Target scalar node number ({max_n_scalar_nodes}) should not be "
                f"less than the number of existing nodes ({len(dag_node_list)})!")
        if NODE_TYPE_DICT['pad'] != 0:
            raise RuntimeError("Node type 'pad' not indexed as zero.")
        dag_node_list.extend([(f'pad{j}', 'pad')
                              for j in range(n_scalar_pad)])
        dag_node_list.extend(function_branch_node_list)

        # create DAG
        dag = nx.DiGraph()
        dag.add_nodes_from([
            (node, {'type': type_, 'typeid': NODE_TYPE_DICT[type_]})
            for node, type_, *predecessors in dag_node_list])
        dag.add_edges_from(edge_list)

        out_tuple = (dag, n_vars, n_functions, n_scalar_pad)
        return out_tuple

    @staticmethod
    def get_attn_bias(node_type: NDArray[int],
                      spatial_pos: NDArray[int],
                      disconn_attn_bias: float) -> NDArray[float]:
        r"""Get the `attn_bias` NumPy array corresponding to the `spatial_pos` NumPy array."""
        n_node, _ = node_type.shape
        attn_bias = np.zeros([n_node, n_node], dtype=float_dtype)
        # optionally disable attention between node pairs that are not
        # connected in the DAG
        connect_mask = spatial_pos == np.max(spatial_pos)  # [n_node, n_node]
        connect_mask = np.logical_and(connect_mask, connect_mask.T)
        attn_bias[connect_mask] = disconn_attn_bias
        # disable attention to padded nodes
        padding_mask = node_type[:, 0] == 0  # [n_node]
        if USE_GLOBAL_NODE:
            attn_bias = np.pad(attn_bias, ((1, 0), (1, 0)))
            padding_mask = np.pad(padding_mask, (1, 0))
        attn_bias[:, padding_mask] = -np.inf
        return attn_bias

    def get_spatial_pos_attn_bias(self, idx_var: int = 0) -> Tuple[NDArray]:
        r"""
        Get the `spatial_pos` and `attn_bias` numpy arrays, in which the
        auxiliary modulation nodes corresponding to the unknown field (uf) with
        index `idx_var` are moved to the first places.
        """
        if idx_var >= self.n_vars:
            raise ValueError(f"PDE Variable index ({idx_var}) out of range")
        if idx_var == 0:
            return (self.spatial_pos, self.attn_bias)
        spatial_pos = np.copy(self.spatial_pos)
        # swap the rows and columns of 'spatial_pos'
        self.mod_node_swapper.apply_(spatial_pos, idx_var, self.node_type)
        attn_bias = self.get_attn_bias(
            self.node_type, spatial_pos, self.disconn_attn_bias)
        return (spatial_pos, attn_bias)

    def plot(self, mode: str = 'type', show_pad: bool = False) -> None:
        r"""Create a plot of the current directed acyclic graph (DAG)."""
        nodes_data = self._dag.nodes.data()
        plot_nodes = [node for node, attr_dict in nodes_data
                      if show_pad or attr_dict['typeid'] != 0]
        hidden_nodes = [node for node, attr_dict in nodes_data
                        if node not in plot_nodes]
        if mode == 'name':
            labels_dict = {node: node for node, attr_dict in nodes_data}
        elif mode == 'type':
            labels_dict = {node: attr_dict['type']
                           for node, attr_dict in nodes_data}
        elif mode == 'name+type':
            labels_dict = {node: f"{node}({attr_dict['type']})"
                           for node, attr_dict in nodes_data}
        elif mode in ['', 'none']:
            labels_dict = {node: '' for node, attr_dict in nodes_data}
        else:
            raise NotImplementedError(
                "Supported values of 'mode' include ['name', 'type', "
                f"'name+type', 'none'], but got '{mode}'.")
        plt.figure(figsize=(6, 6))
        nx.draw_shell(self._dag,
                      with_labels=True,
                      nlist=[plot_nodes, hidden_nodes],
                      nodelist=plot_nodes,
                      labels=labels_dict,
                      node_color=[[0.5, 1, 0.5]])
        plt.show()

    def n_nodes_with_pad(self) -> int:
        r"""Number of nodes in the current DAG, including padded nodes."""
        return self.node_type.shape[0]


class PDENode:
    r"""
    Wrapper of the DAG nodes generated by the `PDENodesCollector` class.

    Args:
        name (Str): Name of the current node.
        src_pde (PDENodesCollector): The `PDENodesCollector` class instance
            from which the current node is generated.
    """

    def __init__(self, name: str, src_pde) -> None:
        self.name = name
        self.src_pde = src_pde

    def __neg__(self):
        return self.src_pde.neg(self)

    def __add__(self, node2):
        if node2 == 0:
            return self
        return self.src_pde.sum(node2, self)

    def __radd__(self, node2):
        return self.__add__(node2)

    def __mul__(self, node2):
        if node2 is self:
            return self.src_pde.square(self)
        return self.src_pde.prod(node2, self)

    def __rmul__(self, node2):
        return self.__mul__(node2)

    def __sub__(self, node2):
        # note that node2 could be a number
        return self.__add__(-node2)

    @property
    @lru_cache(maxsize=1)  # this node is created only once
    def dt(self):
        r"""Create a new node that represents the temporal derivative of this node."""
        return self.src_pde.dt(self)

    @property
    @lru_cache(maxsize=1)  # this node is created only once
    def dx(self):
        r"""Create a new node that represents the spatial derivative of this node."""
        return self.src_pde.dx(self)


NULL_NODE = None


class PDENodesCollector:
    r"""
    This class enables specifying a PDE via Python codes.

    Examples:
    ---------
    The viscous Burgers equation $u_t+(u^2/2 - 0.1 u_x)_x=0$ on $x\in[-1,1]$
    with initial condition $u(0,x) = \sin(\pi x)$:
        >>> x_coord = np.linspace(-1, 1, 257)[:-1]  # define spatial coordinates
        >>> ic_field = np.sin(np.pi * x_coord)  # define initial condition
        >>> pde = PDENodesCollector()
        >>> u = pde.new_uf()  # specify an unknown field
        >>> pde.set_ic(u, x_coord, ic_field)  # specify the initial condition.
        >>> pde.sum_eq0(u.dt, (0.5 * pde.square(u) - 0.1 * u.dx).dx)

    The last line above shows the simplified expression. The corresponding full
    version is:
        >>> kappa = pde.new_coef(0.1)
        >>> dx_u = pde.dx(u)
        >>> viscous_term = pde.mul(kappa, dx_u)
        >>> convection_flux = pde.mul(pde.new_coef(0.5), pde.square(u))
        >>> flux = pde.sum(convection_flux, pde.neg(viscous_term))
        >>> pde.sum_eq0(pde.dt(u), pde.dx(flux))

    We use a scalar coefficient $\kappa=0.1$ above. For the spatial-varying
    case $\kappa(x)=0.5+0.4\cos(\pi x)$, write instead:
        >>> kappa_field = 0.5 + 0.4 * np.cos(np.pi * x_coord)
        >>> kappa = pde.new_coef_field(x_coord, kappa_field)

    The periodic boundary condition is employed by default. For non-periodic
    cases, eg. $u(t,-1)=0,u_x(t,1)=0.2$ (Dirichlet boundary condition on the
    left, Neumann on the right), you may add the following code:
        >>> pde.set_bv_l(u, 0)
        >>> pde.set_bv_r(u.dx, 0.2)

    The Robin boundary condition $(\alpha u_x+\beta u)(t,-1)=-1$ with
    $\alpha=0.6,\beta=-0.8$ can be specified as:
        >>> pde.set_bv_l(0.6 * u.dx + (-0.8) * u, -1)

    To get the best prediction performance, it is recommended to normalize the
    coefficients so that $\alpha^2+\beta^2=1$ and $\alpha>0$, since our
    training data is generated so.

    After the overall PDE is specified, we may pass in the PDEformer
    configurations and construct the DAG data, as follows:
        >>> pde_dag = pde.gen_dag(config)

    Users may plot the resulting DAG by executing
        >>> pde_dag.plot()

    Note 1:
    If a summation involves three or more summands, it is not recommended to
    write
        >>> term_sum = term1 + term2 + term3

    (at least for the current version), which is equivalent to
        >>> term_sum = pde.sum(pde.sum(term1, term2), term3)

    Write instead
        >>> term_sum = pde.sum(term1, term2, term3)

    Note 2:
    It is not recommended to create multiple nodes with the same semantic
    meaning. For example, it is not recommended to write
        >>> pde.sum_eq0(pde.dt(u), 0.1 * pde.dx(u), -pde.dx(pde.dx(u)))

    since each statement `pde.dx(u)` will create a new node representing
    :math:`u_x`. The better practice is to store such a node (that will be used
    multiple times) as a separate variable, like
        >>> dx_u = pde.dx(u)
        >>> pde.sum_eq0(pde.dt(u), 0.1 * dx_u, -pde.dx(dx_u))

    Also, you may make use of the `u.dx` expression as:
        >>> pde.sum_eq0(pde.dt(u), 0.1 * u.dx, -pde.dx(u.dx))

    whose functionality is equivalent to that of
        >>> dx_u = u.dx
        >>> pde.sum_eq0(pde.dt(u), 0.1 * dx_u, -dx_u.dx)

    User may check the resulting DAG by creating a plot of it, as have just
    been introduced above.
    """

    def __init__(self) -> None:
        self.node_list = []
        self.node_scalar = []
        self.node_function = []

    def gen_dag(self, config: DictConfig) -> PDEAsDAG:
        r"""
        Generate the directed acyclic graph (DAG) as well as its structural
        data according to the configuration.
        """
        return PDEAsDAG(config, self.node_list, self.node_scalar, self.node_function)

    def new_uf(self) -> PDENode:
        r"""Specify a new unknown field variable."""
        return self._add_node('uf')

    def new_coef(self, value: float) -> PDENode:
        r"""Specify a new PDE coefficient."""
        return self._add_node('coef', scalar=value)

    def new_varying_coef(self,
                         t_coord: NDArray[float],
                         temporal_coef: NDArray[float]) -> PDENode:
        r"""Specify a new time-dependent PDE coefficient."""
        self._add_function(t_coord, temporal_coef)
        return self._add_node('vc')

    def new_coef_field(self,
                       x_coord: NDArray[float],
                       field_value: NDArray[float]) -> PDENode:
        r"""Specify a new non-constant PDE coefficient."""
        self._add_function(x_coord, field_value)
        return self._add_node('cf')

    def set_ic(self,
               src_node: PDENode,
               x_coord: NDArray[float],
               field_value: NDArray[float]) -> None:
        r"""Specify initial condition."""
        self._add_function(x_coord, field_value)
        self._add_node('ic', predecessors=[src_node.name])

    def set_bv_l(self, src_node: PDENode, value: float = 0.) -> None:
        r"""Specify boundary value at the left endpoint of the interval."""
        self._add_node('bv_l', predecessors=[src_node.name], scalar=value)

    def set_bv_r(self, src_node: PDENode, value: float = 0.) -> None:
        r"""Specify boundary value at the right endpoint of the interval."""
        self._add_node('bv_r', predecessors=[src_node.name], scalar=value)

    def dt(self, src_node: PDENode) -> PDENode:
        r"""Create a new node that represents the temporal derivative of the source node."""
        return self._unary(src_node, 'dt')

    def dx(self, src_node: PDENode) -> PDENode:
        r"""Create a new node that represents the spatial derivative of the source node."""
        return self._unary(src_node, 'dx')

    def neg(self, src_node: PDENode) -> PDENode:
        r"""Create a new node that represents the negation of the source node."""
        return self._unary(src_node, 'neg')

    def square(self, src_node: PDENode) -> PDENode:
        r"""Create a new node that represents the square of the source node."""
        return self._unary(src_node, 'square')

    def sin(self, src_node: PDENode) -> PDENode:
        r"""Create a new node that represents the sine of the source node."""
        return self._unary(src_node, 'sin')

    def cos(self, src_node: PDENode) -> PDENode:
        r"""Create a new node that represents the cosine of the source node."""
        return self._unary(src_node, 'cos')

    def exp(self, src_node: PDENode) -> PDENode:
        r"""Create a new node that represents the exponential of the source node."""
        return self._unary(src_node, 'exp')

    def sum(self, *src_nodes) -> PDENode:
        r"""Create a new node that represents the summation of the existing source nodes, with type 'add'."""
        return self._multi_predecessor('add', src_nodes)

    def prod(self, *src_nodes) -> PDENode:
        r"""Create a new node that represents the product of the existing source nodes, with type 'mul'."""
        return self._multi_predecessor('mul', src_nodes)

    def sum_eq0(self, *src_nodes) -> PDENode:
        r"""Add an equation: the sum of all input nodes equals zero."""
        sum_node = self._multi_predecessor('add', src_nodes)
        return self._unary(sum_node, 'eq0')

    def _add_node(self,
                  type_: str,
                  predecessors: Optional[List[PDENode]] = None,
                  scalar: float = 0.,
                  name: str = "") -> PDENode:
        r"""Add a new node to the DAG."""
        if name == "":
            name = type_ + str(len(self.node_list))
        if predecessors is None:
            self.node_list.append((name, type_))
        else:
            self.node_list.append((name, type_, *predecessors))
        if not np.isscalar(scalar):
            raise ValueError(f"PDE node receives non-scalar value {scalar}.")
        self.node_scalar.append(scalar)
        return PDENode(name, self)

    def _add_function(self,
                      x_coord: NDArray[float],
                      fx_values: NDArray[float]) -> None:
        function = np.stack([x_coord, fx_values], axis=-1)  # [n_x_grid, 2]
        self.node_function.append(function)

    def _unary(self, src_node: PDENode, node_type: str) -> PDENode:
        if src_node is NULL_NODE:
            return NULL_NODE
        return self._add_node(node_type, predecessors=[src_node.name])

    def _multi_predecessor(self, node_type: str, src_nodes: Tuple) -> PDENode:
        r"""
        Input:
            node_type (str): choices {"add", "mul"}
            src_nodes: (node1, node2, ..) or ([node1, node2, ..], )
                or [[node1, node2, ..]], where node1, node2, .. are
                of type Union[PDENode, int, float, NoneType].
        Output:
            out_node (PDENode): The newly created node.
        """
        # ([node1, node2, ..], ) -> [node1, node2, ..]
        if len(src_nodes) == 1 and isinstance(src_nodes[0], (list, tuple)):
            src_nodes, = src_nodes
        # remove NULL_NODE entries
        src_nodes = [node for node in src_nodes if node is not NULL_NODE]
        # type check; convert number entries into 'coef' nodes in the DAG
        for i, node in enumerate(src_nodes):
            if np.isscalar(node):
                src_nodes[i] = self.new_coef(node)
            elif not isinstance(node, PDENode):
                raise ValueError("Nodes involved in sum/prod should have type"
                                 + " {PDENode, float, int, NoneType}, but got"
                                 + f" {type(node)}")
        if len(src_nodes) == 0:  # pylint: disable=C1801
            return NULL_NODE
        if len(src_nodes) == 1:
            return src_nodes[0]  # single predecessor, no new node to create
        src_node_names = [node.name for node in src_nodes]

        return self._add_node(node_type, predecessors=src_node_names)
