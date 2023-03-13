# Copyright 2022 Huawei Technologies Co., Ltd
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
The basic building blocks in model.
"""
import math
import numpy
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore.common.initializer import initializer, XavierUniform
from scipy import stats
from ..util.nn_utils import get_activation_function, AggregateNeighbor


class SelfAttention(nn.Cell):
    """
       Self SelfAttention Layer
    """

    def __init__(self, hidden, in_feature, out_feature):
        """
        The init function.
        :param hidden: the hidden dimension, can be viewed as the number of experts.
        :param in_feature: the input feature dimension.
        :param out_feature: the output feature dimension.
        """
        super(SelfAttention, self).__init__()

        self.w10 = Parameter(Tensor((hidden, in_feature), dtype=ms.float32))
        self.w20 = Parameter(Tensor((out_feature, hidden), dtype=ms.float32))
        self.reset_parameters()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(axis=-1)

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        initializer(XavierUniform(), self.w10)
        initializer(XavierUniform(), self.w20)

    def construct(self, inputs):
        """
        Computing function.
        """
        x = ops.matmul(self.w10, inputs.transpose(1, 0))
        x = self.tanh(x)
        x = ops.matmul(self.w20, x)
        attn = self.softmax(x)
        x = ops.matmul(attn, inputs)
        return x


class Readout(nn.Cell):
    """The readout function. Convert the node embeddings to the graph embeddings."""

    def __init__(self,
                 rtype: str = "none",
                 hidden_size: int = 0,
                 attn_hidden: int = None,
                 attn_out: int = None,
                 ):
        """
        The readout function.
        :param rtype: readout type, can be "mean" and "self_attention".
        :param hidden_size: input hidden size
        :param attn_hidden: only valid if rtype == "self_attention". The attention hidden size.
        :param attn_out: only valid if rtype == "self_attention". The attention out size.
        """
        super(Readout, self).__init__()
        # Cached zeros
        self.cached_zero_vector = ops.Zeros()(hidden_size, ms.float32)
        self.rtype = "mean"

        if rtype == "self_attention":
            self.attn = SelfAttention(hidden=attn_hidden,
                                      in_feature=hidden_size,
                                      out_feature=attn_out)
            self.rtype = "self_attention"

    def construct(self, embeddings, scope):
        """
        Given a batch node/edge embedding and a scope list,
        produce the graph-level embedding by a scope.
        :param embeddings: The embedding matrix, num_atoms or num_bonds \times hidden_size.
        :param scope: a list, in which the element is a list [start, range]. `start` is the index
        :return:
        """
        # Readout

        mol_vecs = []
        for (a_start, a_size) in scope:

            cur_hiddens = embeddings[a_start:(a_start + a_size)]

            if self.rtype == "self_attention":
                cur_hiddens = self.attn(cur_hiddens)
                cur_hiddens = cur_hiddens.flatten()
            else:
                cur_hiddens = cur_hiddens.sum(0) / a_size

            mol_vecs.append(cur_hiddens)

        mol_vecs = ops.stack(mol_vecs, 0)
        return mol_vecs


class MPNEncoder(nn.Cell):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args,
                 atom_messages: bool,
                 init_message_dim: int,
                 attached_fea_fdim: int,
                 hidden_size: int,
                 bias: bool,
                 depth: int,
                 dropout: float,
                 undirected: bool,
                 dense: bool,
                 aggregate_to_atom: bool,
                 attach_fea: bool,
                 input_layer="fc",
                 dynamic_depth='none',
                 is_training=True
                 ):
        """
        Initializes the MPNEncoder.
        :param atom_messages: enables atom_messages or not.
        :param init_message_dim:  the initial input message dimension.
        :param attached_fea_fdim:  the attached feature dimension.
        :param hidden_size: the output message dimension during message passing.
        :param bias: the bias in the message passing.
        :param depth: the message passing depth.
        :param dropout: the dropout rate.
        :param undirected: the message passing is undirected or not.
        :param dense: enables the dense connections.
        :param attach_fea: enables the feature attachment during the message passing process.
        :param dynamic_depth: enables the dynamic depth. Possible choices: "none", "uniform" and "truncnorm"
        """
        super(MPNEncoder, self).__init__()
        self.init_message_dim = init_message_dim
        self.attached_fea_fdim = attached_fea_fdim
        self.hidden_size = hidden_size
        self.bias = bias
        self.depth = depth
        self.dropout = dropout
        self.input_layer = input_layer
        self.layers_per_message = 1
        self.undirected = undirected
        self.atom_messages = atom_messages
        self.dense = dense
        self.aggreate_to_atom = aggregate_to_atom
        self.attached_fea = attach_fea
        self.dynamic_depth = dynamic_depth
        self.is_training = is_training

        # Dropout
        self.dropout_layer = nn.Dropout(keep_prob=1.0 - self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Input
        if self.input_layer == "fc":
            input_dim = self.init_message_dim
            self.wi = nn.Dense(input_dim, self.hidden_size, has_bias=self.bias)

        if self.attached_fea:
            w_h_input_size = self.hidden_size + self.attached_fea_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.wh = nn.Dense(w_h_input_size, self.hidden_size, has_bias=self.bias)

        self.select_neighbor_and_aggregate = AggregateNeighbor()

    def construct(self,
                  init_messages,
                  init_attached_features,
                  a2nei,
                  a2attached,
                  b2a=None,
                  b2revb=None,
                  ):
        """
        :param init_messages:  initial massages, can be atom features or bond features.
        :param init_attached_features: initial attached_features.
        :param a2nei: the relation of item to its neighbors. For the atom message passing, a2nei = a2a. For bond
        messages a2nei = a2b
        :param a2attached: the relation of item to the attached features during message passing. For the atom message
        passing, a2attached = a2b. For the bond message passing a2attached = a2a
        :param b2a: remove the reversed bond in bond message passing
        :param b2revb: remove the reversed atom in bond message passing
        :return: if aggreate_to_atom or self.atom_messages, return num_atoms x hidden.
        Otherwise, return num_bonds x hidden
        """

        # Input
        message = None
        inputs = None
        if self.input_layer == 'fc':
            inputs = self.wi(init_messages)  # num_bonds x hidden_size # f_bond
            message = self.act_func(inputs)  # num_bonds x hidden_size
        elif self.input_layer == 'none':
            inputs = init_messages
            message = inputs

        attached_fea = init_attached_features  # f_atom / f_bond

        # dynamic depth
        # uniform sampling from depth - 1 to depth + 1
        # only works in training.
        if self.training and self.dynamic_depth != "none":
            if self.dynamic_depth == "uniform":
                # uniform sampling
                ndepth = numpy.random.randint(self.depth - 3, self.depth + 3)
            else:
                # truncnorm
                mu = self.depth
                sigma = 1
                lower = mu - 3 * sigma
                upper = mu + 3 * sigma
                x = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                ndepth = int(x.rvs(1))
        else:
            ndepth = self.depth

        # Message passing
        for _ in range(ndepth - 1):
            if self.undirected:
                # two directions should be the same
                message = (message + message[b2revb]) / 2

            nei_message = self.select_neighbor_and_aggregate(message, a2nei)
            a_message = nei_message
            if self.attached_fea:
                attached_nei_fea = self.select_neighbor_and_aggregate(attached_fea, a2attached)
                a_message = ops.concat((nei_message, attached_nei_fea), 1)

            if not self.atom_messages:
                rev_message = message[b2revb]
                if self.attached_fea:
                    atom_rev_message = attached_fea[b2a[b2revb]]
                    rev_message = ops.concat((rev_message, atom_rev_message), 1)
                # Except reverse bond its-self(w) ! \sum_{k\in N(u) \ w}
                message = a_message[b2a] - rev_message  # num_bonds x hidden
            else:
                message = a_message

            message = self.wh(message)

            # BUG here, by default MPNEncoder use the dense connection in the message passing step.
            # The correct form should if not self.dense
            if self.dense:
                message = self.act_func(message)  # num_bonds x hidden_size
            else:
                message = self.act_func(inputs + message)
            message = self.dropout_layer(message)  # num_bonds x hidden

        output = message

        return output  # num_atoms x hidden


class PositionwiseFeedForward(nn.Cell):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, activation="ReLU", dropout=0.1, d_out=None):
        """
        :param d_model: the input dimension.
        :param d_ff: the hidden dimension.
        :param activation: the activation function.
        :param dropout: the dropout rate.
        :param d_out: the output dimension, the default value is equal to d_model.
        """
        super(PositionwiseFeedForward, self).__init__()
        if d_out is None:
            d_out = d_model
        # By default, bias is on.

        self.w1 = nn.Dense(d_model, d_ff)
        self.w2 = nn.Dense(d_ff, d_out)
        self.dropout = nn.Dropout(1 - dropout)
        self.act_func = get_activation_function(activation)

    def construct(self, x):
        x1 = self.w1(x)
        act = self.act_func(x1)
        drop = self.dropout(act)
        out = self.w2(drop)

        return out


class EmbeddTransformLayer(nn.Cell):
    """ Transfer the output of atom/bond multi-head attention to the final atom/bond output."""

    def __init__(self, to_atom, hidden_size, node_fdim, edge_fdim, activation, dropout):

        super(EmbeddTransformLayer, self).__init__()
        self.to_atom = to_atom
        dim = node_fdim
        if not self.to_atom:
            dim = edge_fdim
        self.hidden_size = hidden_size
        self.activation = activation
        self.dropout = dropout

        self.ffn_from_atom = PositionwiseFeedForward(self.hidden_size + dim,
                                                     self.hidden_size * 4,
                                                     activation=self.activation,
                                                     dropout=self.dropout,
                                                     d_out=self.hidden_size)

        self.ffn_from_bond = PositionwiseFeedForward(self.hidden_size + dim,
                                                     self.hidden_size * 4,
                                                     activation=self.activation,
                                                     dropout=self.dropout,
                                                     d_out=self.hidden_size)

        self.from_atom_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout, res_connection=False)
        self.from_bond_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout, res_connection=False)
        self.concat = ops.Concat(axis=1)
        self.select_neighbor_and_aggregate = AggregateNeighbor()

    def construct(self,
                  atomwise_input=None,
                  bondwise_input=None,
                  original_f_atoms=None,
                  original_f_bonds=None,
                  a2a=None,
                  a2b=None,
                  b2a=None,
                  b2revb=None):

        """
        Transfer the output of atom/bond multi-head attention to the final atom/bond output.
        :param atomwise_input: the input embedding of atom/node.
        :param bondwise_input: the input embedding of bond/edge.
        :param original_f_atoms: the initial atom features.
        :param original_f_bonds: the initial bond features.
        :param a2a: mapping from atom index to its neighbors. num_atoms * max_num_bonds
        :param a2b: mapping from atom index to incoming bond indices.
        :param b2a: mapping from bond index to the index of the atom the bond is coming from.
        :param b2revb: mapping from bond index to the index of the reverse bond.
        :return:
        """
        if self.to_atom:
            # input to atom output
            # atom to atom
            aggr_out_aa = self.select_neighbor_and_aggregate(atomwise_input, a2a)
            aggr_atom_aa = self.concat([original_f_atoms, aggr_out_aa])
            atomwise_input = self.ffn_from_atom(aggr_atom_aa)
            atom_in_atom_out = self.from_atom_sublayer(None, atomwise_input)

            # bond to atom
            aggr_out_ba = self.select_neighbor_and_aggregate(bondwise_input, a2b)
            aggr_atom_ba = self.concat([original_f_atoms, aggr_out_ba])
            bondwise_input = self.ffn_from_bond(aggr_atom_ba)

            bond_in_atom_out = self.from_bond_sublayer(None, bondwise_input)

            out_1 = atom_in_atom_out
            out_2 = bond_in_atom_out

        else:
            # atom to bond
            atom_list = ops.expand_dims(b2a, 1)
            atom_list_for_bond = self.concat([atom_list, a2a[b2a]])
            aggr_output_ba = self.select_neighbor_and_aggregate(atomwise_input, atom_list_for_bond)
            # remove rev bond / atom --- need for bond view
            aggr_output_ba = aggr_output_ba - atomwise_input[b2a[b2revb]]
            aggr_bond_ba = self.concat([original_f_bonds, aggr_output_ba])
            atomwise_input = self.ffn_from_atom(aggr_bond_ba)
            atom_in_bond_out = self.from_atom_sublayer(None, atomwise_input)

            # bond input to bond output
            bond_list_for_bond = a2b[b2a]
            aggr_output_bb = self.select_neighbor_and_aggregate(bondwise_input, bond_list_for_bond)
            aggr_output_bb = aggr_output_bb - bondwise_input[b2revb]
            aggr_bond_bb = self.concat([original_f_bonds, aggr_output_bb])
            bondwise_input = self.ffn_from_bond(aggr_bond_bb)

            bond_in_bond_out = self.from_bond_sublayer(None, bondwise_input)

            out_1 = atom_in_bond_out
            out_2 = bond_in_bond_out

        return out_1, out_2


class SublayerConnection(nn.Cell):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout, res_connection):
        """Initialization.

        :param size: the input dimension.
        :param dropout: the dropout ratio.
        """
        super(SublayerConnection, self).__init__()

        self.norm = nn.LayerNorm((size,), begin_norm_axis=-1, begin_params_axis=-1)
        self.dropout = nn.Dropout(1.0 - dropout)
        self.res_connection = res_connection

    def construct(self, inputs, outputs):
        """Apply residual connection to any sublayer with the same size."""
        out = self.norm(outputs)
        out = self.dropout(out)
        if self.res_connection:
            out = out + inputs
        return out


class Attention(nn.Cell):
    """
    Compute 'Scaled Dot Product SelfAttention
    """

    def __init__(self, dropout, dim):
        super(Attention, self).__init__()

        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        self.softmax = nn.Softmax(axis=-1)
        self.dim = dim

    def construct(self, query, key, value):
        """
        Self-attention.
        """
        key = ops.transpose(key, (0, 1, 3, 2))
        scores = ops.matmul(query, key) / self.dim

        p_attn = self.softmax(scores)

        p_attn = self.dropout(p_attn)

        matmul = ops.matmul(p_attn, value)

        return matmul, p_attn


class MultiHeadedAttention(nn.Cell):
    """
    The multi-head attention module. Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h  # number of heads
        dim = math.sqrt(self.d_k)
        self.linear_layers = nn.CellList([nn.Dense(d_model, d_model) for _ in range(3)])  # why 3: query, key, value
        self.output_linear = nn.Dense(d_model, d_model)
        self.attention = Attention(dropout, dim)
        self.dropout = nn.Dropout(keep_prob=1.0 - dropout)

    def construct(self, query, key, value):
        """
        multi-head attention.
        """
        batch_size = query.shape[0]

        # 1) Do all the linear projections in batch from d_model => h x d_k

        linear_q = self.linear_layers[0](query).view(batch_size, -1, self.h, self.d_k)
        query = ops.transpose(linear_q, (0, 2, 1, 3))
        linear_k = self.linear_layers[1](key).view(batch_size, -1, self.h, self.d_k)
        key = ops.transpose(linear_k, (0, 2, 1, 3))
        linear_v = self.linear_layers[2](value).view(batch_size, -1, self.h, self.d_k)
        value = ops.transpose(linear_v, (0, 2, 1, 3))

        # 2) Apply attention on all the projected vectors in batch.
        x, _ = self.attention(query, key, value)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(0, 2, 1, 3).view(batch_size, -1, self.h * self.d_k)

        out = self.output_linear(x)

        return out


class Head(nn.Cell):
    """
    One head for multi-headed attention.
    :return: (query, key, value)
    """

    def __init__(self, args, hidden_size, atom_messages=False):
        """
        Initialization.
        :param args: The argument.
        :param hidden_size: the dimension of hidden layer in Head.
        :param atom_messages: the MPNEncoder type.
        """
        super(Head, self).__init__()
        atom_fdim = hidden_size
        bond_fdim = hidden_size
        hidden_size = hidden_size
        self.atom_messages = atom_messages
        if self.atom_messages:
            init_message_dim = atom_fdim
            attached_fea_dim = bond_fdim
        else:
            init_message_dim = bond_fdim
            attached_fea_dim = atom_fdim

        # Here we use the message passing network as query, key and value.
        self.mpn_q = MPNEncoder(args=args,
                                atom_messages=atom_messages,
                                init_message_dim=init_message_dim,
                                attached_fea_fdim=attached_fea_dim,
                                hidden_size=hidden_size,
                                bias=args.bias,
                                depth=args.depth,
                                dropout=args.dropout,
                                undirected=args.undirected,
                                dense=args.dense,
                                aggregate_to_atom=False,
                                attach_fea=False,
                                input_layer="none",
                                dynamic_depth="truncnorm",
                                is_training=args.is_training)
        self.mpn_k = MPNEncoder(args=args,
                                atom_messages=atom_messages,
                                init_message_dim=init_message_dim,
                                attached_fea_fdim=attached_fea_dim,
                                hidden_size=hidden_size,
                                bias=args.bias,
                                depth=args.depth,
                                dropout=args.dropout,
                                undirected=args.undirected,
                                dense=args.dense,
                                aggregate_to_atom=False,
                                attach_fea=False,
                                input_layer="none",
                                dynamic_depth="truncnorm",
                                is_training=args.is_training)
        self.mpn_v = MPNEncoder(args=args,
                                atom_messages=atom_messages,
                                init_message_dim=init_message_dim,
                                attached_fea_fdim=attached_fea_dim,
                                hidden_size=hidden_size,
                                bias=args.bias,
                                depth=args.depth,
                                dropout=args.dropout,
                                undirected=args.undirected,
                                dense=args.dense,
                                aggregate_to_atom=False,
                                attach_fea=False,
                                input_layer="none",
                                dynamic_depth="truncnorm",
                                is_training=args.is_training)

    def construct(self, f_atoms, f_bonds, a2b, a2a, b2a, b2revb):
        """
        :param f_atoms: the atom features, num_atoms * atom_dim
        :param f_bonds: the bond features, num_bonds * bond_dim
        :param a2b: mapping from atom index to incoming bond indices.
        :param a2a: mapping from atom index to its neighbors. num_atoms * max_num_bonds
        :param b2a: mapping from bond index to the index of the atom the bond is coming from.
        :param b2revb: mapping from bond index to the index of the reverse bond.
        :return:
        """

        init_messages = f_bonds
        init_attached_features = f_atoms
        a2nei = a2b
        a2attached = a2a
        b2a = b2a
        b2revb = b2revb

        if self.atom_messages:
            init_messages = f_atoms
            init_attached_features = f_bonds
            a2nei = a2a
            a2attached = a2b
            b2a = b2a
            b2revb = b2revb

        q = self.mpn_q(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        k = self.mpn_k(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        v = self.mpn_v(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        return q, k, v


class MTBlock(nn.Cell):
    """
    The Multi-headed attention block.
    """

    def __init__(self,
                 args,
                 num_attn_head,
                 input_dim,
                 hidden_size,
                 activation="ReLU",
                 dropout=0.0,
                 bias=True,
                 atom_messages=False,
                 res_connection=False):
        """
        :param num_attn_head: the number of attention head.
        :param input_dim: the input dimension.
        :param hidden_size: the hidden size of the model.
        :param activation: the activation function.
        :param dropout: the dropout ratio
        :param bias: if true: all linear layer contains bias term.
        :param atom_messages: the MPNEncoder type
        :param cuda: if true, the model run with GPU.
        :param res_connection: enables the skip-connection in MTBlock.
        """
        super(MTBlock, self).__init__()
        self.atom_messages = atom_messages
        self.hidden_size = hidden_size
        self.heads = nn.CellList()
        self.input_dim = input_dim
        self.res_connection = res_connection
        self.act_func = get_activation_function(activation)
        self.dropout_layer = nn.Dropout(keep_prob=1.0 - dropout)
        # Note: elementwise_affine has to be consistent with the pre-training phase

        self.layernorm = nn.LayerNorm((self.hidden_size,), begin_norm_axis=-1, begin_params_axis=-1)

        self.wi = nn.Dense(self.input_dim, self.hidden_size, has_bias=bias)
        self.attn = MultiHeadedAttention(h=num_attn_head,
                                         d_model=self.hidden_size,
                                         dropout=dropout)
        self.wo = nn.Dense(self.hidden_size * num_attn_head, self.hidden_size, has_bias=bias)
        self.sublayer = SublayerConnection(self.hidden_size, dropout, self.res_connection)

        heads = []
        for _ in range(num_attn_head):
            head = Head(args, hidden_size=hidden_size, atom_messages=atom_messages)
            heads.append(head)
        self.heads = nn.CellList(heads)

        self.concat = ops.Concat(axis=1)

    def construct(self, batch):
        """
        The block of attention head.
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a2a = batch

        if self.atom_messages:
            # Only add linear transformation in the input feature.
            if f_atoms.shape[1] != self.hidden_size:
                f_atoms = self.wi(f_atoms)
                f_atoms = self.act_func(f_atoms)
                f_atoms = self.layernorm(f_atoms)
                f_atoms = self.dropout_layer(f_atoms)

        else:  # bond messages
            if f_bonds.shape[1] != self.hidden_size:
                f_bonds = self.wi(f_bonds)
                f_bonds = self.act_func(f_bonds)
                f_bonds = self.layernorm(f_bonds)
                f_bonds = self.dropout_layer(f_bonds)

        queries = []
        keys = []
        values = []
        for head in self.heads:
            q, k, v = head(f_atoms, f_bonds, a2b, a2a, b2a, b2revb)
            queries.append(ops.expand_dims(q, 1))
            keys.append(ops.expand_dims(k, 1))
            values.append(ops.expand_dims(v, 1))

        queries = self.concat(queries)
        keys = self.concat(keys)
        values = self.concat(values)

        x_out = self.attn(queries, keys, values)  # multi-headed attention

        x_out = x_out.view(x_out.shape[0], -1)
        x_out = self.wo(x_out)

        x_in = None
        # support no residual connection in MTBlock.
        if self.res_connection:
            if self.atom_messages:
                x_in = f_atoms
            else:
                x_in = f_bonds

        if self.atom_messages:
            f_atoms = self.sublayer(x_in, x_out)
        else:
            f_bonds = self.sublayer(x_in, x_out)

        batch = f_atoms, f_bonds, a2b, b2a, b2revb, a2a

        return batch


class GTransEncoder(nn.Cell):
    """
    Transform Encoder.
    """
    def __init__(self,
                 args,
                 hidden_size,
                 edge_fdim,
                 node_fdim,
                 dropout=0.0,
                 activation="ReLU",
                 num_mt_block=1,
                 num_attn_head=4,
                 atom_emb_output="both",  # options: "none", "atom", "bond", "both"
                 bias=False,
                 res_connection=False):
        """
        :param hidden_size: the hidden size of the model.
        :param edge_fdim: the dimension of additional feature for edge/bond.
        :param node_fdim: the dimension of additional feature for node/atom.
        :param dropout: the dropout ratio
        :param activation: the activation function
        :param num_mt_block: the number of mt block.
        :param num_attn_head: the number of attention head.
        :param atom_emb_output:  enable the output aggregation after message passing.
                                              atom_messages:      True                      False
        -False: no aggregating to atom. output size:     (num_atoms, hidden_size)    (num_bonds, hidden_size)
        -True:  aggregating to atom.    output size:     (num_atoms, hidden_size)    (num_atoms, hidden_size)
        -None:                         same as False
        -"atom":                       same as True
        -"bond": aggragating to bond.   output size:     (num_bonds, hidden_size)    (num_bonds, hidden_size)
        -"both": aggregating to atom&bond. output size:  (num_atoms, hidden_size)    (num_bonds, hidden_size)
                                                         (num_bonds, hidden_size)    (num_atoms, hidden_size)
        :param bias: enable bias term in all linear layers.
        :param res_connection: enables the skip-connection in MTBlock.
        """
        super(GTransEncoder, self).__init__()

        # For the compatibility issue.

        if atom_emb_output == "none":
            atom_emb_output = None

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation
        self.bias = bias
        self.res_connection = res_connection
        self.edge_blocks = nn.CellList()
        self.node_blocks = nn.CellList()

        edge_input_dim = edge_fdim
        node_input_dim = node_fdim
        edge_input_dim_i = edge_input_dim
        node_input_dim_i = node_input_dim

        edge_blocks = []
        node_blocks = []
        for i in range(num_mt_block):
            if i != 0:
                edge_input_dim_i = self.hidden_size
                node_input_dim_i = self.hidden_size
            edge_block = MTBlock(args=args,
                                 num_attn_head=num_attn_head,
                                 input_dim=edge_input_dim_i,
                                 hidden_size=self.hidden_size,
                                 activation=activation,
                                 dropout=dropout,
                                 bias=self.bias,
                                 atom_messages=False,
                                 res_connection=self.res_connection)
            node_block = MTBlock(args=args,
                                 num_attn_head=num_attn_head,
                                 input_dim=node_input_dim_i,
                                 hidden_size=self.hidden_size,
                                 activation=activation,
                                 dropout=dropout,
                                 bias=self.bias,
                                 atom_messages=True,
                                 res_connection=self.res_connection)
            edge_blocks.append(edge_block)
            node_blocks.append(node_block)
        self.edge_blocks = nn.CellList(edge_blocks)
        self.node_blocks = nn.CellList(node_blocks)

        self.atom_emb_output = atom_emb_output

        self.to_atom_transform = EmbeddTransformLayer(to_atom=True,
                                                      hidden_size=self.hidden_size,
                                                      node_fdim=node_fdim,
                                                      edge_fdim=edge_fdim,
                                                      activation=self.activation,
                                                      dropout=self.dropout)

        self.to_bond_transform = EmbeddTransformLayer(to_atom=False,
                                                      hidden_size=self.hidden_size,
                                                      node_fdim=node_fdim,
                                                      edge_fdim=edge_fdim,
                                                      activation=self.activation,
                                                      dropout=self.dropout)

        self.act_func_node = get_activation_function(self.activation)
        self.act_func_edge = get_activation_function(self.activation)

        self.dropout_layer = nn.Dropout(keep_prob=1.0 - self.dropout)

    def construct(self, batch):
        """
        Embedding out.
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a2a = batch

        node_batch = f_atoms, f_bonds, a2b, b2a, b2revb, a2a
        edge_batch = f_atoms, f_bonds, a2b, b2a, b2revb, a2a

        # opt pointwise_feed_forward
        original_f_atoms, original_f_bonds = f_atoms, f_bonds

        # Note: features_batch is not used here.
        for nb in self.node_blocks:  # atom messages. Multi-headed attention
            node_batch = nb(node_batch)
        for eb in self.edge_blocks:  # bond messages. Multi-headed attention
            edge_batch = eb(edge_batch)

        atom_output, _, _, _, _, _ = node_batch  # atom hidden states
        _, bond_output, _, _, _, _ = edge_batch  # bond hidden states

        if self.atom_emb_output is None:
            # output the embedding from multi-head attention directly.
            return atom_output, bond_output

        atom_embeddings, bond_embeddings = None, None
        if self.atom_emb_output == 'atom':
            atom_embeddings = self.to_atom_transform(
                atomwise_input=atom_output,
                bondwise_input=bond_output,
                original_f_atoms=original_f_atoms,
                original_f_bonds=original_f_bonds,
                a2a=a2a,
                a2b=a2b,
                b2a=b2a,
                b2revb=b2revb)

        elif self.atom_emb_output == 'bond':
            bond_embeddings = self.to_bond_transform(
                atomwise_input=atom_output,
                bondwise_input=bond_output,
                original_f_atoms=original_f_atoms,
                original_f_bonds=original_f_bonds,
                a2a=a2a,
                a2b=a2b,
                b2a=b2a,
                b2revb=b2revb)

        else:  # 'both'
            atom_embeddings = self.to_atom_transform(
                atomwise_input=atom_output,
                bondwise_input=bond_output,
                original_f_atoms=original_f_atoms,
                original_f_bonds=original_f_bonds,
                a2a=a2a,
                a2b=a2b,
                b2a=b2a,
                b2revb=b2revb)

            bond_embeddings = self.to_bond_transform(
                atomwise_input=atom_output,
                bondwise_input=bond_output,
                original_f_atoms=original_f_atoms,
                original_f_bonds=original_f_bonds,
                a2a=a2a,
                a2b=a2b,
                b2a=b2a,
                b2revb=b2revb)

        return (atom_embeddings, bond_embeddings)
