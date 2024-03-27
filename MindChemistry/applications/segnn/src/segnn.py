# Copyright 2024 Huawei Technologies Co., Ltd
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
SEGNN module
"""
from mindspore import nn, ops
from mindchemistry.graph.graph import AggregateNodeToGlobal, AggregateEdgeToNode

from src.inspector import Inspector
from src.instance_norm import InstanceNorm
from src.o3_building_blocks import O3TensorProduct, O3TensorProductSwishGate


class SEGNN(nn.Cell):
    """Steerable E(3) equivariant message passing network"""

    def __init__(
            self,
            input_irreps,
            hidden_irreps,
            output_irreps,
            edge_attr_irreps,
            node_attr_irreps,
            num_layers,
            norm=None,
            pool="avg",
            task="graph",
            additional_message_irreps=None,
            dtype=None,
            ncon_dtype=None
    ):
        super().__init__()

        self.task = task

        self.embedding_layer = O3TensorProduct(
            input_irreps, hidden_irreps, node_attr_irreps, dtype, ncon_dtype
        )
        # Message passing layers.
        layers = []
        for _ in range(num_layers):
            layers.append(
                SEGNNLayer(
                    hidden_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    node_attr_irreps,
                    norm=norm,
                    additional_message_irreps=additional_message_irreps,
                    dtype=dtype,
                    ncon_dtype=ncon_dtype
                )
            )
        self.layers = nn.CellList(layers)

        # Prepare for output irreps, since the attrs will disappear after pooling
        if task == "graph":
            pooled_irreps = (
                (output_irreps * hidden_irreps.num_irreps).simplify().sort().irreps
            )
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, node_attr_irreps, dtype, ncon_dtype
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, pooled_irreps, node_attr_irreps, dtype, ncon_dtype
            )
            self.post_pool1 = O3TensorProductSwishGate(pooled_irreps, pooled_irreps, dtype=dtype, ncon_dtype=ncon_dtype)
            self.post_pool2 = O3TensorProduct(pooled_irreps, output_irreps,
                                              dtype=dtype, ncon_dtype=ncon_dtype)
            self.init_pooler(pool)
        elif task == "node":
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, node_attr_irreps, dtype, ncon_dtype
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, output_irreps, node_attr_irreps, dtype, ncon_dtype
            )
        else:
            raise ValueError(f"Task type {task} not support!")

    def init_pooler(self, pool):
        """Initialise pooling mechanism"""
        if pool in ["add", "sum", "avg", "mean"]:
            self.pooler = AggregateNodeToGlobal(pool)
        else:
            raise ValueError(f"Aggregate type {pool} not support!")

    def construct(self, x, node_attr, edge_attr, edge_index, edge_dist,
                  batch, node_mask, edge_mask, batch_size):
        """
        SEGNN forward compute
        """
        x = self.embedding_layer(x, node_attr, node_mask)

        # Pass messages
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, node_attr,
                      batch, edge_dist, batch_size, node_mask, edge_mask)

        x = self.pre_pool1(x, node_attr, node_mask)
        x = self.pre_pool2(x, node_attr, node_mask)

        if self.task == "graph":
            x = self.pooler(x, batch, dim_size=batch_size, mask=node_mask)
            x = self.post_pool1(x)
            x = self.post_pool2(x)
        return x


class SEGNNLayer(nn.Cell):
    """E(3) equivariant message passing layer."""
    special_args = {
        'edge_index', 'adj_t', 'edge_index_i', 'edge_index_j', 'size',
        'size_i', 'size_j', 'ptr', 'index', 'dim_size'
    }

    def __init__(
            self,
            input_irreps,
            hidden_irreps,
            edge_attr_irreps,
            node_attr_irreps,
            norm=None,
            additional_message_irreps=None,
            dtype=None,
            ncon_dtype=None
    ):
        super().__init__()

        self.aggr = "add"
        self.node_dim = -2
        self.decomposed_layers = 1
        self.flow = "source_to_target"
        self.hidden_irreps = hidden_irreps

        dim = 1 if self.flow == 'source_to_target' else 0
        self.scatter = AggregateEdgeToNode(self.aggr, dim)

        self.inspector = Inspector(self)
        self.inspector.inspect(self.message)
        self.inspector.inspect(self.aggregate, pop_first=True)
        self.inspector.inspect(self.update, pop_first=True)
        self.__user_args__ = tuple(self.inspector.keys(
            ['message', 'aggregate', 'update']).difference(self.special_args))
        self.func_parms = {}
        self.func_parms['message'] = self.orderdict_trans('message')
        self.func_parms['aggregate'] = self.orderdict_trans('aggregate')
        self.func_parms['update'] = self.orderdict_trans('update')

        message_input_irreps = (input_irreps + input_irreps + additional_message_irreps).simplify()
        update_input_irreps = (input_irreps + hidden_irreps).simplify()

        self.message_layer_1 = O3TensorProductSwishGate(
            message_input_irreps, hidden_irreps, edge_attr_irreps, dtype, ncon_dtype
        )
        self.message_layer_2 = O3TensorProductSwishGate(
            hidden_irreps, hidden_irreps, edge_attr_irreps, dtype, ncon_dtype
        )
        self.update_layer_1 = O3TensorProductSwishGate(
            update_input_irreps, hidden_irreps, node_attr_irreps, dtype, ncon_dtype
        )
        self.update_layer_2 = O3TensorProduct(
            hidden_irreps, hidden_irreps, node_attr_irreps, dtype, ncon_dtype
        )

        self.norm = norm
        if norm == "instance":
            self.feature_norm = InstanceNorm(self.hidden_irreps, dtype=dtype)
        else:
            raise ValueError(f"{norm} type not support!")

    def construct(
            self,
            x,
            edge_index,
            edge_attr,
            node_attr,
            batch,
            additional_message_features,
            batch_size,
            node_mask,
            edge_mask
    ):
        """Propagate messages along edges"""
        x = self.propagate(
            edge_index,
            x=x,
            node_attr=node_attr,
            node_mask=node_mask,
            edge_attr=edge_attr,
            edge_mask=edge_mask,
            additional_message_features=additional_message_features,
        )
        x = self.feature_norm(x, batch, batch_size, node_mask)
        return x

    def propagate(self, edge_index, size=None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor):
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`Tensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`Tensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        decomposed_layers = self.decomposed_layers
        size = self.__check_input__(edge_index, size)

        out = None
        for _ in range(decomposed_layers):
            coll_dict = self.__collect__(self.__user_args__, edge_index,
                                         size, kwargs)

            msg_kwargs = self.get_func_attr('message', coll_dict)
            out = self.message(**msg_kwargs)

            aggr_kwargs = self.get_func_attr('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.get_func_attr('update', coll_dict)
            out = self.update(out, **update_kwargs)
        return out

    def message(self, x_i, x_j, edge_attr, additional_message_features, edge_mask):
        """Create messages"""
        if additional_message_features is None:
            x_input = ops.concat((x_i, x_j), axis=-1)
        else:
            x_input = ops.concat((x_i, x_j, additional_message_features), axis=-1)

        x_input = x_input * edge_mask.reshape(-1, 1)
        message = self.message_layer_1(x_input, edge_attr, edge_mask)
        message = self.message_layer_2(message, edge_attr, edge_mask)
        return message

    def aggregate(self, inputs, index, dim_size=-1):
        """Aggregates messages"""
        return self.scatter(inputs, index, dim_size=dim_size)

    def update(self, message, x, node_attr, node_mask):
        """Update note features"""
        x_input = ops.concat((x, message), axis=-1)
        update = self.update_layer_1(x_input, node_attr, node_mask)
        update = self.update_layer_2(update, node_attr, node_mask)
        x += update
        return x

    def orderdict_trans(self, func_name):
        out = {}
        params = self.inspector.params
        for key, _ in params[func_name].items():
            out[key] = ()
        return out

    def get_func_attr(self, func_name, kwargs):
        out = {}
        for key in self.func_parms[func_name].keys():
            data = kwargs.get(key, None)
            if data is not None:
                out[key] = data
        return out

    def __check_input__(self, edge_index, size):
        assert edge_index.ndim == 2
        assert edge_index.shape[0] == 2
        the_size = [None, None]
        if size is not None:
            the_size[0] = size[0]
            the_size[1] = size[1]
        return the_size

    def __collect__(self, args, edge_index, size, kwargs):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        out = {}
        for arg in args:
            arg_len = len(arg)
            tmp = arg[arg_len-2] + arg[arg_len-1]
            if tmp not in ['_i', '_j']:
                out[arg] = kwargs.get(arg)
            else:
                dim = 0 if tmp == '_j' else 1
                kk = ''
                for c_i in range(arg_len-2):
                    kk += arg[c_i]
                data = kwargs.get(kk)
                size = self.__set_size__(size, dim, data)
                data = self.__lift__(data, edge_index, j if tmp == '_j' else i)
                out[arg] = data

        out['index'] = edge_index
        out['edge_index'] = edge_index
        out['edge_index_i'] = edge_index[i]
        out['edge_index_j'] = edge_index[j]
        out['size'] = size
        out['size_i'] = size[i]
        out['size_j'] = size[j]
        out['dim_size'] = out['size_i']
        return out

    def __set_size__(self, size, dim, src):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.shape[self.node_dim]
        elif the_size != src.shape[self.node_dim]:
            raise ValueError(
                (f'Encountered tensor with size {src.shape[self.node_dim]} in '
                 f'dimension {self.node_dim}, but expected size {the_size}.')
            )
        return size

    def __lift__(self, src, edge_index, dim):
        index = edge_index[dim]
        return src.index_select(self.node_dim, index)
