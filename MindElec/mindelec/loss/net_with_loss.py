# Copyright 2021 Huawei Technologies Co., Ltd
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
# ==============================================================================
#pylint: disable=C0111
"""
Define net_with_loss cell
"""
from __future__ import absolute_import

import numpy as np
from mindspore import log as logger
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, Parameter


from .losses import get_loss_metric
from .constraints import Constraints
from ..architecture.util import check_mode


def _transfer_tensor_to_tuple(inputs):
    """
    If the input is a tensor, convert it to a tuple. If not, the output is unchanged.
    """
    if isinstance(inputs, Tensor):
        return (inputs,)

    return inputs


class NetWithLoss(nn.Cell):
    r"""
    Encapsulation class of network with loss.

    Args:
        net_without_loss (Cell): The training network without loss definition.
        constraints (Constraints): The constraints function of pde problem.
        loss (Union[str, dict, Cell]): The name of loss function. Defaults: "l2".
        dataset_input_map (dict): The input map of the dataset Defaults: None.
        mtl_weighted_cell (Cell): Losses weighting algorithms based on multi-task learning uncertainty evaluation.
            Default: None.
        latent_vector (Parameter): Tensor of Parameter. The latent vector to encodes the variational parameters in
            governing equation. It will be concated with the sampling data togother as final network inputs.
            Default: None.
        latent_reg (float): The regularization coefficient of latent vector. Default: 0.01.

    Inputs:
        - **inputs** (Tensor) - The input is variable-length argument which contains network inputs.

    Outputs:
        Tensor, a scalar tensor with shape :math:`()`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindelec.loss import Constraints, NetWithLoss
        >>> from mindspore import Tensor, nn
        >>> class Net(nn.Cell):
        ...     def __init__(self, input_dim, output_dim):
        ...         super(NetWithoutLoss, self).__init__()
        ...         self.fc1 = nn.Dense(input_dim, 64)
        ...         self.fc2 = nn.Dense(64, output_dim)
        ...
        ...     def construct(self, *input):
        ...         x = input[0]
        ...         out = self.fc1(x)
        ...         out = self.fc2(out)
        ...         return out
        >>> net = Net(3, 3)
        >>> # For details about how to build the Constraints, please refer to the tutorial
        >>> # document on the official website.
        >>> constraints = Constraints(dataset, pde_dict)
        >>> loss_network = NetWithLoss(net, constraints)
        >>> input = Tensor(np.ones([1000, 3]).astype(np.float32) * 0.01)
        >>> label = Tensor(np.ones([1000, 3]).astype(np.float32))
        >>> output_data = loss_network(input, label)
    """
    def __init__(self, net_without_loss, constraints, loss="l2", dataset_input_map=None,
                 mtl_weighted_cell=None, latent_vector=None, latent_reg=0.01):
        super(NetWithLoss, self).__init__(auto_prefix=False)
        check_mode("NetWithLoss")
        _check_type(net_without_loss, "net_without_loss", nn.Cell, False)
        _check_type(constraints, "constraints", Constraints, False)
        _check_type(loss, "loss", (str, dict, nn.Cell), False)
        _check_type(dataset_input_map, "dataset_input_map", dict)
        _check_type(mtl_weighted_cell, "mtl_weighted_cell", nn.Cell)
        _check_type(latent_vector, "latent_vector", Parameter)
        _check_type(latent_reg, "latent_reg", float)
        self.fn_cell_list = constraints.fn_cell_list
        self.dataset_cell_index_map = constraints.dataset_cell_index_map
        self.dataset_columns_map = constraints.dataset_columns_map
        self.column_index_map = constraints.column_index_map
        self.net_without_loss = net_without_loss
        self.loss_fn_dict = {}
        if dataset_input_map is None:
            logger.info("The input columns was not defined, the first column will be set as input")
            self.dataset_input_map = {}
            for name in self.dataset_columns_map.keys():
                self.dataset_input_map[name] = [self.dataset_columns_map[name][0]]
        else:
            if len(dataset_input_map.keys()) != len(self.dataset_cell_index_map.keys()):
                raise ValueError("Inconsistent dataset input columns info, total datasets number is {}, but only {} "
                                 "sub-dataset's inputs were defined".format(len(self.dataset_cell_index_map.keys()),
                                                                            len(dataset_input_map.keys())))
            input_number = len(dataset_input_map[list(dataset_input_map.keys())[0]])
            for data_name in dataset_input_map.keys():
                if len(dataset_input_map[data_name]) != input_number:
                    raise ValueError("Number of inputs of each dataset should be equal, but got: {}".format(
                        [len(dataset_input_map[key]) for key in dataset_input_map.keys()]))
            self.dataset_input_map = dataset_input_map
        logger.info("Check input columns of each dataset: {}".format(self.dataset_input_map))

        if not isinstance(loss, dict):
            loss_fn = get_loss_metric(loss) if isinstance(loss, str) else loss
            for name in self.dataset_columns_map.keys():
                self.loss_fn_dict[name] = loss_fn
        else:
            for name in self.dataset_columns_map.keys():
                if name in loss.keys():
                    self.loss_fn_dict[name] = get_loss_metric(loss[name]) if isinstance(loss[name], str) \
                        else loss[name]
                else:
                    self.loss_fn_dict[name] = get_loss_metric("l2")
        self.zero = Tensor(np.array([0]).astype(np.float32))
        self.zeros_like = ops.ZerosLike()
        self.reduce_mean = ops.ReduceMean()
        self.pow = ops.Pow()
        self.mtl_cell = mtl_weighted_cell
        self.latent_vector = latent_vector
        self.latent_reg = latent_reg

    def construct(self, *inputs):
        data = _transfer_tensor_to_tuple(inputs)
        loss = {}
        total_loss = self.zero
        for name in self.dataset_columns_map.keys():
            columns_list = self.dataset_columns_map[name]
            input_data = {}
            for column_name in columns_list:
                input_data[column_name] = data[self.column_index_map[column_name]]
            net_input = ()
            for column_name in self.dataset_input_map[name]:
                net_input += (data[self.column_index_map[column_name]],)
            out = self.net_without_loss(*net_input)
            out = _transfer_tensor_to_tuple(out)
            base = self.fn_cell_list[self.dataset_cell_index_map[name]](*out, **input_data)
            temp_loss = self.reduce_mean(self.loss_fn_dict[name](base, self.zeros_like(base)))
            loss[name] = temp_loss
            total_loss += temp_loss

        if self.mtl_cell is not None:
            total_loss = self.mtl_cell(loss.values())

        if self.latent_vector is not None:
            loss_reg = self.latent_reg * self.reduce_mean(self.pow(self.latent_vector, 2))
            loss["reg"] = loss_reg
            total_loss += loss_reg

        loss["total_loss"] = total_loss
        return total_loss


class NetWithEval(nn.Cell):
    r"""
    Encapsulation class of network with loss of eval.

    Args:
        net_without_loss (Cell): The training network without loss definition.
        constraints (Constraints): The constraints function of pde problem.
        loss(Union[str, dict, Cell]): The name of loss function. Defaults: "l2".
        dataset_input_map (dict): The input map of the dataset Defaults: None.

    Inputs:
        - **inputs** (Tensor) - The input is variable-length argument which contains network inputs and label.

    Outputs:
        Tuple, containing a scalar loss Tensor, a network output Tensor of shape :math:`(N, \ldots)` and a label Tensor
        of shape :math:`(N, \ldots)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindelec.loss import Constraints, NetWithEval
        >>> from mindspore import Tensor, nn
        >>> class Net(nn.Cell):
        ...     def __init__(self, input_dim, output_dim):
        ...         super(Net, self).__init__()
        ...         self.fc1 = nn.Dense(input_dim, 64)
        ...         self.fc2 = nn.Dense(64, output_dim)
        ...
        ...     def construct(self, *input):
        ...         x = input[0]
        ...         out = self.fc1(x)
        ...         out = self.fc2(out)
        ...         return out
        >>> net = Net(3, 3)
        >>> # For details about how to build the Constraints, please refer to the tutorial
        >>> # document on the official website.
        >>> constraints = Constraints(dataset, pde_dict)
        >>> loss_network = NetWithEval(net, constraints)
        >>> input = Tensor(np.ones([1000, 3]).astype(np.float32) * 0.01)
        >>> label = Tensor(np.ones([1000, 3]).astype(np.float32))
        >>> output_data = loss_network(input, label)
    """
    def __init__(self, net_without_loss, constraints, loss="l2", dataset_input_map=None):
        super(NetWithEval, self).__init__(auto_prefix=False)
        check_mode("NetWithEval")
        _check_type(net_without_loss, "net_without_loss", nn.Cell, False)
        _check_type(constraints, "constraints", Constraints, False)
        _check_type(loss, "loss", (str, dict, nn.Cell), False)
        _check_type(dataset_input_map, "dataset_input_map", dict)
        self.fn_cell_list = constraints.fn_cell_list
        self.dataset_cell_index_map = constraints.dataset_cell_index_map
        self.dataset_columns_map = constraints.dataset_columns_map
        self.label_key = None
        self.column_index_map = constraints.column_index_map
        for key in self.column_index_map:
            if 'label' in key:
                self.label_key = key
        if self.label_key is None:
            raise NameError("The word 'label' should be in the column name of label.")

        self.net_without_loss = net_without_loss
        self.loss_fn_dict = {}

        if dataset_input_map is None:
            logger.info("The input columns was not defined, the first column will be set as input")
            self.dataset_input_map = {}
            for name in self.dataset_columns_map.keys():
                self.dataset_input_map[name] = [self.dataset_columns_map[name][0]]
        else:
            if len(dataset_input_map.keys()) != len(self.dataset_cell_index_map.keys()):
                raise ValueError("Inconsistent eval dataset input columns info, total datasets number is {}, but only"
                                 " {} sub-dataset's inputs were defined".format(len(self.dataset_cell_index_map.keys()),
                                                                                len(dataset_input_map.keys())))
            input_number = len(dataset_input_map[list(dataset_input_map.keys())[0]])
            for data_name in dataset_input_map.keys():
                if len(dataset_input_map[data_name]) != input_number:
                    raise ValueError("Number of inputs of each eval dataset should be equal, but got: {}".format(
                        [len(dataset_input_map[key]) for key in dataset_input_map.keys()]))
            self.dataset_input_map = dataset_input_map
        logger.info("Check input columns of each eval dataset: {}".format(self.dataset_input_map))

        if not isinstance(loss, dict):
            loss_fn = get_loss_metric(loss) if isinstance(loss, str) else loss
            for name in self.dataset_columns_map.keys():
                self.loss_fn_dict[name] = loss_fn
        else:
            for name in self.dataset_columns_map.keys():
                if name in loss.keys():
                    self.loss_fn_dict[name] = get_loss_metric(loss[name]) if isinstance(loss[name], str) \
                        else loss[name]
                else:
                    self.loss_fn_dict[name] = get_loss_metric("l2")
        self.zero = Tensor(np.array([0]).astype(np.float32))
        self.zeros_like = ops.ZerosLike()
        self.reduce_mean = ops.ReduceMean()

    def construct(self, *inputs):
        data = _transfer_tensor_to_tuple(inputs)
        total_loss = self.zero
        out = 0
        loss = {}
        for name in self.dataset_columns_map.keys():
            columns_list = self.dataset_columns_map[name]
            input_data = {}
            for column_name in columns_list:
                input_data[column_name] = data[self.column_index_map[column_name]]
            net_input = ()
            for column_name in self.dataset_input_map[name]:
                net_input += (data[self.column_index_map[column_name]],)
            out = self.net_without_loss(*net_input)
            out = _transfer_tensor_to_tuple(out)
            base = self.fn_cell_list[self.dataset_cell_index_map[name]](*out, **input_data)
            temp_loss = self.reduce_mean(self.loss_fn_dict[name](base, self.zeros_like(base)))
            loss[name] = temp_loss
            total_loss += temp_loss
        loss["total_loss"] = total_loss
        return loss["total_loss"], out[0], data[self.column_index_map[self.label_key]]

def _check_type(param, param_name, param_type, can_be_none=True):
    """check type"""
    if can_be_none:
        if param is not None and not isinstance(param, param_type):
            raise TypeError("The type of {} should be instance of {}, but got {}".format(
                param_name, param_type, type(param)))
    else:
        if not isinstance(param, param_type):
            raise TypeError("The type of {} should be instance of {}, but got {}".format(
                param_name, param_type, type(param)))
