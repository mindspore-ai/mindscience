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
Metrics for collective variables
"""

from typing import Union
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.communication.management as D
import mindspore.nn as nn
import mindspore.numpy as mnp

from mindspore import Parameter, Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.nn import Metric as _Metric

from ..colvar import Colvar, get_colvar
from ..function import Units


def get_metrics(metrics: Union[dict, set]) -> dict:
    """
    Get metrics used in analysis.

    Args:
        metrics (Union[dict, set]): Dict or set of Metric or Colvar to be evaluated by the model
                                    during MD running or analysis.

    Returns:
        dict, the key is metric name, the value is class instance of metric method.

    Raises:
        TypeError: If the type of argument `metrics` is not ``None``, dict or set.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from sponge.colvar import Distance
        >>> from sponge.metrics import get_metrics
        >>> cv = Distance([0,1])
        >>> metric = get_metrics({"distance": cv})
        >>> coordinate = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        >>> metric.update(coordinate)
        >>> print(metric.eval())
        [1.]
    """
    if metrics is None:
        return metrics

    if isinstance(metrics, dict):
        for name, metric in metrics.items():
            if not isinstance(name, str):
                raise TypeError(f"The key in 'metrics' must be string and but got key: {type(name)}.")
            if isinstance(metric, Colvar):
                metrics[name] = MetricCV(metric)
            elif not isinstance(metric, Metric):
                raise TypeError(f"The value in 'metrics' must be Metric or Colvar, but got: {type(metric)}.")
        return metrics

    if isinstance(metrics, set):
        out_metrics = {}
        for metric in metrics:
            if not isinstance(metric, Colvar):
                raise TypeError(f"When 'metrics' is set, the type of the value in 'metrics must be Colvar, '"
                                f"but got: {type(metric)}")
            out_metrics[metric.name] = MetricCV(metric)
        return out_metrics

    raise TypeError("For 'get_metrics', the argument 'metrics' must be None, dict or set, "
                    "but got {}".format(metrics))


class Metric(_Metric):
    """Metric is fundamental tool used to assess the state and performance of a simulation system. Which provides a mechanism to track the changes in various physical quantities within the simulation system. The base class of Metrics defines a set of methods that are used to update the state information of the simulation system and to calculate the corresponding metrics."""

    def update(self,
               coordinate: Tensor,
               pbc_box: Tensor = None,
               energy: Tensor = None,
               force: Tensor = None,
               potentials: Tensor = None,
               total_bias: Tensor = None,
               biases: Tensor = None,
               ):
        """
        update the state information of the simulation system.

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
                                    Position coordinate of atoms in system.
            pbc_box (Tensor, optional):       Tensor of shape (B, D). Data type is float.
                                    Tensor of PBC box. Default: ``None``.
            energy (Tensor, optional):        Tensor of shape (B, 1). Data type is float.
                                    Total energy of the simulation system. Default: ``None``.
            force (Tensor, optional):         Tensor of shape (B, A, D). Data type is float.
                                    Force on each atoms of the simulation system. Default: ``None``.
            potentials (Tensor, optional):    Tensor of shape (B, U). Data type is float.
                                    All potential energies. Default: ``None``.
            total_bias (Tensor, optional):    Tensor of shape (B, 1). Data type is float.
                                    Total bias energy for reweighting. Default: ``None``.
            biases (Tensor, optional):        Tensor of shape (B, V). Data type is float
                                    All bias potential energies. Default: ``None``.

        Note:
            - B:  Batchsize, i.e. number of walkers in simulation.
            - A:  Number of atoms of the simulation system.
            - D:  Dimension of the space of the simulation system. Usually is 3.
            - U:  Number of potential energies.
            - V:  Number of bias potential energies.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> from sponge.metrics import Metric
            >>> net = Metric()
        """
        #pylint: disable=unused-argument
        raise NotImplementedError


class MetricCV(Metric):
    """Metric for collective variables (CVs)"""

    def __init__(self,
                 colvar: Colvar,
                 ):

        super().__init__()
        self.colvar = get_colvar(colvar)
        self._value = None

    @property
    def shape(self) -> tuple:
        return self.colvar.shape

    @property
    def ndim(self) -> int:
        return self.colvar.ndim

    @property
    def dtype(self) -> type:
        return self.colvar.dtype

    def get_unit(self, units: Units = None) -> str:
        r"""Return unit of the collective variables.

        Args:
            units (Units, optional):  Units of the collective variables. Default: ``None``.

        """
        return self.colvar.get_unit(units)

    def clear(self):
        self._value = 0

    def update(self,
               coordinate: Tensor,
               pbc_box: Tensor = None,
               energy: Tensor = None,
               force: Tensor = None,
               potentials: Tensor = None,
               total_bias: Tensor = None,
               biases: Tensor = None,
               ):
        """
        update the state information of the system.

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
                                    Position coordinate of atoms in system.
            pbc_box (Tensor, optional):       Tensor of shape (B, D). Data type is float.
                                    Tensor of PBC box. Default: ``None``.
            energy (Tensor, optional):        Tensor of shape (B, 1). Data type is float.
                                    Total potential energy of the simulation system. Default: ``None``.
            force (Tensor, optional):         Tensor of shape (B, A, D). Data type is float.
                                    Force on each atoms of the simulation system. Default: ``None``.
            potentials (Tensor, optional):    Tensor of shape (B, U). Data type is float.
                                    Original potential energies from force field. Default: ``None``.
            total_bias (Tensor, optional):    Tensor of shape (B, 1). Data type is float.
                                    Total bias energy for reweighting. Default: ``None``.
            biases (Tensor, optional):        Tensor of shape (B, V). Data type is float.
                                    Original bias potential energies from bias functions. Default: ``None``.

        Note:
            - B:  Batchsize, i.e. number of walkers in simulation.
            - A:  Number of atoms of the simulation system.
            - D:  Dimension of the space of the simulation system. Usually is 3.
            - U:  Number of potential energies.
            - V:  Number of bias potential energies.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> from mindspore import Tensor
            >>> from sponge.colvar import Distance
            >>> from sponge.metrics import MetricCV
            >>> cv = Distance([0,1])
            >>> coordinate = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            >>> metric = MetricCV(cv)
            >>> metric.update(coordinate)
            >>> print(metric.eval())
            [1.]
        """
        #pylint: disable=unused-argument

        colvar = self.colvar(coordinate, pbc_box)

        self._value = self._convert_data(colvar)

    def eval(self):
        return self._value


class Average(Metric):
    """Average of collective variables (CVs)"""

    def __init__(self,
                 colvar: Colvar,
                 ):
        super().__init__()

        self.colvar = get_colvar(colvar)

        self._value = None
        self._average = None
        self._weights = 0

    def clear(self):
        self._value = 0
        self._weights = 0

    def update(self,
               coordinate: Tensor,
               pbc_box: Tensor = None,
               energy: Tensor = None,
               force: Tensor = None,
               potentials: Tensor = None,
               total_bias: Tensor = None,
               biases: Tensor = None,
               ):
        """

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
                                    Position coordinate of atoms in system.
            pbc_box (Tensor, optional):       Tensor of shape (B, D). Data type is float.
                                    Tensor of PBC box. Default: ``None``.
            energy (Tensor, optional):        Tensor of shape (B, 1). Data type is float.
                                    Total potential energy of the simulation system. Default: ``None``.
            force (Tensor, optional):         Tensor of shape (B, A, D). Data type is float.
                                    Force on each atoms of the simulation system. Default: ``None``.
            potentials (Tensor, optional):    Tensor of shape (B, U). Data type is float.
                                    Original potential energies from force field. Default: ``None``.
            total_bias (Tensor, optional):    Tensor of shape (B, 1). Data type is float.
                                    Total bias energy for reweighting. Default: ``None``.
            biases (Tensor, optional):        Tensor of shape (B, V). Data type is float.
                                    Original bias potential energies from bias functions. Default: ``None``.

        Note:
            - B:  Batchsize, i.e. number of walkers in simulation.
            - A:  Number of atoms of the simulation system.
            - D:  Dimension of the space of the simulation system. Usually is 3.
            - U:  Number of potential energies.
            - V:  Number of bias potential energies.
        """
        #pylint: disable=unused-argument

        colvar = self.colvar(coordinate, pbc_box)

        self._average += self._convert_data(colvar)
        self._weights += 1

    def eval(self):
        return self._average / self._weights


class BalancedMSE(nn.Cell):
    r"""
    Balanced MSE error
    Compute Balanced MSE error between the prediction and the ground truth
    to solve unbalanced labels in regression task.

    Refer to `Ren, Jiawei, et al. 'Balanced MSE for Imbalanced Visual Regression' <https://arxiv.org/abs/2203.16427>`_.

    .. math::
        L =-\log \mathcal{N}\left(\boldsymbol{y} ; \boldsymbol{y}_{\text {pred }},
        \sigma_{\text {noise }}^{2} \mathrm{I}\right)
        +\log \sum_{i=1}^{N} p_{\text {train }}\left(\boldsymbol{y}_{(i)}\right)
        \cdot \mathcal{N}\left(\boldsymbol{y}_{(i)} ; \boldsymbol{y}_{\text {pred }},
        \sigma_{\text {noise }}^{2} \mathrm{I}\right)

    Args:
        first_break (float):    The begin value of bin.
        last_break (float):     The end value of bin.
        num_bins (int):         The bin numbers.
        beta (float, optional):           The moving average coefficient, default: ``0.99``.
        reducer_flag (bool, optional):    Whether to aggregate the label values of multiple devices, default: ``False``.

    Inputs:
        - **prediction** (Tensor) - Predict values, shape is :math:`(batch\_size, ndim)`.
        - **target** (Tensor) - Label values, shape is :math:`(batch\_size, ndim)`.

    Outputs:
        Tensor, shape is :math:`(batch\_size, ndim)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from sponge.metrics import BalancedMSE
        >>> from mindspore import Tensor
        >>> net = BalancedMSE(0, 1, 20)
        >>> prediction = Tensor(np.random.randn(32, 10).astype(np.float32))
        >>> target = Tensor(np.random.randn(32, 10).astype(np.float32))
        >>> out = net(prediction, target)
        >>> print(out.shape)
        (32, 10)
    """

    def __init__(self, first_break, last_break, num_bins, beta=0.99, reducer_flag=False):
        super(BalancedMSE, self).__init__()
        self.beta = beta
        self.first_break = first_break
        self.last_break = last_break
        self.num_bins = num_bins

        self.breaks = mnp.linspace(self.first_break, self.last_break, self.num_bins)
        self.width = self.breaks[1] - self.breaks[0]
        bin_width = 2
        start_n = 1
        stop = self.num_bins * 2
        centers = mnp.divide(mnp.arange(start=start_n, stop=stop, step=bin_width), num_bins * 2.0)
        self.centers = centers/(self.last_break-self.first_break) + self.first_break

        self.log_noise_scale = Parameter(Tensor([0.], mstype.float32))
        self.p_bins = Parameter(Tensor(np.ones((self.num_bins)) / self.num_bins, dtype=mstype.float32), \
                                   name='p_bins', requires_grad=False)

        self.softmax = nn.Softmax(-1)
        self.zero = Tensor([0.])

        self.onehot = nn.OneHot(depth=self.num_bins)
        self.reducer_flag = reducer_flag
        if self.reducer_flag:
            self.allreduce = P.AllReduce()
            self.device_num = D.get_group_size()

    def construct(self, prediction, target):
        """construct"""
        p_bins = self._compute_p_bins(prediction)

        log_sigma2 = self.log_noise_scale * 1.
        log_sigma2 = 5. * P.Tanh()(log_sigma2 / 5.)
        sigma2 = mnp.exp(log_sigma2) + 0.25 * self.width
        tau = 2. * sigma2
        a = - F.square(prediction - target) / tau

        ndim = prediction.ndim
        y_bins = mnp.reshape(self.centers * 1., ndim * (1,) + (-1,))
        b_term = - F.square(mnp.expand_dims(prediction, -1) - y_bins) / tau

        p_clip = mnp.clip(p_bins, 1e-8, 1 - 1e-8)
        log_p = mnp.log(p_clip)
        log_p = mnp.reshape(log_p, ndim * (1,) + (-1,))

        b_term += log_p
        b = nn.ReduceLogSumExp(-1, False)(b_term)

        err = -a + b
        return err

    def _compute_p_bins(self, y_gt):
        """compute bins"""
        ndim = y_gt.ndim
        breaks = mnp.reshape(self.breaks, (1,) * ndim + (-1,))
        y_gt = mnp.expand_dims(y_gt, -1)

        y_bins = (y_gt > breaks).astype(mstype.float32)
        y_bins = P.ReduceSum()(y_bins, -1).astype(mstype.int32)
        p_gt = self.onehot(y_bins)

        p_gt = P.Reshape()(p_gt, (-1, self.num_bins))
        p_bins = P.ReduceMean()(p_gt, 0)
        if self.reducer_flag:
            p_bins = self.allreduce(p_bins) / self.device_num

        p_bins = self.beta * self.p_bins + (1 - self.beta) * p_bins
        P.Assign()(self.p_bins, p_bins)

        return p_bins


class MultiClassFocal(nn.Cell):
    r"""Focal error for multi-class classifications.
    Compute the multiple classes focal error between `prediction` and the ground truth `target`.

    Refer to `Lin, Tsung-Yi, et al. 'Focal loss for dense object detection' <https://arxiv.org/abs/1708.02002>`_ .

    Args:
        num_class (int):        The class numbers.
        beta (float, optional):           The moving average coefficient, default: ``0.99``.
        gamma (float, optional):          The hyperparameters, default: ``2.0``.
        e (float, optional):              The proportion of focal loss, default: ``0.1``.
        neighbors(int, optional):         The neighbors to be mask in the target, default ``2``.
        not_focal (bool, optional):       Whether focal loss, default: ``False``.
        reducer_flag (bool, optional):    Whether to aggregate the label values of multiple devices, default: ``False``.

    Inputs:
        - **prediction** (Tensor) - Predict values, shape is :math:`(batch\_size, ndim)`.
        - **target** (Tensor) - Label values, shape is :math:`(batch\_size, ndim)`.

    Outputs:
        Tensor, shape is :math:`(batch\_size, )`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from sponge.metrics import MultiClassFocal
        >>> net = MultiClassFocal(10)
        >>> prediction = Tensor(np.random.randn(32, 10).astype(np.float32))
        >>> target = Tensor(np.random.randn(32, 10).astype(np.float32))
        >>> out = net(prediction, target)
        >>> print(out.shape)
        (32,)
    """

    def __init__(self, num_class, beta=0.99, gamma=2., e=0.1, neighbors=2, not_focal=False, reducer_flag=False):
        super(MultiClassFocal, self).__init__()
        self.num_class = num_class
        self.beta = beta
        self.gamma = gamma
        self.e = e
        self.neighbors = neighbors
        self.not_focal = not_focal

        neighbor_mask = np.ones((self.num_class, self.num_class))
        neighbor_mask = neighbor_mask - np.triu(neighbor_mask, neighbors) - np.tril(neighbor_mask, -neighbors)
        neighbor_mask = neighbor_mask / (np.sum(neighbor_mask, axis=-1, keepdims=True) + 1e-10)
        self.neighbor_mask = Tensor(neighbor_mask, mstype.float32)

        self.class_weights = Parameter(Tensor(np.ones((self.num_class)) / self.num_class, dtype=mstype.float32), \
                                          name='class_weights', requires_grad=False)

        self.softmax = nn.Softmax(-1)
        self.cross_entropy = P.SoftmaxCrossEntropyWithLogits()
        self.zero = Tensor([0.])

        self.reducer_flag = reducer_flag
        if self.reducer_flag:
            self.allreduce = P.AllReduce()

    def construct(self, prediction, target):
        """construct"""
        prediction_tensor = self.softmax(prediction)

        zeros = mnp.zeros_like(prediction_tensor)
        one_minus_p = mnp.where(target > 1e-5, target - prediction_tensor, zeros)
        ft = -1 * mnp.power(one_minus_p, self.gamma) * mnp.log(mnp.clip(prediction_tensor, 1e-8, 1.0))

        classes_num = self._compute_classes_num(target)
        total_num = mnp.sum(classes_num)

        classes_w_t1 = total_num / classes_num
        sum_ = mnp.sum(classes_w_t1)
        classes_w_t2 = classes_w_t1 / sum_
        classes_w_tensor = F.cast(classes_w_t2, mstype.float32)

        weights = self.beta * self.class_weights + (1 - self.beta) * classes_w_tensor
        P.Assign()(self.class_weights, weights)

        classes_weight = mnp.broadcast_to(mnp.expand_dims(weights, 0), target.shape)
        alpha = mnp.where(target > zeros, classes_weight, zeros)

        balanced_fl = alpha * ft
        balanced_fl = mnp.sum(balanced_fl, -1)

        labels = P.MatMul()(target, self.neighbor_mask)
        xent, _ = self.cross_entropy(prediction, target)

        final_loss = (1 - self.e) * balanced_fl + self.e * xent

        if self.not_focal:
            softmax_xent, _ = self.cross_entropy(prediction, labels)
            final_loss = (1 - self.e) * softmax_xent + self.e * xent

        return final_loss

    def _compute_classes_num(self, target):
        "get global classes number"
        classes_num = mnp.sum(target, 0)
        if self.reducer_flag:
            classes_num = self.allreduce(classes_num)
        classes_num = F.cast(classes_num, mstype.float32)
        classes_num += 1.
        return classes_num


class BinaryFocal(nn.Cell):
    r"""
    Focal error for Binary classifications.
    Compute the binary classes focal error between `prediction` and the ground truth `target`.

    Refer to `Lin, Tsung-Yi, et al. 'Focal loss for dense object detection' <https://arxiv.org/abs/1708.02002>`_ .

    .. math::
        \mathrm{FL}\left(p_{\mathrm{t}}\right)=-\alpha_{\mathrm{t}}\left(1-p_{\mathrm{t}}\right)^{\gamma}
        \log \left(p_{\mathrm{t}}\right)

    Args:
        alpha (float, optional):            The weight of cross entropy, default: ``0.25``.
        gamma (float, optional):          The hyperparameters, modulating loss from hard to easy, default: ``2.0``.
        feed_in (bool, optional):         Whether to convert prediction, default: ``False``.
        not_focal (bool, optional):       Whether focal loss, default: ``False``.

    Inputs:
        - **prediction** (Tensor) - Predict values, shape is :math:`(batch\_size, ndim)`.
        - **target** (Tensor) - Label values, shape is :math:`(batch\_size, ndim)`.

    Outputs:
        Tensor, shape is :math:`(batch\_size,)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from sponge.metrics import BinaryFocal
        >>> net = BinaryFocal()
        >>> prediction = Tensor(np.random.randn(32, 10).astype(np.float32))
        >>> target = Tensor(np.random.randn(32, 10).astype(np.float32))
        >>> out = net(prediction, target)
        >>> print(out.shape)
        (32,)
    """

    def __init__(self, alpha=0.25, gamma=2., feed_in=False, not_focal=False):
        super(BinaryFocal, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.feed_in = feed_in
        self.not_focal = not_focal

        self.cross_entropy = P.BinaryCrossEntropy(reduction='none')
        self.sigmoid = P.Sigmoid()
        self.epsilon = 1e-8

    def construct(self, prediction, target):
        """construct"""
        epsilon = self.epsilon
        target = F.cast(target, mstype.float32)
        probs = F.cast(prediction, mstype.float32)
        if self.feed_in:
            probs = self.sigmoid(prediction)
        else:
            prediction = self._convert(prediction)

        ones_tensor = mnp.ones_like(target)
        positive_pt = mnp.where(target > 1e-5, probs, ones_tensor)
        negative_pt = mnp.where(target < 1e-5, 1 - probs, ones_tensor)

        focal_loss = -self.alpha * mnp.power(1 - positive_pt, self.gamma) * \
                     mnp.log(mnp.clip(positive_pt, epsilon, 1.)) - (1 - self.alpha) * \
                     mnp.power(1 - negative_pt, self.gamma) * mnp.log(mnp.clip(negative_pt, epsilon, 1.))
        focal_loss *= 2.

        if self.not_focal:
            focal_loss = self.cross_entropy(prediction, target, ones_tensor)

        return focal_loss

    def _convert(self, probs):
        """convert function"""
        probs = mnp.clip(probs, 1e-5, 1. - 1e-5)
        prediction = mnp.log(probs / (1 - probs))
        return prediction
