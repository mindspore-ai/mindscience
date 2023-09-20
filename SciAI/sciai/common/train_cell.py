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
# ==============================================================================
"""train_cell"""
import os
from numbers import Number

import mindspore as ms
from mindspore import nn, ops, amp
from mindspore.ops import zeros_like, ones_like

from sciai.utils import time_second, time_str, print_log
from sciai.utils.check_utils import to_tuple, _batch_check_type, _check_value_in


class TrainCellWithCallBack:
    r"""
    TrainOneStepCell with callbacks, which can handle multi-losses. Callbacks can be as follows:

    1.loss: print loss(es).
    2.time: print time spent during steps, and time spent from start.
    3.ckpt: save checkpoint.

    Args:
        network (Cell): The training network. The network supports multi-outputs.
        optimizer (Cell): Optimizer for updating the network parameters.
        loss_interval (int): Step interval to print loss. if 0, it wouldn't print loss. Default: 1.
        time_interval (int): Step interval to print time. if 0, it wouldn't print time. Default: 0.
        ckpt_interval (int): Epoch interval to save checkpoint, calculated according to batch_num. If 0,
            it wouldn't save checkpoint. Default: 0.
        loss_names (Union(str, tuple[str], list[str])): Loss names in order of network outputs. It can accept n or n+1
            strings, where n is the count of network outputs. If n, each string corresponds to the loss in the same
            position; if n + 1, the first loss name represents the sum of all outputs. Default:("loss",).
        batch_num (int): How many batches per epoch. Default: 1.
        grad_first (bool): If True, only the first output of the network would participate in the gradient
            descent. Otherwise, the sum of all outputs of the network would be taken into account. Default: False.
        amp_level (str): Mixed precision level, which supports ["O0", "O1", "O2", "O3"]. Default: "O0".
        ckpt_dir (str): Checkpoints saving path. Default: "./checkpoints".
        clip_grad (bool): Whether clip grad or not. Default: False.
        clip_norm (Union(float, int)): The clipping ratio, it should be greater than 0. Only enabled when `clip_grad`
            is True. Default: 1e-3.
        model_name (str): Model name which influences the checkpoint filename.

    Inputs:
        - **\*args** (tuple[Tensor]) - Tuple of input tensors of the network.

    Outputs:
        Union[Tensor, tuple[Tensor]], Tensor(s) of the loss(es).

    Raises:
        TypeError: If the input parameters are not of required types.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn, ops
        >>> from sciai.common import TrainCellWithCallBack
        >>> class LossNet(nn.Cell):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.dense1 = nn.Dense(2, 1)
        >>>         self.dense2 = nn.Dense(2, 1)
        >>>     def construct(self, x):
        >>>         loss1 = self.dense1(x).sum()
        >>>         loss2 = self.dense2(x).sum()
        >>>         return loss1, loss2
        >>> loss_net = LossNet()
        >>> optimizer = nn.Adam(loss_net.trainable_params(), 1e-2)
        >>> train_net = TrainCellWithCallBack(loss_net, optimizer, time_interval=3, loss_interval=1, ckpt_interval=5,
        >>>                                   ckpt_dir='.', loss_names=("total loss", "loss1", "loss2"))
        >>> x = ops.ones((3, 2), ms.float32)
        >>> for epoch in range(8):
        >>>     loss1, loss2 = train_net(x)
        step: 0, loss1: 0.07256523, loss2: 0.010363013, interval: 3.132981061935425s, total: 3.132981061935425s,
            checkpoint saved at: ./model_iter_0_2000-12-31-23-59-59.ckpt
        step: 1, loss1: 0.06356523, loss2: 0.0013630127
        step: 2, loss1: 0.054565262, loss2: 0.007636956
        step: 3, loss1: 0.04556533, loss2: 0.00999487, interval: 0.01753377914428711s, total: 3.150514841079712s
        step: 4, loss1: 0.036565356, loss2: 0.0090501215
        step: 5, loss1: 0.027565379, loss2: 0.0061383317, checkpoint saved at: ./model_iter_5_2000-12-31-23-59-59.ckpt
        step: 6, loss1: 0.018565409, loss2: 0.0019272038, interval: 0.02319502830505371s, total: 3.1737098693847656s
        step: 7, loss1: 0.00956542, loss2: 0.0032018598
    """

    def __init__(self, network, optimizer, loss_interval=1, time_interval=0, ckpt_interval=0, loss_names=("loss",),
                 batch_num=1, grad_first=False, amp_level="O0", ckpt_dir="./checkpoints", clip_grad=False,
                 clip_norm=1e-3, model_name="model"):
        check_type_dict = {
            "network": (network, nn.Cell),
            "optimizer": (optimizer, nn.Cell),
            "loss_interval": (loss_interval, int),
            "time_interval": (time_interval, int),
            "ckpt_interval": (ckpt_interval, int),
            "loss_names": (loss_names, (str, tuple, list)),
            "batch_num": (batch_num, int),
            "grad_first": (grad_first, bool),
            "ckpt_dir": (ckpt_dir, str),
            "clip_grad": (clip_grad, bool),
            "clip_norm": (clip_norm, (int, float)),
            "amp_level": (amp_level, str),
            "model_name": (model_name, str)
        }
        _batch_check_type(check_type_dict)
        _check_value_in(amp_level, "amp_level", ("O0", "O1", "O2", "O3"))
        self.loss_names = to_tuple(loss_names)
        self.batch_num, self.grad_first, self.ckpt_dir, self.amp_level, self.model_name \
            = batch_num, grad_first, ckpt_dir, amp_level, model_name
        network = amp.auto_mixed_precision(network, amp_level=self.amp_level)
        self.train_cell = TrainStepCell(network, optimizer, grad_first=self.grad_first, clip_grad=clip_grad,
                                        clip_norm=clip_norm)
        self.loss_interval, self.time_interval, self.ckpt_interval = loss_interval, time_interval, ckpt_interval
        self.start_time = self._time_second()
        self.last_time = self.start_time
        self.this_time = self.start_time
        self.step, self.epoch = 0, 0
        self._calc_iter_print_prefix()

    def __call__(self, *args):
        """
        Call train_cell with callbacks. See details in __init__.

        Args:
            *args (tuple[Tensor]): Input parameters of train cell.

        Returns:
            Union(Tensor, tuple[Tensor]), representation of loss(es) returned by train_cell.
        """
        self.this_time = self._time_second()
        loss = self.train_cell(*args)
        loss_print = self._print_loss(loss)
        time_print = self._print_time()
        ckpt_print = self._save_ckpt()
        custom_print = list(filter(None, [loss_print, time_print, ckpt_print]))
        if custom_print:
            print_log(", ".join([self.iter_print] + custom_print))
        self._update()
        return loss

    @staticmethod
    def calc_ckpt_name(iter_str, model_name, postfix=""):
        """
        Calculate checkpoint file name.

        Args:
            iter_str (Union[str]): Iteration number or epoch number.
            model_name (str): Model name.
            postfix (str): Filename postfix, generally can be the auto mixed precision level. Default: "".

        Returns:
            str, Filename of checkpoint.
        """
        return f"model_{model_name}_{str(postfix)}_{iter_str}_{time_str()}.ckpt"

    @staticmethod
    def calc_optim_ckpt_name(model_name, postfix=""):
        """
        Calculate the latest checkpoint filename. For example, `Optimal_pinns_O2.ckpt`.

        Args:
            model_name (str): Model name.
            postfix (str): Filename postfix. Default: "".

        Returns:
            str, Filename of checkpoint.
        """
        return f"Optim_{model_name}_{postfix}.ckpt"

    @staticmethod
    def _time_second():
        """
        Timestamp in second.

        Returns:
            long, Second timestamp.
        """
        return time_second()

    def _print_loss(self, loss):
        """
        Return string representation of loss(es) according to loss_names if loss_interval is set > 0.

        Args:
            loss (Union[Tensor, tuple[Tensor]]): Loss(es) returned by train_cell.

        Returns:
            str, Representation of loss(es).
        """
        if self.loss_interval > 0 and self.step % self.loss_interval == 0:
            if isinstance(loss, tuple):
                if self.grad_first:
                    loss_names, losses = self.loss_names, [loss[i] for i in range(len(self.loss_names))]
                elif len(loss) == len(self.loss_names):
                    loss_names, losses = ["total_loss"] + list(self.loss_names), [sum(loss)] + list(loss)
                else:
                    loss_names, losses = [f"loss{i + 1}" for i in range(len(loss))], [loss_value for loss_value in loss]
                loss_tuples = [f"{loss_name}: {loss_value}" for loss_name, loss_value in zip(loss_names, losses)]
                loss_print = ", ".join(loss_tuples)
            elif isinstance(loss, (ms.Tensor, Number)):
                if len(self.loss_names) == 1:
                    loss_print = f"{self.loss_names[0]}: {loss}"
                else:
                    loss_print = f"loss: {loss}"
            else:
                loss_print = f"unsupported loss type: {type(loss)}, value: {loss}"
            return loss_print
        return ""

    def _print_time(self):
        """
        Print time if time_interval is set > 0.

        Returns:
            str, Representation of time interval and time elapsed.
        """
        if self.time_interval > 0 and self.step % self.time_interval == 0:
            this_time = self._time_second()
            interval, total_time = this_time - self.last_time, this_time - self.start_time
            self.last_time = this_time
            time_print = f"interval: {interval}s, total: {total_time}s"
            return time_print
        return ""

    def _save_ckpt(self):
        """
        Save checkpoint if time_interval is set > 0.

        Returns:
            str, Checkpoint saving print string, or exception message when it encounters Exception.
        """
        if self.ckpt_interval > 0:
            if not os.path.exists(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)
            if self.batch_num != 1:
                save_ckpt = self.epoch % self.ckpt_interval == 0 and self.step == 0
                iter_str = f"epoch_{self.epoch}"
            else:
                save_ckpt = self.step % self.ckpt_interval == 0
                iter_str = f"iter_{self.step}"
            if save_ckpt:
                ckpt_name = self.calc_ckpt_name(iter_str, self.model_name, self.amp_level)
                optim_ckpt_name = self.calc_optim_ckpt_name(self.model_name, self.amp_level)
                ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
                optim_ckpt_path = os.path.join(self.ckpt_dir, optim_ckpt_name)
                ckpt_print = f"checkpoint saved at: {ckpt_path}, latest checkpoint re-saved at {optim_ckpt_path}"
                try:
                    ms.save_checkpoint(self.train_cell.network, ckpt_path)
                    ms.save_checkpoint(self.train_cell.network, optim_ckpt_path)
                except IOError as _:
                    ckpt_print = "error: failed to save checkpoint due to system error!"
                return ckpt_print
        return ""

    def _update(self):
        """
        Update step num and iter representation. When batch num is not 1, it clears step counter and increases the
            epoch counter.
        """
        self.step += 1
        if self.batch_num != 1 and self.step % self.batch_num == 0:  # clear step if an epoch is finished
            self.epoch += 1
            self.step = 0
        self._calc_iter_print_prefix()

    def _calc_iter_print_prefix(self):
        """
        Calculate iteration printing prefix message, and store it in self.iter_print.
        """
        if self.batch_num != 1:
            self.iter_print = f"epoch:{self.epoch}, step: {self.step}/{self.batch_num}"
        else:
            self.iter_print = f"step: {self.step}"


class TrainStepCell(nn.Cell):
    r"""
    Cell with gradient descent, similar to nn.TrainOneStepCell, but can accept multi-losses return.

    Args:
        network (Cell): The training network. The network supports multi-outputs.
        optimizer (Union[Cell]): Optimizer for updating the network parameters.
        grad_first (bool): If True, only the first output of the network would participate in the gradient
            descent. Otherwise, the sum of all outputs of the network would be taken into account. Default: False.
        clip_grad (bool): Whether clip grad or not. Default: False.
        clip_norm (Union[float, int]): The clipping ratio, it should be greater than 0. Only enabled when `clip_grad`
            is True. Default: 1e-3.

    Inputs:
        - **\*inputs** (tuple[Tensor]) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Union(Tensor, tuple[Tensor]), tensor(s) of the loss value(s), the shape of which is(are) usually :math:`()`.

    Raises:
        TypeError: If `network` or `optimizer` is not of correct type.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, network, optimizer, grad_first=False, clip_grad=False, clip_norm=1e-3):
        super().__init__()
        _batch_check_type(
            {"network": (network, nn.Cell),
             "optimizer": (optimizer, nn.Optimizer),
             "grad_first": (grad_first, bool),
             "clip_grad": (clip_grad, bool),
             "clip_norm": (clip_norm, (int, float))})
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.grad_fist = grad_first
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)
        self.grad_sens = ops.GradOperation(get_by_list=True, sens_param=True)
        self.clip_grad = clip_grad
        self.clip_norm = clip_norm

    def construct(self, *inputs):
        """construct"""
        loss = self.network(*inputs)
        if self.grad_fist and isinstance(loss, tuple):
            sens = [zeros_like(_) for _ in loss]
            sens[0] = ones_like(loss[0])
            grads = self.grad_sens(self.network, self.weights)(*inputs, tuple(sens))
        else:
            grads = self.grad(self.network, self.weights)(*inputs)
        if self.clip_grad:
            grads = ops.clip_by_global_norm(grads, clip_norm=self.clip_norm)
        self.optimizer(grads)
        return loss
