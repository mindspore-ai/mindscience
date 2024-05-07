"""
Optimizer used to get the minimum value of a given function.
"""

from typing import Union, List, Iterable
from mindspore import Parameter, Tensor
from mindspore.nn.optim.optimizer import Optimizer, opt_init_args_register
from mindspore.ops import functional as F
from mindspore.ops import composite as C
try:
    # MindSpore 2.X
    from mindspore import jit
except ImportError:
    # MindSpore 1.X
    from mindspore import ms_function as jit
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore import _checkparam as validator


_gd_opt = C.MultitypeFuncGraph("sd_opt")


@_gd_opt.register("Tensor", "Tensor", "Tensor")
def _gradient_descent(learning_rate, gradient, weight):
    r"""
    Apply sgd optimizer to the weight parameter using Tensor.

    Args:
        learning_rate (Tensor): The learning rate value.
        gradient (Tensor): The gradient of the weight parameter.
        weight (Tensor): The weight parameter.

    Returns:
        bool, whether the operation is successful.
    """
    success = True
    success = F.depend(success, F.assign_add(weight, -gradient * learning_rate))
    return success


@_gd_opt.register("Tensor", "Float32", "Tensor", "Tensor")
def _gradient_descent_with_shift(learning_rate, shift, gradient, weight):
    r"""
    Apply sgd optimizer to the weight parameter using Tensor.

    Args:
        learning_rate (Tensor): The learning rate value.
        shift (float): The shift value.
        gradient (Tensor): The gradient of the weight parameter.
        weight (Tensor): The weight parameter.

    Returns:
        bool, whether the operation is successful."""
    success = True
    origin_shift = -gradient * learning_rate
    success = F.depend(success, F.assign_add(weight, origin_shift.clip(-shift, shift)))
    return success


class SteepestDescent(Optimizer):
    """
    Implements the steepest descent (gradient descent) algorithm.

    Note:
        If parameters are not grouped, the `weight_decay` in optimizer
        will be applied on the network parameters without 'beta' or 'gamma'
        in their names. Users can group parameters to change the strategy of
        decaying weight. When parameters are grouped, each group can set
        `weight_decay`. If not, the `weight_decay` in optimizer will be
        applied.

    Args:
        params (Union[list[`mindspore.Parameter`], list[dict]]): Must be list of
            `Parameter` or list of `dict`. When the `params` is a list of
            `dict`, the string "params", "lr", "grad_centralization" and
            "order_params" are the keys can be parsed.

            - params: Required. Parameters in current group.
              The value must be a list of `Parameter`.
            - lr: Optional. If "lr" in the keys, the value of corresponding
              learning rate will be used.
              If not, the `learning_rate` in optimizer will be used.
              Fixed and dynamic learning rate are supported.
            - weight_decay: Using different `weight_decay` by grouping
              parameters is currently not supported.
            - grad_centralization: Optional. Must be Boolean. If
              "grad_centralization" is in the keys, the set value
              will be used. If not, the `grad_centralization` is `False`
              by default. This configuration only works on the convolution
              layer.
            - order_params: Optional. When parameters is grouped,
              this usually is used to maintain the order of
              parameters that appeared in the network to improve
              performance. The value should be parameters whose
              order will be followed in optimizer.
              If `order_params` in the keys, other keys will be ignored and
              the element of 'order_params' must be in
              one group of `params`.

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule], optional):

            - float: The fixed learning rate value. Must be equal to or greater than ``0``.
            - int: The fixed learning rate value. Must be equal to or greater than ``0``.
              It will be converted to float.
            - Tensor: Its value should be a scalar or a 1-D vector.
              For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will
              take the i-th value as the learning rate.
            - Iterable: Learning rate is dynamic.
              The i-th step will take the i-th value as the learning rate.
            - `mindspore.nn.LearningRateSchedule`: Learning rate is dynamic.
              During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the
              learning rate of current step.

        weight_decay (Union[float, int], optional): An int or a floating point value for the weight decay.
            It must be equal to or greater than ``0``.
            If the type of `weight_decay` input is int,
            it will be converted to float. Default: ``0.0``.
        loss_scale (float, optional): A floating point value for the loss scale.
            It must be greater than ``0``.
            If the type of `loss_scale` input is int, it will be converted to float.
            In general, use the default value.
            Only when  `mindspore.amp.FixedLossScaleManager` is used
            for training and the `drop_overflow_update` in
            `mindspore.amp.FixedLossScaleManager` is set to ``False``,
            this value needs to be the same as the `loss_scale` in
            `mindspore.amp.FixedLossScaleManager`.
            Refer to class `mindspore.amp.FixedLossScaleManager` for more details.
            Default: 1.0.
        max_shift (float, optional): A floating point value for the max shift. It must be greater than ``0``.
            It is the bound of the shift distance each iteration in the optimizer.
            If the max shift is set to be None, we will do nothing to
            the shift.
            But if max_shift is a given float number,
            thus the bound of shift would be: [-max_shift, max_shift]
            Default: ``None``.

    Inputs:
        - **gradients** (Tensor) - The gradients of the parameters.

    Outputs:
        - **success** (bool) - whether the operation is successful.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor,
            Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `loss_scale` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: If `loss_scale` is less than or equal to ``0``.
        ValueError: If `weight_decay` is less than ``0``.
        ValueError: If `learning_rate` is a Tensor, but the dimension of
            tensor is greater than ``1``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from sponge import Sponge, Molecule, ForceField
        >>> from sponge.optimizer import SteepestDescent
        >>> system = Molecule(template='water.tip3p.yaml')
        >>> potential = ForceField(system, parameters='SPCE')
        >>> optim = SteepestDescent(params=system.trainable_params(), learning_rate=1e-7)
        >>> print(system.coordinate.value())
        >>> # [[[ 0. 0. 0. ]
        >>> # [ 0.07907964 0.06120793 0. ]
        >>> # [-0.07907964 0.06120793 0. ]]]
        >>> md = Sponge(system, potential, optim)
        >>> md.run(1000)
        >>> # [MindSPONGE] Started simulation at 2024-04-29 01:00:42
        >>> # [MindSPONGE] Finished simulation at 2024-04-29 01:00:44
        >>> # [MindSPONGE] Simulation time: 2.02 seconds.
        >>> print(system.coordinate.value())
        >>> # [[[ 5.3361070e-12 2.3146218e-03 0.0000000e+00]
        >>> # [ 8.1648827e-02 6.0050689e-02 0.0000000e+00]
        >>> # [-8.1648827e-02 6.0050689e-02 0.0000000e+00]]]
    """

    @opt_init_args_register
    def __init__(self,
                 params: Union[List[Parameter], List[dict]],
                 learning_rate: Union[float, int, Tensor, Iterable, LearningRateSchedule] = 1e-03,
                 weight_decay: Union[float, int] = 0.0,
                 loss_scale: float = 1.0,
                 max_shift: float = None
                 ):
        super().__init__(
            parameters=params,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
        )
        if max_shift is None:
            self.max_shift = None
        else:
            if isinstance(max_shift, int):
                max_shift = float(max_shift)
            validator.check_value_type("max_shift", max_shift, [float], self.cls_name)
            validator.check_positive_float(max_shift, "max_shift", self.cls_name)
            self.max_shift = max_shift

    @jit
    def construct(self, gradients):
        r"""
        Update the parameters by the gradients

        Args:
            gradients (Tensor): The gradients of the parameters.

        Returns:
            bool, whether the operation is successful."""
        params = self._parameters
        gradients = self.flatten_gradients(gradients)
        gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)
        lr = self.get_lr()
        if self.is_group_lr:
            if self.max_shift is not None:
                success = self.hyper_map_reverse(
                    F.partial(_gd_opt), lr,
                    self.max_shift, gradients, params)
            else:
                success = self.hyper_map_reverse(F.partial(_gd_opt), lr, gradients, params)
        elif self.max_shift is not None:
            success = self.hyper_map_reverse(
                F.partial(_gd_opt, lr,
                          self.max_shift), gradients, params)
        else:
            success = self.hyper_map_reverse(F.partial(_gd_opt, lr), gradients, params)
        return success
