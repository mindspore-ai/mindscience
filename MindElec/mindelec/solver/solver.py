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
# ============================================================================
"""solver"""
from mindspore.train import Model, amp
from mindspore import nn, Tensor
from .callback import EvalCallback
from ..loss import NetWithLoss, NetWithEval, get_loss_metric
from ..architecture.util import check_mode


class Solver:
    """
    High-Level API for training or inference.

    `Solver` groups layers into an object with training and inference features.

    Args:
        network (Cell): A training or testing network.
        optimizer (Cell): Optimizer for updating the weights.
        loss_fn (Union(str, dict, Cell)): Objective function, if loss_fn is None, the network should contain the logic
            of loss and grads calculation,. Note that the dict type of loss_fn is not supported in Data mode.
            Default: "l2".
        mode (str): The type of model. Supports ["Data", "PINNs"]. Default: "Data".

            - Data: The model is data_driven.
            - PINNs: The model is physics_informed.

        train_constraints (Constraints): Definition of the loss for train dataset. Default: None. If mode
            is PINNs, the train_constraints cannot be None.
        test_constraints (Constraints): Definition of the loss for test dataset. Default: None. If mode is
            PINNs and eval is needed, the test_constraints cannot be None.
        train_input_map (dict): Specifies the column names of the data in the corresponding dataset to enter
            into the network while training. The key is name of dataset and the value is column names of the data
            in the  corresponding dataset to enter into the network. Default: None. If the input of model is not
            single, train_input_map can not be None.
        test_input_map (dict): Specifies the column names of the data in the corresponding dataset to enter
            into the network while doing eval. The key is name of dataset and the value is column names of the data
            in the corresponding dataset to enter into the network. Default: None. If the input of model is not
            single and eval is needed, test_input_map can not be None.
        mtl_weighted_cell (Cell): Losses weighting algorithms based on multi-task learning uncertainty evaluation.
            Default: None.
        latent_vector (Parameter): Tensor of Parameter. The latent vector to encodes the variational parameters in
            governing equation. It will be concated with the sampling data togother as final network inputs.
            Default: None.
        latent_reg (float): The regularization coefficient of latent vector. Default: 1e-2.
        metrics (Union[dict, set]): A Dictionary or a set of metrics to be evaluated by the model during
            training and inference. eg: {'accuracy', 'recall'}. Default: None.
        eval_network (Cell): Network for evaluation. If not defined, `network` and `loss_fn` would be wrapped as
            `eval_network` . Default: None. Note that eval_network do not need to be set in PINNs mode.
        eval_indexes (list): When defining the `eval_network`, if `eval_indexes` is None, all outputs of the
            `eval_network` would be passed to metrics, otherwise `eval_indexes` must contain three
            elements, including the positions of loss value, predicted value and label. The loss
            value would be passed to the `Loss` metric, the predicted value and label would be passed
            to other metric. Default: None.
        amp_level (str): Option for argument `level` in `mindspore.amp.build_train_network` , level for mixed
            precision training. Supports ["O0", "O2", "O3", "auto"]. Default: "O0".

            - O0: Do not change.
            - O2: Cast network to float16, keep batchnorm run in float32, using dynamic loss scale.
            - O3: Cast network to float16, with additional property `keep_batchnorm_fp32=False` .
            - auto: Set to level to recommended level in different devices. Set level to O2 on GPU, Set
              level to O3 Ascend. The recommended level is choose by the export experience, cannot
              always general. User should specify the level for special network.

            O2 is recommended on GPU, O3 is recommended on Ascend.The more detailed explanation of `amp_level` setting
            can be found at `mindspore.amp.build_train_network`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindelec.solver import Solver
        >>> import mindspore
        >>> from mindspore import nn
        ...
        >>> class Net(nn.Cell):
        ...     def __init__(self, num_class=10, num_channel=1):
        ...         super(Net, self).__init__()
        ...         self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        ...         self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        ...         self.fc1 = nn.Dense(16*5*5, 120, weight_init='ones')
        ...         self.fc2 = nn.Dense(120, 84, weight_init='ones')
        ...         self.fc3 = nn.Dense(84, num_class, weight_init='ones')
        ...         self.relu = nn.ReLU()
        ...         self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        ...         self.flatten = nn.Flatten()
        ...
        ...     def construct(self, x):
        ...         x = self.max_pool2d(self.relu(self.conv1(x)))
        ...         x = self.max_pool2d(self.relu(self.conv2(x)))
        ...         x = self.flatten(x)
        ...         x = self.relu(self.fc1(x))
        ...         x = self.relu(self.fc2(x))
        ...         x = self.fc3(x)
        ...         return x
        ...
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> solver = Solver(net, loss_fn=loss, optimizer=optim, metrics=None)
    """
    def __init__(self, network, optimizer, loss_fn="l2", mode="Data", train_constraints=None, test_constraints=None,
                 train_input_map=None, test_input_map=None, mtl_weighted_cell=None, latent_vector=None, latent_reg=1e-2,
                 metrics=None, eval_network=None,
                 eval_indexes=None, amp_level="O0", **kwargs):
        check_mode("Solver")
        if not isinstance(mode, str):
            raise TypeError("the type of mode should be str but got {}".format(type(mode)))
        self._mode = mode
        self._network = network
        if mode == "PINNs":
            if train_constraints is None:
                raise ValueError("train_constraints can not be None when mode is PINNs")
            self._loss_scale_manager = None
            self._loss_scale_manager_set = False
            self._keep_bn_fp32 = True
            self._amp_level = amp_level
            self._eval_network = eval_network
            self._process_amp_args(kwargs)
            self._train_constraints = train_constraints
            self._train_input_map = train_input_map
            self._optimizer = optimizer
            self._loss_fn = loss_fn
            self._test_constraints = test_constraints
            self.mtl_weighted_cell = mtl_weighted_cell
            self.latent_vector = latent_vector
            self.latent_reg = latent_reg
            train_network = self._build_train_network()
            if not test_constraints:
                eval_network = train_network
            else:
                eval_network = NetWithEval(self._network, test_constraints, loss_fn, test_input_map)
                eval_indexes = [0, 1, 2] if not eval_indexes else eval_indexes
            self.model = Model(network=train_network, eval_network=eval_network, metrics=metrics,
                               eval_indexes=eval_indexes)
        elif mode == "Data":
            if not isinstance(loss_fn, (str, nn.Cell)):
                raise TypeError("For `Data` mode, the type of loss_fn should be str or an instance of Cell but got {}"
                                .format(type(loss_fn)))
            loss_fn = get_loss_metric(loss_fn) if isinstance(loss_fn, str) else loss_fn
            self.model = Model(self._network, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics,
                               eval_network=eval_network, eval_indexes=eval_indexes, amp_level=amp_level, **kwargs)
        else:
            raise ValueError("the value of `mode` should be Data or PINNs, but got {}".format(mode))

    def _build_train_network(self):
        """Build train network"""
        loss_network = NetWithLoss(self._network, self._train_constraints, self._loss_fn,
                                   dataset_input_map=self._train_input_map,
                                   mtl_weighted_cell=self.mtl_weighted_cell,
                                   latent_vector=self.latent_vector,
                                   latent_reg=self.latent_reg)
        if self._loss_scale_manager_set:
            network = amp.build_train_network(loss_network,
                                              self._optimizer,
                                              level=self._amp_level,
                                              loss_scale_manager=self._loss_scale_manager,
                                              keep_batchnorm_fp32=self._keep_bn_fp32)
        else:
            network = amp.build_train_network(loss_network,
                                              self._optimizer,
                                              level=self._amp_level,
                                              keep_batchnorm_fp32=self._keep_bn_fp32)
        return network

    def _process_amp_args(self, kwargs):
        if self._amp_level in ["O0", "O3"]:
            self._keep_bn_fp32 = False
        if 'keep_batchnorm_fp32' in kwargs:
            self._keep_bn_fp32 = kwargs['keep_batchnorm_fp32']
        if 'loss_scale_manager' in kwargs:
            self._loss_scale_manager = kwargs['loss_scale_manager']
            self._loss_scale_manager_set = True

    def train(self, epoch, train_dataset, callbacks=None, dataset_sink_mode=True, sink_size=-1):
        """
        Training API where the iteration is controlled by python front-end.

        Note:
            If dataset_sink_mode is True, data will be sent to device. If device is Ascend, features
            of data will be transferred one by one. The limitation of data transmission per time is 256M.
            If sink_size > 0, each epoch the dataset can be traversed unlimited times until you get sink_size
            elements of the dataset. Next epoch continues to traverse from the end position of the previous traversal.

        Args:
            epoch (int): Generally, total number of iterations on the data per epoch.
                         When dataset_sink_mode is set to true and sink_size>0, each epoch sink sink_size
                         steps on the data instead of total number of iterations.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned.
            callbacks (Optional[list[Callback], Callback]): List of callback objects or callback object,
                                                            which should be executed while training.
                                                            Default: None.
            dataset_sink_mode (bool): Determines whether to pass the data through dataset channel.
                                      Configure pynative mode or CPU, the training process will be performed with
                                      dataset not sink. Default: True.
            sink_size (int): Control the amount of data in each sink.
                             If sink_size = -1, sink the complete dataset for each epoch.
                             If sink_size > 0, sink sink_size data for each epoch.
                             If dataset_sink_mode is False, set sink_size as invalid.
                             Default: -1.

        Examples:
            >>> # For details about how to build the dataset, please refer to the tutorial
            >>> # document on the official website.
            >>> dataset = create_custom_dataset()
            >>> solver.train(2, dataset)
        """
        self.model.train(epoch, train_dataset, callbacks=callbacks, dataset_sink_mode=dataset_sink_mode,
                         sink_size=sink_size)

    def train_with_eval(self, epoch, train_dataset, test_dataset, eval_interval,
                        callbacks=None, dataset_sink_mode=True, sink_size=-1):
        """
        Train_with_eval API where the iteration is controlled by python front-end.

        Note:
            If dataset_sink_mode is True, data will be sent to device. If device is Ascend, features
            of data will be transferred one by one. The limitation of data transmission per time is 256M.
            If sink_size > 0, each epoch the dataset can be traversed unlimited times until you get sink_size
            elements of the dataset. Next epoch continues to traverse from the end position of the previous traversal.

        Args:
            epoch (int): Generally, total number of iterations on the data per epoch.
                         When dataset_sink_mode is set to true and sink_size>0, each epoch sink sink_size
                         steps on the data instead of total number of iterations.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            test_dataset (Dataset): Dataset to evaluate the model.
            eval_interval (int): Specifies eval interval.
            callbacks (Optional[list[Callback], Callback]): List of callback objects or callback object,
                                                            which should be executed while training.
                                                            Default: None.
            dataset_sink_mode (bool): Determines whether to pass the data through dataset channel. Default: True.
                                      Configure pynative mode or CPU, the training process will be performed with
                                      dataset not sink. Default: True.
            sink_size (int): Control the amount of data in each sink.
                             If sink_size = -1, sink the complete dataset for each epoch.
                             If sink_size > 0, sink sink_size data for each epoch.
                             If dataset_sink_mode is False, set sink_size as invalid.
                             Default: -1.

        Examples:
            >>> # For details about how to build the dataset, please refer to the tutorial
            >>> # document on the official website.
            >>> dataset = create_custom_dataset()
            >>> solver.train_with_eval(20, dataset, dataset, 10)
        """
        if self._mode == "PINNs" and not self._test_constraints:
            raise ValueError("test_constraints cannot be None while train_with_eval in PINNs mode")
        eval_callback = EvalCallback(self.model, test_dataset, eval_interval)
        if not callbacks:
            callbacks = [eval_callback]
        else:
            callbacks.append(eval_callback)
        self.model.train(epoch, train_dataset, callbacks=callbacks,
                         dataset_sink_mode=dataset_sink_mode, sink_size=sink_size)

    def eval(self, valid_dataset, callbacks=None, dataset_sink_mode=True):
        """
        Evaluation API where the iteration is controlled by python front-end.

        Configure to pynative mode or CPU, the evaluating process will be performed with dataset non-sink mode.

        Note:
            If dataset_sink_mode is True, data will be sent to device. If device is Ascend, features
            of data will be transferred one by one. The limitation of data transmission per time is 256M.

        Args:
            valid_dataset (Dataset): Dataset to evaluate the model.
            callbacks (Optional[list(Callback)]): List of callback objects which should be executed
                while training. Default: None.
            dataset_sink_mode (bool): Determines whether to pass the data through dataset channel.
                Default: True.

        Returns:
            Dict, whose key is name of metric and value is value of metric.

        Examples:
            >>> # For details about how to build the dataset, please refer to the tutorial
            >>> # document on the official website.
            >>> dataset = create_custom_dataset()
            >>> acc = solver.eval(dataset, dataset_sink_mode=False)
        """
        if self._mode == "PINNs" and not self._test_constraints:
            raise ValueError("test_constraints cannot be None while eval in PINNs mode")
        return self.model.eval(valid_dataset, callbacks=callbacks, dataset_sink_mode=dataset_sink_mode)

    def predict(self, *predict_data):
        """
        Calculate model predictions based on input.

        Data could be a single tensor, a list of tensor, or a tuple of tensor.

        Note:
            This is a pre-compile function. The arguments should be the same with model.predict() function.

        Args:
            predict_data: The predict data can be tensor or tuple of tensor.

        Returns:
            Tensor, array(s) of predictions.

        Raises:
            TypeError: if predict_data is not Tensor of tuple of tensor.

        Examples:
            >>> input_data = Tensor(np.random.randint(0, 255, [1, 1, 32, 32]), mindspore.float32)
            >>> result = solver.predict(input_data)
            >>> print(result.shape)
            (1, 10)
        """
        if isinstance(predict_data, tuple):
            for item in predict_data:
                if not isinstance(item, Tensor):
                    raise TypeError("The element of predict_data should be tensor, but got {}".format(type(item)))
        else:
            if not isinstance(predict_data, Tensor):
                raise TypeError("predict_data should be Tensor of tuple of tensor but got {}"
                                .format(type(predict_data)))
        return self._network(*predict_data)
