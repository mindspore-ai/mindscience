# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
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
"""Engine"""

import os
import sys
import logging
from typing import List

import mindspore as ms
from mindspore import ops

from ..configs import Config
from . import meter
from ..configs import Registry as R

module = sys.modules.get(__name__)
logger = logging.getLogger(__name__)


@R.register('core.Metrics')
class Metrics:
    """
    Metrics to evaluate the performance of deep learning model.

    Args:
        metrics (dict, list, tuple): a set of metrics.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, metrics):
        if isinstance(metrics, (list, tuple)):
            self.metrics = {key: key for key in metrics}
        else:
            self.metrics = metrics
        for key, metric in self.metrics.items():
            if isinstance(metric, str):
                symbol = metric.split('.')[-1]
                self.metrics[key] = R.search('metric.' + symbol)()

    def __call__(self, preds, targets):
        self.update(preds, targets)
        results = self.eval()
        self.clear()
        return results

    @classmethod
    def stack(cls, objs, *args, **kwargs):
        """
        Stack a list of nested containers with the same structure.
        """
        obj = objs[0]
        if isinstance(obj, ms.Tensor):
            return ops.stack(objs, *args, **kwargs)
        if isinstance(obj, dict):
            return {k: cls.stack([x[k] for x in objs], *args, **kwargs) for k in obj}
        if isinstance(obj, (list, tuple)):
            return type(obj)(cls.stack(xs, *args, **kwargs) for xs in zip(*objs))

        raise TypeError("Can't perform stack over object type `%s`" % type(obj))

    @classmethod
    def mean(cls, obj, *args, **kwargs):
        """
        Compute mean of tensors in any nested container.
        """
        if hasattr(obj, "mean"):
            return obj.mean(*args, **kwargs)
        if isinstance(obj, dict):
            return type(obj)({k: cls.mean(v, *args, **kwargs) for k, v in obj.items()})
        if isinstance(obj, (list, tuple)):
            return type(obj)(cls.mean(x, *args, **kwargs) for x in obj)

        raise TypeError("Can't perform mean over object type `%s`" % type(obj))

    def update(self, preds, targets):
        if isinstance(preds, List):
            preds = ops.concat(preds)
        if isinstance(targets, List):
            targets = ops.concat(targets)
        if self.metrics is None:
            pass
        for fn in self.metrics.values():
            fn.update(preds, targets)

    def eval(self):
        results = {}
        for key, metric in self.metrics.items():
            results[key] = metric.eval()
        return results

    def clear(self):
        for metric in self.metrics.values():
            metric.clear()


@R.register("core.Engine")
class Engine:
    """
    General class that handles everything about training and test of a task.
    If :meth:`preprocess` is defined by the task, it will be applied to ``train_set``, ``valid_set`` and ``test_set``.

    args:
        scenario (nn.Cell): The scenario of the neural network models
        train_set (data.Dataset): training set
        valid_set (data.Dataset): validation set
        test_set (data.Dataset): test set
        optimizer (optim.Optimizer): optimizer.
        scheduler (lr_scheduler._LRScheduler, optional): scheduler to control the change of learning rate.
        batch_size (int, optional): batch size of a single CPU / GPU / Ascend
        loss_fn (Callable): The loss function for training models in scenario. Default: scenario.loss_fn
        gradient_interval (int, optional): perform a gradient update every n batches.
            This creates an equivalent batch size of ``batch_size * gradient_interval`` for optimization.
        log_interval (int, optional): log every n gradient updates
    """

    def __init__(self,
                 scenario,
                 optimizer=None,
                 train_set=None,
                 valid_set=None,
                 test_set=None,
                 metrics=None,
                 scheduler=None,
                 gradient_interval=1,
                 log_interval=1,
                 loss_fn=None,
                 batch_size=None,
                 device='GPU'):

        if hasattr(scenario, "preprocess"):
            result = scenario.preprocess(train_set, valid_set, test_set)
            if result is not None:
                train_set, valid_set, test_set = result
        self.batch_size = batch_size
        self.meter = meter.Meter(log_interval=log_interval, silent=False, logger=logger)
        self.scenario = scenario
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.metrics = metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_interval = gradient_interval
        self.device = device
        self.loss_fn = loss_fn if loss_fn is not None else self.scenario.loss_fn
        self.grad = ops.value_and_grad(self.loss_fn, grad_position=None, has_aux=False,
                                       weights=self.scenario.trainable_params())

    @classmethod
    def load_config(cls, config):
        """
        Construct an instance from the configuration dict.
        """
        optim_config = config.pop("optimizer")
        metric_config = config.pop('metrics')
        kwargs = {}
        for k, v in config.items():
            if isinstance(v, dict) and "cls_name" in v:
                v = Config.load_config(v)
            if k != "cls_name":
                kwargs[k] = v
        engine = cls(**kwargs)
        optim_config['params'] = kwargs.get('scenario').trainable_params()
        engine.optimizer = Config.load_config(optim_config)
        engine.metrics = Metrics(metric_config)
        return engine

    def fit(self, epochs=100):
        """
        Fit the model. The training set and validation set should be given during the initialization of engine.

        Args:
            epochs (int, optional): number of epochs

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``
        """
        for _ in self.meter(epochs):
            loss = 0
            # the last gradient update may contain less than gradient_interval batches
            dataloader = self.train_set.dict_iterator(self.batch_size, drop_last=True)
            for idx, batch in enumerate(dataloader):
                (bloss, _), grads = self.grad(**batch)
                bloss = ops.depend(bloss, self.optimizer(grads))
                loss += bloss

                if idx % self.gradient_interval == 0 and idx != 0:
                    valid_metrics = self.eval(self.valid_set)
                    valid_metrics['train_loss'] = loss / self.gradient_interval
                    self.meter.update(valid_metrics)
                    loss = 0
            if self.scheduler:
                self.scheduler.step()

    def eval(self, dataset):
        """
        Evaluate the model.

        Args:
            dataset (data.BaseDataset): dataset to evaluate.

        Returns:
            dict: metrics

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``
        """
        loader = dataset.dict_iterator(self.batch_size, drop_last=True)
        loss = 0
        n_batch = 0
        metrics = []
        for batch in loader:
            bloss, output = self.loss_fn(**batch)
            if self.metrics is None:
                metric = self.scenario.eval(*output)
                metrics.append(metric)
            else:
                self.metrics.update(*output)
                loss += bloss
            n_batch += 1
        if self.metrics is not None:
            metrics = self.metrics.eval()
            self.metrics.clear()
        else:
            metrics = Metrics.stack(metrics, axis=0)
            metrics = Metrics.mean(metrics, axis=0)
        metrics['valid_loss'] = loss / n_batch
        return metrics

    def load(self, checkpoint):
        """
        Load a checkpoint from file.

        Args:
            checkpoint (file-like): checkpoint file
        """
        checkpoint = os.path.expanduser(checkpoint)
        ms.load_checkpoint(checkpoint, self.scenario)

    def save(self, checkpoint):
        """
        Save checkpoint to file.

        Args:
            checkpoint (file-like): checkpoint file
        """
        checkpoint = os.path.expanduser(checkpoint)
        ms.save_checkpoint(self.scenario, checkpoint)
