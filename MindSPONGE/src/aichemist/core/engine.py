# Copyright 2023 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
from itertools import islice
from typing import List
import inspect

import mindspore as ms
from mindspore import ops
from mindspore import nn
from mindspore import context
from mindspore.common.api import _pynative_executor
from mindspore.common.parameter import Parameter

from .meta import args_to_dict
from .config import Configurable
from . import meter
from .config import Registry as R
from .. import util


module = sys.modules.get(__name__)
logger = logging.getLogger(__name__)


# pylint: disable=W0703
# pylint: disable=W0212
class Cell(nn.Cell):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __call__(self, *args, **kwargs):
        """_summary_

        Raises:
            err: _description_

        Returns:
            _type_: _description_
        """
        if self.__class__.construct is nn.Cell.construct:
            logger.warning("The '%s' does not override the method 'construct', \
                           it will call the super class(Cell) 'construct'.", self.__class__)
        if kwargs:
            bound_arguments = inspect.signature(
                self.construct).bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            args = bound_arguments.args
            kwargs = bound_arguments.kwargs

        # Run in Graph mode.
        if context._get_mode() == context.GRAPH_MODE:
            if self._hook_fn_registered():
                logger.warning("For 'Cell', it's not support hook function in graph mode. If you want to use hook "
                               "function, please use context.set_context to set pynative mode.")
            out = self.compile_and_run(*args)
            return out

        # Run in PyNative mode.
        if _pynative_executor.is_first_cell():
            _pynative_executor.set_lazy_build(True)
            _pynative_executor._optimizer = getattr(self, "optimizer", None)
            _pynative_executor._top_cell = self
            # There many Casts in parameter_broadcast. Enable lazy_build and build faster.
            self._do_parameter_broadcast()

        if self.requires_grad:
            _pynative_executor.set_grad_flag(True)

        if self._dynamic_shape_inputs is not None:
            self._check_compile_dynamic_shape(*args)

        try:
            _pynative_executor.new_graph(self, *args, **kwargs)
            output = self._run_construct(args, kwargs)
            _pynative_executor.end_graph(self, output, *args, **kwargs)
        except Exception as err:
            _pynative_executor.clear_res()
            raise err

        if isinstance(output, Parameter):
            output = output.data
        return output


class Grad:
    def __init__(self, fn, grad_position=None, weights=None, has_aux=True):
        self.grad = ops.value_and_grad(fn,
                                       grad_position=grad_position,
                                       weights=weights,
                                       has_aux=has_aux)

    def __call__(self, *args, **kwargs):
        args, kwargs = args_to_dict(*args, **kwargs)
        return self.grad(*args, **kwargs)


@R.register('core.Metrics')
class Metrics:
    """_summary_
    """

    def __init__(self, metrics):
        self.metrics = {}
        for metric in metrics:
            if isinstance(metric, str):
                symbol = metric.split('.')[-1]
                self.metrics[metric] = R.search('metric.' + symbol)()

    def __call__(self, preds, targets):
        self.update(preds, targets)
        results = self.eval()
        self.clear()
        return results

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
    This class can perform synchronous distributed parallel training over multiple CPUs or GPUs.
    To invoke parallel training, launch with one of the following commands.
    1. Single-node multi-process case.
    .. code-block:: bash
        python -m torch.distributed.launch --nproc_per_node={number_of_gpus} {your_script.py} {your_arguments...}
    2. Multi-node multi-process case.
    .. code-block:: bash
        python -m torch.distributed.launch --nnodes={number_of_nodes} --node_rank={rank_of_this_node}
        --nproc_per_node={number_of_gpus} {your_script.py} {your_arguments...}
    If :meth:`preprocess` is defined by the task, it will be applied to ``train_set``, ``valid_set`` and ``test_set``.
    Parameters:
        task (nn.Module): task
        train_set (data.Dataset): training set
        valid_set (data.Dataset): validation set
        test_set (data.Dataset): test set
        optimizer (optim.Optimizer): optimizer
        scheduler (lr_scheduler._LRScheduler, optional): scheduler
        gpus (list of int, optional): GPU ids. By default, CPUs will be used.
            For multi-node multi-process case, repeat the GPU ids for each node.
        batch_size (int, optional): batch size of a single CPU / GPU
        gradient_interval (int, optional): perform a gradient update every n batches.
            This creates an equivalent batch size of ``batch_size * gradient_interval`` for optimization.
        num_worker (int, optional): number of CPU workers per GPU
        logger (str or core.LoggerBase, optional): logger type or logger instance.
            Available types are ``logging`` and ``wandb``.
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
                 gradient_interval=5,
                 log_interval=1,
                 detach=False):

        if hasattr(scenario, "preprocess"):
            result = scenario.preprocess(train_set, valid_set, test_set)
            if result is not None:
                train_set, valid_set, test_set = result
        self.meter = meter.Meter(log_interval=log_interval, silent=False, logger=logger)
        self.scenario = scenario
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.metrics = metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_interval = gradient_interval
        self.detach = detach
        self.grad = Grad(self.scenario.loss_fn, grad_position=None,
                         weights=self.scenario.trainable_params(), has_aux=False)

    @classmethod
    def load_config(cls, config):
        """
        Construct an instance from the configuration dict.
        """
        optim_config = config.pop("optimizer")
        metric_config = config.pop('metrics')
        kwargs = {}
        for k, v in config.items():
            if isinstance(v, dict) and "__class__" in v:
                v = Configurable.load_config(v)
            if k != "__class__":
                kwargs[k] = v
        engine = cls(**kwargs)
        optim_config['params'] = kwargs.get('scenario').trainable_params()
        engine.optimizer = Configurable.load_config(optim_config)
        engine.metrics = Metrics(metric_config)
        return engine

    def fit(self, epochs=100, batch_per_epoch=None):
        """
        Train the model.
        If ``batch_per_epoch`` is specified, randomly draw a subset of the training set for each epoch.
        Otherwise, the whole training set is used for each epoch.
        Parameters:
            num_epoch (int, optional): number of epochs
            batch_per_epoch (int, optional): number of batches per epoch
        """
        batch_per_epoch = batch_per_epoch or len(self.train_set) // self.train_set.batch_size

        for _ in self.meter(epochs):
            interval = 0
            loss = 0
            # the last gradient update may contain less than gradient_interval batches
            gradient_interval = min(batch_per_epoch - interval, self.gradient_interval)

            for idx, batch in enumerate(islice(self.train_set, batch_per_epoch)):
                batch = util.batch_to_device(batch, detach=self.detach)
                (bloss, output), grads = self.grad(*batch)
                bloss = ops.depend(bloss, self.optimizer(grads))
                loss += bloss
                self.metrics.update(*output)

                if idx - interval + 1 == gradient_interval:
                    train_metrics = self.metrics.eval()
                    self.metrics.clear()
                    train_metrics['loss'] = loss / gradient_interval
                    self.meter.log(train_metrics)
                    valid_metrics = self.eval('valid')
                    self.meter.update(valid_metrics)
                    interval = idx + 1
                    loss = 0
                    gradient_interval = min(batch_per_epoch - interval, self.gradient_interval)
            if self.scheduler:
                self.scheduler.step()

    def eval(self, split):
        """
        Evaluate the model.
        Parameters:
            split (str): split to evaluate. Can be ``train``, ``valid`` or ``test``.
            log (bool, optional): log metrics or not
        Returns:
            dict: metrics
        """
        dataloader = getattr(self, f"{split}_set")
        loss = 0
        n_batch = 0
        for batch in dataloader:
            batch = util.batch_to_device(batch, detach=self.detach)
            bloss, output = self.scenario.eval(*batch)
            self.metrics.update(*output)
            loss += bloss
            n_batch += 1
        results = self.metrics.eval()
        self.metrics.clear()
        results['loss'] = loss / n_batch
        return results

    def load(self, checkpoint):
        """
        Load a checkpoint from file.
        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
        """
        checkpoint = os.path.expanduser(checkpoint)
        ms.load_checkpoint(checkpoint, self.scenario)

    def save(self, checkpoint):
        """
        Save checkpoint to file.
        Parameters:
            checkpoint (file-like): checkpoint file
        """
        checkpoint = os.path.expanduser(checkpoint)
        ms.save_checkpoint(self.scenario, checkpoint)
