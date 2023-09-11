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

"""mgnet train"""
import time
import os

import numpy as np
import mindspore as ms
from mindspore import ops, nn, Model, ModelCheckpoint, get_context
from mindspore.train import Callback

from sciai.context import init_project
from sciai.utils import print_log, amp2datatype, calc_ckpt_name
from sciai.utils.python_utils import print_time

from src.network import MgNet
from src.process import load_data, prepare


class LossMonitor(Callback):
    """Loss Monitor"""
    def __init__(self, per_print_times=1):
        super().__init__()
        self.start_time = None
        self.step_time = None
        self.per_print_times = per_print_times
        self.last_print_time = 0

    def on_train_begin(self, _):
        """
        On train begin.
        """
        self.start_time = time.time()
        print_log('start training ...')

    def on_train_step_begin(self, _):
        """
        On train step begin.
        """
        self.step_time = time.time()

    def on_train_step_end(self, run_context):
        """
        Print training info at the end of each train step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        callback_params = run_context.original_args()
        step_millisecond = (time.time() - self.step_time) * 1000
        loss = callback_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], ms.Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, ms.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (callback_params.cur_step_num - 1) % callback_params.batch_num + 1

        # Boundary check.
        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError(f"Invalid loss, terminate training.")

        def print_info():
            print_log(f"epoch: {callback_params.cur_epoch_num - 1}/{callback_params.epoch_num}, "
                      f"step: {cur_step_in_epoch}/{callback_params.batch_num}, "
                      f"loss: {loss:5.3f}, "
                      f"interval: {step_millisecond:5.3f} ms, "
                      f"total: {time.time() - self.start_time:.3f} s")

        if (callback_params.cur_step_num - self.last_print_time) >= self.per_print_times:
            self.last_print_time = callback_params.cur_step_num
            print_info()


def train(args, net, train_set):
    """Model training"""
    criterion = nn.CrossEntropyLoss()
    num_batch = train_set.get_dataset_size()
    exponential_decay_lr = nn.ExponentialDecayLR(args.lr, 0.1, num_batch * 50, is_stair=True)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=exponential_decay_lr, momentum=0.9, weight_decay=0.0005)
    print_log("total {} parameters".format(sum(ops.size(x) for x in net.trainable_params())))
    config_ck = ms.CheckpointConfig(save_checkpoint_steps=args.ckpt_interval, keep_checkpoint_max=10)
    callbacks = [ModelCheckpoint(directory=args.save_ckpt_path, config=config_ck), LossMonitor(args.print_interval)]
    model = Model(network=net, loss_fn=criterion, optimizer=optimizer, metrics={'accuracy'}, amp_level=args.amp_level)
    model.train(epoch=args.epochs, train_dataset=train_set, callbacks=callbacks)
    return model


@print_time("train")
def main(args):
    if get_context('device_target') == 'GPU' and args.amp_level in ('O1', 'O3'):
        raise ValueError(f'For MgNet, auto mixed precision level {args.amp_level} is not supported on GPU. '
                         'Please use level O0 instead, or use with Ascend devices.')
    dtype = amp2datatype(args.amp_level)
    train_set, test_set, num_classes = load_data(args.load_data_path, args.batch_size, args.dataset)
    net = MgNet(args, dtype, num_classes=num_classes)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, net)
    model = train(args, net, train_set)
    if args.save_ckpt:
        ms.save_checkpoint(net, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))
    test_accuracy = model.eval(test_set)
    print_log("test accuracy: {}".format(test_accuracy))


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
