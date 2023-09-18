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
"""train"""

import os
import time
import datetime
import mindspore
from mindspore import nn
from mindspore import save_checkpoint

from sciai.context import init_project
from sciai.utils import print_time, print_log, calc_ckpt_name
from src.network import prepare_network
from src.process import prepare_dataset, prepare
from src.utils import MetricLogger, SmoothedValue
from src.scheduler import WarmupMultiStepLR


def train(args, network, criterion, dataset_train, dataset_val):
    """Model training"""
    args.epochs = args.epoch_block * args.num_block
    # scheduler
    warmup_iters = args.lr_warmup_epochs * dataset_train.get_dataset_size()
    lr_milestones = [dataset_train.get_dataset_size() * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(args.lr,
                                     milestones=lr_milestones,
                                     gamma=args.lr_gamma,
                                     warmup_iters=warmup_iters,
                                     warmup_factor=1e-5)
    lr_list = lr_scheduler.get_lr(dataset_train.get_dataset_size(), args.start_epoch, args.epochs)

    # optimizer
    optimizer = nn.optim.AdamWeightDecay(
        network.trainable_params(),
        learning_rate=lr_list,
        weight_decay=args.weight_decay)

    # Define forward function
    def forward_fn(data, label):
        logits = network(data)
        loss, loss_g1v, loss_g2v = criterion(logits, label)
        return loss, loss_g1v, loss_g2v, logits

    # Get gradient function
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(data, label):
        (loss, loss_g1v, loss_g2v, _), grads = grad_fn(data, label)
        optimizer(grads)
        return loss, loss_g1v, loss_g2v

    def train_one_epoch(model, dataset, epoch, print_freq):
        model.set_train(True)

        # Logger setup
        metric_logger = MetricLogger(delimiter='  ')
        metric_logger.add_meter('samples/s', SmoothedValue(window_size=10, fmt='{value:.3f}'))
        header = 'Epoch: [{}]'.format(epoch)

        dataset_iter = metric_logger.log_every(dataset.create_tuple_iterator(), print_freq, header)
        for data, label in dataset_iter:
            start_time = time.time()
            loss, loss_g1v, loss_g2v = train_step(data, label)
            metric_logger.update(loss=loss,
                                 loss_g1v=loss_g1v,
                                 loss_g2v=loss_g2v)
            metric_logger.meters['samples/s'].update(data.shape[0] / (time.time() - start_time))

    def evaluate(model, dataset, criterion):
        model.set_train(False)
        metric_logger = MetricLogger(delimiter='  ')
        header = 'Test:'
        total = 0
        dataset_iter = metric_logger.log_every(dataset.create_tuple_iterator(), 20, header)
        for data, label in dataset_iter:
            output = model(data)
            total += len(data)
            loss, loss_g1v, loss_g2v = criterion(output, label)
            metric_logger.update(loss=loss,
                                 loss_g1v=loss_g1v,
                                 loss_g2v=loss_g2v)
        print_log(' * Loss {loss.global_avg:.8f}\n'.format(loss=metric_logger.loss))
        return metric_logger.loss.global_avg

    print_log('Start training')
    start_time = time.time()
    best_loss = 10
    chp = 1

    for epoch in range(args.start_epoch, args.epochs):

        train_one_epoch(network, dataset_train, epoch, args.print_freq)
        loss = evaluate(network, dataset_val, criterion)

        # Save checkpoint per epoch
        if args.save_ckpt and loss < best_loss:
            save_checkpoint(network, os.path.join(args.save_ckpt_path, 'model.ckpt'))
            print_log('saving checkpoint at epoch: ', epoch)
            chp = epoch
            best_loss = loss
        # Save checkpoint every epoch block
        print_log('current best loss: ', best_loss)
        print_log('current best epoch: ', chp)
        if args.save_ckpt and args.save_ckpt_path and (epoch + 1) % args.epoch_block == 0:
            save_checkpoint(network, os.path.join(args.save_ckpt_path, 'model_{}.ckpt'.format(epoch + 1)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print_log('Training time {}'.format(total_time_str))


@print_time("train")
def main(args, data_config):
    print_log('mindspore version: ', mindspore.__version__)

    # prepare dataset
    dataset_train, dataset_val = prepare_dataset(args, data_config)

    # prepare neuaral networks
    network, loss_fn = prepare_network(args)

    if args.load_ckpt:
        mindspore.load_checkpoint(args.load_ckpt_path, network)
        print_log('Loaded model checkpoint at {}'.format(args.load_ckpt_path))

    # training
    train(args, network, loss_fn, dataset_train, dataset_val)
    if args.save_ckpt:
        save_checkpoint(network, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
