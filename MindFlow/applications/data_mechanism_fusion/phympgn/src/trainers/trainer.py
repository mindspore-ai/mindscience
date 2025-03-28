# Copyright 2025 Huawei Technologies Co., Ltd
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
"""trainer"""
import os
import os.path as osp
import time

import numpy as np
import mindspore as ms
from mindspore import ops
from mindflow.utils import print_log

from .utils import AverageMeter, compute_average_correlation, compute_armse
from ..loaders.data_loader import batch_graph
from ..datasets.dataset import PDECFDataset


class Trainer:
    """
    Args:
        model: the instance of mindspore Cell's child class
        optimizer: mindspore optimizer
        scheduler: mindspore scheduler
        config: the config of project
        loss_func: loss function
    """

    def __init__(self, model, optimizer, scheduler, config, loss_func):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.loss_func = loss_func

    def train(self, tr_loader, val_loader):
        """train"""
        # continuous train
        if self.config.continuous_train:
            self.load_checkpoint()

        min_val_loss = 1.0e+6
        tr_batch_time = AverageMeter()
        tr_data_time = AverageMeter()
        tr_graph_time = AverageMeter()
        tr_grad_time = AverageMeter()
        tr_optim_time = AverageMeter()

        val_batch_time = AverageMeter()
        val_data_time = AverageMeter()
        val_graph_time = AverageMeter()

        def forward_fn(graph):
            target = graph.y.transpose(1, 0, 2)  # (t, n, 2)
            graph.y = target[0]  # (n, 2)
            pred = self.model(graph, steps=target.shape[0]-1)  # (t, n, 2)
            loss = self.loss_func(pred, target)
            return loss, pred

        grad_fn = ops.value_and_grad(forward_fn, None,
                                     self.model.trainable_params(),
                                     has_aux=True)
        for epoch in range(self.config.optim.start_epoch,
                           self.config.optim.epochs + self.config.optim.start_epoch):
            tr_loss = self._train_loop(tr_loader, grad_fn, tr_batch_time,
                                       tr_data_time, tr_graph_time, tr_grad_time, tr_optim_time)

            if epoch == self.config.optim.start_epoch or \
                    epoch % self.config.optim.val_freq == 0:
                self.save_checkpoint()
                val_loss = self._evaluate_loop(val_loader, val_batch_time, val_data_time,
                                               val_graph_time)

                tr_time_str = '[Epoch {:>4}/{}] Batch Time: {:.3f} ({:.3f})  ' \
                           'Data Time: {:.3f} ({:.3f})  Graph Time: {:.3f} ({:.3f})  ' \
                           'Grad Time: {:.3f} ({:.3f})  Optim Time: {:.3f} ({:.3f})'.format(
                               epoch, self.config.optim.start_epoch + self.config.optim.epochs - 1,
                               tr_batch_time.val, tr_batch_time.avg, tr_data_time.val,
                               tr_data_time.avg,
                               tr_graph_time.val, tr_graph_time.avg, tr_grad_time.val,
                               tr_grad_time.avg,
                               tr_optim_time.val, tr_optim_time.avg)
                val_time_str = '[Epoch {:>4}/{}] Batch Time: {:.3f} ({:.3f})  ' \
                           'Data Time: {:.3f} ({:.3f})  Graph Time: {:.3f} ({:.3f})'.format(
                               epoch, self.config.optim.start_epoch + self.config.optim.epochs - 1,
                               val_batch_time.val, val_batch_time.avg, val_data_time.val,
                               val_data_time.avg,
                               val_graph_time.val, val_graph_time.avg)
                info_str = '[Epoch {:>4}/{}] tr_loss: {:.2e} ' \
                           '\t\tval_loss: {:.2e} {}'.format(
                               epoch, self.config.optim.start_epoch + self.config.optim.epochs - 1,
                               tr_loss, val_loss, '{}')
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    info_str = info_str.format('[MIN]')
                    self.save_checkpoint(val=True)
                else:
                    info_str = info_str.format('     ')
                print_log(tr_time_str)
                print_log(val_time_str)
                print_log(info_str)

    # @jit
    def _train_loop(self, tr_loader, grad_fn, batch_time, data_time, graph_time, grad_time,
                    optim_time):
        """train loop"""
        self.model.set_train()
        loss_list = []
        end = time.time()
        for _, data in enumerate(tr_loader.create_dict_iterator()):
            # measure time
            data_t_end = time.time()
            data_time.update(data_t_end - end)

            graph = batch_graph(data)
            graph_t_end = time.time()
            graph_time.update(graph_t_end - data_t_end)

            (loss, _), grads = grad_fn(graph)
            # clap grad norm
            grads = ops.clip_by_norm(grads, max_norm=0.15)
            grad_t_end = time.time()
            grad_time.update(grad_t_end - graph_t_end)

            self.optimizer(grads)
            optim_t_end = time.time()
            optim_time.update(optim_t_end - grad_t_end)

            loss_list.append(loss.asnumpy())

            # measure time
            batch_time.update(time.time() - end)
            end = time.time()
        return np.mean(loss_list)

    # @jit
    def _evaluate_loop(self, val_loader, batch_time, data_time, graph_time):
        """evaluate loop"""
        self.model.set_train(False)
        loss_list = []
        end = time.time()
        for _, data in enumerate(val_loader.create_dict_iterator()):
            # measure time
            data_time.update(time.time() - end)

            start = time.time()
            graph = batch_graph(data)
            graph_time.update(time.time() - start)

            target = graph.y.transpose(1, 0, 2)  # (t, n, 2)
            graph.y = target[0]  # (n, 2)
            pred = self.model(graph, steps=target.shape[0]-1)  # (t, n, 2)
            loss = self.loss_func(pred, target)
            loss_list.append(loss.asnumpy())

            # measure time
            batch_time.update(time.time() - end)
            end = time.time()
        return np.mean(loss_list)

    def test(self, te_loader):
        """test"""
        self.model.set_train(False)
        self.load_checkpoint(val=True)

        inference_time_list = []
        mse_list = []
        armse_list = []
        corre_list = []
        pred_list = []
        target_list = []
        te_num = len(te_loader)
        for b_i, data in enumerate(te_loader.create_dict_iterator()):
            graph = batch_graph(data)
            target = graph.y.transpose(1, 0, 2)
            t = target.shape[0]
            graph.y = target[0]
            start_time = time.time()
            pred = self.model(graph, steps=target.shape[0]-1)
            inference_time = time.time() - start_time
            inference_time_list.append(inference_time)
            target = ops.index_select(target, 1, graph.truth_index)

            # dimensional
            pos = ops.index_select(graph.pos, 0, graph.truth_index)
            pred, target, pos = PDECFDataset.dimensional(
                u_pred=pred,
                u_gt=target,
                pos=pos,
                u_m=graph.u_m,
                d=graph.r * 2
            )

            te_loss = ops.mse_loss(pred, target)
            armse = compute_armse(pred, target)
            mse_list.append(te_loss.asnumpy())
            armse_list.append(armse[-1].asnumpy())
            pred_list.append(pred.asnumpy())
            target_list.append(target.asnumpy())

            info_str = '[TEST {:>2}/{}] MSE at {}t: {:.2e}, armse: {:.3f}, time: {:.2f}s' \
                .format(b_i, te_num, t, te_loss.asnumpy().mean(), armse[-1].item(), inference_time)
            print_log(info_str)

        corre_list = compute_average_correlation(pred_list, target_list)
        corre = np.mean(corre_list)
        info_str = '[Test {}] Mean Loss: {:.2e}, Mean armse: {:.3f}, corre: {:.3f}, time: {:.2f}' \
            .format(len(te_loader), np.mean(mse_list), np.mean(armse_list), corre,
                    np.mean(inference_time_list))
        print_log(info_str)

    def save_checkpoint(self, val=False):
        """save checkpoint"""
        if val:
            ckpt_path = osp.join(self.config.path.ckpt_path,
                                 f'ckpt-{self.config.experiment_name}-val/')
        else:
            ckpt_path = osp.join(self.config.path.ckpt_path,
                                 f'ckpt-{self.config.experiment_name}-tr/')

        if not osp.exists(ckpt_path):
            os.makedirs(ckpt_path)
        ms.save_checkpoint(self.model.parameters_dict(),
                           osp.join(ckpt_path, 'model.ckpt'))
        ms.save_checkpoint(self.optimizer.parameters_dict(),
                           osp.join(ckpt_path, 'optim.ckpt'))

    def load_checkpoint(self, val=False):
        """load checkpoint"""
        if val:
            ckpt_path = osp.join(self.config.path.ckpt_path,
                                 f'ckpt-{self.config.experiment_name}-val/')
        else:
            ckpt_path = osp.join(self.config.path.ckpt_path,
                                 f'ckpt-{self.config.experiment_name}-tr/')

        ckpt_model = ms.load_checkpoint(
            osp.join(ckpt_path, 'model.ckpt'))
        ms.load_param_into_net(self.model, ckpt_model)
        if self.optimizer is not None:
            ckpt_optim = ms.load_checkpoint(
                osp.join(ckpt_path, 'optim.ckpt'))
            ms.load_param_into_net(self.optimizer, ckpt_optim)
