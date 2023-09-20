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
"""InferenceModule and EvaluateCallBack"""

import os
import time

import numpy as np
import properscoring as ps
from scipy.stats import norm
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.train.serialization import save_checkpoint
from mindspore.communication import get_rank, get_group_size


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id


class InferenceModule:
    """
    Perform the model inference in Dgmr.
    """

    def __init__(self, logger, thresholds):
        self.logger = logger
        self.loss = nn.MSELoss(reduction='mean')
        self.thresholds = thresholds

    def cal_tp(self, pred=None, target=None, th=None):
        return ops.where(ops.logical_and(pred >= th, target >= th), Tensor(np.ones(target.shape)),
                         Tensor(np.zeros(target.shape))).sum(axis=(-1, -2))

    def cal_tn(self, pred=None, target=None, th=None):
        return ops.where(ops.logical_and(pred < th, target < th), Tensor(np.ones(target.shape)),
                         Tensor(np.zeros(target.shape))).sum(axis=(-1, -2))

    def cal_fp(self, pred=None, target=None, th=None):
        return ops.where(ops.logical_and(pred >= th, target < th), Tensor(np.ones(target.shape)),
                         Tensor(np.zeros(target.shape))).sum(axis=(-1, -2))

    def cal_fn(self, pred=None, target=None, th=None):
        return ops.where(ops.logical_and(pred < th, target >= th), Tensor(np.ones(target.shape)),
                         Tensor(np.zeros(target.shape))).sum(axis=(-1, -2))

    def cal_csi(self, pred=None, target=None, tp=None, fp=None, fn=None):
        if tp is None and fp is None and fn is None:
            tp = self.cal_tp(pred=pred, target=target, th=self.thresholds)
            fp = self.cal_fp(pred=pred, target=target, th=self.thresholds)
            fn = self.cal_fn(pred=pred, target=target, th=self.thresholds)
        csi = (tp / (tp + fp + fn)).mean(axis=0)
        return csi

    def cal_crps_max(self, pred=None, target=None, scale=None):
        pred = ops.avg_pool2d(pred, kernel_size=scale)
        target = ops.avg_pool2d(target, kernel_size=scale)
        target_cdf = norm.cdf(x=target.asnumpy(), loc=0, scale=1)
        pred_cdf = norm.cdf(x=pred.asnumpy(), loc=0, scale=1)
        forecast_score = ps.crps_ensemble(target_cdf, pred_cdf).mean(axis=(0, -1, -2))
        return forecast_score

    def eval(self, dataset, model):
        self.logger.info("================================Start Evaluation================================")
        for data in dataset.create_dict_iterator():
            images = ops.cast(data["inputs"], ms.float32)
            future_images = ops.cast(data["labels"], ms.float32)
            pred = self.forecast(model, images)
            mean_mse_step = self.loss(pred, future_images)
            self.logger.info(f"mean mse per step: {mean_mse_step.asnumpy()}")
        self.logger.info("================================End Evaluation================================")

    def forecast(self, model, inputs):
        pred = model(inputs)
        return pred


class EvaluateCallBack:
    """
    This class includes several functions that can save images/checkpoints and print/save logging information.
    """

    def __init__(self, config, dataset_size=5000, logger=None):
        self.logger = logger
        self.output_path = config['summary']["summary_dir"]
        self.log_dir = os.path.join(self.output_path, 'log')
        self.imgs_dir = os.path.join(self.output_path, "imgs")
        self.ckpts_dir = os.path.join(self.output_path, "ckpt")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        if not os.path.exists(self.ckpts_dir):
            os.makedirs(self.ckpts_dir, exist_ok=True)

        rank_size, self.rank_id = _get_rank_info()
        if rank_size > 1:
            self.run_distribute = True
        else:
            self.run_distribute = False
        self.epoch = 0
        self.batch_size = config["data"]["batch_size"]
        self.dataset_size = dataset_size
        self.predict_interval = config["summary"]["eval_interval"]
        self.keep_checkpoint_max = config["summary"]["keep_checkpoint_max"]
        self.ckpt_list = []
        self.epoch_times = []
        self.eval_net = InferenceModule(logger, config["summary"]["csi_thresholds"])

    def epoch_start(self):
        self.epoch_start_time = time.time()
        self.epoch += 1

    def print_loss(self, res_g, res_d):
        """print log when step end."""
        loss_d = float(res_d[0].asnumpy())
        loss_g = float(res_g[0].asnumpy())

        epoch_cost = (time.time() - self.epoch_start_time) * 1000
        losses = "D_loss: {:.2f}, G_loss:{:.2f}".format(loss_d, loss_g)
        info = "epoch[{}] epoch cost: {:.2f} ms, {}".format(
            self.epoch, epoch_cost, losses)
        if self.run_distribute:
            info = "Rank[{}] , {}".format(self.rank_id, info)
        self.logger.info(info)
        self.epoch_start_time = time.time()

    def epoch_end(self, dataset, model):
        """Evaluate the model at the end of epoch."""
        epoch_cost = (time.time() - self.epoch_start_time) * 1000
        self.epoch_times.append(epoch_cost)
        if self.epoch % self.predict_interval == 0:
            self.eval_net.eval(dataset, model)

    def save_ckpt(self, net):
        """save the model at the end of epoch."""
        g_name = os.path.join(self.ckpts_dir, f"generator{self.epoch}.ckpt")
        d_name = os.path.join(self.ckpts_dir, f"discriminator{self.epoch}.ckpt")
        save_checkpoint(net.network.generator, g_name)
        save_checkpoint(net.network.discriminator, d_name)
        self.ckpt_list.append(self.epoch)
        if len(self.ckpt_list) > self.keep_checkpoint_max:
            del_epoch = self.ckpt_list[0]
            os.remove(os.path.join(self.ckpts_dir, f"generator{del_epoch}.ckpt"))
            os.remove(os.path.join(self.ckpts_dir, f"discriminator{del_epoch}.ckpt"))
            self.ckpt_list.remove(del_epoch)

    def summary(self):
        """train summary at the end of epoch."""
        len_times = len(self.epoch_times)
        sum_times = sum(self.epoch_times)
        epoch_times = sum_times / len_times
        info = 'total {} epochs, cost {:.2f} ms, pre epoch cost {:.2f}'.format(len_times, sum_times, epoch_times)
        if self.run_distribute:
            info = "Rank[{}] {}".format(self.rank_id, info)
        self.logger.info(info)
        self.logger.info('==========end train ===============')
