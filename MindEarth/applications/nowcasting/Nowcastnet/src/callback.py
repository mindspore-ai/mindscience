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
"""Callback"""
import os
import time

import mindspore.communication.management as D
from mindspore.train.callback import Callback
from mindspore.train.serialization import save_checkpoint
from mindspore.communication.management import get_rank, get_group_size
from mindspore.train.summary import SummaryRecord
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint

from .forecast import EvolutionPredictor


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


class NowcastCallBack:
    """
    This class includes several functions that can save images/checkpoints and print/save logging information.
    """
    def __init__(self, config, dataset_size=5000, logger=None):
        self.logger = logger
        self.summary_params = config.get("summary")
        self.data_params = config.get("data")
        self.train_params = config.get("train")
        self.output_path = self.summary_params.get("summary_dir", "")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.ckpt_dir = os.path.join(self.output_path, "ckpt")
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        rank_size, self.rank_id = _get_rank_info()
        if rank_size > 1:
            self.run_distribute = True
        else:
            self.run_distribute = False
        self.epoch = 0
        self.epoch_start_time = None
        self.step = 0
        self.step_start_time = None
        self.batch_size = self.data_params.get("batch_size")
        self.dataset_size = dataset_size
        self.predict_interval = self.summary_params.get("eval_interval")
        self.keep_checkpoint_max = self.summary_params.get("keep_checkpoint_max")
        self.ckpt_list = []
        self.epoch_times = []

    def epoch_start(self):
        self.epoch_start_time = time.time()
        self.epoch += 1

    def step_start(self):
        self.step_start_time = time.time()
        self.step += 1

    def print_loss(self, res_g, res_d, step=False):
        """print log when step end."""
        loss_d = float(res_d)
        loss_g = float(res_g)
        losses = "D_loss: {:.3f}, G_loss:{:.3f}".format(loss_d, loss_g)
        if step:
            step_cost = (time.time() - self.step_start_time) * 1000
            info = "epoch[{}] step {}, cost: {:.2f} ms, {}".format(
                self.epoch, self.step, step_cost, losses)
        else:
            epoch_cost = (time.time() - self.epoch_start_time) * 1000
            info = "epoch[{}] epoch cost: {:.2f} ms, {}".format(
                self.epoch, epoch_cost, losses)
        if self.run_distribute:
            info = "Rank[{}] , {}".format(self.rank_id, info)
        self.logger.info(info)
        if not step:
            self.epoch_start_time = time.time()

    def epoch_end(self):
        """Evaluate the model at the end of epoch."""
        epoch_cost = (time.time() - self.epoch_start_time) * 1000
        self.epoch_times.append(epoch_cost)
        self.step = 0

    def save_generation_ckpt(self, net):
        """save the model at the end of epoch."""
        if self.train_params.get('distribute', False):
            rank_id = D.get_rank()
            ckpt_name = f"generator-device{rank_id}"
        else:
            ckpt_name = "generator"
        g_name = os.path.join(self.ckpt_dir, f"{ckpt_name}_{self.epoch}.ckpt")
        save_checkpoint(net.network.generator, g_name)
        self.ckpt_list.append(f"{ckpt_name}_{self.epoch}.ckpt")
        if len(self.ckpt_list) > self.keep_checkpoint_max:
            del_ckpt = self.ckpt_list[0]
            os.remove(os.path.join(self.ckpt_dir, del_ckpt))
            self.ckpt_list.remove(del_ckpt)

    def summary(self):
        """train summary at the end of epoch."""
        len_times = len(self.epoch_times)
        sum_times = sum(self.epoch_times)
        try:
            epoch_times = sum_times / len_times
        except ZeroDivisionError:
            self.logger.info('==========no epoch===============')
        info = 'total {} epochs, cost {:.2f} ms, pre epoch cost {:.2f}'.format(len_times, sum_times, epoch_times)
        if self.run_distribute:
            info = "Rank[{}] {}".format(self.rank_id, info)
        self.logger.info(info)
        self.logger.info('==========end train ===============')


class EvolutionCallBack(Callback):
    """
    Monitor the prediction accuracy in training.
    """

    def __init__(self,
                 model,
                 valid_dataset,
                 config,
                 logger,
                 ):
        super(EvolutionCallBack, self).__init__()
        summary_params = config.get('summary')
        self.summary_params = config.get("summary")
        self.train_params = config.get("train")
        self.output_path = self.summary_params.get("summary_dir", "")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.ckpt_dir = os.path.join(self.output_path, "ckpt")
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.summary_dir = summary_params.get('summary_dir', "")
        self.predict_interval = summary_params.get('eval_interval', 10)
        self.epochs = config.get('optimizer-evo').get("epochs", 200)
        self.valid_dataset = valid_dataset
        self.eval_net = EvolutionPredictor(config, model, logger)
        self.eval_time = 0

    def __enter__(self):
        self.summary_record = SummaryRecord(self.summary_dir)
        return self

    def __exit__(self, *exc_args):
        self.summary_record.close()

    def on_train_epoch_end(self, run_context):
        """
        Evaluate the model at the end of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.predict_interval == 0 or cb_params.cur_epoch_num == self.epochs - 1:
            self.eval_time += 1
            self.eval_net.eval(self.valid_dataset)

    def save_evolution_ckpt(self):
        """
        Get the checkpoint callback of the model.

        Returns:
            Callback, The checkpoint callback of the model.
        """
        if self.train_params.get('distribute', False):
            rank_id = D.get_rank()
            ckpt_name = f"evolution-device{rank_id}"
        else:
            ckpt_name = "evolution"
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=self.summary_params.get("save_checkpoint_epochs", 2),
            keep_checkpoint_max=self.summary_params.get("keep_checkpoint_max", 4))
        ckpt_cb = ModelCheckpoint(prefix=ckpt_name, directory=self.ckpt_dir, config=ckpt_config)
        return ckpt_cb
