# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
train pde net
"""
import os
import argparse
import numpy as np

from mindspore.common import set_seed
from mindspore import nn, Tensor, Model, context
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindflow.cell.neural_operators import PDENet

from src.callbacks import PredictCallback
from src.data_generator import DataGenerator
from src.dataset import DataPrepare
from src.loss import LpLoss
from src.utils import create_logger, check_file_path
from config import train_config

set_seed(0)
np.random.seed(0)

print("pid:", os.getpid())

parser = argparse.ArgumentParser(description="train")
parser.add_argument('--device_target', type=str, default="Ascend")
parser.add_argument('--device_id', type=int, default=0)
opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False,
                    save_graphs_path='./graph',
                    device_target=opt.device_target,
                    device_id=opt.device_id
                    )


def _callback(model, eval_dataset, cur_logger, step, config):
    pred_cb = PredictCallback(model=model, eval_dataset=eval_dataset, logger=cur_logger, step=step, config=config)
    time_cb = TimeMonitor()
    return pred_cb, time_cb


def _checkpoint(ckpt_dir, step_interval):
    ckpt_config = CheckpointConfig(save_checkpoint_steps=step_interval, keep_checkpoint_max=50)
    ckpt_cb = ModelCheckpoint(prefix='pdenet', directory=ckpt_dir, config=ckpt_config)
    return ckpt_cb


def _scheduler(lr_scheduler_step, step, lr):
    if step % lr_scheduler_step == 0:
        lr *= 0.5
        print("learning rate reduced to {}".format(lr))
    return lr


def train_single_step(step, cur_logger, config, lr):
    """train PDE-Net with advancing steps"""
    dataset = DataPrepare(config=config, data_file="data/train_step{}.mindrecord".format(step))
    train_dataset, eval_dataset = dataset.train_data_prepare()
    print("dataset size: {}".format(train_dataset.get_dataset_size()))

    model = init_model(config, step)

    epoch = config["epochs"]
    warm_up_epoch_scale = 10
    if step == 1:
        model.if_fronzen = True
        epoch = warm_up_epoch_scale * epoch
    elif step == 2:
        param_dict = load_checkpoint(
            "./summary_dir/summary/ckpt/step_{}/pdenet-{}_1.ckpt".format(step - 1, epoch * 10))
        load_param_into_net(model, param_dict)
        print("Load pre-trained model successfully")
    else:
        param_dict = load_checkpoint(
            "./summary_dir/summary/ckpt/step_{}/pdenet-{}_1.ckpt".format(step - 1, epoch))
        load_param_into_net(model, param_dict)
        print("Load pre-trained model successfully")

    optimizer = nn.Adam(model.trainable_params(), learning_rate=Tensor(lr))
    loss_func = LpLoss(size_average=False)
    loss_scale = DynamicLossScaleManager()
    solver = Model(model,
                   optimizer=optimizer,
                   loss_scale_manager=loss_scale,
                   loss_fn=loss_func
                   )

    pred_cb, time_cb = _callback(model, eval_dataset, cur_logger, step, config)
    ckpt_cb = _checkpoint("./summary_dir/summary/ckpt/step_{}".format(step), config["save_epoch_interval"])

    solver.train(epoch=epoch,
                 train_dataset=train_dataset,
                 callbacks=[LossMonitor(), pred_cb, time_cb, ckpt_cb],
                 dataset_sink_mode=True
                 )


def init_model(config, step):
    return PDENet(height=config["mesh_size"],
                  width=config["mesh_size"],
                  channels=config["channels"],
                  kernel_size=config["kernel_size"],
                  max_order=config["max_order"],
                  step=step,
                  dx=2 * np.pi / config["mesh_size"],
                  dy=2 * np.pi / config["mesh_size"],
                  dt=config["dt"],
                  periodic=config["perodic_padding"],
                  enable_moment=config["enable_moment"],
                  if_fronzen=config["if_frozen"],
                  )


def train(config, cur_logger):
    lr = config["lr"]
    for i in range(1, config["multi_step"] + 1):
        data = DataGenerator(step=i, config=config, mode="train", data_size=2 * config["batch_size"],
                             file_name="data/train_step{}.mindrecord".format(i))
        data.process()
        lr = _scheduler(int(config["multi_step"] / config["learning_rate_reduce_times"]), step=i, lr=lr)
        train_single_step(step=i, config=config, cur_logger=cur_logger, lr=lr)


if __name__ == '__main__':
    check_file_path('logs')
    check_file_path('data')
    logger = create_logger(path=os.path.join('logs', "results.log"))
    MESSAGE = "pid: {}".format(os.getpid())
    logger.info(MESSAGE)
    train(train_config, logger)
