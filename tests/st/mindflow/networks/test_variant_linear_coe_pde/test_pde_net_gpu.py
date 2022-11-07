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
import time
import numpy as np
import pytest

from mindspore.common import set_seed
from mindspore import nn, Tensor, Model, context
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindflow.cell.neural_operators import PDENet

from src.callbacks import PredictCallback
from src.data_generator import DataGenerator
from src.dataset import DataPrepare
from src.loss import LpLoss
from src.utils import create_logger, check_file_path
from config import train_config as config

set_seed(0)
np.random.seed(0)

print("pid:", os.getpid())

context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False,
                    device_target='GPU',
                    )


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_variant_linear_coe_pde_net():
    """test function of pde net"""
    step = 1
    check_file_path('logs')
    check_file_path('data')
    logger = create_logger(path=os.path.join('logs', "results.log"))

    data = DataGenerator(step=step, config=config, mode="train", data_size=2 * config["batch_size"],
                         file_name="data/train_step{}.mindrecord".format(step))
    data.process()

    dataset = DataPrepare(config=config, data_file="data/train_step{}.mindrecord".format(step))
    train_dataset, eval_dataset = dataset.train_data_prepare()
    print("dataset size: {}".format(train_dataset.get_dataset_size()))

    model = PDENet(height=config["mesh_size"],
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
                   if_fronzen=True,
                  )

    optimizer = nn.Adam(model.trainable_params(), learning_rate=Tensor(config["lr"]))
    loss_func = LpLoss(size_average=False)
    loss_scale = DynamicLossScaleManager()
    solver = Model(model,
                   optimizer=optimizer,
                   loss_scale_manager=loss_scale,
                   loss_fn=loss_func
                   )

    pred_cb = PredictCallback(model=model, eval_dataset=eval_dataset, logger=logger, step=step, config=config)
    time_cb = TimeMonitor()

    start_time = time.time()
    solver.train(epoch=100,
                 train_dataset=train_dataset,
                 callbacks=[LossMonitor(), pred_cb, time_cb],
                 dataset_sink_mode=True
                 )
    full_time = (time.time() - start_time)*1000
    per_epoch_time = full_time/100
    lp_error = pred_cb.get_lploss_error()

    print(f'lp error: {lp_error:.10f}')
    print(f'per epoch time: {per_epoch_time:.10f}')

    assert lp_error <= 15
