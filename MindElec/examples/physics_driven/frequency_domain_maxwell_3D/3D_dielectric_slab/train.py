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
# ==============================================================================
"""
train
"""
import numpy as np

from mindspore import context
import mindspore as ms
import mindspore.nn as nn
from mindspore import Model
from mindspore.train.callback import LossMonitor
from mindspore.common.initializer import HeUniform
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindelec.architecture import MultiScaleFCCell

from src.dataset import create_train_dataset
from src.maxwell import MaxwellCavity
from src.config import maxwell_3d_config
from src.callback import TimeMonitor, SaveCkptMonitor

context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False, device_target="Ascend")


def load_paramters_into_net(param_path, net):
    """
    Load pre-trained parameter into net.
    """
    param_dict = load_checkpoint(param_path)
    convert_ckpt_dict = {}
    for _, param in net.parameters_and_names():
        convert_name1 = "jac2.model.model.cell_list." + param.name
        convert_name2 = "jac2.model.model.cell_list." + \
            ".".join(param.name.split(".")[2:])
        for key in [convert_name1, convert_name2]:
            if key in param_dict:
                convert_ckpt_dict[param.name] = param_dict[key]
    load_param_into_net(net, convert_ckpt_dict)
    print("Load parameters finished!")


def train():
    """
    Train model.
    """
    # Train dataset.
    train_dataset = create_train_dataset()
    train_loader = train_dataset.create_dataset(batch_size=maxwell_3d_config["batch_size"],
                                                shuffle=True, drop_remainder=True)
    # Network.
    net = MultiScaleFCCell(in_channel=maxwell_3d_config["in_channel"],
                           out_channel=maxwell_3d_config["out_channel"],
                           layers=maxwell_3d_config["layers"],
                           neurons=maxwell_3d_config["neurons"],
                           weight_init=HeUniform(negative_slope=np.sqrt(5)),
                           act="sin",
                           )
    net.to_float(ms.dtype.float16)
    if maxwell_3d_config["pretrained"]:
        load_paramters_into_net(maxwell_3d_config["param_path"], net)

    # Loss network.
    net_with_criterion = MaxwellCavity(net, maxwell_3d_config)

    # Optimizer.
    opt = nn.Adam(net.trainable_params(),
                  learning_rate=maxwell_3d_config["lr"])

    # Model.
    model = Model(net_with_criterion, loss_fn=None, optimizer=opt)

    # Callback functions.
    time_cb = TimeMonitor()
    save_cb = SaveCkptMonitor(loss=10.0, comment="slab")
    loss_cb = LossMonitor()

    # Train.
    model.train(maxwell_3d_config["epochs"],
                train_loader,
                callbacks=[loss_cb, time_cb, save_cb])


if __name__ == "__main__":
    train()
