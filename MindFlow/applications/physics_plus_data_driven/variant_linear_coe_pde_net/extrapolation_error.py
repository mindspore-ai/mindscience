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
extrapolation test and show error
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from mindspore import context, ops
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindflow.cell.neural_operators import PDENet

from src.data_generator import DataGenerator
from src.dataset import DataPrepare
from src.loss import LpLoss
from src.utils import check_file_path
from config import train_config as config

parser = argparse.ArgumentParser(description="extrapolation_error")
parser.add_argument('--ckpt_step', type=int, default=20, help="step of ckpt")
parser.add_argument('--ckpt_epoch', type=int, default=500, help="epoch of ckpt")
parser.add_argument('--max_step', type=int, default=20, help="max test step")
parser.add_argument('--sample_size', type=int, default=16, help="data size for test")
parser.add_argument('--device_target', type=str, default="Ascend", help="device target")
parser.add_argument('--device_id', type=int, default=0, help="device id")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False,
                    device_target=args.device_target,
                    device_id=args.device_id
                    )


def extapolation(extra_step, ckpt_step, ckpt_epoch):
    """long time prediction test for given ckpt"""
    error_list = []

    model = PDENet(height=config["mesh_size"],
                   width=config["mesh_size"],
                   channels=config["channels"],
                   kernel_size=config["kernel_size"],
                   max_order=config["max_order"],
                   step=extra_step,
                   dx=2 * np.pi / config["mesh_size"],
                   dy=2 * np.pi / config["mesh_size"],
                   dt=config["dt"],
                   periodic=config["perodic_padding"],
                   enable_moment=config["enable_moment"],
                   if_fronzen=config["if_frozen"],
                   )
    if extra_step == 1:
        model.if_fronzen = True

    param_dict = load_checkpoint(
        "./summary_dir/summary/ckpt/step_{}/pdenet-{}_1.ckpt".format(ckpt_step, ckpt_epoch))
    load_param_into_net(model, param_dict)
    d = DataPrepare(config=config, data_file="data/extrapolation.mindrecord")
    eval_dataset = d.test_data_prepare(extra_step)
    iterator = eval_dataset.create_dict_iterator()
    cast = ops.Cast()
    loss_func = LpLoss(size_average=False)

    for item in iterator:
        u0 = item["u0"]
        ut = item["u_step{:.0f}".format(extra_step)]
        u0 = cast(u0, mstype.float32)
        ut = cast(ut, mstype.float32)
        ut_predict = model(u0.reshape(-1, 1, config["mesh_size"], config["mesh_size"]))
        error_list.append(loss_func(ut_predict, ut).asnumpy().reshape(1)[0])
    return error_list


if __name__ == '__main__':
    error_data = []
    plot_data = np.zeros([args.max_step, 3])
    check_file_path("data")
    check_file_path("figure")
    data = DataGenerator(step=args.max_step, config=config, mode="test", data_size=args.sample_size,
                         file_name="data/extrapolation.mindrecord")
    data.process()

    for i in range(1, args.max_step + 1):
        error = extapolation(i, args.ckpt_step, args.ckpt_epoch)
        error_data.append(error)
        p25 = np.percentile(error, 25)
        p75 = np.percentile(error, 75)
        print("step = {:.0f}, p25 = {:.5f}, p75 = {:.5f}".format(i, p25, p75))
        plot_data[i - 1, :] = [i, p25, p75]

    plt.semilogy(plot_data[:, 0], plot_data[:, 1], color='orange')
    plt.semilogy(plot_data[:, 0], plot_data[:, 2], color='orange')
    plt.fill_between(plot_data[:, 0], plot_data[:, 1], plot_data[:, 2], facecolor='orange', alpha=0.5)
    plt.xlim(1, args.max_step)
    plt.ylim(0.01, 100)
    plt.savefig("figure/extrapolation.jpg")
