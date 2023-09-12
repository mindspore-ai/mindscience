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

"""enso train"""
import os
from collections import defaultdict
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn

from sciai.common.dataset import DatasetGenerator
from sciai.context import init_project
from sciai.utils import calc_ckpt_name
from sciai.utils.python_utils import print_time
from sciai.common import TrainCellWithCallBack
from src.network import ENSO, after_train
from src.plot import evaluate, plot_loss
from src.process import fetch_dataset_nino34, prepare


def train(*inputs):
    """train"""
    args, net, ip_train, nino34_train, obs_ip_train, obs_nino34_train, ip_var, nino34_var = inputs

    loss_func = nn.MSELoss()

    optim = nn.optim.SGD(params=net.trainable_params(), learning_rate=args.lr)
    train_dataset = ds.GeneratorDataset(source=DatasetGenerator(ip_train, nino34_train),
                                        shuffle=True, column_names=["data", "label"])
    train_dataset = train_dataset.batch(batch_size=args.batch_size, drop_remainder=True)
    var_dataset = ds.GeneratorDataset(source=DatasetGenerator(ip_var, nino34_var),
                                      column_names=["data", "label"])
    var_dataset = var_dataset.batch(batch_size=len(ip_var))
    loss_record = defaultdict(list)
    loss_cell = nn.WithLossCell(net, loss_func)
    train_cell = TrainCellWithCallBack(loss_cell, optimizer=optim, loss_interval=args.print_interval,
                                       time_interval=args.print_interval, amp_level=args.amp_level,
                                       ckpt_interval=args.ckpt_interval if args.save_ckpt else 0,
                                       ckpt_dir=f"{args.save_ckpt_path}/exp2/")
    for _ in range(args.epochs):
        for x, y in train_dataset:
            loss_train = train_cell(x, y)
            loss_record["train_loss_record"].append(loss_train)
        for x, y in var_dataset:
            loss_val = loss_cell(x, y)
            loss_record["val_loss_record"].append(loss_val)

    if args.save_figure:
        plot_loss(loss_record, args.figures_path, "Training and Validation Loss")
    if args.save_ckpt:
        ms.save_checkpoint(net, f"{args.save_ckpt_path}/exp2/{calc_ckpt_name(args)}")

    if not args.skip_aftertrain:
        after_train(args, net, obs_ip_train, obs_nino34_train, ip_var, nino34_var)
        if args.save_ckpt:
            os.makedirs(f"{args.save_ckpt_path}/exp2_aftertrain/", exist_ok=True)
            ms.save_checkpoint(net, f"{args.save_ckpt_path}/exp2_aftertrain/{calc_ckpt_name(args)}")


@print_time("train")
def main(args):
    ip_train, nino34_train, ip_var, nino34_var, obs_ip_train, obs_nino34_train = \
        fetch_dataset_nino34(args.load_data_path)
    net = ENSO()
    if args.load_ckpt:
        ckpt = ms.load_checkpoint(args.load_ckpt_path)
        ms.load_param_into_net(net, ckpt)
    train(args, net, ip_train, nino34_train, obs_ip_train, obs_nino34_train, ip_var, nino34_var)
    # Test with validation data
    evaluate(args, net, ip_var, nino34_var)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
