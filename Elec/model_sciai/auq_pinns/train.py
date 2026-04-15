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

"""auq pinns train"""
import numpy as np
from mindspore import nn
import mindspore as ms

from sciai.common import TrainCellWithCallBack
from sciai.context import init_project
from sciai.utils import to_tensor
from sciai.utils.python_utils import print_time
from src.network import get_all_networks
from src.process import TrainDataset, ValDataset, post_process, prepare


def train(args, dataset, *networks):
    """Model training"""
    encoder, decoder, discriminator, generator_loss, discriminator_loss = networks
    generator_params = encoder.trainable_params() + decoder.trainable_params()
    optimizer_kl = nn.Adam(generator_params, args.lr)
    train_net_kl = TrainCellWithCallBack(generator_loss, optimizer_kl,
                                         loss_interval=args.print_interval,
                                         time_interval=args.print_interval,
                                         loss_names=("G_loss", "KL_loss", "recon_loss", "pde_loss"),
                                         ckpt_interval=args.ckpt_interval * args.term_kl if args.save_ckpt else 0,
                                         ckpt_dir=f"{args.save_ckpt_path}/generator",
                                         grad_first=True,
                                         amp_level=args.amp_level)

    discriminator_params = discriminator.trainable_params()
    optimizer_t = nn.Adam(discriminator_params, args.lr)
    train_net_t = TrainCellWithCallBack(discriminator_loss, optimizer_t,
                                        loss_interval=args.print_interval,
                                        time_interval=args.print_interval,
                                        ckpt_interval=args.ckpt_interval * args.term_t if args.save_ckpt else 0,
                                        ckpt_dir=f"{args.save_ckpt_path}/discriminator",
                                        loss_names="t_loss",
                                        amp_level=args.amp_level,
                                        model_name=args.model_name)
    x_bound = dataset.x_bound
    y_bound = dataset.y_bound
    x_col = dataset.x_col
    z_dim = args.layers_q[-1]
    for _ in range(args.epochs):
        z_bound, z_col = to_tensor((np.random.randn(2 * args.n_bound, z_dim),
                                    np.random.randn(args.n_col, z_dim)), dtype=dataset.dtype)
        for _ in range(args.term_t):
            train_net_t(x_bound, y_bound, z_bound)
        for _ in range(args.term_kl):
            train_net_kl(x_col, z_col, x_bound, z_bound)

    if args.save_ckpt:
        ms.save_checkpoint(train_net_t.train_cell.network,
                           f"{args.save_ckpt_path}/Optim_{args.model_name}_discriminator_{args.amp_level}.ckpt")
        ms.save_checkpoint(train_net_kl.train_cell.network,
                           f"{args.save_ckpt_path}/Optim_{args.model_name}_generator_{args.amp_level}.ckpt")


@print_time("train")
def main(args):
    train_dataset = TrainDataset(args)
    encoder, decoder, discriminator, generator_loss, discriminator_loss = get_all_networks(args, train_dataset)
    train(args, train_dataset, encoder, decoder, discriminator, generator_loss, discriminator_loss)
    val_dataset = ValDataset(args)
    post_process(args, decoder, train_dataset, val_dataset)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
