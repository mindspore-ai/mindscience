"""maxwell net train"""
import os

import mindspore as ms
from mindspore import nn

from sciai.common import TrainCellWithCallBack
from sciai.context import init_project
from sciai.utils import print_log, data_type_dict_amp, calc_ckpt_name
from sciai.utils.python_utils import print_time
from src.network import MaxwellNet, LossNet
from src.process import prepare, load_data


def train(args, net, scat_pot_ms, ri_value_ms):
    """train"""
    loss_cell = LossNet(net)
    lr = nn.exponential_decay_lr(learning_rate=args.lr, decay_rate=args.lr_decay, decay_epoch=args.lr_decay_step,
                                 total_step=args.epochs, step_per_epoch=1, is_stair=True)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)
    train_net = TrainCellWithCallBack(loss_cell, optimizer, clip_grad=True,
                                      loss_interval=args.print_interval, time_interval=args.print_interval,
                                      ckpt_interval=args.ckpt_interval if args.save_ckpt else 0,
                                      model_name=args.model_name)
    print_log("Training start")
    loss_train = []
    for epoch in range(args.epochs):
        loss_v = train_net(scat_pot_ms, ri_value_ms)
        loss_train.append(loss_v.asnumpy())
        if args.save_ckpt and epoch % 200 == 0:
            print_log("'latest' checkpoint saved at {} epoch.".format(epoch))
            ms.save_checkpoint(train_net.train_cell.network, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))


@print_time("train")
def main(args):
    """main"""
    dtype = data_type_dict_amp.get(args.amp_level, ms.float32)
    scat_pot_ms, ri_value_ms = load_data(args, dtype)
    net = MaxwellNet(args)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, net)
    if dtype == ms.float16:
        net.to_float(ms.float16)
    train(args, net, scat_pot_ms, ri_value_ms)


if __name__ == '__main__':
    args_problem = prepare()
    init_project(args=args_problem[0])
    main(*args_problem)
