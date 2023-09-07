"""pi deeponet train"""
import os

import mindspore as ms

from sciai.context import init_project
from sciai.utils import data_type_dict_amp, calc_ckpt_name
from sciai.utils.python_utils import print_time

from src.network import PiDeepONet
from src.plot import plot_loss, plot_train
from src.process import generate_training_data, save_loss_logs, prepare
from eval import evaluate


@print_time("train")
def main(args):
    dtype = data_type_dict_amp.get(args.amp_level, ms.float32)
    # GRF
    d_t = 1.0
    ics_dataset, bcs_dataset, res_dataset = generate_training_data(d_t, args.batch_size, args.n_train, dtype)
    model = PiDeepONet(args.branch_layers, args.trunk_layers, d=0.001, k=0.001)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, model)
    model.train(ics_dataset, bcs_dataset, res_dataset, args)
    if args.save_ckpt:
        ms.save_checkpoint(model, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))
    if args.save_data:
        save_loss_logs(args.save_data_path, model)
    n_test, nt_test, nx_test, s_pred_, s_test = evaluate(d_t, model)
    if args.save_fig:
        plot_loss(model.loss_ics_log, model.loss_bcs_log, model.loss_res_log, args.figures_path)
        plot_train(d_t, n_test, nt_test, nx_test, s_pred_, s_test, args.figures_path)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
