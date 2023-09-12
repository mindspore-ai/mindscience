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
"""pinn heat transfer train"""
import mindspore as ms
import numpy as np

from eval import evaluate
from sciai.context import init_project
from sciai.utils.python_utils import print_time
from src.network import NeuralNetwork
from src.plot import plot_inf_cont_results
from src.process import prepare_data, Logger, prepare


@print_time("train")
def main(args):
    x, t, x_mesh, t_mesh, u, x_star, u_star, x_train, u_train, x_f_train, ub, lb \
        = prepare_data(args.load_data_path, args.n_t, args.n_f)

    def error():
        u_pred = pinn.predict(x_star)
        return np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

    logger = Logger(error)
    pinn = NeuralNetwork(args, logger, x_f_train, ub, lb)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, pinn.model)
    pinn.train(x_train, u_train)
    u_pred = evaluate(pinn, u_star, x_star)
    if args.save_fig:
        plot_inf_cont_results(x_star, u_pred.flatten(), x_train, u_train, u, x_mesh, t_mesh, x, t,
                              save_path=args.figures_path, save_hp=args)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
