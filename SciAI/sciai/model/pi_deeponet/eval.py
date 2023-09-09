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
"""pi deeponet eval"""
import mindspore as ms
import mindspore.numpy as mnp
from scipy.interpolate import griddata

from sciai.context import init_project
from sciai.utils import print_log
from sciai.architecture import SSE
from sciai.utils.python_utils import print_time
from src.network import PiDeepONet
from src.plot import plot_loss, plot_train
from src.process import solve_adr, load_loss_logs, prepare


def evaluate(d_t, model):
    """evaluate"""
    n_test = 50
    nx_test = 100
    nt_test = 100 * n_test
    p_test = 100
    length_scale_test = 0.2
    x, t, uu, u = solve_adr(n_test * d_t, nx_test, nt_test, length_scale_test)
    u_test = mnp.tile(u, (p_test ** 2, 1))
    s_test = uu
    x = mnp.linspace(0, 1, nx_test)
    t = mnp.linspace(0, d_t, nx_test)
    xx, tt = mnp.meshgrid(x, t)
    y_test = mnp.hstack([xx.flatten()[:, None], tt.flatten()[:, None]])
    s_pred = model.predict_s(u_test, y_test)
    s_pred_ = ms.Tensor(griddata(y_test, s_pred.flatten(), (xx, tt), method='cubic'), ms.float32)
    s_pred_, error = calc_error(n_test, p_test, s_pred_, s_test, tt, xx, model, y_test)
    print_log(f'Relative l2 error: {error.asnumpy()}')
    return n_test, nt_test, nx_test, s_pred_, s_test


@print_time("eval")
def main(args):
    d_t = 1.0
    model = PiDeepONet(args.branch_layers, args.trunk_layers, d=0.001, k=0.001)
    ms.load_checkpoint(args.load_ckpt_path, model)
    n_test, nt_test, nx_test, s_pred_, s_test = evaluate(d_t, model)
    if args.save_fig:
        loss_ics, loss_bcs, loss_res = load_loss_logs(args.save_data_path)
        plot_loss(loss_ics, loss_bcs, loss_res, args.figures_path)
        plot_train(d_t, n_test, nt_test, nx_test, s_pred_, s_test, args.figures_path)


def calc_error(*inputs):
    n_test, p_test, s_pred, s_test, tt, xx, model, y_test = inputs
    for _ in range(n_test - 1):
        u_k = s_pred[-1, :]
        u_test_k = mnp.tile(u_k, (p_test ** 2, 1))
        s_pred_k = model.predict_s(u_test_k, y_test)
        s_pred_k_ = ms.Tensor(griddata(y_test, s_pred_k.flatten(), (xx, tt), method='cubic'), ms.float32)
        s_pred = mnp.vstack([s_pred, s_pred_k_])
    error = SSE()(s_pred - s_test.T) / SSE()(s_test.T)
    return s_pred, error


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
