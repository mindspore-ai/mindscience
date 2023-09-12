
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

"""label free dnn surrogate eval"""
import numpy as np

from sciai.context import init_project
from sciai.utils import print_log
from sciai.utils.python_utils import print_time

from src.plot import plot_contour, plot_wall_shear, det_test, load_with_default
from src.process import prepare


def evaluate(args, case_idx, data_cfd, data_nn, scale):
    x = data_cfd['x']
    y = data_cfd['y']
    u_cfd = data_cfd['U']
    u_all = data_nn['U']
    print_log("data is loaded successfully")
    u, v, _ = det_test(x, y, scale, case_idx, args)
    w = np.zeros_like(u)
    u_all = np.concatenate([u, v, w], axis=1)
    return u_all, u_cfd, x, y


@print_time("eval")
def main(args):
    args.epochs_val = 90
    std_type = '3sigma'
    # geometry
    data = load_with_default(args, f'aneurysm_scale0005to002_eval0to002mean001{std_type}.npz')
    scale_test = data['scale']

    ns = len(scale_test)
    case_count = [1.0, 151.0, 486.0]

    w_ctl = np.zeros([ns, 1])
    w_ctl_ml = np.zeros([ns, 1])

    for case_idx in case_count:
        # geo_case
        scale = scale_test[int(case_idx - 1)]

        print_log(f'path is: {args.load_data_path}/{case_idx}')
        data_cfd = np.load(f'{args.load_data_path}/{case_idx}CFD_contour.npz')
        data_nn = np.load(f'{args.load_data_path}/{case_idx}NN_contour.npz')

        u_all, u_cfd, x, y = evaluate(args, case_idx, data_cfd, data_nn, scale)
        mse = np.square(np.mean(u_all - u_cfd))
        print_log(f"MSE: {mse}")
        # Contour Comparison
        if args.save_fig:
            plot_contour(args, u_all, u_cfd, x, y, case_idx, scale)
            plot_wall_shear(args, case_idx, w_ctl, w_ctl_ml)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
