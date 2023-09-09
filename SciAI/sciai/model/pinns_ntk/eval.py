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
"""pinns ntk eval"""
import numpy as np
import mindspore as ms

from sciai.context import init_project
from sciai.utils import print_log, to_tensor, amp2datatype
from sciai.utils.python_utils import print_time
from src.network import PINN
from src.plot import plot_loss
from src.process import generate_data, u, u_xx, prepare


def evaluate(a, dom_coords, model, dtype):
    """evaluate"""
    nnum = 1000
    x_star = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nnum)[:, None]
    u_star = u(x_star, a)
    r_star = u_xx(x_star, a)
    x_star = to_tensor(x_star, dtype)
    # Predictions
    u_pred = model.predict_u(x_star)
    r_pred = model.predict_r(x_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_r = np.linalg.norm(r_star - r_pred, 2) / np.linalg.norm(r_star, 2)
    print_log('Relative L2 error_u: ' + str(error_u))
    print_log('Relative L2 error_r: ' + str(error_r))
    return x_star.asnumpy(), u_pred, u_star


@print_time("eval")
def main(args):
    dtype = amp2datatype(args.amp_level)
    a = 4
    dom_coords = np.array([[0.0], [1.0]])
    x_r, x_u, y_r, y_u = generate_data(a, dom_coords, args.num, dtype)
    model = PINN(args.layers, x_u, y_u, x_r, y_r, dtype)
    if dtype == ms.float16:
        model.to_float(ms.float16)
    ms.load_checkpoint(args.load_ckpt_path, model)
    if args.save_fig:
        plot_loss(args, model)
    evaluate(a, dom_coords, model, dtype)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
