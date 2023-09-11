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

"""cpinns eval"""
import numpy as np
import mindspore as ms
from sciai.context import init_project
from sciai.utils import amp2datatype, datatype2np, print_log
from sciai.utils.python_utils import print_time

from src.network import PINN
from src.plot import plot
from src.process import get_data, get_star_inputs, prepare


def evaluate(model, np_dtype, total_dict):
    """evaluate"""
    x_star1, x_star2, x_star3, x_star4, u1_star, u2_star, u3_star, u4_star = get_star_inputs(np_dtype, total_dict)
    u_star = np.concatenate([u1_star, u2_star, u3_star, u4_star])
    u1_pred, u2_pred, u3_pred, u4_pred = model.predict(x_star1, x_star2, x_star3, x_star4)
    u_pred = np.concatenate([u1_pred, u2_pred, u3_pred, u4_pred])
    error_u = np.linalg.norm(u_star.astype(np.float) - u_pred.astype(np.float), 2) \
              / np.linalg.norm(u_star.astype(np.float), 2)
    print_log("error_u: ", error_u)


@print_time("eval")
def main(args):
    dtype = amp2datatype(args.amp_level)
    np_dtype = datatype2np(dtype)
    nu = 0.01 / np.pi  # 0.0025
    nn_layers_total, t_mesh, x_mesh, x_star, u_star, x_interface, total_dict = get_data(args, np_dtype)
    model = PINN(nn_layers_total, nu, x_interface, dtype)
    if dtype == ms.float16:
        model.to_float(ms.float16)
    ms.load_checkpoint(args.load_ckpt_path, model)
    evaluate(model, np_dtype, total_dict)
    if args.save_fig:
        plot(None, t_mesh, x_mesh, x_star, args, model, u_star, x_interface, total_dict)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
