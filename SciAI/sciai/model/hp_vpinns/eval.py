
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

"""hp vpinns eval"""
import numpy as np
import mindspore as ms

from sciai.context import init_project
from sciai.utils import datatype2np, amp2datatype, print_log
from sciai.utils.python_utils import print_time
from src.network import VPINN
from src.plot import plot_fig
from src.process import get_data, prepare


@print_time("eval")
def main(args):
    dtype = amp2datatype(args.amp_level)
    np_dtype = datatype2np(dtype)
    f_ext_total, w_quad_train, x_quad_train, x_test, x_u_train, grid, u_test, _ \
        = get_data(args, dtype, np_dtype)

    # Model and Training
    net = VPINN(x_quad_train, w_quad_train, f_ext_total, grid, args, dtype)
    if dtype == ms.float16:
        net.to_float(ms.float16)
    ms.load_checkpoint(args.load_ckpt_path, net)
    u_pred = net.net_u(x_test)

    mse = np.mean(np.square(u_pred.asnumpy() - u_test))
    print_log(f"MSE: {mse}")

    if args.save_fig:
        plt_elems = grid, u_pred, u_test, x_test
        plot_fig(x_quad_train, x_u_train, args, None, plt_elems)


if __name__ == "__main__":
    args_ = prepare()
    init_project(mode=ms.PYNATIVE_MODE, args=args_[0])
    main(*args_)
