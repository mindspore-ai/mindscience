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
"""xpinns eval"""
import numpy as np
import mindspore as ms
from sciai.context import init_project
from sciai.utils import print_log, amp2datatype
from sciai.utils.python_utils import print_time

from src.network import XPINN
from src.plot import plot, PlotElements
from src.process import generate_data, prepare


def evaluate(model, u_exact, x_star1, x_star2, x_star3):
    """evaluate"""
    u_pred1, u_pred2, u_pred3 = model.predict(x_star1, x_star2, x_star3)
    u_pred = np.concatenate([u_pred1, u_pred2, u_pred3])
    error_u_total = np.linalg.norm(np.squeeze(u_exact) - u_pred.flatten(), 2) / np.linalg.norm(np.squeeze(u_exact), 2)
    print_log('Error u_total: ' + str(error_u_total))
    return u_pred


@print_time("eval")
def main(args):
    dtype = amp2datatype(args.amp_level)
    layers1, layers2, layers3 = args.layers1, args.layers2, args.layers3
    x_f1_train, x_f2_train, x_f3_train, x_fi1_train, x_fi2_train, x_star1, x_star2, x_star3, x_ub_train, \
        _, u_exact, _, _, xb, xi1, xi2, yb, yi1, yi2 = generate_data(args.load_data_path, dtype)
    model = XPINN(layers1, layers2, layers3)
    if dtype == ms.float16:
        model.to_float(ms.float16)
    ms.load_checkpoint(args.load_ckpt_path, model)
    u_pred = evaluate(model, u_exact, x_star1, x_star2, x_star3)
    if args.save_fig:
        elements = PlotElements(epochs=args.epochs,
                                x_f_trains=(x_f1_train, x_f2_train, x_f3_train),
                                x_fi_trains=(x_fi1_train, x_fi2_train),
                                x_ub_train=x_ub_train,
                                u=(u_exact, u_pred),
                                x_stars=(x_star1, x_star2, x_star3),
                                x=(xb, xi1, xi2),
                                y=(yb, yi1, yi2),
                                figures_path=args.figures_path)
        plot(elements)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
