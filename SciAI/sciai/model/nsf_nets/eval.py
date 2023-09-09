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
"""nsf nets eval"""
import numpy as np
import mindspore as ms

from sciai.context import init_project
from sciai.utils import print_log, amp2datatype
from sciai.utils.python_utils import print_time
from src.dataset import generate_data, generate_test_data
from src.network import VPNSFnet
from src.process import prepare


def evaluate(lam, net):
    """evaluate"""
    # Test Data
    p_star, u_star, v_star, x_star, y_star = generate_test_data(lam)
    u_pred, v_pred, p_pred = net.neural_net(x_star, y_star)
    # Error
    error_u = np.linalg.norm(u_star - u_pred.asnumpy(), 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred.asnumpy(), 2) / np.linalg.norm(v_star, 2)
    error_p = np.linalg.norm(p_star - p_pred.asnumpy(), 2) / np.linalg.norm(p_star, 2)
    print_log('Error u: ' + str(error_u))
    print_log('Error v: ' + str(error_v))
    print_log('Error p: ' + str(error_p))


@print_time("eval")
def main(args):
    dtype = amp2datatype(args.amp_level)
    lam, ub_train, vb_train, x_train, xb_train, y_train, yb_train = generate_data(args, dtype)
    net = VPNSFnet(xb_train, yb_train, ub_train, vb_train, x_train, y_train, args.layers)
    if dtype == ms.float16:
        net.to_float(ms.float16)
    ms.load_checkpoint(args.load_ckpt_path, net)
    evaluate(lam, net)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
