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

"""deeponet eval"""
import yaml

import mindspore as ms
import numpy as np

from sciai.context import init_project
from sciai.utils import print_log, to_tensor, amp2datatype
from sciai.utils.python_utils import print_time
from src.network import DeepONet, SampleNet
from src.plot import plot_prediction
from src.process import load_data, generate_args


def evaluation(feed_test, feed_test0, loss_net):
    """validation"""
    _, _, y_test = feed_test
    _, _, y_test0 = feed_test0
    x_u_test_ms, x_y_test_ms, y_test_ms = to_tensor(feed_test, ms.float32)
    x_u_test0_ms, x_y_test0_ms, y_test0_ms = to_tensor(feed_test0, ms.float32)
    valid_loss, _ = loss_net.net(x_u_test_ms, x_y_test_ms, y_test_ms)
    test_loss, y_pred = loss_net.net(x_u_test_ms, x_y_test_ms, y_test_ms)
    test_loss0, y_pred0 = loss_net.net(x_u_test0_ms, x_y_test0_ms, y_test0_ms)
    test_err = np.linalg.norm(y_pred.asnumpy() - y_test) / np.linalg.norm(y_test)
    test_err0 = np.linalg.norm(y_pred0.asnumpy() - y_test0) / np.linalg.norm(y_test0)
    print_log(f"Validation loss: {valid_loss},  Test loss: {test_loss},  Test loss0: {test_loss0},  RelErr: "
              f"{test_err},  RelErr0: {test_err0}\n\n")
    return y_pred, y_test


@print_time("eval")
def main(args):
    """main"""
    dtype = amp2datatype(args.amp_level)
    _, feed_test, feed_test0 = load_data(args.load_data_path)
    net = DeepONet(args.layers_u, args.layers_y)
    loss_net = SampleNet(net)
    loss_net.to_float(dtype)
    ms.load_checkpoint(args.load_ckpt_path, loss_net)
    y_pred, y_test = evaluation(feed_test, feed_test0, loss_net)
    if args.save_fig:
        plot_prediction(y_pred, y_test, args)


if __name__ == "__main__":
    with open("config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = generate_args(config_dict)
    init_project(args=args_)
    main(args_)
