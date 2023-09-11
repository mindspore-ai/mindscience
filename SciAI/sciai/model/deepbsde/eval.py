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
"""DeepBSDE evaluation script"""
from mindspore import load_checkpoint

from sciai.context import init_project
from sciai.utils import print_time, print_log, amp2datatype
from src.net import DeepBSDE, WithLossCell
from src.config import prepare
from src.equation import get_bsde
from src.eval_utils import apply_eval


@print_time("eval")
def main(args):
    dtype = amp2datatype(args.amp_level)

    bsde = get_bsde(args)
    net = DeepBSDE(args, bsde)
    net_loss = WithLossCell(net)
    load_checkpoint(args.load_ckpt_path, net=net_loss)
    net_loss.to_float(dtype)

    eval_param = {"model": net_loss, "valid_data": bsde.sample(args.valid_size)}
    loss, y_init = apply_eval(eval_param, dtype)
    print_log("eval loss: {}, Y0: {}".format(loss, y_init))


if __name__ == '__main__':
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
