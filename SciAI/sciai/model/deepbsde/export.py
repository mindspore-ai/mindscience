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
"""DeepBSDE export model script"""
from mindspore import load_checkpoint, export

from sciai.context import init_project
from sciai.utils import amp2datatype, to_tensor
from src.net import DeepBSDE
from src.config import prepare
from src.equation import get_bsde


def main(args):
    dtype = amp2datatype(args.amp_level)

    bsde = get_bsde(args)
    net = DeepBSDE(args, bsde)
    load_checkpoint(args.load_ckpt_path, net=net)
    net.to_float(dtype)

    dw, x = bsde.sample(args.valid_size)
    dw, x = to_tensor((dw, x), dtype=dtype)
    export(net, dw, x, file_name=f"deepbsde_{args.eqn_name}_{dtype}", file_format=args.file_format)


if __name__ == '__main__':
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
