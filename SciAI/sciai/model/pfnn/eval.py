# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Evaler"""
import numpy as np
from mindspore import Tensor
from mindspore import load_param_into_net, load_checkpoint

from sciai.context import init_project
from sciai.utils import print_log, print_time, amp2datatype
from src import pfnnmodel
from src.process import prepare, calerror
from data import gendata


@print_time("eval")
def main(args):
    dtype = amp2datatype(args.amp_level)

    bound = np.array(args.bound).reshape(2, 2)
    test_set = gendata.TestSet(bound, args.teset_nx, args.problem_type)
    len_fac = pfnnmodel.LenFac(Tensor(args.bound, dtype).reshape(2, 2), 1)

    netg = pfnnmodel.NetG()
    netf = pfnnmodel.NetF()

    netg.to_float(dtype)
    netf.to_float(dtype)

    load_param_into_net(netg, load_checkpoint(args.load_ckpt_path[0]))
    load_param_into_net(netf, load_checkpoint(args.load_ckpt_path[1]))

    error = calerror(netg, netf, len_fac, test_set)
    print_log(f"The Test Error: {error}")


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
