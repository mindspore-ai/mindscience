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

"""enso eval"""
import mindspore as ms

from sciai.context import init_project
from sciai.utils import amp2datatype
from sciai.utils.python_utils import print_time
from src.network import ENSO
from src.plot import evaluate
from src.process import fetch_dataset_nino34, prepare


@print_time("eval")
def main(args):
    dtype = amp2datatype(args.amp_level)
    net = ENSO()
    if dtype == ms.float16:
        net.to_float(ms.float16)
    _, _, ip_var, nino34_var, _, _ = fetch_dataset_nino34(args.load_data_path)
    ms.load_checkpoint(args.load_ckpt_path, net)
    evaluate(args, net, ip_var, nino34_var)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
