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

"""hfm eval"""
import mindspore as ms
import scipy.io

from sciai.context import init_project
from sciai.utils import amp2datatype
from sciai.utils.python_utils import print_time
from src.process import full_evaluation, prepare_network, prepare_training_data, prepare


@print_time("eval")
def main(args):
    dtype = amp2datatype(args.amp_level)
    data = scipy.io.loadmat(f'{args.load_data_path}/Cylinder2D_flower.mat')
    t_star = data['t_star']  # t_num x 1
    x_star = data['x_star']  # x_num x 1
    y_star = data['y_star']  # x_num x 1
    u_star = data['U_star']  # x_num x t_num
    v_star = data['V_star']  # x_num x t_num
    p_star = data['P_star']  # x_num x t_num
    c_star = data['C_star']  # x_num x t_num
    del data  # data too large

    t_data, x_data, y_data, _, _, _, _ = prepare_training_data(args, c_star, t_star, x_star, y_star)

    model = prepare_network(args, dtype, t_data, x_data, y_data)
    if dtype == ms.float16:
        model.to_float(ms.float16)
    ms.load_checkpoint(args.load_ckpt_path, net=model)
    full_evaluation(args, model, c_star, p_star, u_star, v_star, t_star, x_star, y_star)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
