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

"""train"""
import os.path
import time
import numpy as np
import mindspore as ms
import mindspore.context as context

from sciai.context import init_project
from sciai.utils import print_log, print_time, calc_ckpt_name
from src.utils import prepare
from src.read_var import read_nc
from src.GOMO import GOMOInit, GOMO, read_init


@print_time("train")
def main(args):
    context.set_context(enable_graph_kernel=True)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    variable = read_nc(f"{args.load_data_path}/seamount65_49_21.nc")
    im = args.im
    jm = args.jm
    kb = args.kb
    stencil_width = args.stencil_width

    # variable init
    dx, dy, dz, uab, vab, elb, etb, sb, tb, ub, vb, dt, h, w, wubot, wvbot, vfluxb, utb, vtb, dhb, egb, vfluxf, _, zz, \
        _, _, fsm = read_init(variable, im, jm, kb)

    # define grid and init variable update
    net_init = GOMOInit(im, jm, kb, stencil_width)
    init_res = net_init(dx, dy, dz, uab, vab, elb, etb, sb, tb, ub, vb, h, w, vfluxf, zz, fsm)
    for res_tensor in init_res:
        if isinstance(res_tensor, (list, tuple)):
            for rt in res_tensor:
                rt.data_sync(True)
        else:
            res_tensor.data_sync(True)
    ua, va, el, et, etf, d, dt, _, q2b, q2lb, kh, km, kq, aam, w, q2, q2l, t, s, u, v, cbc, rmean, rho, x_d, y_d, z_d \
        = init_res

    # define GOMO model
    gomo_model = GOMO(im=im, jm=jm, kb=kb, stencil_width=stencil_width, variable=variable, x_d=x_d, y_d=y_d, z_d=z_d,
                      q2b=q2b, q2lb=q2lb, aam=aam, cbc=cbc, rmean=rmean)

    # time step of GOMO Model
    begin_time = time.time()
    for step in range(1, args.epochs + 1):
        (_, etf, ua, uab, va, vab, el, elb, d, u, v, w, kq, km, kh, q2, q2l, tb, t, sb, s, rho, wubot, wvbot, ub, vb,
         egb, etb, dt, dhb, utb, vtb, vfluxb, et, _, _, q2b, q2lb) = gomo_model(etf, ua, uab, va, vab, el, elb,
                                                                                d, u, v, w, kq, km, kh, q2, q2l, tb, t,
                                                                                sb, s, rho, wubot, wvbot, ub, vb, egb,
                                                                                etb, dt, dhb, utb, vtb, vfluxb, et)
        print_log("per step cost time", time.time() - begin_time)
        # save output
        if step % 5 == 0:
            np.save(args.output_path + "/u_" + str(step) + ".npy", u.asnumpy())
            np.save(args.output_path + "/v_" + str(step) + ".npy", v.asnumpy())
            np.save(args.output_path + "/t_" + str(step) + ".npy", t.asnumpy())
            np.save(args.output_path + "/et_" + str(step) + ".npy", et.asnumpy())
        begin_time = time.time()

    if args.save_ckpt:
        ms.save_checkpoint(gomo_model, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
