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
"""Train PINNs"""
from sciai.context import init_project
from sciai.utils import print_time, print_log
from src.utils import prepare
from src.navier_stokes.train_ns import train_navier
from src.schrodinger.train_sch import train_sch


@print_time("train")
def main(args):
    if args.problem == "Schrodinger":
        train_sch(args.epoch, args.lr, args.n0, args.nb, args.nf, args.num_neuron, args.seed, args.load_data_path,
                  args.save_ckpt_path)
    elif args.problem == "NavierStokes":
        train_navier(args.epoch, args.lr, args.batch_size, args.n_train, args.load_data_path, args.noise,
                     args.num_neuron, args.save_ckpt_path, args.seed)
    else:
        print_log(f"problem {args.problem} is not supported, please choose from: [Schrodinger, NavierStokes]")


if __name__ == '__main__':
    args_ = prepare()
    init_project(args=args_)
    main(args_)
