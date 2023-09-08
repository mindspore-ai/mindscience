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
"""Eval"""
from sciai.context import init_project
from sciai.utils import print_time, print_log
from src.utils import prepare
from src.navier_stokes.eval_ns import eval_pinns_navier
from src.schrodinger.eval_sch import eval_pinns_sch


@print_time("eval")
def main(args):
    if args.problem == "Schrodinger":
        eval_pinns_sch(args.load_ckpt_path, args.num_neuron, args.load_data_path)
    elif args.problem == "NavierStokes":
        eval_pinns_navier(args.load_ckpt_path, args.load_data_path, args.num_neuron)
    else:
        print_log(f"problem {args.problem} is not supported, please choose from: [Schrodinger, NavierStokes]")


if __name__ == '__main__':
    args_ = prepare()
    init_project(args=args_)
    main(args_)
