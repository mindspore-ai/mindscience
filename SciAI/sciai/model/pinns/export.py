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
"""export checkpoint file into mindir models"""
from sciai.context import init_project
from sciai.utils import print_log
from src.utils import prepare
from src.schrodinger.export_sch import export_sch
from src.navier_stokes.export_ns import export_ns


def main(args):
    file_format = 'MINDIR'
    file_name = args.export_file_name
    if args.problem == 'Schrodinger':
        export_sch(args.num_neuron, n0=args.n0, nb=args.nb, nf=args.nf, ck_file=args.load_ckpt_path,
                   export_format=file_format, export_name=file_name)
    elif args.problem == 'NavierStokes':
        export_ns(args.num_neuron, path=args.load_data_path, ck_file=args.load_ckpt_path, batch_size=args.batch_size,
                  export_format=file_format, export_name=file_name)
    else:
        print_log(f'{args.problem} scenario in PINNs is not supported to export for now')


if __name__ == '__main__':
    args_ = prepare()
    init_project(args=args_)
    main(args_)
